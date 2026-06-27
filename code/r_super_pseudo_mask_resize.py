#!/usr/bin/env python3
# restore_ct_shapes_v2.py · June‑2025
#
# Bring every prediction back to the *original* CT geometry
# – undo the random CropForeground crop via the original masks (if given)
# – otherwise by translation‑only registration from the cropped CT
# – if no crop ever happened, just resample back to the original grid
# – always error if neither masks nor cropped‑CT are available when needed

import argparse, shutil, sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops
import copy
import math
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ← your utils must be on PYTHONPATH
from dataset_conversion.utils import ResampleLabelToRef, reorient_image, ResampleXYZAxis

def verify_saved(args, case_dir: Path) -> bool:
    bid = case_dir.name
    # NEW: preserve subset structure
    out_case = Path(args.output_folder) / case_dir.relative_to(args.input_folder)

    missing = []
    for src in case_dir.rglob("*.nii.gz"):
        rel = src.relative_to(case_dir)
        dst = out_case / rel
        if not dst.exists():
            missing.append(rel)
    return not bool(missing)

def ResampleToSize(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear, size=None):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

    imRefImage = sitk.Image(size, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())

    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage

def is_binary(img: sitk.Image) -> bool:
    arr = sitk.GetArrayViewFromImage(img)
    u = np.unique(arr)
    return np.array_equal(u, [0]) or np.array_equal(u, [0,1])


def translation_from_registration(fixed: sitk.Image,
                                  moving: sitk.Image) -> sitk.Transform:
    tx_init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.TranslationTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=4.0, minStep=0.1,
        numberOfIterations=200, relaxationFactor=0.5
    )
    reg.SetInitialTransform(tx_init, inPlace=False)
    reg.SetInterpolator(sitk.sitkLinear)
    return reg.Execute(fixed, moving)

def resample_like(src: sitk.Image,
                  reference: sitk.Image,
                  transform: sitk.Transform,
                  interp) -> sitk.Image:
    return sitk.Resample(src, reference, transform, interp, 0.0, src.GetPixelID())


def process_case(case_dir: Path):
    bid = case_dir.name
    print(f"→ {bid}", flush=True)

    # NEW: preserve subset structure
    rel_case = case_dir.relative_to(args.input_folder)
    out_case = args.output_folder / rel_case

    if not args.overwrite and verify_saved(args, case_dir):
        print(f"   ✅  already restored case {bid} → skipping", flush=True)
        return

    # 1) load the “native” CT
    ct_src = case_dir / "ct.nii.gz"
    if not ct_src.exists():
        raise ValueError(f"   ⚠️  {bid}: missing ct.nii.gz")
        
    ct_img     = sitk.ReadImage(str(ct_src))
    original_CT = copy.deepcopy(ct_img)  # keep the original CT for later
    dirs = ct_img.GetDirection()
    orienter = sitk.DICOMOrientImageFilter()
    orientation = orienter.GetOrientationFromDirectionCosines(dirs)
    ct_img = reorient_image(ct_img, 'RAI') # we must reorient back to the original at the end
    ct_size    = ct_img.GetSize()
    ct_spacing = ct_img.GetSpacing()

    # 2) prepare output dir
    out_case = args.output_folder / bid
    if out_case.exists():
        shutil.rmtree(out_case)
    out_case.mkdir(parents=True)
    shutil.copy(ct_src, out_case / "ct.nii.gz")
    # also copy any .txt from the original folder
    for txt in case_dir.glob("*.txt"):
        shutil.copy(txt, out_case / txt.name)

    # 3) quick‐copy if *all* preds already match the CT
    all_match = True
    for f in case_dir.rglob("*.nii.gz"):
        if f.name == "ct.nii.gz":
            continue
        img = sitk.ReadImage(str(f))
        if img.GetSize() != ct_size or img.GetSpacing() != ct_spacing:
            all_match = False
            break
    if all_match:
        print("   ➡️  already on native grid – copying entire folder", flush=True)
        for f in case_dir.rglob("*"):
            rel = f.relative_to(case_dir)
            dst = out_case / rel
            if f.is_dir():
                dst.mkdir(exist_ok=True)
            elif f.name != "ct.nii.gz":
                shutil.copy(f, dst)
        print(f"   ✅  done → {out_case}\n", flush=True)
        return

            
    # 6) process each prediction
    tx_crop2full = None
    for src_file in case_dir.rglob("*.nii.gz"):
        if src_file.name == "ct.nii.gz":
            continue
        rel      = src_file.relative_to(case_dir)
        dst_file = out_case / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        pred = sitk.ReadImage(str(src_file))
        print(f'Pred file and shape: {src_file} {pred.GetSize()}', flush=True)
        
        

        # 6a) already aligned?
        if pred.GetSize() == ct_size and pred.GetSpacing() == ct_spacing:
            shutil.copy(src_file, dst_file)
            continue

        interp = sitk.sitkNearestNeighbor if is_binary(pred) else sitk.sitkLinear
        
        # 5) detect whether a real crop ever happened
        #what would be the size of the original CT if we just resized the 1x1x1 cropped CT to the original CT spacing?
        original_spacing = np.array(list(ct_img.GetSpacing()))
        original_size = np.array(list(ct_img.GetSize()), float)
        #img is one of the labels
        final_spacing = np.array(list(pred.GetSpacing()))
        final_size = np.array(list(pred.GetSize()), float)
        metric_size_original = original_size * original_spacing
        metric_size_final = final_size * final_spacing
        
        
        #is it close?
        rel_err = np.abs(metric_size_original - metric_size_final) / metric_size_original
        same_metric_size = np.all(rel_err < 0.04)
        # 6b) if we know no crop ever happened → just resample back
        if same_metric_size:
            out = resample_like(pred, ct_img,
                                sitk.Transform(3, sitk.sitkIdentity),
                                interp)
            #send to original orientation
            if orientation != 'RAI':
                out = reorient_image(out, orientation)
            out.CopyInformation(original_CT)
            w = out.GetSize(); o = out.GetOrigin(); s=out.GetSpacing()
            assert w == original_CT.GetSize(), f"size mismatch {w} vs {original_CT.GetSize()}"
            assert np.allclose(s, original_CT.GetSpacing()), f"spacing mismatch {s}"
            sitk.WriteImage(out, str(dst_file))
            print(f"   🔄  {rel} resampled (no crop)", flush=True)
            continue
        
        
        #if not, we have 2 options: 1- resize and crop. 2- resize, crop and pad

        # 6c) try mask‑based undo if both masks_path & cropped CT exist
        if args.masks_path:
            #check if pred is RAI, it must be
            
            d = pred.GetDirection()
            ori = sitk.DICOMOrientImageFilter()
            ori_pred = orienter.GetOrientationFromDirectionCosines(d)
            if ori_pred != 'RAI':
                raise ValueError(
                    f"❌ {bid}/{rel}: expected RAI orientation for mask, got {ori_pred}"
                )
            
            
            mask_dir = args.masks_path / bid / 'predictions'
            if not mask_dir.exists():
                raise ValueError(f'cannot find masks for {bid} in {args.masks_path}')
            masks = []
            for mp in mask_dir.rglob("*.nii.gz"):
                if 'background' in mp.name:
                    continue
                m_img = sitk.ReadImage(str(mp))
                # ensure orientation
                m_img = reorient_image(m_img, 'RAI')
                m_rs = ResampleXYZAxis(m_img,(1., 1., 1.), sitk.sitkNearestNeighbor)
                masks.append(sitk.GetArrayFromImage(m_rs) > 0)
            if not masks:
                raise ValueError(f'Missing masks for {bid} in {args.masks_path}')
            
            npLab = np.stack(masks, axis=0)
            union = (npLab.sum(0) > 0).astype(np.uint8)  # Foreground mask
            
            
            regions = regionprops(union)
            assert len(regions) == 1
            zz, yy, xx = union.shape
            z_min, y_min, x_min, z_max_excl, y_max_excl, x_max_excl = regions[0].bbox

            # YOUR context_size must be in [z, y, x] order:
            cz, cy, cx = 20, 30, 30

            # compute the same padded‐bbox that CropForeground used
            z0p = max(0, z_min - cz)
            z1p = min(zz,  z_max_excl + cz)
            y0p = max(0, y_min - cy)
            y1p = min(yy,  y_max_excl + cy)
            x0p = max(0, x_min - cx)
            x1p = min(xx,  x_max_excl + cx)

            foreground_size = (z1p - z0p, y1p - y0p, x1p - x0p)
            print(f'mask‑based crop {foreground_size}, pred size {tuple(pred.GetSize()[::-1])}', flush=True)
            
            if tuple(pred.GetSize()[::-1]) != foreground_size:
                #maybe the expected size was too low (<128). In this case, we may have done a padding. If we pad, does the size match?
                if foreground_size[0]>=128 and foreground_size[1]>=128 and foreground_size[2]>=128:
                    raise RuntimeError(
                        f"{bid}/{rel}: mask‑based crop {foreground_size} ≠ pred size "
                        f"{tuple(pred.GetSize()[::-1])}"
                    )
                print('Unpadding 128')
                #we need to crop, to undo the pad.
                z, y, x = foreground_size
                assert len(pred.GetSize()) == 3, 'Expect 3d mask'
                tmp = sitk.GetArrayFromImage(pred)
                if z < 128:
                    diff = int(math.ceil((128. - z) / 2)) 
                    tmp = tmp[diff:z-diff, :, :]
                    #crop 
                if y < 128:
                    diff = int(math.ceil((128. - y) / 2)) 
                    tmp = tmp[:, diff:y-diff, :]
                if x < 128:
                    diff = int(math.ceil((128. - x) / 2)) 
                    tmp = tmp[:, :, diff:x-diff]
                #back to sitk
                new_origin = np.array(pred.GetOrigin()) + np.array([z0p, y0p, x0p]) * np.array(pred.GetSpacing())
                cropped = sitk.GetImageFromArray(tmp)
                cropped.SetOrigin(tuple(new_origin))
                cropped.SetDirection(pred.GetDirection())
                cropped.SetSpacing(pred.GetSpacing())
                pred = cropped
                
                
            #check if now the physical size matches
            final_spacing = np.array(list(pred.GetSpacing()))
            final_size = np.array(list(pred.GetSize()), float)
            metric_size_final = final_size * final_spacing
            rel_err = np.abs(metric_size_original - metric_size_final) / metric_size_original
            same_metric_size = np.all(rel_err < 0.04)
            
            if not same_metric_size:
                #we still need padding
                # pad back out to full CT shape
                print('Uncropping', flush=True)
                arr     = sitk.GetArrayFromImage(pred)
                pad_z   = (z0p, union.shape[0]-z1p)
                pad_y   = (y0p, union.shape[1]-y1p)
                pad_x   = (x0p, union.shape[2]-x1p)
                arr_pad = np.pad(arr, (pad_z, pad_y, pad_x),
                                    mode='constant', constant_values=0)
                expected_full = (   # z,y,x of the cropped CT grid
                    union.shape[0],
                    union.shape[1],
                    union.shape[2]
                )
                if arr_pad.shape != expected_full:
                    raise RuntimeError(
                        f"{bid}/{rel}: after pad {arr_pad.shape} ≠ CT {tuple(reversed(ct_size))}"
                    )
                
                # how many voxels you padded *at the front* along each axis:
                pz0, pz1 = pad_z
                py0, py1 = pad_y
                px0, px1 = pad_x
                new_origin = np.array(pred.GetOrigin()) + np.array([pz0, py0, px0]) * np.array(pred.GetSpacing())
                padded = sitk.GetImageFromArray(arr_pad)
                padded.SetOrigin(tuple(new_origin))
                padded.SetDirection(pred.GetDirection())
                padded.SetSpacing(pred.GetSpacing())
                pred = padded

            
                
            #resample
            out = ResampleToSize(pred, 
                                    space=ct_spacing,
                                    interp=interp,
                                    size=ct_size)
            
            # send to original orientation
            
            if orientation != 'RAI':
                out = reorient_image(out, orientation)
            out.CopyInformation(original_CT)
            #assert same size
            if out.GetSize() != original_CT.GetSize():
                raise RuntimeError(
                    f"{bid}/{rel}: after mask crop+pad {out.GetSize()} ≠ CT {original_CT.GetSize()}"
                )
            w = out.GetSize(); o = out.GetOrigin(); s=out.GetSpacing()
            assert w == original_CT.GetSize(), f"size mismatch {w} vs {original_CT.GetSize()}"
            assert np.allclose(s, original_CT.GetSpacing()), f"spacing mismatch {s}"
            sitk.WriteImage(out, str(dst_file))
            print(f"   ✔️  {rel} restored via masks", flush=True)
            continue


        # compute tx_crop2full once
        if tx_crop2full is None:
            #get cropped CT
            if not args.cropped_cts:
                raise ValueError(
                    f"❌ {bid}/{rel}: no cropped CT available for registration"
                )
            crop_ct = args.cropped_cts / f"{bid}.nii.gz"
            if crop_ct.exists():
                cropped_img = sitk.ReadImage(str(crop_ct))
            else:
                raise ValueError(
                    f"❌ {bid}/{rel}: no cropped CT available for registration"
                )
            ct_on_crop = (resample_like(ct_img, cropped_img,
                                        sitk.Transform(3,sitk.sitkIdentity),
                                        sitk.sitkBSpline)
                            if cropped_img.GetSpacing()!=ct_spacing else ct_img)
            tx_crop2full = translation_from_registration(ct_on_crop, cropped_img)

        restored = resample_like(pred, ct_img, tx_crop2full, interp)
        # send to original orientation
        if orientation != 'RAI':
            restored = reorient_image(restored, orientation)
        restored.CopyInformation(original_CT)
        w = restored.GetSize(); o = restored.GetOrigin(); s = restored.GetSpacing()
        assert w == original_CT.GetSize()
        assert np.allclose(s, original_CT.GetSpacing()), f"spacing mismatch {s}"
        sitk.WriteImage(restored, str(dst_file))
        print(f"   🔄  {rel} restored via registration", flush=True)

    print(f"   ✅  done → {out_case}\n", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Undo crop+resample so all masks align with the original CT."
    )
    p.add_argument("--input_folder",  required=True, type=Path,
                   help="Output of r_super_pseudo_masks.py (per‐case subfolders)")
    p.add_argument("--cropped_cts",   required=False, type=Path,
                   help="Folder of the *cropped* CTs used at inference (ID.nii.gz)")
    p.add_argument("--masks_path",    required=False, type=Path,
                   help="Folder of original pre‑crop masks (one subfolder per ID)")
    p.add_argument("--output_folder", required=True, type=Path)
    p.add_argument("--overwrite", action='store_true',help="Overwrite existing output_folder")
    p.add_argument("--workers", type=int, default=10,
               help="Number of parallel workers (default=1)")
    p.add_argument("--parts", type=int, default=1,
               help="Split the workload into this many parts")
    p.add_argument("--part",  type=int, default=0,
               help="Which part (1‑based) to run")
    args = p.parse_args()

    # Validate optional inputs
    if args.cropped_cts and not args.cropped_cts.exists():
        sys.exit(f"❌ cropped_cts {args.cropped_cts} not found")
    if args.masks_path and not args.masks_path.exists():
        sys.exit(f"❌ masks_path {args.masks_path} not found")

    args.output_folder.mkdir(parents=True, exist_ok=True)


    # Find every case dir that contains a ct.nii.gz (supports subset/ID and ID)
    case_dirs = sorted({p.parent for p in args.input_folder.rglob("ct.nii.gz")})
    if not case_dirs:
        sys.exit("❌ No cases found (looked for */ct.nii.gz under input_folder)")

    # Drop already-done (pass the path, not the name)
    if not args.overwrite:
        case_dirs = [d for d in case_dirs if not verify_saved(args, d)]

    # split into parts and pick the 0‑based slice
    if args.parts > 1:
        n = len(case_dirs)
        base, rem = divmod(n, args.parts)
        chunks, start = [], 0
        for i in range(args.parts):
            size = base + (1 if i < rem else 0)
            chunks.append(case_dirs[start:start+size])
            start += size

        if not (0 <= args.part < args.parts):
            sys.exit(f"❌ --part must be in [0..{args.parts-1}]")
        case_dirs = chunks[args.part]

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as exe:
            list(tqdm(exe.map(process_case, case_dirs),
                    total=len(case_dirs),
                    desc="Restoring cases"))
    else:
        for case_dir in tqdm(case_dirs, desc="Restoring cases"):
            process_case(case_dir)

    print(f"\n🎉  All cases restored → {args.output_folder}", flush=True)