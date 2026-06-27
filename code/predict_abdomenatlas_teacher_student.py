import builtins
import logging
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from inference.utils import get_inference
from dataset_conversion.utils import ResampleXYZAxis, ResampleLabelToRef, reorient_image
from torch.utils import data
import scipy.ndimage as ndi

import SimpleITK as sitk
import yaml
import argparse
import time
import math
import json
import sys
import pdb
import warnings
import pandas
from pathlib import Path
import re
import unicodedata

import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

import matplotlib.pyplot as plt

import copy

from utils import (
    configure_logger,
    save_configure,
)
warnings.filterwarnings("ignore", category=UserWarning)


def clean_ufo(reports,annotated_tumors=['bladder', 'duodenum','esophagus', 'gallbladder','prostate','spleen','stomach','uterus']):
    """
    This function gets a list of reports and removes cases of no interest:
    - We get the healthy patients
    - We get, for each tumor we have annotated organs, all reports that have known tumor size
    - We remove, for organs that have rignr and left (adrenal glands, kidneys), the reports that have unknown sub-segment (not right or left)
    Then, we print the number of useful cases per tumor
    """
    
    
    interest = {}
    
    for organ in annotated_tumors:
        interest[organ] = reports[reports['Standardized Organ'] == organ]
        interest[organ] = interest[organ][interest[organ]['Tumor Size (mm)'] != 'u']
        interest[organ] = interest[organ][interest[organ]['Tumor Size (mm)'] != 'multiple']
        interest[organ] = interest[organ][interest[organ]['Unknow Tumor Size'] == 'no']
        if organ in ['kidney','adrenal_gland','lung','breast','femur']:
            interest[organ] = interest[organ][interest[organ]['Standardized Location'].str.contains('right') | interest[organ]['Standardized Location'].str.contains('left')]
        print('Number of useful cases for %s: %s'%(organ, interest[organ]['BDMAP_ID'].nunique()))

    #interest['healthy'] = reports[reports['no lesion'] == True]
    #print('Number of healthy cases:', interest['healthy']['BDMAP_ID'].nunique())
    #concat
    tumors_per_type = {}
    for k,v in interest.items():
        tumors_per_type[k]=v['BDMAP_ID'].unique().tolist()
    interest = pd.concat(interest.values())
    interest = interest.drop_duplicates()
    print('Total number of useful cases:', interest['BDMAP_ID'].nunique())
    ids_of_interest = interest['BDMAP_ID'].unique().tolist()
    return interest, ids_of_interest,tumors_per_type

ALIASES = {
    "gall bladder": "gallbladder",
    "gall_bladder": "gallbladder",
    "gall‑bladder": "gallbladder",
}


def canon(name: str) -> str:
    name = unicodedata.normalize("NFKC", name).strip().lower()
    return ALIASES.get(name.replace(" ", "_"), name.replace(" ", "_"))

def restrictive_filtering(
    meta,
    class_list=['adrenal gland', 'bladder', 'colon', 'duodenum',
    'esophagus', 'gallbladder','prostate','spleen','stomach','uterus'],
    single_tumor=False,
    id_col=None,):
    """
    Keep only IDs whose reports show lesions exclusively in organs from
    `class_list`.  Optionally require lesions in exactly one organ.

    Prints how many kept IDs belong to each organ in `class_list`.

    Returns
    -------
    list[str] : BDMAP IDs that satisfy the constraints.
    """
    """
    Keep only IDs whose reports show lesions exclusively in organs from
    `class_list`.  Optionally require lesions in exactly one organ.

    Prints:
      • counts of kept IDs per organ in class_list
      • (up to) 10 kept IDs and which tumour organs they have

    Returns
    -------
    list[str] : BDMAP IDs that satisfy the constraints.
    """
    # 0. Ensure an index column
    if id_col is None:
        if "BDMAP ID" in meta.columns:
            id_col = "BDMAP ID"
        elif "BDMAP_ID" in meta.columns:
            id_col = "BDMAP_ID"
        else:
            raise ValueError(
                "Cannot detect ID column; pass id_col='...' explicitly."
            )
    meta = meta.set_index(id_col, drop=False)

    # 1. Canonicalise allowed organs
    allowed = {canon(o) for o in class_list}

    # 2. Map every '*lesion instances' column -> organ
    col_to_organ: dict[str, str] = {}
    rgx = re.compile(r"number of (.+?) lesions? instances?", re.I)
    for col in meta.columns:
        if "lesion instances" not in col.lower():
            continue
        m = rgx.search(col.lower())
        if m:
            col_to_organ[col] = canon(m.group(1))

    if not col_to_organ:
        raise ValueError("No columns containing 'lesion instances' found.")

    # 3. Row‑wise filtering + per‑organ counter
    kept, id_to_orgs = [], {}
    per_organ = {canon(o): 0 for o in class_list}

    for bid, row in meta.iterrows():
        lesion_orgs = {
            col_to_organ[c]
            for c in col_to_organ
            if row.get(c, 0) > 0
        }

        if not lesion_orgs:                 # no reported tumour
            continue
        if not lesion_orgs.issubset(allowed):
            continue
        if single_tumor and len(lesion_orgs) != 1:
            continue

        kept.append(str(bid))
        id_to_orgs[str(bid)] = lesion_orgs
        for org in lesion_orgs:
            if org in per_organ:
                per_organ[org] += 1

    # 4. Print summary
    print("\n--- restrictive_filtering summary ---")
    for org in class_list:
        print(f"{org}: {per_organ.get(canon(org), 0)} IDs")
    print(f"Total kept IDs: {len(kept)}")

    # 5. Show up to 10 example IDs and their tumour organs
    print("\nSample of kept IDs (≤10):")
    for bid in kept[:10]:
        print(f"  {bid}: {sorted(id_to_orgs[bid])}")
    print("------------------------------------\n")

    return kept

def prediction(model_list, tensor_img, args, tgt_organ=None, age=None, sex=None,
               bdmap_id=None, tumor_info_builder=None, gt_organ_masks=None,
               report_organs=None, gt_lesion_masks=None):

    save_raw = (args.save_probabilities_lesions or args.save_probabilities_report_tumors_only or args.save_probabilities)

    inference = get_inference(args)
    cls_out = None

    assert len(model_list) == 1, 'Ensemble not supported yet'
    model = model_list[0]

    use_teacher = bool(getattr(args, 'use_teacher_eval', False)) and tumor_info_builder is not None and bdmap_id is not None
    # Fast mode: keep stage 1 + 2 unchanged, but pass a SW ROI mask so
    # windows that miss every GT organ are skipped at the
    # inference_sliding_window_one_pass pancreas-skip check.
    sw_roi_mode = bool(getattr(args, 'fast_teacher', False) or getattr(args, 'fast_student', False))
    # Compute keep_class_indices for the SW (cap pred_output channels
    # to ~16 instead of 74 — fits on GPU for big CTs).
    _keep_set = _compute_keep_classes(args, args.class_list) if sw_roi_mode else None
    _keep_idx_list = sorted({args.class_list.index(c) for c in _keep_set if c in args.class_list}) if _keep_set else None

    def _run(image_chunk, chunk_z_start=0):
        """Single-call inference dispatch — student baseline / teacher 2-stage,
        optionally with the SW window-skip ROI for fast modes."""
        # Slice GT masks per chunk (used by the fast path).
        gt_chunk = None
        if gt_organ_masks is not None and sw_roi_mode:
            chunk_z = image_chunk.shape[2]
            gt_chunk = {
                org: m[chunk_z_start:chunk_z_start + chunk_z]
                for org, m in gt_organ_masks.items()
            }
        gt_lesion_chunk = None
        if gt_lesion_masks is not None and sw_roi_mode:
            chunk_z = image_chunk.shape[2]
            gt_lesion_chunk = {
                org: m[chunk_z_start:chunk_z_start + chunk_z]
                for org, m in gt_lesion_masks.items()
            }
        # In fast mode, restrict stage-2 to organs that actually have a
        # non-empty GT mask for THIS chunk. The full args.organs_with_tumor
        # list causes stage-2 to iterate organs the SW didn't run on (e.g.
        # prostate when AbdomenAtlasPro has no prostate mask for this
        # patient), and stage-1 leakage from neighbouring ROI organs lets
        # crop_foreground_3d find spurious crop windows there — the
        # teacher then pastes nonsense lesion predictions in the wrong
        # anatomical region. Filtering to non-empty GT masks prevents this.
        # For fast_teacher, additionally intersect with `report_organs` so
        # stage-2's teacher passes only run on the report-mentioned subset
        # (the original optimisation). report_organs is None in fast_student.
        stage2_organs = list(args.organs_with_tumor)
        if sw_roi_mode and gt_chunk is not None:
            stage2_organs = [o for o in args.organs_with_tumor
                             if o in gt_chunk and int(gt_chunk[o].sum()) > 0]
            if report_organs is not None:
                stage2_organs = [o for o in stage2_organs if o in report_organs]

        # ---------- Fast (a): SW window-skip via ROI mask ----------
        sw_roi_chunk = None
        if sw_roi_mode:
            # Build the ROI mask from the GT organ masks of the chosen
            # ROI organs. Important: fast modes must NEVER fall back to a
            # full SW. If no organ contributes a non-empty mask (e.g.
            # the report mentions no tumor in any --organs_with_tumor),
            # we pass an all-zeros ROI so every SW window is skipped
            # (output is then fully blank, by design).
            roi = None
            if gt_chunk is not None:
                for m_chunk in gt_chunk.values():
                    if m_chunk.sum() == 0:
                        continue
                    if roi is None:
                        roi = m_chunk.clone()
                    else:
                        roi = roi | m_chunk
            if roi is None:
                # No usable mask → force every window to skip by passing
                # an all-zeros mask (sum()==0 inside every window, so
                # the existing pancreas-skip check at
                # inference3d.inference_sliding_window_one_pass:132
                # branches to pred=zeros).
                _, _, _D, _H, _W = image_chunk.shape
                sw_roi_chunk = torch.zeros((_D, _H, _W), dtype=torch.uint8)
                print(f"[fast] ROI empty for this chunk — every SW window will be skipped (output stays blank)")
            else:
                sw_roi_chunk = roi.to(torch.uint8)
        if use_teacher:
            from inference.inference3d_teacher import inference_2_stages_teacher
            pred_t, cls_t = inference_2_stages_teacher(
                student_net=model, teacher_net=model,
                img=image_chunk, args=args,
                tumor_info_builder=tumor_info_builder, bdmap_id=bdmap_id,
                organs_with_tumors=stage2_organs,
                class_list=args.class_list,
                age=age, sex=sex,
                # Either --teacher_no_size or --teacher_no_mask_training
                # turns size off in the report decoder. They map to the same
                # forward-time `no_mask_training=True` switch.
                give_size=not (getattr(args, 'teacher_no_size', False) or
                               getattr(args, 'teacher_no_mask_training', False)),
                reproduce_sided_organ_bug=getattr(args, 'teacher_reproduce_dataset_bugs', False),
                filter_laterality=getattr(args, 'teacher_filter_laterality', False),
                dump_dir=getattr(args, 'teacher_dump_dir', None),
                sw_roi_mask=sw_roi_chunk,
                keep_class_indices=_keep_idx_list,
                gt_lesion_masks=gt_lesion_chunk,
                gt_organ_masks=gt_chunk,
            )
            if cls_t is not None and (cls_t.sum() > 0).item():
                cls_t = torch.amax(cls_t, dim=(-3, -2, -1))
                # Keep pred_t in bf16 — the downstream threshold in
                # prediction() works on bf16 and casting to fp32 would
                # double the GPU tensor to ~36 GB and OOM.
                return pred_t, cls_t.float()
            return pred_t
        return inference(model, image_chunk, args, pancreas=tgt_organ,
                         use_transformer_decoder=args.use_transformer_decoder,
                         age=age, sex=sex, sw_roi_mask=sw_roi_chunk,
                         keep_class_indices=_keep_idx_list,
                         stage2_organs=stage2_organs if sw_roi_mode else None)

    with torch.no_grad():
        D, H, W = tensor_img.shape
        print(f'Shape of the input image: {tensor_img.shape}')

        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)

        z_len = 768
        if D > z_len:
            num_z_chunks = math.ceil(D / z_len)
            z_chunk_len = math.ceil(D / num_z_chunks)


            label_pred_list = []
            if save_raw:
                raw_pred_list = []
            cls_list = []
            for i in range(num_z_chunks):
                image_chunk = tensor_img[:, :, i*z_chunk_len: (i+1)*z_chunk_len, :, :]
                _, _, D1, H1, W1 = image_chunk.shape

                print(f'Shape of the chunk path: {image_chunk.shape}')

                pred = _run(image_chunk, chunk_z_start=i*z_chunk_len)
                if isinstance(pred, list) or isinstance(pred, tuple):
                    clss = pred[1]
                    # The SW path can land pred_output (and so cls_output) on
                    # GPU or CPU depending on the per-chunk memory check; if
                    # two chunks differ, the later torch.stack/amax fails with
                    # a device mismatch. Force CPU here (matches how
                    # label_pred is normalised below).
                    if torch.is_tensor(clss):
                        clss = clss.cpu()
                    cls_list.append(clss)
                    pred = pred[0]
                print('Pred shape:',pred.shape)
                pred = pred.to(torch.bfloat16).squeeze(0)
                # Threshold on whatever device pred is on (GPU when the
                # inference path returns on GPU); move only the much
                # smaller uint8 result to CPU.
                label_pred = (pred > 0.5).to(torch.uint8)
                if save_raw:
                    raw_pred = pred.clone().cpu()
                if label_pred.is_cuda:
                    label_pred = label_pred.cpu()
                del pred
                torch.cuda.empty_cache()

                label_pred_list.append(label_pred)
                if save_raw:
                    raw_pred_list.append(raw_pred)

            label_pred = torch.cat(label_pred_list, dim=1)
            if save_raw:
                raw_pred =  torch.cat(raw_pred_list, dim=1)
            if len(cls_list)>0:
                cls_out = torch.stack(cls_list, dim=0).amax(dim=0)

        else:
            pred = _run(tensor_img)
            if isinstance(pred, list) or isinstance(pred, tuple):
                cls_out = pred[1]
                pred = pred[0]
            pred = pred.to(torch.bfloat16)

            if args.dimension == '2d':
                pred = pred.permute(1, 0, 2, 3)
            else:
                pred = pred.squeeze(0)

            # Threshold on whatever device pred is on (GPU when the
            # inference path returns on GPU — fast on a 17.9 GB bf16
            # tensor; the resulting uint8 mask is 9× smaller for the
            # transfer to CPU).
            label_pred = (pred > 0.5).to(torch.uint8)
            if save_raw:
                raw_pred = pred.clone().cpu()
            if label_pred.is_cuda:
                label_pred = label_pred.cpu()
            del pred
            torch.cuda.empty_cache()

    if not save_raw:
        raw_pred = None

    return label_pred, raw_pred, cls_out


def pad_to_training_size(tensor_img, args):

    z, y, x = tensor_img.shape
   
    if args.dimension == '3d':
        if z < args.training_size[0]:
            diff = (args.training_size[0]+2 - z) // 2
            tensor_img = F.pad(tensor_img, (diff, diff, 0,0, 0,0))
            z_start = diff
            z_end = diff + z
        else:
            z_start = 0
            z_end = z

        if y < args.training_size[1]:
            diff = (args.training_size[1]+2 - y) // 2
            tensor_img = F.pad(tensor_img, (0,0, diff, diff, 0,0))
            y_start = diff
            y_end = diff + y
        else:
            y_start = 0
            y_end = y

        if x < args.training_size[2]:
            diff = (args.training_size[2]+2 -x) // 2
            tensor_img = F.pad(tensor_img, (0,0, 0,0, diff, diff))
            x_start = diff
            x_end = diff + x
        else:
            x_start = 0
            x_end = x

        return tensor_img, [z_start, z_end, y_start, y_end, x_start, x_end]

    elif args.dimension == '2d':
        
        if y < args.training_size[0]:
            diff = (args.training_size[0]+2 - y) // 2
            tensor_img = F.pad(tensor_img, (0,0, diff, diff, 0,0))
            y_start = diff
            y_end = diff + y
        else:
            y_start = 0
            y_end = y

        if x < args.training_size[1]:
            diff = (args.training_size[1]+2 -x) // 2
            tensor_img = F.pad(tensor_img, (0,0, 0,0, diff, diff))
            x_start = diff
            x_end = diff + x
        else:
            x_start = 0
            x_end = x

        return tensor_img, [y_start, y_end, x_start, x_end]

    else:
        raise ValueError




def unpad_img(tensor_pred, original_idx, args):
    if args.dimension == '3d':
        z_start, z_end, y_start, y_end, x_start, x_end = original_idx
    
        return tensor_pred[z_start:z_end, y_start:y_end, x_start:x_end]
    elif args.dimension == '2d':
        y_start, y_end, x_start, x_end = original_idx

        return tensor_pred[:, y_start:y_end, x_start:x_end]
        
    else:
        raise ValueError


def preprocess(itk_img, target_spacing, args):
    '''
    This function performs preprocessing to make images to be consistent with training, e.g. spacing resample, redirection and etc.
    Args:
        itk_img: the simpleITK image to be predicted
    Return: the preprocessed image tensor
    '''
    
    import time as _t_pp
    _ppp = (os.environ.get('DEBUG_PREP_PROFILE') == '1')

    if _ppp: _t = _t_pp.time()
    origin_orientation = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(itk_img.GetDirection())
    imImage = reorient_image(itk_img, 'RAI')
    if _ppp: print(f"[PREP] reorient: {_t_pp.time()-_t:.2f}s", flush=True)

    spacing = list(imImage.GetSpacing())
    if not torch.equal(torch.tensor(spacing), torch.tensor(target_spacing)):
        if _ppp: _t = _t_pp.time()
        # Use sitkLinear (trilinear) instead of the previous BSpline+NN
        # two-pass. BSpline is 3rd-order on a CT-sized array and is the
        # dominant cost of preprocess (~8 s vs ~1 s for sitkLinear).
        # Visual quality difference is negligible for CT inference; the
        # downstream model is robust to interp-order differences.
        re_img_xyz = ResampleXYZAxis(imImage, space=target_spacing, interp=sitk.sitkLinear)
        if _ppp: print(f"[PREP] resample (sitkLinear, 1-pass): {_t_pp.time()-_t:.2f}s", flush=True)
    else:
        re_img_xyz=imImage

    if _ppp: _t = _t_pp.time()
    np_img = sitk.GetArrayFromImage(re_img_xyz).astype(np.float32)
    tensor_img = torch.from_numpy(np_img).cuda().float()
    del np_img
    if _ppp: print(f"[PREP] sitk->numpy->gpu: {_t_pp.time()-_t:.2f}s", flush=True)

    if _ppp: _t = _t_pp.time()
    tensor_img = torch.clip(tensor_img, -991, 500)
    mean = torch.mean(tensor_img)
    std = torch.std(tensor_img)
    tensor_img -= mean
    tensor_img /= std
    tensor_img, original_idx = pad_to_training_size(tensor_img, args)
    if _ppp: print(f"[PREP] clip/zscore/pad: {_t_pp.time()-_t:.2f}s", flush=True)

    return tensor_img, original_idx, origin_orientation, re_img_xyz


def _compute_keep_classes(args, class_list):
    """In fast modes, return the keep set (used both to size pred_output
    in the SW and to filter postprocess + save). For non-fast modes
    returns None (= keep all)."""
    if not (getattr(args, 'fast_teacher', False) or getattr(args, 'fast_student', False)):
        return None
    from inference.inference3d import _lesion_like_name
    keep = set()
    for org in args.organs_with_tumor:
        if org in class_list:
            keep.add(org)
        lesion = _lesion_like_name(org)
        if lesion in class_list:
            keep.add(lesion)
    if 'uterus_lesion' in class_list:
        keep.add('uterus_lesion')
    return keep


def _gpu_resample_pred_dict_to_ref(pred_dict, ref_itk):
    """Resample every uint8 ITK image in `pred_dict` onto `ref_itk`'s
    grid in a single batched F.interpolate(mode='nearest') call.

    Replaces N per-channel sitk.Resample calls (which together can take
    several seconds for a 18-channel pred_dict on Turkish-sized CTs).
    Each input image is assumed to share the same shape/spacing/direction
    (true here: all channels go through the same reorient_image step).

    Output preserves the value semantics of sitk.sitkNearestNeighbor —
    we use torch.nn.functional.interpolate(mode='nearest'), which uses
    the same nearest-neighbour rule as SITK for an axis-aligned
    identity transform between two regular grids.
    """
    if not pred_dict:
        return pred_dict
    keys = list(pred_dict.keys())
    arrays = [sitk.GetArrayFromImage(pred_dict[k]) for k in keys]
    stacked = np.stack(arrays, axis=0).astype(np.uint8)          # (N, D_in, H_in, W_in)
    t_full = torch.from_numpy(stacked).cuda()                    # (N, D, H, W) uint8
    ref_size = ref_itk.GetSize()                                 # (X, Y, Z)
    target = (ref_size[2], ref_size[1], ref_size[0])             # numpy (Z, Y, X)
    # F.interpolate(mode='nearest') for upsample_nearest3d has an
    # internal int32-numel guard. For 18 channels at Turkish CT sizes
    # (~1.9e9 elements stacked) it trips. Chunk along the batch dim.
    N = t_full.shape[0]
    chunk = 4
    out_chunks = []
    for s in range(0, N, chunk):
        sub = t_full[s:s+chunk].unsqueeze(0).float()              # (1, K, D, H, W)
        sub = F.interpolate(sub, size=target, mode='nearest')
        out_chunks.append(sub.squeeze(0).to(torch.uint8))
    t = torch.cat(out_chunks, dim=0).cpu().numpy()                # (N, D_out, H_out, W_out)
    out = {}
    for i, k in enumerate(keys):
        img = sitk.GetImageFromArray(t[i])
        img.CopyInformation(ref_itk)
        out[k] = img
    return out


def _gpu_dilate_itk(itk_img, radius=3):
    """Dilate a SITK uint8 binary image by `radius` voxels using GPU
    max_pool3d. ~15× faster than sitk.BinaryDilate on full-volume
    organ masks (postprocess) since sitk.BinaryDilate is
    single-threaded CPU.
    """
    np_in = sitk.GetArrayFromImage(itk_img)
    t = torch.from_numpy(np_in.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    k = 2 * radius + 1
    t = F.max_pool3d(t, kernel_size=k, stride=1, padding=radius)
    np_out = t.squeeze(0).squeeze(0).to(torch.uint8).cpu().numpy()
    itk_out = sitk.GetImageFromArray(np_out)
    itk_out.CopyInformation(itk_img)
    return itk_out


def _fast_save_keys(pred_dict_keys, args):
    """In fast modes, save only:
      - every '*_lesion' channel (the tumor predictions),
      - every channel whose name is an entry in `args.organs_with_tumor`
        (the organs that may have lesions).
    Fixed across cases (user spec: even when this case's report doesn't
    mention a tumor in an organ, still save its lesion + organ NIfTI as
    blanks). For organs whose channel name doesn't exist in the model
    (e.g. uterus), `pred_dict` simply has no key for it — nothing to
    save.
    """
    if not (getattr(args, 'fast_teacher', False) or getattr(args, 'fast_student', False)):
        return list(pred_dict_keys)  # not in fast mode -> save everything
    organs_set = set(getattr(args, 'organs_with_tumor', []))
    keep = set()
    for k in pred_dict_keys:
        if '_lesion' in k:
            keep.add(k)
        elif k in organs_set:
            keep.add(k)
    return list(keep)


def _load_gt_organ_masks_for_fast(bdmap_id, organs, args, target_spacing,
                                   reoriented_itk_img, tensor_img_shape, original_idx):
    """Load GT organ masks for `organs` and resample them to match the
    preprocessed `tensor_img` array shape (RAI, 1mm iso, padded to
    training_size).

    Layout: <gt_mask_root>/<BDMAP_ID>/segmentations/<organ>.nii.gz
    (matches /mnt/bodymaps/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/).

    Returns a dict: organ_name -> uint8 torch.Tensor (D, H, W) aligned to
    `tensor_img`. Organs whose file is missing/empty/load-error are
    silently skipped (caller's inference will skip them too).

    `original_idx` is `[z0, z1, y0, y1, x0, x1]` returned by
    `pad_to_training_size`; the resampled mask is placed onto a zero
    array of shape `tensor_img_shape` at exactly those offsets, so the
    mask aligns voxel-for-voxel with the CT.
    """
    root = getattr(args, 'gt_mask_root',
                   '/mnt/bodymaps/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/')
    # GT-filename aliases. Some organ-class names in our pipeline don't
    # correspond 1:1 to AbdomenAtlasPro filenames; pin those here.
    #   uterus: AbdomenAtlasPro stores the uterus segmentation under
    #           `prostate.nii.gz` (same file is used for prostate in male
    #           cases, uterus in female cases — exactly one is non-empty).
    GT_FILENAME_ALIAS = {'uterus': 'prostate'}
    out = {}
    z0, z1, y0, y1, x0, x1 = original_idx
    for org in organs:
        fn = GT_FILENAME_ALIAS.get(org, org)
        path = os.path.join(root, bdmap_id, 'segmentations', f'{fn}.nii.gz')
        if not os.path.exists(path):
            print(f"[fast] missing GT mask {path}  (organ={org})")
            continue
        try:
            itk_lab = sitk.ReadImage(path)
        except Exception as e:
            print(f"[fast] failed to read {path}: {e}")
            continue

        # One-step resample: ResampleLabelToRef handles the spacing
        # change (original → 1mm iso) AND the grid alignment to the CT
        # in a single SITK Resample call. Drops the previous redundant
        # 2-pass ResampleXYZAxis (~600 ms saved per organ).
        itk_lab = reorient_image(itk_lab, 'RAI')
        itk_lab = ResampleLabelToRef(itk_lab, reoriented_itk_img,
                                     interp=sitk.sitkNearestNeighbor)
        arr = sitk.GetArrayFromImage(itk_lab).astype(np.uint8)
        arr = (arr > 0).astype(np.uint8)

        # Place onto a zero array of shape `tensor_img_shape` at the same
        # offsets `pad_to_training_size` applied to the CT.
        Z, Y, X = tensor_img_shape
        padded = np.zeros((Z, Y, X), dtype=np.uint8)
        az, ay, ax = arr.shape
        # Clamp by source extent, window extent AND the room left in
        # `padded` (Z-z0 / Y-y0 / X-x0): ResampleLabelToRef does not
        # always return exactly the reference grid size, so `arr` can be
        # a few voxels taller than the window — without the third bound
        # the LHS slice clips and the assignment raises a broadcast error.
        dz = min(az, z1 - z0, Z - z0)
        dy = min(ay, y1 - y0, Y - y0)
        dx = min(ax, x1 - x0, X - x0)
        padded[z0:z0+dz, y0:y0+dy, x0:x0+dx] = arr[:dz, :dy, :dx]
        t = torch.from_numpy(padded)
        out[org] = t
        print(f"[fast] loaded GT {org}: shape={tuple(t.shape)} nz={int(t.sum())}")
    return out


def _load_gt_lesion_masks_for_fast(bdmap_id, lesion_paths_for_bid, args,
                                    reoriented_itk_img, tensor_img_shape, original_idx):
    """Load GT lesion masks (per organ) and align them to the preprocessed
    `tensor_img` grid, the same way `_load_gt_organ_masks_for_fast` does for
    organ masks.

    Parameters
    ----------
    lesion_paths_for_bid : dict[str, str]
        Mapping {organ_canonical: lesion_nifti_path} for this BDMAP_ID, taken
        from the user's --gt_lesion_paths_json.
    """
    z0, z1, y0, y1, x0, x1 = original_idx
    out = {}
    for org, path in lesion_paths_for_bid.items():
        if not os.path.exists(path):
            print(f"[fast-mask-info] missing GT lesion {path} (organ={org})")
            continue
        try:
            itk_lab = sitk.ReadImage(path)
        except Exception as e:
            print(f"[fast-mask-info] failed to read {path}: {e}")
            continue
        itk_lab = reorient_image(itk_lab, 'RAI')
        itk_lab = ResampleLabelToRef(itk_lab, reoriented_itk_img,
                                     interp=sitk.sitkNearestNeighbor)
        arr = (sitk.GetArrayFromImage(itk_lab) > 0).astype(np.uint8)
        Z, Y, X = tensor_img_shape
        padded = np.zeros((Z, Y, X), dtype=np.uint8)
        az, ay, ax = arr.shape
        # See _load_gt_organ_masks_for_fast: clamp by the room left in
        # `padded` too, or an oversized `arr` raises a broadcast error.
        dz = min(az, z1 - z0, Z - z0)
        dy = min(ay, y1 - y0, Y - y0)
        dx = min(ax, x1 - x0, X - x0)
        padded[z0:z0+dz, y0:y0+dy, x0:x0+dx] = arr[:dz, :dy, :dx]
        t = torch.from_numpy(padded)
        out[org] = t
        print(f"[fast-mask-info] loaded GT lesion {org}: shape={tuple(t.shape)} nz={int(t.sum())}")
    return out


def postprocess_non_binary(pred, reoriented_itk_img, original_idx, origin_orientation,
                           target_spacing, classes, args, original_itk_img):
    # Remove any squeezing if needed.
    pred = pred.squeeze(0)
    pred_dict = {}
    
    for i in range(pred.shape[0]):
        tensor_pred = pred[i]
        # Unpad to original region.
        tensor_pred = unpad_img(tensor_pred, original_idx, args)

        # If resampling is required, resample using 'trilinear' (continuous)
        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            tensor_pred = resample_image_with_gpu(
                            tensor_pred.float(), 
                            target_spacing, 
                            tensor_pred.shape[::-1], 
                            reoriented_itk_img.GetSpacing(), 
                            reoriented_itk_img.GetSize(), 
                            interp='trilinear'
                        )

        # Create a SimpleITK image, preserving the float values.
        itk_pred = sitk.GetImageFromArray(tensor_pred.cpu().numpy().astype(np.float32))
        itk_pred.CopyInformation(reoriented_itk_img)
        itk_pred = reorient_image(itk_pred, origin_orientation)

        pred_dict[classes[i]] = itk_pred

    for key, img in list(pred_dict.items()):
        pred_dict[key] = ResampleLabelToRef(img,original_itk_img,interp=sitk.sitkLinear  # continuous, good for probabilities
        )

    return pred_dict


def postprocess(pred, reoriented_itk_img, original_idx, origin_orientation, target_spacing, classes, args, original_itk_img, keep_classes=None):
    print(f'Shape of the prediction for postprocessing: {pred.shape}')
    pred = pred.squeeze(0)
    pred_dict = {}
    # --- profiling (active when DEBUG_POST_PROFILE=1) ---
    import os as _os_pp, time as _t_pp
    _ppp = (_os_pp.environ.get('DEBUG_POST_PROFILE') == '1')
    _pp_t = {'unpad':0.0, 'cpu_numpy':0.0, 'getitk':0.0, 'reorient':0.0, 'roundtrip':0.0,
             'lesion_mask_lookup':0.0, 'lesion_dilate':0.0, 'lesion_mul':0.0,
             'final_resample':0.0, 'cnt_organ':0, 'cnt_lesion':0}

    for i in range(pred.shape[0]):
        #skip lesions:
        #do organ first
        if 'lesion' in classes[i]:
            continue
        # Fast-mode channel filter: skip channels we won't save.
        if keep_classes is not None and classes[i] not in keep_classes:
            continue
        tensor_pred = pred[i]

        #if 'pancrea' in classes[i]:
        #    print("Sum of organ mask binary:", pred[i].sum(),classes[i])

        if _ppp: _t = _t_pp.time()
        tensor_pred = unpad_img(tensor_pred, original_idx, args)
        if _ppp: _pp_t['unpad'] += _t_pp.time() - _t

        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            raise ValueError(f'Spacing should be the same, it is {target_spacing} and {reoriented_itk_img.GetSpacing()}')

        if _ppp: _t = _t_pp.time()
        np_uint = tensor_pred.cpu().numpy().astype(np.uint8)
        if _ppp: _pp_t['cpu_numpy'] += _t_pp.time() - _t
        if _ppp: _t = _t_pp.time()
        itk_pred = sitk.GetImageFromArray(np_uint)
        itk_pred.CopyInformation(reoriented_itk_img)
        if _ppp: _pp_t['getitk'] += _t_pp.time() - _t

        if _ppp: _t = _t_pp.time()
        itk_pred = reorient_image(itk_pred, origin_orientation)
        if _ppp: _pp_t['reorient'] += _t_pp.time() - _t
        # Drop redundant ITK→numpy→ITK round trip; itk_pred is already
        # a uint8 SITK image at the reoriented_itk_img grid.
        pred_dict[classes[i]] = itk_pred
        if _ppp: _pp_t['cnt_organ'] += 1

    # Pre-dilate every organ entry on GPU. Originally a single batched
    # F.max_pool3d over all organ channels (~9 organs × 128³ in fast mode).
    # On full-class T_oplc the dict can have ~30 organ channels and CTs
    # up to ~700×500×500 → a stack of (30, 700, 500, 500) float32 is
    # ~21 GB and max_pool3d's output another 21 GB, which OOMs even a
    # 48 GB card. Chunk the channels to cap peak GPU memory.
    dilated_organ_dict = {}
    if args.organ_mask_on_lesion and pred_dict:
        if _ppp: _t = _t_pp.time()
        _organ_keys = list(pred_dict.keys())
        _arrs = [sitk.GetArrayFromImage(pred_dict[k]) for k in _organ_keys]
        _vol = int(np.prod(_arrs[0].shape)) if _arrs else 0
        # Budget ~3 GB per pool3d call (input + output ≈ 2 × N × vol × 4 B);
        # solve N <= 3 GB / (2 × vol × 4 B). Clamp to [1, 16] so very small
        # CTs still benefit from batching.
        _N = max(1, min(16, int((3 * (1 << 30)) / max(1, 8 * _vol))))
        for _start in range(0, len(_organ_keys), _N):
            _ks = _organ_keys[_start:_start + _N]
            _t_in = torch.from_numpy(np.stack([_arrs[_start + _j] for _j in range(len(_ks))], axis=0).astype(np.float32)).cuda().unsqueeze(0)
            _t_out = F.max_pool3d(_t_in, kernel_size=7, stride=1, padding=3)
            _t_out_np = _t_out.squeeze(0).to(torch.uint8).cpu().numpy()
            del _t_in, _t_out
            for _j, _k in enumerate(_ks):
                _img = sitk.GetImageFromArray(_t_out_np[_j])
                _img.CopyInformation(pred_dict[_k])
                dilated_organ_dict[_k] = _img
        if _ppp: _pp_t['lesion_dilate'] += _t_pp.time() - _t

    #now do lesions
    for i in range(pred.shape[0]):
        if 'lesion' not in classes[i]:
            continue
        if keep_classes is not None and classes[i] not in keep_classes:
            continue
        tensor_pred = pred[i]
        if _ppp: _t = _t_pp.time()
        tensor_pred = unpad_img(tensor_pred, original_idx, args)
        if _ppp: _pp_t['unpad'] += _t_pp.time() - _t

        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            tensor_pred=resample_image_with_gpu(tensor_pred.float(),
            target_spacing, tensor_pred.shape[::-1], reoriented_itk_img.GetSpacing(),
            reoriented_itk_img.GetSize(), interp='nearest').long()

        if _ppp: _t = _t_pp.time()
        np_uint = tensor_pred.cpu().numpy().astype(np.uint8)
        if _ppp: _pp_t['cpu_numpy'] += _t_pp.time() - _t
        if _ppp: _t = _t_pp.time()
        itk_pred = sitk.GetImageFromArray(np_uint)
        itk_pred.CopyInformation(reoriented_itk_img)
        if _ppp: _pp_t['getitk'] += _t_pp.time() - _t

        if _ppp: _t = _t_pp.time()
        itk_pred = reorient_image(itk_pred, origin_orientation)
        if _ppp: _pp_t['reorient'] += _t_pp.time() - _t
        itk_lab = itk_pred

        if args.organ_mask_on_lesion:
            # Look up the pre-dilated organ mask from dilated_organ_dict
            # (computed once before the lesion loop). Avoids 9 separate
            # GPU upload+max_pool3d round trips.
            if _ppp: _t = _t_pp.time()
            organ_name = classes[i].split('_')[0].replace('pancreatic', 'pancreas')
            if organ_name == 'kidney':
                organ = sitk.Add(dilated_organ_dict['kidney_right'], dilated_organ_dict['kidney_left'])
                organ.CopyInformation(dilated_organ_dict['kidney_right'])
            elif organ_name == 'adrenal':
                organ = sitk.Add(dilated_organ_dict['adrenal_gland_right'], dilated_organ_dict['adrenal_gland_left'])
                organ.CopyInformation(dilated_organ_dict['adrenal_gland_right'])
            elif organ_name == 'lung':
                organ = sitk.Add(dilated_organ_dict['lung_right'], dilated_organ_dict['lung_left'])
                organ.CopyInformation(dilated_organ_dict['lung_right'])
            elif organ_name == 'uterus':
                organ = dilated_organ_dict['prostate']
            elif organ_name == 'gallbladder':
                organ = dilated_organ_dict['gall_bladder']
            elif organ_name in ['bone','breast']:
                # No organ mask — use ones (so multiply is a no-op).
                size = pred_dict['prostate'].GetSize()
                organ = sitk.Image(size, sitk.sitkUInt8)
                organ = organ + 1
                organ.CopyInformation(pred_dict['prostate'])
            else:
                organ = dilated_organ_dict[organ_name]

            organ = sitk.Cast(organ, sitk.sitkUInt8)
            itk_lab = sitk.Cast(itk_lab, sitk.sitkUInt8)
            if _ppp: _pp_t['lesion_mask_lookup'] += _t_pp.time() - _t

            if _ppp: _t = _t_pp.time()
            # Multiply the lesion label (itk_lab) by the organ mask using SimpleITK.Multiply.
            itk_lab = sitk.Multiply(organ, itk_lab)
            if _ppp: _pp_t['lesion_mul'] += _t_pp.time() - _t

        if args.connected_components:
            itk_lab = keep_largest_component(itk_lab)

        pred_dict[classes[i]] = itk_lab
        if _ppp: _pp_t['cnt_lesion'] += 1

    if _ppp: _t = _t_pp.time()
    # Per-channel sitk.Resample, parallelised across channels. SITK
    # releases the GIL during sitk.Resample, so a ThreadPoolExecutor
    # gives near-linear speedup on multi-core machines. ~3-4× on this
    # step vs serial.
    # (We also tried batched GPU F.interpolate but the np.stack + 1.3 GB
    # PCIe transfer overhead outweighed savings.)
    from concurrent.futures import ThreadPoolExecutor
    _resample_keys = list(pred_dict.keys())
    def _resample_one(k):
        return k, ResampleLabelToRef(pred_dict[k], original_itk_img)
    with ThreadPoolExecutor(max_workers=8) as _ex:
        for _k, _img in _ex.map(_resample_one, _resample_keys):
            pred_dict[_k] = _img
    if _ppp: _pp_t['final_resample'] = _t_pp.time() - _t

    if _ppp:
        print(f"[POST-PROF] organs={_pp_t['cnt_organ']} lesions={_pp_t['cnt_lesion']}  "
              f"unpad={_pp_t['unpad']:.2f}s  cpu_numpy={_pp_t['cpu_numpy']:.2f}s  "
              f"getitk={_pp_t['getitk']:.2f}s  reorient={_pp_t['reorient']:.2f}s  "
              f"lesion_mask_lookup={_pp_t['lesion_mask_lookup']:.2f}s  "
              f"lesion_dilate={_pp_t['lesion_dilate']:.2f}s  "
              f"lesion_mul={_pp_t['lesion_mul']:.2f}s  "
              f"final_resample={_pp_t['final_resample']:.2f}s", flush=True)
    return pred_dict


def postprocess_non_binary_lesion(pred, reoriented_itk_img, original_idx, origin_orientation, target_spacing,
                                  classes, args, original_itk_img):
    print(f'Shape of the prediction for postprocessing: {pred.shape}')
    pred = pred.squeeze(0)
    pred_dict = {}
    
    for i in range(pred.shape[0]):
        #skip lesions:
        #do organ first
        if 'lesion' in classes[i]:
            continue

        #if 'pancrea' in classes[i]:
        #    print("Sum of organ mask raw:", pred[i].sum(),classes[i])
        tensor_pred = unpad_img(pred[i], original_idx, args)
        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            tensor_pred = resample_image_with_gpu(
                tensor_pred.float(), target_spacing, tensor_pred.shape[::-1],
                reoriented_itk_img.GetSpacing(), reoriented_itk_img.GetSize(),
                interp="nearest")

        itk_pred = sitk.GetImageFromArray(tensor_pred.cpu().numpy())   # scores
        itk_pred.CopyInformation(reoriented_itk_img)
        itk_pred = reorient_image(itk_pred, origin_orientation)

        # hard mask – keeps spacing / origin / direction intact
        itk_lab  = sitk.Cast( itk_pred > 0.5 , sitk.sitkFloat32 )
        #if 'pancrea' in classes[i]:
        #    print("Sum of organ mask:", float(sitk.GetArrayViewFromImage(itk_lab).sum()),classes[i])
        pred_dict[classes[i]] = itk_lab

    #now do lesions
    for i in range(pred.shape[0]):
        if 'lesion' not in classes[i]:
            continue
        tensor_pred = pred[i]
        tensor_pred = unpad_img(tensor_pred, original_idx, args)

        resized=[]
        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            tensor_pred=resample_image_with_gpu(tensor_pred.float(), target_spacing, 
            tensor_pred.shape[::-1], reoriented_itk_img.GetSpacing(), 
            reoriented_itk_img.GetSize(), interp='trilinear')
        
        itk_pred = sitk.GetImageFromArray(tensor_pred.cpu().numpy().astype(np.float32))
        itk_pred.CopyInformation(reoriented_itk_img)

        itk_pred = reorient_image(itk_pred, origin_orientation)

        np_pred = sitk.GetArrayFromImage(itk_pred)

        lab_arr = np_pred
        lab_arr = lab_arr.astype(np.float32)
        itk_lab = sitk.GetImageFromArray(lab_arr)
        itk_lab.CopyInformation(itk_pred)
        #print('Sum of lesion mask:',lab_arr.sum(), classes[i])

        if args.organ_mask_on_lesion:
            # remove anything outside of the organ
            organ_name = classes[i].split('_')[0].replace('pancreatic', 'pancreas')
            if organ_name == 'kidney':
                # Combine kidney_right and kidney_left using SimpleITK's Add function.
                organ = sitk.Add(pred_dict['kidney_right'], pred_dict['kidney_left'])
            elif organ_name == 'adrenal':
                organ = sitk.Add(pred_dict['adrenal_gland_right'], pred_dict['adrenal_gland_left'])
            elif organ_name == 'lung':
                organ = sitk.Add(pred_dict['lung_right'], pred_dict['lung_left'])
            elif organ_name == 'uterus':
                organ = pred_dict['prostate']
            elif organ_name == 'gallbladder':
                organ = pred_dict['gall_bladder']
            elif organ_name in ['bone','breast']:
                #we do not have organ masks for these, make a mask of ones
                size = pred_dict['prostate'].GetSize()
                organ = sitk.Image(size, sitk.sitkFloat32)
                organ = organ + 1  # This adds 1 to every voxel, making an image of ones.
            else:
                organ = pred_dict[organ_name]

            #print('Sum of organ mask after getting it:', sitk.GetArrayFromImage(organ).sum(),organ_name)
                
            # Convert organ to NumPy, threshold to binary, and convert back to SimpleITK.
            organ_np = sitk.GetArrayFromImage(organ)
            organ_np = (organ_np > 0).astype(np.float32)
            organ = sitk.GetImageFromArray(organ_np)
            # Copy spatial information (assuming pred_dict[organ_name] or kidney_right has correct info)
            if organ_name == 'kidney':
                organ.CopyInformation(pred_dict['kidney_right'])
            elif organ_name == 'adrenal':
                organ.CopyInformation(pred_dict['adrenal_gland_right'])
            elif organ_name == 'lung':
                organ.CopyInformation(pred_dict['lung_right'])
            elif organ_name == 'uterus':
                organ.CopyInformation(pred_dict['prostate'])
            elif organ_name == 'gallbladder':
                organ.CopyInformation(pred_dict['gall_bladder'])
            elif organ_name in ['bone','breast']:
                #we do not have organ masks for these, make a mask of ones
                organ.CopyInformation(pred_dict['prostate'])
            else:
                organ.CopyInformation(pred_dict[organ_name])
            
            # Dilate the organ mask using a radius of 3 voxels.
            organ  = sitk.Cast( organ > 0.5 , sitk.sitkUInt8 )   # binarise in‑place
            organ  = sitk.BinaryDilate(organ, (3, 3, 3))

            #print('Sum of organ and lesion mask before gating:',sitk.GetArrayFromImage(itk_lab).sum(),sitk.GetArrayFromImage(organ).sum())     

            organ = sitk.Cast(organ, sitk.sitkFloat32)
            itk_lab = sitk.Cast(itk_lab, sitk.sitkFloat32)   

            
            #print('Sum of organ and lesion mask before gating f32:',sitk.GetArrayFromImage(itk_lab).sum(),sitk.GetArrayFromImage(organ).sum())     
            
            # Multiply the lesion label (itk_lab) by the organ mask using SimpleITK.Multiply.
            itk_lab = sitk.Multiply(organ, itk_lab)

        
        print('Sum of gated lesion mask:',sitk.GetArrayFromImage(itk_lab).sum())

        itk_lab = sitk.Cast(itk_lab, sitk.sitkFloat32)   
        
        pred_dict[classes[i]] = itk_lab
        
    # Map probabilities back to the original CT geometry
    for key, img in list(pred_dict.items()):
        pred_dict[key] = ResampleLabelToRef(
            img,
            original_itk_img,
            interp=sitk.sitkLinear  # correct for probabilities
        )

    return pred_dict

def postprocess_npz(pred, classes, args):
    print(f'Shape of the prediction for postprocessing: {pred.shape}')
    pred = pred.squeeze(0)
    pred_dict = {}
    
    for i in range(pred.shape[0]):
        #skip lesions:
        #do organ first
        if 'lesion' in classes[i]:
            continue
        tensor_pred = pred[i]
        np_pred = tensor_pred.float().cpu().numpy()
        pred_dict[classes[i]] = np_pred

    #now do lesions
    for i in range(pred.shape[0]):
        if 'lesion' not in classes[i]:
            continue
        np_pred = pred[i].cpu().numpy()

        if args.organ_mask_on_lesion:
            # remove anything outside of the organ
            organ_name = classes[i].split('_')[0].replace('pancreatic', 'pancreas')
            if organ_name == 'kidney':
                # Combine kidney_right and kidney_left using SimpleITK's Add function.
                organ = pred_dict['kidney_right']+pred_dict['kidney_left']
            elif organ_name == 'adrenal':
                organ = pred_dict['adrenal_gland_right']+pred_dict['adrenal_gland_left']
            elif organ_name == 'lung':
                organ = pred_dict['lung_right']+pred_dict['lung_left']
            elif organ_name == 'uterus':
                organ = pred_dict['prostate']
            elif organ_name == 'gallbladder':
                organ = pred_dict['gall_bladder']
            elif organ_name in ['bone','breast']:
                #we do not have organ masks for these, make a mask of ones
                organ = np.ones_like(pred_dict['prostate'], dtype=np.uint8)
            else:
                organ = pred_dict[organ_name]
                
            #threshold to binary
            organ = (organ > 0.5).astype(np.uint8)
            
            # Dilate the organ mask using a radius of 3 voxels in numpy
            organ = ndi.binary_dilation(organ, structure=np.ones((3, 3, 3)))
            
            #type as np_pred
            organ = organ.astype(np_pred.dtype)
            
            np_pred = organ * np_pred
        
        pred_dict[classes[i]] = np_pred

    return pred_dict

def keep_largest_component(label_map):
    # Convert the label map to a binary image
    binary_image = label_map > 0
    
    # Run connected component analysis
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_image = cc_filter.Execute(binary_image)
    
    # Get the number of connected components
    num_components = cc_filter.GetObjectCount()
    
    # Find the largest connected component
    largest_component_label = 0
    largest_component_size = 0
    for i in range(1, num_components + 1):
        component_mask = sitk.Equal(cc_image, i)
        component_size = sitk.GetArrayFromImage(component_mask).sum()
        if component_size > largest_component_size:
            largest_component_label = i
            largest_component_size = component_size
    
    # Create a new label map with only the largest component
    largest_component_map = sitk.Equal(cc_image, largest_component_label)
    
    return largest_component_map

def resample_image_with_gpu(tensor_img, old_spacing=(2., 2., 2.), old_size=(512, 512, 512), new_spacing=(1., 1., 1.), new_size=None, interp='trilinear'):
    # space order: x, y, z. numpy/pytorch tensor order z, y, x
    
    new_spacing = np.array(new_spacing)[::-1] # -> z, y, x

    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0) # -> b, c, z, y, x

    old_spacing = np.array(old_spacing)[::-1]
    old_size = np.array(old_size, dtype=np.float32)[::-1] # -> z, y, x
    
    if new_size == None:
        new_size = old_size * (old_spacing / new_spacing)
        new_size = new_size.round().astype(int).tolist()
    else:
        new_size = np.array(new_size)[::-1].tolist() # -> z, y, x
    
    if interp in  ['linear', 'bilinear', 'bicubic', 'trilinear']:
        resampled_tensor_img = F.interpolate(tensor_img, size=new_size, mode=interp, align_corners=True)
    else:
        resampled_tensor_img = F.interpolate(tensor_img, size=new_size, mode=interp)
    resampled_tensor_img = resampled_tensor_img.squeeze(0).squeeze(0)
    
    torch.cuda.empty_cache()

    return resampled_tensor_img


    
def get_age_and_sex(report):
    try:
        age = report['Patient Age'].iloc[0]
        sex = report['Patient Sex'].iloc[0]
    except:
        age = report['age'].iloc[0]
        sex = report['sex'].iloc[0]
        
    
    # --- age ---
    # handle NaN/None early
    if pd.isna(age):
        age_val = 0.0
    else:
        # numeric already
        if isinstance(age, (int, float, np.integer, np.floating)):
            age_val = float(age)
        else:
            # string: extract first number like "65", "65.0", "065", "65Y"
            s = str(age).strip()
            m = re.search(r"(\d+(\.\d+)?)", s)
            age_val = float(m.group(1)) if m else 0.0

    # clamp to a sane range (optional but recommended)
    if not np.isfinite(age_val) or age_val < 0 or age_val > 120:
        age_val = 0.0

    age = torch.tensor([age_val / 100.0], dtype=torch.float32)

    if not isinstance(sex, str):
        sex = 'unknown'
    if sex.lower() == 'male':
        sex = torch.tensor([1]).float()
    elif sex.lower() == 'female':
        sex = torch.tensor([0.5]).float()
    elif sex.lower() == 'unknown':
        sex = torch.tensor([0]).float()
    else:
        raise ValueError(f'Unexpected value for patient sex, {sex}, it should be Male, Female or unknown/NaN')
    return age.unsqueeze(0), sex.unsqueeze(0)

def init_model(args,classes,old_classes=None):
    #checkpoint = torch.load(args.load)
    #net.load_state_dict(checkpoint['model_state_dict'])
    #args.start_epoch = checkpoint['epoch']
    print(f"Number of classes: {len(classes)}")
    if old_classes is not None:
        print(f"Number of old classes: {len(old_classes)}")
    if args.update_output_layer:
        c = old_classes # we must load the checkpoint with the old classes
    else:
        c = classes
        
    if args.update_output_layer or args.malignancy_classification:
        from model.dim3.medformer import update_output_layer_onk
        print('Classes for onk:', classes)
        if args.malignancy_classification and old_classes is None:
            old_classes = classes
        if args.malignancy_classification:
            lesion_classes = [c for c in sorted(classes) if 'lesion' in c]
            malignants = [c.replace('lesion', 'malignant') for c in lesion_classes]
            benigns = [c.replace('lesion', 'benign') for c in lesion_classes]
            new_classes = classes + malignants + benigns
        else:
            new_classes = classes
    else:
        new_classes = classes
    
    

    model_list = []
    for ckp_path in args.load:
        print('Number of classes for model loading: ', len(c))
        model = get_model(args,classes=c)
        if args.update_output_layer or args.malignancy_classification:
            from model.dim3.medformer import update_output_layer_onk
            model=update_output_layer_onk(model, original_classes=old_classes, new_classes=new_classes,
                                            age_and_sex=args.age_and_sex_provided)
        if not args.EMA:
            pth = torch.load(ckp_path, map_location=torch.device('cpu'))['model_state_dict']
        else:
            pth = torch.load(ckp_path, map_location=torch.device('cpu'))['ema_model_state_dict']
            
            
        from model.dim3.medformer import RTSuper
        model_base = model
        
   
        model = RTSuper(model_base,train_mode = args.train_mode, num_inpt_ch = args.num_inpt_ch,
                                teacher_report_info_prob=args.teacher_report_info_prob,
                                student_report_info_prob=args.student_report_info_prob,
                                EMA_net = copy.deepcopy(model_base),
                                MLP_in_dim= args.MLP_in_dim,
                                use_transformer_decoder=not(args.remove_transformer_decoder),
                                use_dynamic_conv=not(args.remove_dynamic_conv),
                                use_transformer_conv3 = args.use_transformer_conv3,
                                age_and_sex_provided=args.age_and_sex_provided,
                                ablate_dynamic_decoder=args.ablate_dynamic_decoder,
                                ablate_longitudinal_attention=args.ablate_longitudinal_attention)
        
        
        if args.time_points > 1:
            model.make_longitudinal(
                time_points=args.time_points,
                use_transformer_decoder=not(args.remove_transformer_decoder),
                use_dynamic_conv=not(args.remove_dynamic_conv),
                age_and_sex_provided=args.age_and_sex_provided,
                use_transformer_conv3=args.use_transformer_conv3,
                concat_time_info=args.concat_time_info)

        # If the checkpoint was trained with --use_time_consistency_loss, its
        # state_dict contains registration_module weights. Build the module so
        # those keys load (otherwise they end up as "unexpected_keys" and the
        # registration weights are silently dropped).
        if getattr(args, 'use_time_consistency_loss', False):
            model.make_registration(
                registration_mode=getattr(args, 'registration_mode', 'image'),
                registration_input_shape=[175, 175, 175],
                initialization_registration=getattr(args, 'initialization_registration', 'unigradicon'),
                train_registration=False,
            )

        missing, unexpected = model.load_state_dict(pth, strict=False)
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:    print(f"  first missing:    {missing[:5]}")
        if unexpected: print(f"  first unexpected: {unexpected[:5]}")
            
        model=model.return_inference_net()
        model.cuda()
        model.eval()
        # torch.compile was tried (mode='reduce-overhead') but recompiles
        # on this model's dynamic shapes pushed SW loop from 5 s to 97 s
        # — leaving it disabled.
        model_list.append(model)
        print(f"Model loaded from {ckp_path}")

    
        

    return model_list, new_classes

def _nii_stem(fn: str) -> str:
    # convert 'adrenal_gland_left.nii.gz' -> 'adrenal_gland_left'
    return fn[:-7] if fn.endswith('.nii.gz') else os.path.splitext(fn)[0]

def nib_load(path_to_nii):
    """
    Attempt to load a NIfTI file with nibabel in a way that more closely
    matches SimpleITK's default LPS orientation from sitk.ReadImage().
    
    Returns:
        tmp_itk_img (SimpleITK.Image)
    """
    # We import nibabel here to avoid "UnboundLocalError"
    import nibabel as nb
    from nibabel.orientations import (
        io_orientation,
        axcodes2ornt,
        ornt_transform,
        apply_orientation
    )

    # 1) Load with nibabel
    nib_img = nb.load(path_to_nii)
    affine = nib_img.affine
    np_array = nib_img.get_fdata(dtype=np.float32)  # shape: (x, y, z) typically

    # 2) Determine the orientation of the current image
    orig_ornt = io_orientation(affine)          # e.g. RAS, LAS, etc.
    lps_ornt  = axcodes2ornt(("L", "P", "S"))    # LPS convention

    # 3) Compute the transform that takes us from the original orientation to LPS
    trans_ornt = ornt_transform(orig_ornt, lps_ornt)

    # 4) Reorient the data array to LPS axis ordering
    np_array_lps = apply_orientation(np_array, trans_ornt)
    # apply_orientation() only reorders the array (axes / flips); 
    # it does NOT return a new affine in nibabel’s current versions.

    # 5) By default, SimpleITK’s GetArrayFromImage() returns data in (z, y, x).
    #    Our LPS array here is in (L, P, S) which typically aligns with (x, y, z).
    #    We can transpose to get (z, y, x) if your code expects that ordering:
    np_array_zyx = np.transpose(np_array_lps, (2, 1, 0))  # now shape: (z, y, x)

    # 6) Make a SimpleITK image from that array
    tmp_itk_img = sitk.GetImageFromArray(np_array_zyx)

    return tmp_itk_img

def get_parser():

    def parse_spacing_list(string):
        return tuple([float(spacing) for spacing in string.split(',')])
    def parse_model_list(string):
        return string.split(',')
    parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
    parser.add_argument('--parts', type=int, default=1, help='For running multiple instances of this script. Parts divides the dataset, and this script runs on current_part')
    parser.add_argument('--current_part', type=int, default=0, help='For running multiple instances of this script. Parts divides the dataset, and this script runs on current_part')
    parser.add_argument('--dataset', type=str, default='abdomenatlas', help='dataset name')
    parser.add_argument('--model', type=str, default='medformer', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')

    parser.add_argument('--load', type=parse_model_list, default='./exp/abdomenatlas/former_batch4_pth128/fold_0_latest.pth', help='the path of trained model checkpoint. Use \',\' as the separator if load multiple checkpoints for ensemble')
    parser.add_argument('--img_path', type=str, default='/projects/bodymaps/Data/Dataset244_smallAtlasUCSF/imagesTr/', help='the path of the directory of images to be predicted')
    parser.add_argument('--save_path', type=str, default='./result/UFO/', help='the path to save predicted label')
    parser.add_argument('--learnable_loss_weights', action='store_true', help='Allows learnable loss weigths (https://arxiv.org/pdf/1705.07115).')  

    parser.add_argument('--ids', type=str, default=None, help='ids of testing samples')
    
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--class_list', type=str, default='/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml')
    parser.add_argument('--connected_components', action='store_true', help='whether to keep the largest connected component')
    parser.add_argument('--organ_mask_on_lesion', action='store_true', help='whether to keep the largest connected component')
    parser.add_argument('--classification_branch', action='store_true', help='whether to use the classification branch')
    parser.add_argument('--cls_gate', action='store_true', help='multiplies the segmentation sigmoid output by the classification sigmoid output--gate')
    parser.add_argument('--cls_gate_norm', action='store_true', help='before applying the the cls gate, the segmentation output is normalized, making its maximum value above 0.5 become 1')
    parser.add_argument('--update_output_layer', action='store_true', help='update the output layer to have the same number of classes as the number of classes in the class_list')
    parser.add_argument('--old_classes', type=str, default=None, help='old classes, we will keep weights/kernels of the old classes. This parameter should be a location of a yaml file with the old classes, we will sort them!')
    parser.add_argument('--mtl', type=str, default=None, help='multi-task learning method. If None, no MTL. Uses method from https://github.com/SamsungLabs/MTL/')
    parser.add_argument('--save_probabilities', action='store_true', help='saves probabilities')
    parser.add_argument('--not_save_binary', action='store_true', help='does not save binary')
    parser.add_argument('--save_probabilities_lesions', action='store_true', help='saves probabilities only for lesion classes')
    parser.add_argument('--save_probabilities_report_tumors_only', action='store_true', help='saves probabilities')
    parser.add_argument('--overwrite', action='store_true', help='overwrites last saved results')
    parser.add_argument('--save_pancreas_lesion_only', action='store_true', help='overwrites last saved results')
    parser.add_argument('--predict_pancreas_only', action='store_true', help='overwrites last saved results')
    parser.add_argument('--epai_stage_2', action='store_true', help='only for testing epai stage 2')
    #meta
    parser.add_argument('--meta', type=str, help='meta from reports')
    parser.add_argument('--reports', type=str, help='meta from reports')
    parser.add_argument('--filter_cases_ufo', action='store_true', help='predict only cases of interest (no missing size)')
    parser.add_argument('--restrictive_filter', action='store_true', help='only consider cases that have no tumor outside of a given class list')
    parser.add_argument('--restrictive_filter_one_organ', action='store_true', help='only consider cases that tumors in a single organ')

    parser.add_argument('--aggregator_mode', type=str, default='concat', help='mode for the aggregator')
    parser.add_argument('--cls_on_output', action='store_true', help='if true, the classification branch is on the output of the model, otherwise it is on the bottleneck')
    parser.add_argument('--cls_on_segmentation', action='store_true', help='if true, the classification branch is on the output of the model, otherwise it is on the bottleneck')
    parser.add_argument('--binarize_cls_on_segmentation', action='store_true', help='if true, the classification branch on the segmentation output receives binary inputs (straight through trick)')

    #extra classifiers on top of the segmentation output
    parser.add_argument('--attenuation_classifier', type=str, default='none')
    parser.add_argument('--train_att_MLP_on_mask_only', action='store_true', help='if true, the attenuation classifier MLP is trained only on the mask (segmentation) output. Otherwise, it is trained on mask and model outputs.')
    parser.add_argument('--att_weight', type=float, default=0.01, help='weight for the tumor attenuation loss')
    parser.add_argument('--tumor_classifier', action='store_true', help='if true, adds a tumor classifier on top of the segmentation output. The classifier classifies tumor number and diameters.')
    parser.add_argument('--cls_weight', type=float, default=0.01, help='weight for the tumor classifier loss')
    parser.add_argument('--organs_with_tumor', type=str, nargs='+', default=['bladder','gall_bladder','esophagus','duodenum','stomach','adrenal_gland_right','adrenal_gland_left','prostate','spleen'], help='organs that may have tumors, used in the two pass inference')
    parser.add_argument('--disable_inference_2_stages', action='store_true', help='if not set, we run one normal inference pass, then a second pass cropping on all organs that may have tumors')
    parser.add_argument('--EMA', action='store_true', help='If set, we test the EMA model')
    
    #parser.add_argument('--class_list', type=str, default='/projects/bodymaps/Pedro/data/atlas_300_medformer_multi_ch_tumor_npy/list/label_names.yaml')

    parser.add_argument('--malignancy_classification', action='store_true', help='will train to differentiate between benign and malignant tumors, adds benign and malignant classes beyond the lesion classes')
    
    #RT Super
    parser.add_argument('--train_mode', type=str, default='both', help='train mode for RT Super')
    parser.add_argument('--num_inpt_ch', type=int, default=21, help='number of input channels')
    parser.add_argument('--teacher_report_info_prob', type=float, default=0.0, help='This is only used for loading, match the setting used in training. In inference, no report info will be passed anyway.')
    parser.add_argument('--student_report_info_prob', type=float, default=0.0, help='This is only used for loading, match the setting used in training. In inference, no report info will be passed anyway.')
    parser.add_argument('--MLP_in_dim', type=int, default=23, help='dimension of the MLP in RT Super')
    parser.add_argument('--use_transformer_decoder', action='store_true', help='if set, the model uses the transformer decoder (without giving it report information)')
    parser.add_argument('--give_tumor_size_input', action='store_true', help='important for matching dimension when model was trained with this flag, we never actually give tumor size information during inference')
    parser.add_argument('--remove_transformer_decoder', action='store_true', help='removes the transformer decoder in RTSuper teacher_decoder mode')
    parser.add_argument('--remove_dynamic_conv', action='store_true', help='removes the dynamic conv layers in RTSuper teacher_decoder mode')
    parser.add_argument('--use_transformer_conv3', action='store_true', help='uses the transformer decoder to select the kernels used in the 3x3x3 convolutions in the report-informed decoder')
    parser.add_argument('--time_points', type=int, default=1, help='number of time points that the model will see as input')
    parser.add_argument('--use_time_consistency_loss', action='store_true',
                        help='Set if the checkpoint was trained with --use_time_consistency_loss. '
                             'Builds the registration module so its weights load from the state_dict.')
    parser.add_argument('--registration_mode', type=str, default='image',
                        help="Match the training value of --registration_mode (default 'image').")
    parser.add_argument('--initialization_registration', type=str, default='unigradicon',
                        help="Match the training value of --initialization_registration (default 'unigradicon').")
    parser.add_argument('--time_fusion', type=str, default='transformer_decoder', help='type of time fusion: early or transformer_decoder')
    parser.add_argument('--age_and_sex_provided', action='store_true', help='provides patient age and sex during inference')
    parser.add_argument('--concat_time_info', action='store_true', help='uses the transformer decoder to select the kernels used in the 3x3x3 convolutions in the report-informed decoder')
    parser.add_argument('--ablate_dynamic_decoder', action='store_true',
                        help='Match training: replace the dynamic report-informed decoder '
                             'with StaticUNetDecoder3D (medformer.py:900). Required for '
                             'checkpoints trained with --ablate_dynamic_decoder, otherwise '
                             'state_dict load will fail with shape/key mismatches.')
    parser.add_argument('--ablate_longitudinal_attention', action='store_true',
                        help='Match training: disable inter-image (other-time) cross-attention '
                             'in the teacher transformer decoder (medformer.py:2920). No '
                             'parameter-shape change, but omitting this for a checkpoint '
                             'trained with it produces a silent semantic mismatch.')

    # Teacher-eval (2-stage report-informed inference)
    parser.add_argument('--use_teacher_eval', action='store_true',
                        help='If set, stage 2 calls the teacher (report-informed decoder) per organ '
                             'with a tumor_info dict built by TumorInfoBuilder. Stage 1 is the same '
                             'student sliding-window pass as the baseline.')
    parser.add_argument('--teacher_no_size', action='store_true',
                        help='[A/B test] Pass no_mask_training=True to the teacher so its '
                             'report_informed_decoder forces give_size_mask=0, zeroing '
                             'diameter/volume tokens while keeping every other report field. '
                             'Lets us measure whether the teacher actually uses tumor sizes.')
    parser.add_argument('--teacher_no_mask_training', action='store_true',
                        help='REQUIRED if the checkpoint was trained with --no_mask. '
                             'Sets no_mask_training=True in the teacher forward, which '
                             'forces give_size_mask=zeros (medformer.py:1191-1193) — i.e. the '
                             'report-informed decoder never sees diameter/volume tokens, '
                             'matching the training distribution of a --no_mask checkpoint. '
                             'Functionally equivalent to --teacher_no_size; use whichever '
                             'name fits your mental model.')
    parser.add_argument('--reports_malignancy_col', type=str,
                        default='pathology_and_radiology_malignant',
                        help='Column name in --reports for the strict (pathology-confirmed) '
                             "malignancy flag. Default matches the dataset's default. "
                             'If the column is missing in the CSV and --reports_relaxed_malignancy_col '
                             "exists, the relaxed column is promoted to primary.")
    parser.add_argument('--reports_benign_col', type=str,
                        default='radiology_benign_ICD_pathology_ok',
                        help='Column name in --reports for the strict benign flag. '
                             "Default matches the dataset's default.")
    parser.add_argument('--reports_relaxed_malignancy_col', type=str, default=None,
                        help='Column name in --reports for a relaxed (radiology-based) '
                             'malignancy fallback. Training launchers set this to "malignancy". '
                             'Match what the checkpoint was trained with for distribution parity.')
    parser.add_argument('--teacher_reproduce_dataset_bugs', action='store_true',
                        help='[A/B test] Reproduce the dataset bug where estimate_tumor_volume '
                             'returns zero rows for sided sub-segment crops '
                             '(adrenal_gland_*, kidney_*, lung_*, femur_*) — feed those crops the '
                             'no_tumor branch, matching the distribution the model saw at training.')
    parser.add_argument('--teacher_filter_laterality', action='store_true',
                        help='[A/B test] When stage-2 crops on a sided organ '
                             '(adrenal_gland_right, etc.), restrict the report rows fed to the '
                             "teacher to that side's tumors only. Without this, both sides' tumors are merged.")
    parser.add_argument('--teacher_dump_dir', type=str, default=None,
                        help='[debug] If set, dump per-crop tumor_info JSON files to this directory '
                             'so we can compare against the dataset-side dumps.')
    parser.add_argument('--fast_teacher', action='store_true',
                        help='In stage 1 (sliding window), skip every window that does NOT '
                             'intersect any organ where the per-tumor report mentions a tumor '
                             'for this case (ROI = union of GT organ masks for those organs). '
                             'Stage 2 is unchanged. Saves NIfTIs only for *_lesion channels '
                             'and for organ channels in the ROI subset.')
    parser.add_argument('--fast_student', action='store_true',
                        help='In stage 1 (sliding window), skip every window that does NOT '
                             'intersect any organ in --organs_with_tumor (ROI = union of GT '
                             'organ masks for those organs). Stage 2 is unchanged. Saves '
                             'NIfTIs only for *_lesion channels and for organ channels in '
                             '--organs_with_tumor.')
    parser.add_argument('--gt_mask_root', type=str,
                        default='/mnt/bodymaps/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/',
                        help='Root for GT organ masks consumed by --fast_teacher / --fast_student. '
                             'Layout: <root>/<BDMAP_ID>/segmentations/<organ>.nii.gz')
    parser.add_argument('--fast_teacher_mask_info', action='store_true',
                        help='In --fast_teacher: instead of deriving tumor sizes from the '
                             'per-tumor report text (e.g. "u" → sentinel -0.5), derive them '
                             'from the GT lesion mask via connected-component analysis '
                             '(mirrors the dataset atlas branch). Requires --gt_lesion_paths_json '
                             'to locate per-case lesion NIfTIs.')
    parser.add_argument('--gt_lesion_paths_json', type=str, default=None,
                        help='Path to JSON mapping {BDMAP_ID: {organ_canonical: lesion_path}}. '
                             'Used by --fast_teacher_mask_info to locate GT lesion NIfTIs.')
    parser.add_argument('--fast_teacher_roi', choices=['full', 'report'], default='full',
                        help='--fast_teacher SW ROI policy: '
                             '"full" = union of GT masks for every organ in '
                             '--organs_with_tumor (stage-2 still filtered to '
                             'report organs, so the optimisation is preserved). '
                             '"report" = union of GT masks for only the organs '
                             'the per-tumor report mentions for this case (faster '
                             'but misses tumors whose organ has an empty GT mask '
                             'in AbdomenAtlasPro). Default: full.')

    args = parser.parse_args()

    # Mutually-exclusive fast modes. fast_teacher requires the teacher
    # branch (report builder + --use_teacher_eval); fast_student is a
    # pure-student path and must NOT have --use_teacher_eval.
    if getattr(args, 'fast_teacher', False) and getattr(args, 'fast_student', False):
        parser.error('--fast_teacher and --fast_student are mutually exclusive')
    if getattr(args, 'fast_teacher', False) and not getattr(args, 'use_teacher_eval', False):
        parser.error('--fast_teacher requires --use_teacher_eval (we need the report builder loaded)')
    if getattr(args, 'fast_student', False) and getattr(args, 'use_teacher_eval', False):
        parser.error('--fast_student must NOT be combined with --use_teacher_eval (use --fast_teacher instead)')
    if getattr(args, 'fast_teacher_mask_info', False):
        if not getattr(args, 'fast_teacher', False):
            parser.error('--fast_teacher_mask_info requires --fast_teacher')
        if not getattr(args, 'gt_lesion_paths_json', None):
            parser.error('--fast_teacher_mask_info requires --gt_lesion_paths_json')
        if not os.path.exists(args.gt_lesion_paths_json):
            parser.error(f'--gt_lesion_paths_json file not found: {args.gt_lesion_paths_json}')
    if (getattr(args, 'fast_teacher', False) or getattr(args, 'fast_student', False)) and getattr(args, 'predict_pancreas_only', False):
        parser.error('--fast_teacher / --fast_student are incompatible with --predict_pancreas_only '
                     '(both want to use the SW window-skip mask).')


    args.clip_loss = False
    args.load_clip = False

    #latest folder of load
    if isinstance(args.load, list):
        model_name = os.path.basename(os.path.dirname(args.load[0]))  # Gets the last folder name
    else:
        model_name = os.path.basename(os.path.dirname(args.load))  # Handles single path case

    args.save_path = os.path.join(args.save_path, args.dataset, model_name)
    os.makedirs(args.save_path, exist_ok=True)
    print('Save path: %s'%args.save_path)

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    args.inference_2_stages=True
    if args.disable_inference_2_stages:
        args.inference_2_stages=False
        print('Inference 2 stages disabled! (Why?)')
    

    return args
def filter_already_predicted(ids, save_path, class_list, overwrite):
    print('Filtering predicted')
    if overwrite:
        return ids

    def _nii_stem(fn: str) -> str:
        # convert 'adrenal_gland_left.nii.gz' -> 'adrenal_gland_left'
        return fn[:-7] if fn.endswith('.nii.gz') else os.path.splitext(fn)[0]

    kept = []
    for img_name in ids:
        case_id = img_name.replace('/ct.nii.gz','').replace('.nii.gz','').replace('.npz','')
        pred_dir = os.path.join(save_path, case_id, 'predictions')

        if not os.path.isdir(pred_dir):
            kept.append(img_name)
            continue

        # Collect all .nii.gz stems
        stems = set()
        with os.scandir(pred_dir) as it:
            for entry in it:
                if entry.name.endswith('.nii.gz'):
                    stems.add(_nii_stem(entry.name))

        # Require every class to be present as a .nii.gz
        if set(class_list).issubset(stems):
            print(f"Skipping {img_name}: found all {len(class_list)} classes (.nii.gz only)")
        else:
            kept.append(img_name)

    print(f"Ids after filtering: {len(kept)} / {len(ids)}")
    return kept

if __name__ == '__main__':
    
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.sliding_window = True
    args.window_size = args.training_size

    if args.ids is None:
        ids = [i for i in os.listdir(args.img_path) if 'gt.' not in i]
    else:
        if 'BDMAP_ID' in pandas.read_csv(args.ids).columns:
            col = 'BDMAP_ID'
        else:
            col = 'BDMAP ID'
        ex = pandas.read_csv(args.ids)[col].tolist()[0]
        if os.path.exists(args.img_path+'/'+ex+'.npz'):
            ids = [f+'.npz' for f in pandas.read_csv(args.ids)[col].tolist()]
        elif os.path.exists(args.img_path+'/'+ex+'/ct.nii.gz'):
            ids = [f+'/ct.nii.gz' for f in pandas.read_csv(args.ids)[col].tolist()]
        elif os.path.exists(args.img_path+'/'+ex+'.nii.gz'):
            ids = [f+'.nii.gz' for f in pandas.read_csv(args.ids)[col].tolist()]
        else:
            raise ValueError('Path not found')

    #remove cases not in data folder:
    valid_ids = []
    removed = []
    for img_name in ids:
        full_path = os.path.join(args.img_path, img_name)
        if os.path.exists(full_path):
            valid_ids.append(img_name)
        else:
            logging.warning(f"Skipping '{img_name}': not found at {full_path}")
            removed.append(img_name)
    ids = valid_ids
    print(f'After removing missing cases, {len(ids)} ids remain. Number of removed files: {len(removed)}')

    if args.age_and_sex_provided:
        reports_original = pd.read_csv(args.reports)
        assert 'BDMAP ID' in reports_original.columns, "Expected 'BDMAP ID' column in reports CSV"

    if args.filter_cases_ufo:
        reports = pd.read_csv(args.reports)
        _,ids_of_interest,_ = clean_ufo(reports)
        print('Number of IDs before filtering IDs of interest:', len(ids))
        print(f'Examples of IDs of interest: {ids_of_interest[:10]}')
        print(f'Examples of our IDs: {ids[:10]}')
        ids = [i for i in ids if i[:len('BDMAP_00032584')] in ids_of_interest]
        print('Number of IDs after filtering IDs of interest:', len(ids))
        
    if args.restrictive_filter:
        meta = pd.read_csv(args.meta)
        ids_of_interest = restrictive_filtering(meta, single_tumor=args.restrictive_filter_one_organ)
        print('restrictive: number of IDs of interest:', len(ids_of_interest))
        print('restrictive: Number of IDs before filtering IDs of interest:', len(ids))
        print(f'restrictive: Examples of IDs of interest : {ids_of_interest[:10]}')
        print(f'restrictive: Examples of our IDs: {ids[:10]}')
        ids = [i for i in ids if i[:len('BDMAP_00032584')] in ids_of_interest]
        print('restrictive: Number of IDs after filtering IDs of interest:', len(ids))
        
    # Keep the YAML path around — args.class_list is about to be overwritten
    # with the sorted list, but TumorInfoBuilder needs the path.
    args.class_list_yaml_path = args.class_list
    with open(args.class_list, 'r') as f:
        class_list = yaml.load(f, Loader=yaml.SafeLoader)
        #sort--we sorted when saving in nii2npy.py
    args.class_list = sorted(class_list)
    class_list=args.class_list
    print('Class list:', class_list)

    args.classes = len(class_list)

    if args.old_classes is not None:
        with open(args.old_classes, 'r') as f:
            old_classes = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
        args.old_classes = sorted(old_classes)
        old_classes=args.old_classes
    else:
        old_classes = None

    # In --fast_teacher / --fast_student modes the inference loop only
    # writes the subset of channels returned by _compute_keep_classes
    # (each organ in --organs_with_tumor + its lesion counterpart, plus
    # uterus_lesion). The resume filter must compare against the same
    # subset, otherwise no case is ever considered complete and every
    # run starts from scratch.
    _keep = _compute_keep_classes(args, class_list)
    expected_classes = sorted(_keep) if _keep is not None else class_list

    #filter ids here
    if not args.overwrite:
        print('Ids before filtering:', len(ids))
        # ─── filter out already‐predicted cases ─────────────────────────────────────────────
        ids = filter_already_predicted(ids,
                                   save_path=args.save_path,
                                   class_list=expected_classes,
                                   overwrite=args.overwrite)
        print('Ids after filtering:', len(ids))
        #raise ValueError('This is just a test, please remove this line to run the code.')
        # ────────────────────────────────────────────────────────────────────────────────────

    if args.parts>1:
        window = len(ids)//args.parts
        start = args.current_part*window
        end = args.current_part*window + window
        if (end+window)>len(ids):
            #last part
            ids = ids[start:]
        else:
            ids = ids[start:end]
        print('Ids to predict after splitting:',len(ids))
        
    if args.save_probabilities_report_tumors_only:
        meta = pd.read_csv(args.meta)
        meta = meta.set_index('BDMAP ID')



    model_list, class_list = init_model(args,classes=class_list,old_classes=old_classes)
    args.class_list = class_list
    args.classes = len(class_list)

    tumor_info_builder = None
    if args.use_teacher_eval:
        from tumor_info_builder import TumorInfoBuilder
        assert args.reports, "--use_teacher_eval requires --reports (per-tumor metadata CSV)"
        tumor_info_builder = TumorInfoBuilder(
            reports_csv_path=args.reports,
            label_names_yaml_path=args.class_list_yaml_path,
            malignancy_column=args.reports_malignancy_col,
            benign_column=args.reports_benign_col,
            relaxed_malignancy_col=args.reports_relaxed_malignancy_col,
            reproduce_volumes_gt_zero_bug=getattr(args, 'teacher_reproduce_dataset_bugs', False),
        )
        print(f'TumorInfoBuilder initialized from reports={args.reports} '
              f'(malig={args.reports_malignancy_col!r}, '
              f'benign={args.reports_benign_col!r}, '
              f'relaxed={args.reports_relaxed_malignancy_col!r})')

        # Extended bug-reproduce: also reproduce the clean_ufo `_was_multiple`
        # bug. At training the column was never reliably set (size 'multiple'
        # rows were dropped before clean_ufo could flag them), so the model
        # was trained with unknown_tumor_count=False for cases that actually
        # had multiple tumors. Drop the column here to match.
        if getattr(args, 'teacher_reproduce_dataset_bugs', False):
            if '_was_multiple' in tumor_info_builder.reports.columns:
                n_true = int(tumor_info_builder.reports['_was_multiple'].sum())
                tumor_info_builder.reports = tumor_info_builder.reports.drop(columns=['_was_multiple'])
                print(f'[bug-repro] dropped _was_multiple column from builder.reports '
                      f'(was True on {n_true} rows; now unknown_tumor_count=False everywhere)')

    random.shuffle(ids)

    import tqdm
    for img_name in tqdm.tqdm(ids):
        case_id = img_name.replace('/ct.nii.gz','').replace('.nii.gz','').replace('.npz','')
        pancreas = None
        tic = time.time()
        if not args.overwrite:
            pred_dir = os.path.join(args.save_path, case_id, 'predictions')
            if os.path.isdir(pred_dir):
                present = {_nii_stem(f) for f in os.listdir(pred_dir) if f.endswith('.nii.gz')}
                if set(expected_classes).issubset(present):
                    print('Already predicted:', img_name)
                    continue
    	
        print(f"Start processing {img_name}")
        if 'nii.gz' in img_name:
            if args.predict_pancreas_only:
                raise ValueError('You cannot predict only pancreas with nii.gz files, please use npz files.')
            
            try:
                itk_img = os.path.join(args.img_path, img_name)
                if '.nii.gz' not in img_name:
                    itk_img = os.path.join(args.img_path, img_name, 'ct.nii.gz')
                itk_img=sitk.ReadImage(itk_img)
                tmp_itk_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(itk_img))
                tmp_itk_img.CopyInformation(itk_img)
            except:
                #for itk errors, load with nib.
                itk_img = os.path.join(args.img_path, img_name)
                if '.nii.gz' not in img_name:
                    itk_img = os.path.join(args.img_path, img_name, 'ct.nii.gz')
                tmp_itk_img=nib_load(itk_img)

            import time as _t_pre
            _t0_pre = _t_pre.time()
            tensor_img, original_idx, origin_orientation, reoriented_itk_img = preprocess(tmp_itk_img, [1.0,1.0,1.0], args)
            if os.environ.get('DEBUG_PREP_PROFILE') == '1':
                print(f'[TIMING] preprocess: {_t_pre.time()-_t0_pre:.2f}s', flush=True)
        else:
            #for npz files
            tensor_img = np.load(os.path.join(args.img_path, img_name))['arr_0']
            tensor_img = torch.from_numpy(tensor_img).cuda().float()
            
            if args.predict_pancreas_only:
                #load pancreas mask
                labels = np.load(os.path.join(args.img_path, img_name.replace('.npz', '_gt.npz')))['arr_0']
                if labels.shape[0] != args.classes:
                    labels = np.unpackbits(labels, axis=0)
                    assert labels.shape[0] < (args.classes+10)
                    assert labels.shape[0] >= (args.classes)
                    labels = labels[:args.classes]
                pancreas = labels[class_list.index('pancreas')]
                pancreas = torch.from_numpy(pancreas).cuda().float()
                
        age,sex = None, None
        bdmap_id = img_name[:len('BDMAP_00052990')]
        if args.age_and_sex_provided:
            rep = reports_original[reports_original['BDMAP ID']==bdmap_id]
            print(f'Report for {bdmap_id}:',flush=True)
            print(rep[['BDMAP ID','Patient Age','Patient Sex']],flush=True)
            print()

            age, sex = get_age_and_sex(rep)

        # Fast modes need GT organ masks aligned to the preprocessed
        # tensor_img. NPZ inputs do not carry the geometry needed to
        # resample the masks, so we refuse them.
        gt_organ_masks = None
        fast_roi_organs = None  # subset of organs_with_tumor used by the SW ROI + save filter
        if getattr(args, 'fast_teacher', False) or getattr(args, 'fast_student', False):
            if 'nii.gz' not in img_name:
                raise ValueError('--fast_teacher / --fast_student require nii.gz inputs '
                                 '(npz inputs do not carry the geometry needed to resample GT masks).')

            # `report_organs`: the report-filtered subset (fast_teacher) or
            # the full list (fast_student). Used to gate stage-2's teacher
            # forward passes.
            # `fast_roi_organs`: which GT masks contribute to the SW ROI.
            #   --fast_teacher_roi=full   (default): every organ in
            #                              --organs_with_tumor — robust to
            #                              cases where the report-mentioned
            #                              organ has an empty AbdomenAtlas
            #                              GT mask (e.g. prostate.nii.gz is
            #                              all-zero on some Turkish cases),
            #                              at the cost of more SW windows.
            #   --fast_teacher_roi=report: only the report-mentioned organs
            #                              — faster on sparse-report cases
            #                              but misses tumors when the GT
            #                              for the only ROI candidate is
            #                              empty.
            # fast_student is unaffected (always uses the full list).
            REPORT_ORGAN_TO_ROI = {'prostate': ['prostate', 'uterus']}
            from inference.inference3d_teacher import _canonical_organ, _organ_side
            if getattr(args, 'fast_teacher', False):
                _flat = getattr(args, 'teacher_filter_laterality', False)
                report_organs = []
                for org in args.organs_with_tumor:
                    side = _organ_side(org) if _flat else None
                    aliases = REPORT_ORGAN_TO_ROI.get(org, [org])
                    has_rows = False
                    for alias in aliases:
                        try:
                            rows = tumor_info_builder._select_rows(
                                bdmap_id, _canonical_organ(alias), side=side)
                        except Exception as e:
                            print(f"[fast] _select_rows failed for {bdmap_id}/{alias}: {e} — skipping")
                            continue
                        if len(rows) > 0:
                            has_rows = True
                            break
                    if has_rows:
                        report_organs.append(org)
                if getattr(args, 'fast_teacher_roi', 'full') == 'full':
                    fast_roi_organs = list(args.organs_with_tumor)
                    print(f"[fast-teacher] ROI=full --organs_with_tumor; stage-2 restricted to report organs {report_organs} for {bdmap_id}")
                else:
                    fast_roi_organs = list(report_organs)
                    print(f"[fast-teacher] ROI=report (report-mentioned organs only): {fast_roi_organs} for {bdmap_id}")
            else:
                fast_roi_organs = list(args.organs_with_tumor)
                report_organs = list(args.organs_with_tumor)
                print(f"[fast-student] ROI organs for {bdmap_id} (all organs_with_tumor): {fast_roi_organs}")

            import time as _t_gt
            _t0_gt = _t_gt.time()
            gt_organ_masks = _load_gt_organ_masks_for_fast(
                bdmap_id=bdmap_id, organs=fast_roi_organs, args=args,
                target_spacing=[1.0, 1.0, 1.0],
                reoriented_itk_img=reoriented_itk_img,
                tensor_img_shape=tuple(tensor_img.shape),
                original_idx=original_idx,
            )
            if os.environ.get('DEBUG_PREP_PROFILE') == '1':
                print(f'[TIMING] gt_organ_masks load ({len(fast_roi_organs)} organs): {_t_gt.time()-_t0_gt:.2f}s', flush=True)
            if not gt_organ_masks:
                print(f"[fast] no GT masks loaded for {bdmap_id} — every SW window will be skipped, "
                      f"output will be fully blank (by design)")

        # --fast_teacher_mask_info: also load GT lesion masks so the
        # tumor_info builder can derive volumes/diameters from real mask
        # voxels (atlas-mode parity) instead of from the report text.
        gt_lesion_masks = None
        if getattr(args, 'fast_teacher_mask_info', False):
            import json as _json_les
            with open(args.gt_lesion_paths_json) as _f:
                _lesion_paths_all = _json_les.load(_f)
            _les_for_bid = _lesion_paths_all.get(bdmap_id)
            if _les_for_bid is None:
                print(f"[fast-mask-info] no entry for {bdmap_id} in --gt_lesion_paths_json — "
                      f"will fall back to report-derived tumor sizes for this case")
            else:
                gt_lesion_masks = _load_gt_lesion_masks_for_fast(
                    bdmap_id=bdmap_id, lesion_paths_for_bid=_les_for_bid, args=args,
                    reoriented_itk_img=reoriented_itk_img,
                    tensor_img_shape=tuple(tensor_img.shape),
                    original_idx=original_idx,
                )

        #count time
        prediction_start = time.time()
        pred_label, pred_raw, cls_out = prediction(
            model_list, tensor_img, args,
            tgt_organ=pancreas, age=age, sex=sex,
            bdmap_id=bdmap_id, tumor_info_builder=tumor_info_builder,
            gt_organ_masks=gt_organ_masks,
            report_organs=report_organs if (getattr(args, 'fast_teacher', False) or getattr(args, 'fast_student', False)) else None,
            gt_lesion_masks=gt_lesion_masks,
        )
        #try:
        #    pred_label, pred_raw = prediction(model_list, tensor_img, args, tgt_organ=pancreas, age=age, sex=sex)
        #    print('Predicted case:', img_name)
        #except:
        #    print('FAILED')
        #    #raise ValueError('Failed to predict case:', img_name)
        #    with open('prediction_errors.txt', "a") as f:
        #        f.write(str(os.path.join(args.img_path, img_name, 'ct.nii.gz')) + "\n")
        #    continue
        prediction_end = time.time()
        print(f"Time for prediction of {img_name}: {prediction_end - prediction_start} seconds")
        


        # In fast modes, restrict postprocess + save to a fixed set:
        #   - every '*_lesion' class (the tumor channels — user spec: always
        #     save lesions, even when blank because the report didn't mention
        #     a tumor in that organ for this case),
        #   - every organ class in args.organs_with_tumor (user spec: organs
        #     that have lesions),
        #   - auxiliary organ classes needed by --organ_mask_on_lesion
        #     (prostate, gall_bladder, adrenal_gland_*, kidney_*, lung_*)
        #     so the lesion-organ multiplication in postprocess works.
        keep_classes = None
        if getattr(args, 'fast_teacher', False) or getattr(args, 'fast_student', False):
            from inference.inference3d import _lesion_like_name
            keep_classes = set()
            # Each organ in --organs_with_tumor and its corresponding
            # lesion channel. For our default 9 organs this gives 9 organs
            # + 7 unique lesions (adrenal_lesion is shared by both
            # adrenal_gland_*) = 16 channels. Every organ that
            # `organ_mask_on_lesion` needs for these lesions is already
            # in --organs_with_tumor, so NO auxiliary organs required.
            for org in args.organs_with_tumor:
                if org in class_list:
                    keep_classes.add(org)
                lesion = _lesion_like_name(org)
                if lesion in class_list:
                    keep_classes.add(lesion)
            # Hardcode for the AbdomenAtlasPro name quirk: prostate.nii.gz
            # actually stores the patient's prostate OR uterus (one or the
            # other is non-empty per patient). `prostate` is already in
            # --organs_with_tumor → kept. But `uterus_lesion` isn't pulled
            # in by any organ in the CLI list, and the
            # `organ_mask_on_lesion` step for uterus_lesion uses
            # pred_dict['prostate'] (postprocess.py:657-658), which IS in
            # keep_classes. So just hardcode `uterus_lesion` in.
            if 'uterus_lesion' in class_list:
                keep_classes.add('uterus_lesion')
            print(f'[fast] postprocess keep_classes ({len(keep_classes)}): {sorted(keep_classes)}')

        # In fast mode `pred_label` was sliced to only the kept channels in
        # sorted-full-index order (see _compute_keep_classes + the SW
        # keep_class_indices refactor). Postprocess iterates
        # `for i in range(pred.shape[0])` using `classes[i]`, so the
        # `classes` arg has to be the kept-name list in that same order —
        # not the full 74-name class_list. Build it here.
        if keep_classes is not None:
            _keep_idx_sorted = sorted(class_list.index(c) for c in keep_classes if c in class_list)
            classes_for_pp = [class_list[i] for i in _keep_idx_sorted]
        else:
            classes_for_pp = class_list

        postprocess_start = time.time()
        try:
            if 'nii.gz' in img_name:
                pred_dict = postprocess(pred_label, reoriented_itk_img, original_idx,
                origin_orientation, [1.0,1.0,1.0], classes_for_pp, args, tmp_itk_img,
                keep_classes=keep_classes)
            else:
                pred_dict = postprocess_npz(pred_label, class_list, args)
        except Exception as _post_e:
            import traceback as _tb
            print('FAILED postprocess')
            print(f'  exception: {type(_post_e).__name__}: {_post_e}')
            print(_tb.format_exc())
            with open('prediction_errors.txt', "a") as f:
                f.write(str(os.path.join(args.img_path, img_name, 'ct.nii.gz')) + "\n")
            continue

        toc = time.time()
        print(f"Time for postprocessing and saving of {img_name}: {toc - postprocess_start} seconds")
        
        if not os.path.exists(os.path.join(args.save_path, case_id, 'predictions')):
            os.makedirs(os.path.join(args.save_path, case_id, 'predictions'))
            
            
        if args.save_pancreas_lesion_only:
            tmp = {}
            for key in pred_dict.keys():
                if 'pancrea' in key:
                    tmp[key] = pred_dict[key]
            pred_dict = tmp
            
        if not args.not_save_binary:
            _binary_save_keys = _fast_save_keys(pred_dict.keys(), args)
            for key in _binary_save_keys:
                if 'nii.gz' in img_name:
                    sitk.WriteImage(pred_dict[key], os.path.join(args.save_path, case_id, 'predictions', f"{key}.nii.gz"))
                else:
                    #np.savez(os.path.join(args.save_path, case_id, 'predictions', f"{key}.npz"), pred_dict[key])
                    #use nib to save a nii.gz version too
                    nib.save(nib.Nifti1Image(pred_dict[key], np.eye(4)), os.path.join(args.save_path, case_id, 'predictions', f"{key}.nii.gz"))

        if args.save_probabilities:
            if 'nii.gz' in img_name:
                pred_raw_dict = postprocess_non_binary(pred_raw, reoriented_itk_img, original_idx,
                                            origin_orientation, [1.0,1.0,1.0], classes_for_pp, args,
                                            tmp_itk_img)
            else:
                pred_raw_dict = postprocess_npz(pred_raw, class_list, args)
            if args.save_pancreas_lesion_only:
                tmp = {}
                for key in pred_raw_dict.keys():
                    if 'pancrea' in key:
                        tmp[key] = pred_raw_dict[key]
                pred_raw_dict = tmp
            if not os.path.exists(os.path.join(args.save_path, case_id, 'predictions_raw')):
                os.makedirs(os.path.join(args.save_path, case_id, 'predictions_raw'))
            _raw_save_keys = _fast_save_keys(pred_raw_dict.keys(), args)
            for key in _raw_save_keys:
                if 'nii.gz' in img_name:
                    sitk.WriteImage(pred_raw_dict[key], os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.nii.gz"))
                else:
                    #np.savez(os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.npz"), pred_raw_dict[key])
                    #use nib to save a nii.gz version too
                    nib.save(nib.Nifti1Image(pred_raw_dict[key], np.eye(4)), os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.nii.gz"))

        if args.save_probabilities_lesions or args.save_probabilities_report_tumors_only:
            if 'nii.gz' in img_name:
                pred_raw_dict = postprocess_non_binary_lesion(pred_raw, reoriented_itk_img, original_idx,
                                            origin_orientation, [1.0,1.0,1.0], classes_for_pp, args, tmp_itk_img)
            else:
                pred_raw_dict = postprocess_npz(pred_raw, class_list, args)
            if args.save_pancreas_lesion_only:
                tmp = {}
                for key in pred_raw_dict.keys():
                    if 'pancrea' in key:
                        tmp[key] = pred_raw_dict[key]
                pred_raw_dict = tmp

            if not os.path.exists(os.path.join(args.save_path, case_id, 'predictions_raw')):
                os.makedirs(os.path.join(args.save_path, case_id, 'predictions_raw'))
            for key in pred_raw_dict.keys():
                if ('lesion' not in key) and ('pdac' not in key) and ('pnet' not in key) and ('cyst' not in key):
                    continue
                if args.save_probabilities_report_tumors_only:
                    column = f'number of {key.replace("_", " ").replace("adrenal","adrenal gland")} instances'
                    lesions= meta.loc[img_name[:len('BDMAP_00052990')], column]
                    # If more than one row matched, keep the first value (clean your data!!)
                    if isinstance(lesions, pd.Series):
                        lesions = lesions.iloc[0]
                    if lesions == 0:
                        continue
                
                if 'nii.gz' in img_name:
                    sitk.WriteImage(pred_raw_dict[key], os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.nii.gz"))
                else:
                    #np.savez(os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.npz"), pred_raw_dict[key])
                    #use nib to save a nii.gz version too
                    nib.save(nib.Nifti1Image(pred_raw_dict[key], np.eye(4)), os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.nii.gz"))
        
        if cls_out is not None:
            #save classification output
            lesion_classes = [c for c in class_list if (('lesion' in c) or ('malignant' in c) or ('benign' in c))] #same as we do in training (inside the loss function)
            cls_out = cls_out.squeeze(0)
            assert cls_out.shape[0]==len(lesion_classes), f'Classification output shape {cls_out.shape} does not match number of lesion classes {len(lesion_classes)}'
            #create a dict of class probabilities
            cls_prob_dict = {}
            for i, c in enumerate(lesion_classes):
                cls_prob_dict[c] = cls_out[i].detach().float().cpu().item()
            #save as yaml
            with open(os.path.join(args.save_path, case_id, 'cls_probs.yaml'), 'w') as f:
                yaml.dump(cls_prob_dict, f)
        
        print(f"{img_name} finished. Process time: {toc-tic}s")




