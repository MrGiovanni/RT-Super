"""
debug:
python abdomenatlas_3d.py --src_path /projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ --label_path /projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/ --tgt_path /projects/bodymaps/Pedro/data/debug_slice_preprocess/ --ids /projects/bodymaps/Pedro/data/examples_of_UCSF_with_slice.csv --label_names /projects/bodymaps/Pedro/foundational/MedFormer/dataset_conversion/label_names_all_tumors_no_lesion.yaml

for real ucsf:
python abdomenatlas_3d.py --src_path /projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ \
--label_path /projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/ \
--tgt_path projects/bodymaps/Data/UCSF_batch_1_to_5_medformer/ \
--ids /projects/bodymaps/Data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_live_latest.csv \
--label_names /projects/bodymaps/Pedro/foundational/MedFormer/dataset_conversion/label_names_all_tumors_no_lesion.yaml


"""


import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, reorient_image
import os
import random
import yaml
import copy
import numpy as np
import pdb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
from functools import partial

sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(16)  # Set the number of threads (adjust to your hardware)

essential_organs = ['spleen', 'bladder', 'gall', 'pancrea', 'kidney', 'liver', 'adrenal', 'esophagus', 'stomach', 'duodenum', 'colon', 'uterus', 'prostate']


def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.),file=''):

    imImage = reorient_image(imImage, 'RAI')
    for key in imLabel.keys():
        imLabel[key] = reorient_image(imLabel[key], 'RAI')
        if imLabel[key].GetSize() != imImage.GetSize():
            raise ValueError(f'size mismatch for {key} in {file}')

    spacing = imImage.GetSpacing()

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))


    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    im_size = re_img_xy.GetSize()
    im_spacing = re_img_xy.GetSpacing()
    re_lab_xy = {}
    for key in imLabel.keys():
        re_lab_xy[key]=ResampleLabelToRef(imLabel[key], re_img_xy, interp=sitk.sitkNearestNeighbor)
        assert re_lab_xy[key].GetSize() == im_size
        assert re_lab_xy[key].GetSpacing() == im_spacing
        
    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = {}
    for key in imLabel.keys():
        out=ResampleLabelToRef(re_lab_xy[key], re_img_xyz, interp=sitk.sitkNearestNeighbor)
        if key == 'slices' and out.GetPixelID() != sitk.sitkUInt16:
            out = sitk.Cast(out, sitk.sitkUInt16)
        re_lab_xyz[key] = out
    sitk.WriteImage(re_img_xyz, '%s/%s.nii.gz'%(save_path, name))
    for key in re_lab_xyz.keys():
        os.makedirs('%s/%s'%(save_path, name), exist_ok=True)
        sitk.WriteImage(re_lab_xyz[key], '%s/%s/%s.nii.gz'%(save_path, name, key))
        
        
def organ_fill(lab_dict, *, skip_bilateral=('kidney','lung','femur','adrenal'), on_missing='skip'):
    """
    Expand organ labels to include their lesion masks.
    - Treat 'uterus' lesions as 'prostate'.
    - Normalize 'pancreatic' -> 'pancreas'.
    - Skip bilateral organsy.
    
    lab_dict: {str -> sitk.Image} with binary (0/1 or >0) masks per organ/lesion.
    on_missing: 'skip' or 'raise' if organ mask not present.
    """
    for key in list(lab_dict.keys()):  # list() to be safe if values are updated
        if 'lesion' in key and key != 'any_lesion':
            organ_name = (
                key.replace('_lesion', '')
                   .replace('uterus', 'prostate')
                   .replace('pancreatic', 'pancreas')
                   .replace('gallbladder','gall_bladder')
            )

            # Skip bilateral organs here---we are unsure in which side the lesion is.
            if any(x in organ_name for x in skip_bilateral):
                continue

            if organ_name not in lab_dict:
                if on_missing == 'raise':
                    raise KeyError(f'Organ mask "{organ_name}" not found for lesion "{key}"')
                print(f'Skipping lesion "{key}" for missing organ "{organ_name}"')
                continue

            # Binarize and union: organ := organ OR lesion
            organ_bin  = lab_dict[organ_name] > 0
            lesion_bin = lab_dict[key]        > 0
            merged     = sitk.Cast(sitk.Or(organ_bin, lesion_bin), sitk.sitkUInt8)

            lab_dict[organ_name] = merged

    return lab_dict
    

# Thin wrapper that only changes behavior when --skip_problems is set.
def process_case_bdmap_format(name, args, overwrite=False):
    if not getattr(args, "skip_problems", False):
        # Original strict behavior: let exceptions raise
        return process_case_bdmap_format_real(name, args, overwrite)

    # Lenient path: catch and return a single-line problem report
    try:
        return process_case_bdmap_format_real(name, args, overwrite)
    except Exception as e:
        msg = f"[{name}] {type(e).__name__}: {e}"
        print(msg)
        return [msg]

# Define the processing function
def process_case_bdmap_format_real(name, args, overwrite=False):
    # Define paths for the output files
    output_ct_path = os.path.join(tgt_path, f"{name}.nii.gz")
    output_label_dir = os.path.join(tgt_path, name)

    # Check if the output CT and all labels already exist
    if os.path.exists(output_ct_path) and all(
        os.path.exists(os.path.join(output_label_dir, f"{lab_name}.nii.gz")) for lab_name in lab_name_list+['slices']
    ) and (not overwrite):
        print(f"Skipping {name}: All outputs already exist.")
        return

    # Load the CT image
    img_name = os.path.join(src_path, name, 'ct.nii.gz')
    itk_img = sitk.ReadImage(img_name)


    # --------------------------------------------------
    # 1) Create the 'any_lesion' by summing
    #    the lesion segmentations, then thresholding
    # --------------------------------------------------
    if args.global_lesion_class:
        lesion_images = []
        for lesion_name in [c for c in lab_name_list if 'lesion' in c]:
            lesion_path = os.path.join(label_path, name, 'segmentations', f"{lesion_name}.nii.gz")
            if not os.path.exists(lesion_path):
                # try predictions folder as fallback
                lesion_path = os.path.join(label_path, name, 'predictions', f"{lesion_name}.nii.gz")

            if not os.path.exists(lesion_path):
                zero_img = sitk.Image(itk_img.GetSize(), sitk.sitkUInt8)
                zero_img.CopyInformation(itk_img)
                lesion_images.append(zero_img)
            else:
                lesion_img = sitk.ReadImage(lesion_path)
                lesion_img = ResampleLabelToRef(lesion_img, itk_img, interp=sitk.sitkNearestNeighbor)
                lesion_images.append(lesion_img)

        # Sum up all three lesion images
        
        sum_lesion = sitk.Cast(lesion_images[0], sitk.sitkFloat32)
        for i in range(1, len(lesion_images)):
            sum_lesion = sitk.Add(sum_lesion, sitk.Cast(lesion_images[i], sitk.sitkFloat32))

        # Threshold > 0 to produce a binary union
        union_lesion_bin = sitk.Cast(sum_lesion > 0, sitk.sitkUInt8)

        # --------------------------------------------------
        # 2) Load all other labels in lab_name_list (except pancreatic_lesion),
        #    or create zero if missing, then store them in lab_dict.
        # --------------------------------------------------
        lab_dict = {'any_lesion': union_lesion_bin}
    else:
        lab_dict = {}


    # Prepare the label dictionary
    for lab_name in lab_name_list:
        pth = os.path.join(label_path, name, 'segmentations', f"{lab_name}.nii.gz")
        if not os.path.exists(pth):
            pth = os.path.join(label_path, name, 'predictions', f"{lab_name}.nii.gz")
        if not os.path.exists(pth):
            print(f"File {pth} does not exist")
            # Create a zero label
            l = sitk.Image(itk_img.GetSize(), sitk.sitkUInt8)
            l.CopyInformation(itk_img)
        else:
            try:
                l = sitk.ReadImage(pth)
            except:
                #label not readable. raise if the label is essential
                lesion_labels = [c for c in lab_name_list if 'lesion' in c]
                tmp = [c.replace('_lesion','').replace('gallbladder','gall_bladder').replace('uterus','prostate').replace('pancreatic','pancreas') for c in lesion_labels]
                organs_w_lesions = [c for c in lab_name_list if any(o in c for o in tmp)]
                essential=lesion_labels+organs_w_lesions+essential_organs
                if any(e in pth for e in essential):
                    raise ValueError(f'Could not read essential label {pth}')
                else: #non-essential label, continue
                    print(f'Non essential label will not open, creating an empty one: {pth}')
                    l = sitk.Image(itk_img.GetSize(), sitk.sitkUInt8)
                    l.CopyInformation(itk_img)
            l = ResampleLabelToRef(l, itk_img, interp=sitk.sitkNearestNeighbor)
        lab_dict[lab_name] = l
        
        
    # fill organs: consider that lesions in an organ is part of the organ
    if any(['lesion' in c for c in lab_name_list]):
        lab_dict = organ_fill(lab_dict)
        

    # --------------------------------------------------
    # 3) Create 'slices' label (UInt16): 1..Z from head→toe.
    #    Uses physical Z only to decide which end is the head.
    #    The stored values are slice indices (integers), not distances.
    # --------------------------------------------------
    size_x, size_y, size_z = itk_img.GetSize()

    # Determine which end (k=0 or k=Z-1) is closer to the head (higher LPS-Z).
    ci, cj = size_x // 2, size_y // 2
    z0  = itk_img.TransformIndexToPhysicalPoint((ci, cj, 0))[2]
    zN  = itk_img.TransformIndexToPhysicalPoint((ci, cj, size_z - 1))[2]
    head_at_k0 = (z0 > zN)   # in LPS, larger Z is more superior (toward the head)

    # Build per-slice integers: 1..Z with head slice = 1
    vals = np.arange(1, size_z + 1, dtype=np.uint16)
    if not head_at_k0:
        vals = vals[::-1]  # if head is at k=Z-1, reverse so that k=Z-1 → 1

    # Fill a (z, y, x) array with those integers
    slices_arr = np.empty((size_z, size_y, size_x), dtype=np.uint16)
    for k, v in enumerate(vals):
        slices_arr[k, :, :] = v

    slices_img = sitk.GetImageFromArray(slices_arr)  # UInt16 label
    slices_img.CopyInformation(itk_img)
    slices_img = sitk.Cast(slices_img, sitk.sitkUInt16)
    lab_dict['slices'] = slices_img

    # Resample the image and labels
    ResampleImage(itk_img, lab_dict, tgt_path, name, (1.0, 1.0, 1.0), name)
    print(f"{name} processed successfully.")
    return [f"{name} processed successfully."]


def process_case_nnunet_format(name,overwrite=False):
    raise ValueError('Deprecated: missing any_lesion, slices classes, organ_filling')
    lab_name_list=sorted(list(labels_nnunet.keys()))
    name = name.replace("_0000.nii.gz","").replace(".nii.gz","")
    
    try:
        # Define paths for the output files
        output_ct_path = os.path.join(tgt_path, f"{name}.nii.gz")
        output_label_dir = os.path.join(tgt_path, name)

        # Check if the output CT and all labels already exist
        if os.path.exists(output_ct_path) and all(
            os.path.exists(os.path.join(output_label_dir, f"{lab_name}.nii.gz")) for lab_name in lab_name_list
        ) and (not overwrite):
            print(f"Skipping {name}: All outputs already exist.")
            return

        # Load the CT image
        img_name = os.path.join(src_path, name+'_0000.nii.gz')
        if not os.path.exists(img_name):
            img_name = os.path.join(src_path, name+'.nii.gz')
        if not os.path.exists(img_name):
            img_name = os.path.join(src_path, name, 'ct.nii.gz')
        if not os.path.exists(img_name):
            raise ValueError(f"File {img_name} does not exist")
        itk_img = sitk.ReadImage(img_name)

        # load the nnunet labels
        pth = os.path.join(label_path, name+'_0000.nii.gz')
        if not os.path.exists(pth):
            pth = os.path.join(label_path, name+'_0000.nii.gz.nii.gz')
        if not os.path.exists(pth):
            pth = os.path.join(label_path, name+'.nii.gz')
        if not os.path.exists(pth):
            raise ValueError(f"File {pth} does not exist")
        
        itk_lab = sitk.ReadImage(pth)
        # Prepare the label dictionary
        lab_dict = {}

        for lab_name, value in labels_nnunet.items():
            itk_arr = sitk.GetArrayFromImage(itk_lab)  # Extract array

            #if the label is a list, then we need to create a mask for each element in the list
            if isinstance(value, list):
                m = np.zeros_like(itk_arr)
                for v in value:
                    m += (itk_arr == v).astype(np.uint8)
                itk_arr = (m>0).astype(np.uint8)
            else:
                # Efficient masking using single operation
                itk_arr = (itk_arr == value).astype(np.uint8)

            # Convert back to SimpleITK Image
            l = sitk.GetImageFromArray(itk_arr)

            # Preserve metadata
            l.SetOrigin(itk_lab.GetOrigin())
            l.SetSpacing(itk_lab.GetSpacing())
            l.SetDirection(itk_lab.GetDirection())

            # Store in dictionary
            lab_dict[lab_name] = l
        # Resample the image and labels
        ResampleImage(itk_img, lab_dict, tgt_path, name, (1.0, 1.0, 1.0),name)
        print(f"{name} processed successfully.")

    except Exception as e:
        print(f"Error processing {name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process AbdomenAtlas cases for MedFormer.")
    parser.add_argument("--src_path", type=str, default="/projects/bodymaps/Data/AbdomenAtlasPro/",
                        help="Source path for the CT images.")
    parser.add_argument("--label_path", type=str, default="/projects/bodymaps/Data/UFO_OrganSubSegments_nnUNet/subSegments_output/",
                        help="Label path for the segmentation masks.")
    parser.add_argument("--tgt_path", type=str, default="/projects/bodymaps/Data/UFO_27k_medformer/",
                        help="Target path for the processed outputs.")
    parser.add_argument("--parts", type=int, default=1,
                        help="Number of parts to split the dataset into (default 1, meaning no split).")
    parser.add_argument("--current_part", type=int, default=0,
                        help="The index (0-based) of the current part to process.")
    parser.add_argument("--nnunet_labels_used", action="store_true",
                        help="Flag to indicate whether to use nnUNet label formatting.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of workers")
    parser.add_argument("--overwrite", action="store_true",
                        help="Flag to indicate whether to overwrite existing files.")
    parser.add_argument("--add_lesions", type=str, default=None,
                        help="Path to a yaml file with the lesions to add to the classes list here.")
    parser.add_argument("--label_names", type=str, default=None,
                        help="Path to a yaml file with all class names to consider here. Optional.")
    parser.add_argument("--global_lesion_class", action="store_true",
                        help="Adds a lesion class, which is the sum of all labels with 'lesion' in their names.")    
    parser.add_argument("--ids", default=None,
                        help="IDS in the dataset. path to a csv with a column BDMAP ID")                 
    parser.add_argument(
        "--skip_problems",
        action="store_true",
        help="If set, do not raise on errors; collect and log problems instead."
    )

    args = parser.parse_args()

    src_path = args.src_path
    label_path = args.label_path
    tgt_path = args.tgt_path
    #cases=pd.read_csv('/projects/bodymaps/Data/UCSF_metadata_filled.csv')['BDMAP ID'].to_list()
    name_list = os.listdir(label_path)
    if args.ids is not None:
        ids=pd.read_csv(args.ids)
        name_list = [f for f in name_list if f.replace('.nii.gz','').replace('.npz','') in ids['BDMAP ID'].to_list()]
    else: 
        name_list = [f for f in name_list if 'BDMAP' in f]
    nnunet_labels_used = args.nnunet_labels_used
    print('Number of cases:', len(name_list))
    # If splitting is requested, divide the name_list accordingly.
    if args.parts > 1:
        splits = np.array_split(name_list, args.parts)
        # Ensure current_part is a valid index.
        if args.current_part < 0 or args.current_part >= len(splits):
            raise ValueError(f"current_part must be between 0 and {len(splits)-1}")
        name_list = splits[args.current_part].tolist()
        print(f"Processing part {args.current_part+1}/{args.parts} with {len(name_list)} cases.")
        
    #print([file for file in os.listdir(src_path) if file.endswith('.nii.gz') and not file.startswith('BDMAP_0000')])
    #remove the cases already predicted and saved in the tgt_path
    workers = args.workers

    if args.label_names is not None:
        with open(args.label_names, 'r') as file:
            lab_name_list = yaml.safe_load(file)
    else:
        lab_name_list = ['kidney_right',
                        'kidney_left',
                        'kidney_lesion',
                        'pancreas',
                        'pancreas_head',
                        'pancreas_body',
                        'pancreas_tail',
                        'pancreatic_lesion',
                        'liver',
                        'liver_segment_1',
                        'liver_segment_2',
                        'liver_segment_3',
                        'liver_segment_4',
                        'liver_segment_5',
                        'liver_segment_6',
                        'liver_segment_7',
                        'liver_segment_8',
                        'liver_lesion',
                        'spleen',
                        'colon',
                        'stomach',
                        'duodenum',
                        'common_bile_duct',
                        'intestine',
                        'aorta',
                        'postcava',
                        'adrenal_gland_left',
                        'adrenal_gland_right',
                        'gall_bladder',
                        'bladder',
                        'celiac_trunk',
                        'esophagus',
                        'hepatic_vessel',
                        'portal_vein_and_splenic_vein',
                        'lung_left',
                        'lung_right',
                        'prostate',
                        'rectum',
                        'femur_left',
                        'femur_right',
                        'superior_mesenteric_artery',
                        'veins']
    

    if args.add_lesions is not None:
        with open(args.add_lesions, 'r') as f:
            lesions = yaml.load(f, Loader=yaml.SafeLoader)
            lab_name_list.extend(lesions)
        #remove duplicates
        lab_name_list = list(set(lab_name_list))
        #sort
        lab_name_list = sorted(lab_name_list)

    nnunet_labels_saved = {'background': 0,
	 'aorta': 1,
	 'gall_bladder': 2,
	 'kidney_left': 3,
	 'kidney_right': 4,
	 'postcava': 5,
	 'spleen': 6,
	 'stomach': 7,
	 'adrenal_gland_left': 8,
	 'adrenal_gland_right': 9,
	 'bladder': 10,
	 'celiac_trunk': 11,
	 'colon': 12,
	 'duodenum': 13,
	 'esophagus': 14,
	 'femur_left': 15,
	 'femur_right': 16,
	 'hepatic_vessel': 17,
	 'intestine': 18,
	 'lung_left': 19,
	 'lung_right': 20,
	 'portal_vein_and_splenic_vein': 21,
	 'prostate': 22,
	 'rectum': 23,
     'liver_segment_1': 24,
     'liver_segment_2': 25,
     'liver_segment_3': 26,
     'liver_segment_4': 27,
     'liver_segment_5': 28,
     'liver_segment_6': 29,
     'liver_segment_7': 30,
     'liver_segment_8': 31,
     'pancreas_head': 32,
     'pancreas_body': 33,
     'pancreas_tail': 34,
     }
    
    joint_labels_nnunet = {'liver': [f'liver_segment_{i}' for i in range(1, 9)]+['hepatic_vessel'],
                           'pancreas': ['pancreas_head', 'pancreas_body', 'pancreas_tail']}
    
    names_labels_nnunet = set(list(nnunet_labels_saved.keys())+list(joint_labels_nnunet.keys()))-{'background','hepatic_vessel'}

    tmp = {}
    for key in names_labels_nnunet:
        if key in nnunet_labels_saved:
            tmp[key] = nnunet_labels_saved[key]
        elif key in joint_labels_nnunet:
            group_labels = [nnunet_labels_saved[name] for name in joint_labels_nnunet[key]]
            tmp[key] = group_labels
    labels_nnunet = tmp

    #name_list = os.listdir(src_path)

    os.makedirs(tgt_path+"/list/", exist_ok=True)
    if args.global_lesion_class:
        lab_name_list.append('any_lesion')
    if nnunet_labels_used:
        lab_name_list = sorted(list(labels_nnunet.keys()))
    with open(tgt_path+"/list/label_names.yaml", "w",encoding="utf-8") as f:
        yaml.dump(lab_name_list, f)
    with open(tgt_path+"/list/dataset.yaml", "w",encoding="utf-8") as f:
        yaml.dump(os.listdir(src_path), f)

    os.chdir(src_path)
    
    if nnunet_labels_used:
        process_case = process_case_nnunet_format
    else:
        process_case = process_case_bdmap_format
        

    process_case_with_overwrite = partial(process_case, args=args, overwrite=args.overwrite)

    all_problems = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in tqdm(executor.map(process_case_with_overwrite, name_list),
                        total=len(name_list), desc="Processing Cases"):
            if not res:
                continue
            # Your wrapper returns a list on success (with a "processed successfully." line)
            # and also a list of problem strings on failure. Only keep the problems.
            if isinstance(res, list):
                for line in res:
                    if isinstance(line, str) and "processed successfully." not in line:
                        all_problems.append(line)
            elif isinstance(res, str):
                # If any worker returns a single string, treat it as a problem line
                all_problems.append(res)

    if args.skip_problems:
        if all_problems:
            os.makedirs(tgt_path, exist_ok=True)
            log_path = os.path.join(tgt_path, "problems.log")
            with open(log_path, "w") as f:
                f.write("\n".join(all_problems) + "\n")
            print(f"\n{len(all_problems)} problems logged to {log_path}")
        else:
            print("\n0 problems.")