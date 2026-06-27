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


def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    imImage = reorient_image(imImage, 'RAI')
    for key in imLabel.keys():
        imLabel[key] = reorient_image(imLabel[key], 'RAI')

    spacing = imImage.GetSpacing()

    mx = []
    for key in imLabel.keys():
        mx.append(sitk.GetArrayFromImage(imLabel[key]).astype(np.uint8).max())
    mx = np.max(mx)

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
        re_lab_xyz[key]=ResampleLabelToRef(re_lab_xy[key], re_img_xyz, interp=sitk.sitkNearestNeighbor)
    
    if np.random.uniform() < 0.25:
        pass
    else:
        if mx == 0:
            pass
        else:
            re_img_xyz, re_lab_xyz = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 30, 30])

    sitk.WriteImage(re_img_xyz, '%s/%s.nii.gz'%(save_path, name))
    for key in re_lab_xyz.keys():
        os.makedirs('%s/%s'%(save_path, name), exist_ok=True)
        sitk.WriteImage(re_lab_xyz[key], '%s/%s/%s.nii.gz'%(save_path, name, key))

# Define the processing function
def process_case_bdmap_format(name,overwrite=False):
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
        img_name = os.path.join(src_path, name, 'ct.nii.gz')
        itk_img = sitk.ReadImage(img_name)



        # --------------------------------------------------
        # 1) Create the 'pancreatic_lesion' by summing
        #    the three lesion segmentations, then thresholding
        # --------------------------------------------------
        lesion_images = []
        for lesion_name in ['pancreatic_cyst', 'pancreatic_pdac', 'pancreatic_pnet']:
            lesion_path = os.path.join(label_path, name, 'segmentations', f"{lesion_name}.nii.gz")
            if not os.path.exists(lesion_path):
                # try predictions folder as fallback
                lesion_path = os.path.join(label_path, name, 'predictions', f"{lesion_name}.nii.gz")

            if not os.path.exists(lesion_path):
                # Create a zero label if not found
                zero_img = sitk.Image(itk_img.GetSize(), sitk.sitkUInt8)
                zero_img.CopyInformation(itk_img)  # match metadata
                lesion_images.append(zero_img)
            else:
                lesion_img = sitk.ReadImage(lesion_path)
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
        lab_dict = {'pancreatic_lesion': union_lesion_bin}

        for lab_name in lab_name_list:
            if lab_name=='pancreatic_lesion':
                continue
            pth = os.path.join(label_path, name, 'segmentations', f"{lab_name}.nii.gz")
            if not os.path.exists(pth):
                pth = os.path.join(label_path, name, 'predictions', f"{lab_name}.nii.gz")
            if not os.path.exists(pth):
                print(f"File {pth} does not exist")
                # Create a zero label
                l = sitk.Image(itk_img.GetSize(), sitk.sitkUInt8)
                l.SetSpacing(itk_img.GetSpacing())  # Match spacing
                l.SetOrigin(itk_img.GetOrigin())    # Match origin
                l.SetDirection(itk_img.GetDirection())  # Match orientation
            else:
                l = sitk.ReadImage(pth)
            lab_dict[lab_name] = l

        # Resample the image and labels
        ResampleImage(itk_img, lab_dict, tgt_path, name, (1.0, 1.0, 1.0))
        print(f"{name} processed successfully.")

    except Exception as e:
        print(f"Error processing {name}: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process AbdomenAtlas cases for MedFormer.")
    parser.add_argument("--src_path", type=str, default="/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/",
                        help="Source path for the CT images.")
    parser.add_argument("--label_path", type=str, default="/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/",
                        help="Label path for the segmentation masks.")
    parser.add_argument("--tgt_path", type=str, default="/projects/bodymaps/Pedro/data/JHH_medformer/",
                        help="Target path for the processed outputs.")
    parser.add_argument("--parts", type=int, default=1,
                        help="Number of parts to split the dataset into (default 1, meaning no split).")
    parser.add_argument("--current_part", type=int, default=0,
                        help="The index (0-based) of the current part to process.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of workers")
    parser.add_argument("--overwrite", action="store_true",
                        help="Flag to indicate whether to overwrite existing files.")
    parser.add_argument("--add_lesions", type=str, default=None,
                        help="Path to a yaml file with the lesions to add to the classes list here.")


    args = parser.parse_args()

    src_path = args.src_path
    label_path = args.label_path
    tgt_path = args.tgt_path
    #cases=pd.read_csv('/projects/bodymaps/Data/UCSF_metadata_filled.csv')['BDMAP ID'].to_list()
    name_list = [file for file in os.listdir(label_path) if (('BDMAP_A' in file) or ('BDMAP_V' in file))]
    print('Number of cases:', len(name_list))
    # If splitting is requested, divide the name_list accordingly.
    
    
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
    
    
    #remove from name_list cases already processed:
    if not args.overwrite:
        filtered_name_list = []
        for name in name_list:
            output_ct_path = os.path.join(tgt_path, f"{name}.nii.gz")
            output_label_dir = os.path.join(tgt_path, name)
            # Check if the main CT file and all label files exist
            if os.path.exists(output_ct_path) and all(
                os.path.exists(os.path.join(output_label_dir, f"{lab_name}.nii.gz")) for lab_name in lab_name_list
            ):
                print(f"Skipping {name}: All outputs already exist.")
            else:
                filtered_name_list.append(name)
        name_list = filtered_name_list
    
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
    

    with open(tgt_path+"/list/label_names.yaml", "w",encoding="utf-8") as f:
        yaml.dump(lab_name_list, f)

    #name_list = os.listdir(src_path)

    os.makedirs(tgt_path+"/list/", exist_ok=True)
    with open(tgt_path+"/list/dataset.yaml", "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)
    

    os.chdir(src_path)
    
    process_case = process_case_bdmap_format
        

    process_case_with_overwrite = partial(process_case, overwrite=args.overwrite)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm(executor.map(process_case_with_overwrite, name_list),
                    total=len(name_list), desc="Processing Cases"):
            pass


