import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
from tqdm import tqdm

def pad(img, lab):
    z, y, x = img.shape
    # pad if the image size is smaller than trainig size
    if z < 128:
        diff = int(math.ceil((128. - z) / 2)) 
        img = np.pad(img, ((diff, diff), (0,0), (0,0)))
        lab = np.pad(lab, ((0,0), (diff, diff), (0,0), (0,0)))
    if y < 128:
        diff = int(math.ceil((128. - y) / 2)) 
        img = np.pad(img, ((0,0), (diff,diff), (0,0)))
        lab = np.pad(lab, ((0,0), (0,0), (diff, diff), (0,0)))
    if x < 128:
        diff = int(math.ceil((128. - x) / 2)) 
        img = np.pad(img, ((0,0), (0,0), (diff, diff)))
        lab = np.pad(lab, ((0,0), (0,0), (0,0), (diff, diff)))

    return img, lab

dataset_list = [
            ('abdomenatlas', 'ct'),
            ]

source_path = '/mnt/ccvl15/pedro/atlas_300_medformer/'
target_path = '/mnt/ccvl15/pedro/atlas_300_medformer_npy'

os.makedirs(os.path.join(target_path), exist_ok=True)

for dataset, modality in dataset_list:
    
    #if not os.path.exists(os.path.join(target_path, 'list')):
    #    shutil.copytree(os.path.join(source_path, dataset, 'list'), os.path.join(target_path, 'list'))
    names=[name for name in os.listdir(os.path.join(source_path)) if '.nii.gz' in name]

    for idx, name in enumerate(tqdm(names)):
        img = sitk.ReadImage(os.path.join(source_path, name))
        img = sitk.GetArrayFromImage(img).astype(np.float32)

        lab=[]
        for file in os.listdir(os.path.join(source_path, name.replace('.nii.gz', ''))):
            pth=os.path.join(source_path, name.replace('.nii.gz', ''), file)
            item = sitk.ReadImage(pth)
            item = sitk.GetArrayFromImage(item).astype(np.int8)
            lab.append(item)
        lab = np.stack(lab, axis=0) #makes label multi-channel
                     
        #lab = sitk.ReadImage(os.path.join(source_path, f"BDMAP_{idx:0>8}_gt.nii.gz"))
        #lab = sitk.GetArrayFromImage(lab).astype(np.int8)
       
        if modality == 'ct':
            img = np.clip(img, -991, 500)
        else:
            percentile_2 = np.percentile(img, 2, axis=None)
            percentile_98 = np.percentile(img, 98, axis=None)
            img = np.clip(img, percentile_2, percentile_98)
            
        mean = np.mean(img)
        std = np.std(img)

        img -= mean
        img /= std

        img, lab = pad(img, lab)
        
        img, lab = img.astype(np.float32), lab.astype(np.int8)
        
        np.save(os.path.join(target_path, f"{name.replace('.nii.gz', '')}.npy"), img)
        np.save(os.path.join(target_path, f"{name.replace('.nii.gz', '')}_gt.npy"), lab)

