import os
import shutil
import tqdm
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor

# Target directory
tgt = '/projects/bodymaps/Pedro/data/Radiologist_annotated_300_UCSF/'
m_loc = '/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/'
ct_loc = '/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/'
pseudo_masks_ucsf = '/projects/bodymaps/Data/UCSF_pseudo_masks_subsegments/'
rad_loc = '/projects/bodymaps/Data/UFO_radiologist_annotations/Multi-Cancer_or1/'


files = [
    folder
    for folder in os.listdir(rad_loc)
    if (os.path.isdir(os.path.join(rad_loc, folder)) and ('BDMAP' in folder))
]
print(files)


files=list(set(files))
print(' files:', len(files))

predicted_ucsf=[f for f in os.listdir(pseudo_masks_ucsf) if 'BDMAP' in f]
files_atlas=[f for f in os.listdir('/projects/bodymaps/Data/AbdomenAtlas1.1Mini') if 'BDMAP' in f]

missing = list(set(files)-set(predicted_ucsf+files_atlas))
print(' missing:', len(missing))
valid_files = list(set(files)-set(missing))



print('Valid files:', len(valid_files))

# Ensure the target directory exists
os.makedirs(tgt, exist_ok=True)


# Assuming valid_files, files_atlas, files, and pth are defined
for file in tqdm.tqdm(valid_files):
    # Create required directories
    os.makedirs(os.path.join(tgt, file, 'segmentations'), exist_ok=True)
    t = os.path.join(tgt, file)
    t_seg = os.path.join(tgt, file, 'segmentations')


    shutil.copy(os.path.join(ct_loc,file, 'ct.nii.gz'), os.path.join(t, 'ct.nii.gz'))
    ct = nib.load(os.path.join(ct_loc,file, 'ct.nii.gz'))

    # Copy from AbdomenAtlas if the file exists there
    if file in files_atlas:
        for label in os.listdir(os.path.join(m_loc,file, 'segmentations')):
            if 'lung_tumor' in label:
                continue
            anno = nib.load(os.path.join(m_loc,file, 'segmentations', label))
            assert ct.shape == anno.shape, f'Size mismatch in sample {file}, label {label}'
            shutil.copy(os.path.join(m_loc,file, 'segmentations', label), os.path.join(t_seg, label))

    # Copy from UFO_27k_medformer if the file exists there
    else:
        for label in os.listdir(os.path.join(pseudo_masks_ucsf, file, 'predictions')):
            anno = nib.load(os.path.join(pseudo_masks_ucsf, file, 'predictions',label))
            assert ct.shape == anno.shape, f'Size mismatch in sample {file}, label {label}'
            shutil.copy(os.path.join(pseudo_masks_ucsf, file, 'predictions',label), os.path.join(t_seg, label))

    # Copy radiologist annotations
    for label in os.listdir(os.path.join(rad_loc, file, 'segmentations')):
        anno = nib.load(os.path.join(rad_loc, file, 'segmentations', label))
        assert ct.shape == anno.shape, f'Size mismatch in sample {file}, label {label}'
        shutil.copy(os.path.join(rad_loc, file, 'segmentations', label), os.path.join(t_seg, label))