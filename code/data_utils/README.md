# Inference nnU-Net - Create organ and sub-segment labels we need for the report loss

1- Install nnUnet (see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).

2- Run inference the trained model:

```bash
python PredictSubOrgansnUnet.py --num_parts 1 --part_id 0 --gpu 0 --pth path/to/ct/scans --outdir path/to/output --checkpoint path/to/model/weights
```

<details>
  <summary>Multi-gpu acceleration</summary>

You can run this code on multiple GPUs, each one inferencing on a different part of the dataset. Just change the `part_id` and `gpu` arguments to split the task into multiple parts.
Example of splitting it in 4 parts:

```bash
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 0 --gpu 0
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 1 --gpu 1
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 2 --gpu 2
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 3 --gpu 3
```
  
</details>

Important: If your input folder (images) are not in the standard nnU-Net format, you need to change files_input inside PredictSubOrgansnUnet.py. files_input should be a list of lists. Each of these lists should contain the path to one nii.gz file you want to inference. The variable files_output is a list of strings. It has the output locations for each of the input files. See https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/readme.md for more information.
```python
files_input = [['path/to/first/ct.nii.gz'],['path/to/second/ct.nii.gz'],...,['path/to/last/ct.nii.gz']]
files_output = ['path/to/output/first/ct.nii.gz','path/to/output/second/ct.nii.gz',...,'path/to/output/last/ct.nii.gz']
```

---
## Train nnU-Net to sub-segment organs

### prepare dataset

**This code will convert a dataset from the BDMAP format to the nnU-Net format.**

0. Define nnunet paths
```bash
export nnUNet_raw=/projects/bodymaps/Pedro/data/nnUNet_raw/
export nnUNet_preprocessed=/projects/bodymaps/Pedro/data/nnUNet_preprocessed/
export nnUNet_results=/projects/bodymaps/Pedro/data/nnUNet_results/
```

1. Combine labels. The script merges the BDMAP labes into combined labels. To change the labes used, edit the label map in combine_labels.py. The output are combined label in the BDMAP structure.

```bash
python3 combine_labels.py --dataset /path/to/dataset/in/BDMAP/format --destination /path/to/output/of/step1/ --cases /path/to/csv/with/BDMAP/ids --num_workers 10
```

2. Copy dataset to nnUNet raw folder, chaning file names to the nnUNet standard. Change paths in the beginning of the copy_dataset.py script. Target path must be in the nnunet_raw folder, and include the a dataset_id (use any number above 300) and name. E.g.: Dataset300_smallAtlas has id 300 and name smallAtlas.

```bash
python3 copy_dataset.py
```

3. Verify if mask and CT shapes match. Remove/solve unmatching cases.

```bash
python verify_data.py --dataset_dir /path/to/nnUNet_raw/dataset_with_id_and_name/imagesTr
```

4. Create a dataset json. Change the Dataset300_smallAtlas.py, change target_dataset_id, target_dataset_name and raw_dir (nnUNet raw directory). For id, put any number above 300. You will use this dataset_id in the other steps. Change the ids variable: the label map here should **match the one in step 1**. If you have label superposition, you may need to change superposing_groups too (see script)

```bash
python Dataset300_smallAtlas.py
```

5. Extract fingerprint. NP is just the number of processes.

```bash
nnUNetv2_extract_fingerprint -d dataset_id -np 15
```

6. Create plans for the nnUNet training. Here, we use ResEncL with isotropic spacing.

```bash
nnUNetv2_plan_experiment -d dataset_id -overwrite_target_spacing 1 1 1 -overwrite_plans_name nnUNetPlannerResEncL_torchres_isotropic -pl nnUNetPlannerResEncL_torchres
```

7. Preprocess the dataset. This takes a long time.

```bash
nnUNetv2_preprocess -d dataset_id -npfp 64 -np 64 -c 3d_fullres -pl nnUNetPlannerResEncL_torchres_isotropic --npz
```

### Train

```bash
nnUNetv2_train dataset_id 3d_fullres all -p nnUNetPlannerResEncL_torchres_isotropic --npz
```
