#### Install requirements

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">
    
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```
</div>
</details>

Create a new virtual environment and install all dependencies by:
```bash
conda create -n medformer python=3.10
conda activate medformer
pip install -r requirements.txt
```
#### Data preparation

Change lab_name_list in abdomenatlas_3d.py, write the names of the labels you are actually going to use. Do not worry about overlapping labels: we use sigmoid.
```bash
cd dataset_conversion
python abdomenatlas_3d.py --src_path /path/to/cts/in/BDMAP/format --label_path /path/to/labels/in/BDMAP/format --tgt_path /path/to/output --workers 16 --parts 1 --part 0
```

Change label list in nii2npy.py too. Change paths source_path and target_path, source_path should be the tgt_path of the command above
```bash
python nii2npz.py
```

#### Configuration
Change `config/abdomenatlas/medformer_3d.yaml`, change data_root to the tgt_path you set in nii2npy.py. Also change classes to the number of labels you want to use.

The training details, e.g. model hyper-parameters, training epochs, learning rate, optimizer, data augmentation, etc., can be altered here. You can try your own congiguration or use the default configure, the one we used in the MedFormer paper, which should have a decent performance. The only thing to care is the `data_root`, make sure it points to the processed dataset directory.

#### Data Augmentation --- THIS MUST BE ALWAYS KEPT RUNNING WHILE YOU ARE TRAINING MEDFORMER!

Data augmentation seems a major bottleneck here. Thus, I (Pedro) created a standalone data augmentation pipeline. This pipeline will eternally be cropping the CTs and labels, and saving the results to disk. This is a CPU-only operation, so I suggest using a CPU server for it. The training code (below) will read the saved crops and do only minimal augmentation (e.g., contrast and noise).

Go to AugmentFOREVER.sh and change the paths to data_root (the same you wrote in config/abdomenatlas/medformer_3d.yaml) and for save_destination, save_destination is the path where the crops will be saved, use a fast storage (SSD).

```bash
bash AugmentFOREVER.sh
```

#### Training

```bash
python train_ddp_standard.py --dataset abdomenatlas --model medformer --dimension 3d --batch_size 4 --unique_name name_for_your_save --crop_on_tumor --gpu '0,1' --workers 4 --load_augmented --save_destination /path/to/where/the/augmented/patches/are/saved/in/the/previous/step/
```

To continue training from an interrupted run, add:  --resume --load exp/abdomenatlas/name_for_your_save/fold_0_latest.pth

- gpus: list of gpus you want to use
- batch: total batch size, not batch per gpu. We usually can set batch as the number of GPUs times 2, but check memory usage to define this.
- Epochs and other params: check the Configuration file config/abdomenatlas/medformer_3d.yaml

#### Inference

For testing, currently we format the dataset as a single folder with many CTs inside, similar to the nnU-Net raw datasets. You do not need to, you can modify the predict_abdomenatlas.py file to load the dataset in other formats. You can have the files in nii.gz, no need for conversion.

```bash
python predict_abdomenatlas.py --load exp/abdomenatlas/name_for_your_save/fold_0_latest.pth \
--img_path /path/to/test/dataset/ \
--class_list path/to/file/with/names/of/your/labels/label_names.yaml
```

#### Evaluation (Sensitivity and Specificity)

The code below checks the saved predictions, and calculates tumor volume for a given confidence threshold (th). It saves these volumes, per sample, to a csv.

```bash
python test_with_reports.py --outputs_folder result/abdomenatlas/name_for_your_save/ \
--ct_folder /path/to/test/dataset/ --th 0.5
```

The next code read the saved volumes and a metdatada csv file that says if each case has cancer or not. It uses this information to calculate sensitivity and specificity at many **volume** thresholds. I.e., the code above uses confidence thresholds (logits->[sigmoid]->condifence scores->[confidence th]->binary mask). This code uses volume thresholds (binary mask->[volume th]->cancer/no cancer).
```bash
python calculate_sensitivity_specificity.py --ground_truth_csv /path/to/UCSF_metadata_filled.csv \
--predictions_csv result/abdomenatlas/name_for_your_save/tumor_detection_results.csv \
--output_csv result/abdomenatlas/name_for_your_save/metrics.csv
```

### Report-Enhanced Pseudo-Masks

You can use the reports to improve the final predictions of the network.

```bash
python r_super_pseudo_masks.py --pred_root /projects/bodymaps/Pedro/foundational/MedFormer/result/ufo_train_set/abdomenatlas/MULTI_TUMOR_db_Atlas_UCSF_no_liver_kidney_pancreas_basic_balanced_cropper/ --source_ct /projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ --output_folder /projects/bodymaps/Pedro/data/r_super_pseudo_masks_ufo/
```

If you predicted from resized masks (faster, using the output folder of abdomenatlas_3d.py), you need to resize the masks back to the ct size:
```bash
python r_super_pseudo_mask_resize.py --input_folder /projects/bodymaps/Pedro/data/r_super_pseudo_masks_ufo/ --output_folder /projects/bodymaps/Pedro/data/r_super_pseudo_masks_ufo_resized/ --cropped_cts /projects/bodymaps/Data/UFO_27k_medformer/ --masks_path /projects/bodymaps/Data/UCSF_pseudo_masks_subsegments/
```