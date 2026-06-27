import argparse
import os
import yaml
from torch.utils.data import DataLoader
import tqdm

#python dataset_abdomenatlas.py --dataset abdomenatlas --model medformer --dimension 3d --batch_size 2 --crop_on_tumor --save_destination /fastwork/psalvador/JHU/data/atlas_300_medformer_augmented_npy_augmented_multich_crop_on_tumor/ --crop_on_tumor --multi_ch_tumor --workers_overwrite 10


def main():
    """
    - Parses arguments (including 'save_destination').
    - Creates dataset and dataloader.
    - Runs infinite loop to keep generating/saving augmented data to disk.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='atlas300_UFO', help='dataset name: atlas300 or atlas300_UFO')
    parser.add_argument('--model', type=str, default='medformer', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--all_train', default=True, help='Uses all dataset in training')
    parser.add_argument('--crop_on_tumor', action='store_true', help='Uses all dataset in training')
    parser.add_argument('--multi_ch_tumor', action='store_true', help='Use when predicting tumor instances, uses Hungarian algorithm for matching predictions')
    parser.add_argument('--multi_ch_tumor_classes', type=int, default=61, help='number of classes for multi channel tumor dataset') 
    parser.add_argument('--workers_overwrite', type=int, default=6, help='overwrites number of workers in config file') 
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_destination', type=str, default='/projects/bodymaps/Pedro/data/atlas_300_medformer_npy_augmented_UFO/', help='destination to save augmented data')
    parser.add_argument('--dataset_path', type=str, default=None, help='destination to save augmented data')
    parser.add_argument('--UFO_root', type=str, default=None, help='destination to save augmented data')
    parser.add_argument('--ucsf_ids', type=str, default=None, help='location of a csv file with the UFO IDs to use for training')
    parser.add_argument('--jhh_root', type=str, default=None, help='data root for JHH dataset')
    parser.add_argument('--balanced_cropper', action='store_true', help='use the new balanced cropper')
    parser.add_argument('--balance_pos_neg', action='store_true', help='blances positives and negatives')
    parser.add_argument('--class_weights', action='store_true', help='blances positives and negatives')
    parser.add_argument('--Atlas_only', action='store_true', help='blances positives and negatives')
    parser.add_argument('--UFO_only', action='store_true', help='blances positives and negatives')
    parser.add_argument('--pancreas_only', action='store_true', help='blances positives and negatives')
    parser.add_argument('--kidney_only', action='store_true', help='blances positives and negatives')
    parser.add_argument('--no_pancreas_subseg', action='store_true', help='blances positives and negatives')
    parser.add_argument('--no_mask', action='store_true', help='blances positives and negatives')
    parser.add_argument('--tumor_classes',nargs='+',
                        default=None,
                        help="List of tumor types to process"
                        )
    parser.add_argument('--reports',
                        default=None,
                        help="Path to csv with reports"
                        )
    parser.add_argument('--slice_loss', action='store_true', help='use the slice loss for training')
    parser.add_argument('--train_on_slices_only', action='store_true', help='use the slice loss for training')
    parser.add_argument('--sanity_path_debug', default='./DatasetSanityMultiTumorOnePerTumor/')
    parser.add_argument('--use_all_data', action='store_true', help='uses all data, including reports w/o tumor, for training')
    parser.add_argument('--balance_supervision_report_quality', action='store_true', help='balances the supervision according to report quality, so that reports with more tumor information are seen more often. Quality tiers: tumor size and slice, tumor size, no tumor size')
    parser.add_argument('--exclude_ids', type=str, default=None, help='ids to exclude from training, not really needed here, you can jsut use this argument in train_ddp')
    parser.add_argument('--load_malignancy', action='store_true', help='loads information about malignancy of lesions')
    parser.add_argument('--benign_maligant_only', action='store_true', help='loads only data confirmed as benign or malignant, excluding uncertain cases')
    parser.add_argument('--malignant_col', type=str, default='pathology_and_radiology_malignant', help='for a less strict malignancy definition, set as malignancy (radiology based)')
    parser.add_argument('--benign_col', type=str, default='radiology_benign_ICD_pathology_ok', help='column indicating benign cases')
    parser.add_argument('--load_per_patient', action='store_true', help='loads data per patient.')
    parser.add_argument('--maximum_loaded_per_patient', type=int, default=2, help='if loading per patient, maximum number of samples to load per patient')
    
    
    args = parser.parse_args()

    reports = args.reports

    dp = args.dataset_path
    ufo_root = args.UFO_root
    jhh_root = args.jhh_root
    args.model_genesis_pretrain=False
    args.load_clip=False
    args.clip_loss=False
    
    config_path = 'config/%s/%s_%s.yaml'%('abdomenatlas', args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    if args.multi_ch_tumor:
        #overwrites the arguments in config file
        args.classes = args.multi_ch_tumor_classes
        print('Using multi channel tumor dataset')
        print(f'Using multi channel tumor, overwriting classes to {args.classes}')
        print(f'Using multi channel tumor, overwriting data root to {args.data_root}')

    if dp is not None:
        args.data_root = dp
    if ufo_root is not None:
        args.UFO_root = ufo_root
    if jhh_root is not None:
        args.jhh_root = jhh_root
    if reports is not None:
        args.reports = reports

    # Create the training dataset
    if args.dataset == 'atlas300' or args.dataset == 'atlas':
        import training.dataset.dim3.dataset_abdomenatlas as abdomenatlas
    elif args.dataset == 'atlas300_UFO' or args.dataset == 'atlas_ufo':
        import training.dataset.dim3.dataset_abdomenatlas_UFO as abdomenatlas
        print('Running UFO+atlas')
    elif args.dataset == 'atlas_ufo_multi_tumor':
        import training.dataset.dim3.dataset_abdomenatlas_UFO_multi_tumor as abdomenatlas
    elif args.dataset == 'atlas_jhh':
        import training.dataset.dim3.dataset_abdomenatlas_JHH as abdomenatlas
        print('Running JHH+atlas')
    elif args.dataset == 'atlas_jhh_lesion_types':
        import training.dataset.dim3.dataset_abdomenatlas_JHH_types as abdomenatlas
        print('Running JHH+atlas with pdac, pnet and cyst')
    elif args.dataset == 'atlas_jhh_ufo':
        import training.dataset.dim3.dataset_abdomenatlas_JHH_UFO as abdomenatlas
        print('Running JHH+atlas+UFO')
    else:
        raise ValueError("The specified dataset doesn't exist: %s"%args.dataset)

    tumor_classes=None
    if args.tumor_classes is not None:
        tumor_classes=args.tumor_classes
        print(f'Using tumor classes: {tumor_classes}')
    if args.pancreas_only:
        tumor_classes=['pancreas']
    elif args.kidney_only:
        tumor_classes=['kidney']

    if args.Atlas_only or args.UFO_only:
        if tumor_classes is not None:
            train_dataset = abdomenatlas.AbdomenAtlasDataset(
                args=args,
                mode='train',
                crop_on_tumor=args.crop_on_tumor,
                save_destination=args.save_destination,
                load_augmented=False,  # set to True if you want to load from previously saved data
                gigantic_length=False,
                all_train=True,
                save_augmented=True,
                Atlas_only=args.Atlas_only,
                UFO_only=args.UFO_only,
                tumor_classes=tumor_classes,
                load_slices=args.slice_loss
            )
        else:
            train_dataset = abdomenatlas.AbdomenAtlasDataset(
                    args=args,
                    mode='train',
                    crop_on_tumor=args.crop_on_tumor,
                    save_destination=args.save_destination,
                    load_augmented=False,  # set to True if you want to load from previously saved data
                    gigantic_length=False,
                    all_train=True,
                    save_augmented=True,
                    Atlas_only=args.Atlas_only,
                    UFO_only=args.UFO_only,
                    load_slices=args.slice_loss
                )
    else:
        if tumor_classes is not None:
            train_dataset = abdomenatlas.AbdomenAtlasDataset(
                args=args,
                mode='train',
                crop_on_tumor=args.crop_on_tumor,
                save_destination=args.save_destination,
                load_augmented=False,  # set to True if you want to load from previously saved data
                gigantic_length=False,
                all_train=True,
                save_augmented=True,
                tumor_classes=tumor_classes,
                load_slices=args.slice_loss
            )
        else:
            train_dataset = abdomenatlas.AbdomenAtlasDataset(
                    args=args,
                    mode='train',
                    crop_on_tumor=args.crop_on_tumor,
                    save_destination=args.save_destination,
                    load_augmented=False,  # set to True if you want to load from previously saved data
                    gigantic_length=False,
                    all_train=True,
                    save_augmented=True,
                    load_slices=args.slice_loss
                )


    # Create a dataloader for the infinite loop
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers_overwrite)

    print("Starting infinite loop to generate and save augmentations...")

    try:
        # Example infinite loop
        while True:
            step_count = 0
            for batch in tqdm.tqdm(train_loader):
                # Here 'images' and 'labels' are augmented (and saved if save_destination is set)
                step_count += 1
                #print(f"Step {step_count}: Processed a batch of size {images.size(0)}")
                # You might want to break or do some other logic here

                # If you only want a certain number of steps, you can do:
                # if step_count >= 1000:
                #     break
            # or remove the above break logic to truly go infinite
    except KeyboardInterrupt:
        # This ensures we exit gracefully on Ctrl+C
        print("Caught Ctrl+C! Shutting down dataloader workers and exiting.")

if __name__ == "__main__":
    main()

