import builtins
import logging
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from training.losses import DiceLoss, DiceLossMultiClass
from training.validation import validation  # Might be a new or adjusted validation func
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result, 
    get_optimizer, 
    filter_validation_results,
    unwrap_model_checkpoint,
    update_ema_variables
)
import yaml
import argparse
import sys
import warnings
import matplotlib.pyplot as plt
import time
import pdb

import HungarianAlgorithm as HA  # <--- If your code is "import HungarianAlgorithm as HA"

from utils import (
    configure_logger,
    save_configure,
    is_master,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
)

warnings.filterwarnings("ignore", category=UserWarning)


def train_net(net, trainset, testset, args, ema_net=None, fold_idx=0):
    """
    Equivalent to your single-GPU 'train_net', but with DDP logic plus all the 
    multi-channel tumor, debug_val, and advanced timing features integrated.
    """

    # -------------------------------------------
    # 1) Create Distributed Samplers/Dataloaders
    # -------------------------------------------
    train_sampler = DistributedSampler(trainset) if args.distributed else None
    trainLoader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=(args.aug_device != 'gpu'),
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    test_sampler = DistributedSampler(testset) if args.distributed else None
    # batch_size=1 for 3D testing or variable input sizes
    testLoader = data.DataLoader(
        testset,
        batch_size=1,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        pin_memory=True,
        num_workers=min(2, args.num_workers),  # Or your choice
    )

    logging.info(f"Created Dataset and DataLoader (DDP mode={args.distributed})")

    # ------------------------------
    # 2) Setup TensorBoard + Opt
    # ------------------------------
    if is_master(args):
        writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}")
    else:
        writer = None

    optimizer = get_optimizer(args, net)

    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)

    # --------------------------------------------------
    # 3) Setup Loss Functions (inc. multi-ch / Hungarian)
    # --------------------------------------------------
    if args.multi_ch_tumor:
        # For multi-channel tumor, import your label_names, set up Hungarian
        with open(f"{args.data_root}/list/label_names.yaml", 'r') as f:
            class_list = yaml.load(f, Loader=yaml.SafeLoader)
            class_list = sorted(class_list)
            grouped_classes = HA.group_classes(class_list)  # depends on your code
        matcher = HA.HungarianMatcher(grouped_classes)

        # Typically we use DiceLossMultiClass + BCEWithLogits for multi-ch
        criterion_dl = DiceLossMultiClass()
    else:
        matcher = None
        criterion_dl = DiceLoss()

    # You used BCEWithLogitsLoss in your single-GPU script
    criterion_bce = nn.BCEWithLogitsLoss()

    # For AMP
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Tracking best metrics
    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    # ----------------------------------------------
    # 4) Training Loop
    # ----------------------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        curr_lr = exp_lr_scheduler_with_warmup(
            optimizer, init_lr=args.base_lr, epoch=epoch,
            warmup_epoch=5, max_epoch=args.epochs
        )
        logging.info(f"Current lr: {curr_lr:.4e}")

        # Optional debug_val: run validation before training each epoch
        if args.debug_val:
            dice_list_test, ASD_list_test, HD_list_test = validation(net, testLoader, args, matcher=matcher)
            if is_master(args):
                dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(
                    dice_list_test, ASD_list_test, HD_list_test, args
                )
                log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test_debug', epoch, args)
                logging.info(f"[DEBUG VAL] Mean dice: {dice_list_test.mean():.4f}")

        train_epoch(
            trainLoader, net, ema_net, optimizer, epoch,
            writer, criterion_bce, criterion_dl, scaler, args, matcher
        )

        # net_for_eval is either net or ema_net
        net_for_eval = ema_net if args.ema else net

        # Save the latest checkpoint from the master process
        if is_master(args):
            net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net_state_dict,
                'ema_model_state_dict': ema_net_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_latest.pth")

        # Evaluate every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            dice_list_test, ASD_list_test, HD_list_test = validation(
                net_for_eval, testLoader, args, matcher=matcher
            )
            if is_master(args):
                dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(
                    dice_list_test, ASD_list_test, HD_list_test, args
                )
                log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)

                if dice_list_test.mean() >= best_Dice.mean():
                    best_Dice = dice_list_test
                    best_HD = HD_list_test
                    best_ASD = ASD_list_test

                    net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': net_state_dict,
                        'ema_model_state_dict': ema_net_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")

                logging.info("Evaluation Done")
                logging.info(f"Dice: {dice_list_test.mean():.4f} / Best Dice: {best_Dice.mean():.4f}")

                if writer is not None:
                    writer.add_scalar('LR', curr_lr, epoch + 1)

    return best_Dice, best_HD, best_ASD


def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer,
                criterion_bce, criterion_dl, scaler, args, matcher=None):
    """
    Merged logic from your single-GPU train_epoch, including:
    - time measurement (AverageMeter, ProgressMeter)
    - multi-ch / Hungarian matching
    - dice + BCE losses
    """

    # Meters for timing and losses
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    elapsed_time_meter = AverageMeter("Elapsed Time", ":6.2f")
    remaining_time_meter = AverageMeter("Remaining Time", ":6.2f")
    progress = ProgressMeter(
        len(trainLoader) if args.dimension == '2d' else args.iter_per_epoch,
        [batch_time, epoch_loss, elapsed_time_meter, remaining_time_meter],
        prefix=f"Epoch: [{epoch + 1}]",
    )

    net.train()
    start_epoch_time = time.time()
    tic = time.time()

    iter_num_per_epoch = 0
    for i, batch_data in enumerate(trainLoader):
        img, label = batch_data[0], batch_data[1].long()

        # Move data to correct device
        img = img.cuda(args.proc_idx, non_blocking=True)
        label = label.cuda(args.proc_idx, non_blocking=True)

        step = i + epoch * len(trainLoader)  # global step
        optimizer.zero_grad()

        # -------------- Forward & Loss --------------
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = net(img)
                loss = compute_multich_loss(
                    result, label, criterion_bce, criterion_dl, matcher, args
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            result = net(img)
            loss = compute_multich_loss(
                result, label, criterion_bce, criterion_dl, matcher, args
            )
            loss.backward()
            optimizer.step()

        # EMA update
        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        # -------------- Logging & Timing --------------
        epoch_loss.update(loss.item(), img.size(0))

        batch_time.update(time.time() - tic)
        tic = time.time()

        elapsed_time = time.time() - start_epoch_time
        estimated_total_time = elapsed_time / (i + 1) * len(trainLoader)
        remaining_time = estimated_total_time - elapsed_time
        elapsed_time_meter.update(elapsed_time, n=1)
        remaining_time_meter.update(remaining_time, n=1)

        if i % args.print_freq == 0:
            progress.display(i)

        # -------------- 3D stopping early if needed  --------------
        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break

        # -------------- TB logging  --------------
        if is_master(args) and writer is not None:
            # Only do this on rank=0
            writer.add_scalar('Train/Loss', epoch_loss.avg, epoch + 1)


def compute_multich_loss(result, label, criterion_bce, criterion_dl, matcher, args):
    """
    Merges your Hungarian logic (if multi_ch_tumor=True) or normal logic otherwise.
    """
    loss = 0.0
    # result might be a tuple/list if using deep supervision
    if isinstance(result, (tuple, list)):
        for j, out_j in enumerate(result):
            if j == 0 and matcher is not None:
                out_ids, label_ids = matcher(out_j, label)
            if matcher is not None:
                r = out_j[out_ids]
                l = label[label_ids]
            else:
                r = out_j
                l = label
            # Weighted sum of BCE and dice
            loss += args.aux_weight[j] * (criterion_bce(r, l.float()) + criterion_dl(r, l))
    else:
        if matcher is not None:
            out_ids, label_ids = matcher(result, label)
            result = result[out_ids]
            label = label[label_ids]
        loss = criterion_bce(result, label.float()) + criterion_dl(result, label)
    return loss


def main_worker(proc_idx, ngpus_per_node, fold_idx, args, result_dict=None, trainset=None, testset=None):
    """
    This is the distributed entry-point function. 
    We integrate all of your new dataset arguments and logic here.
    """
    # Reproducibility
    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)
        if hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args.proc_idx = proc_idx
    args.ngpus_per_node = ngpus_per_node

    # If multiprocessing_distributed, adjust rank
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + proc_idx

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.proc_idx)

        # Adjust per‐GPU batch size/workers
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = int((args.num_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)

    # Prepare logging on master
    if is_master(args):
        args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
        os.makedirs(args.cp_dir, exist_ok=True)
        configure_logger(args.rank, args.cp_dir + f"/fold_{fold_idx}.txt")
        save_configure(args)
        logging.info(f"\nDataset: {args.dataset},\nModel: {args.model},\nDimension: {args.dimension}")
    else:
        # Non-master can also log if you want, but typically we disable or separate logs
        args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
        os.makedirs(args.cp_dir, exist_ok=True)

    # Initialize the network with everything from the single-GPU version
    net, ema_net = init_network(args)

    net.to('cuda')
    if args.ema:
        ema_net.to('cuda')

    if args.distributed:
        # Convert BN to SyncBatchNorm and wrap in DDP
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DistributedDataParallel(net, device_ids=[args.proc_idx], find_unused_parameters=True)
        if args.ema:
            ema_net = nn.SyncBatchNorm.convert_sync_batchnorm(ema_net)
            ema_net = DistributedDataParallel(ema_net, device_ids=[args.proc_idx], find_unused_parameters=True)
            for p in ema_net.parameters():
                p.requires_grad_(False)

    if is_master(args):
        logging.info("Created Model (DDP). Starting training...")

    best_Dice, best_HD, best_ASD = train_net(net, trainset, testset, args, ema_net, fold_idx=fold_idx)

    if is_master(args):
        logging.info(f"Training and evaluation on Fold {fold_idx} is done")

    # Gather results
    if args.distributed:
        if is_master(args):
            result_dict['best_Dice'] = best_Dice
            result_dict['best_HD'] = best_HD
            result_dict['best_ASD'] = best_ASD
    else:
        return best_Dice, best_HD, best_ASD


def init_network(args):
    """
    Mirrors your single-GPU init_network with multi_ch_tumor logic or not.
    """
    net = get_model(args, pretrain=args.pretrain)
    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None

    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)

    if args.torch_compile:
        net = torch.compile(net)

    return net, ema_net


def main():
    """
    Entry point that handles:
      - Parsing args
      - Setting up environment, GPU
      - Possibly running k-fold
      - Launching distributed processes
      - Summarizing cross-val results
    """
    args = get_parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_start_method('spawn', force=True)
    args.log_path = args.log_path + '%s/' % args.dataset

    # If dist_url is 'env://', world_size might come from outside
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    # K-Fold setup
    Dice_list, HD_list, ASD_list = [], [], []
    if args.use_k_fold:
        folds = range(args.k_fold)
    else:
        folds = [0]

    for fold_idx in folds:
        if args.multiprocessing_distributed:
            import multiprocessing as mp
            with mp.Manager() as manager:
                result_dict = manager.dict()
                # For multi-proc distributed, total rank = gpus * world_size
                args.world_size = ngpus_per_node * args.world_size

                # Build dataset (including 'all_train', 'crop_on_tumor')
                trainset = get_dataset(
                    args, mode='train',
                    fold_idx=fold_idx,
                    all_train=args.all_train,
                    crop_on_tumor=args.crop_on_tumor
                )
                testset = get_dataset(args, mode='test', fold_idx=fold_idx)

                mp.spawn(
                    main_worker, 
                    nprocs=ngpus_per_node,
                    args=(ngpus_per_node, fold_idx, args, result_dict, trainset, testset)
                )

                best_Dice = result_dict['best_Dice']
                best_HD = result_dict['best_HD']
                best_ASD = result_dict['best_ASD']
            args.world_size = 1  # reset if you do multiple folds
        else:
            # Single-process or single-node multi-GPU
            trainset = get_dataset(
                args, mode='train', 
                fold_idx=fold_idx,
                all_train=args.all_train,
                crop_on_tumor=args.crop_on_tumor
            )
            testset = get_dataset(args, mode='test', fold_idx=fold_idx)

            best_Dice, best_HD, best_ASD = main_worker(
                0, ngpus_per_node, fold_idx, args,
                trainset=trainset, testset=testset
            )

        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)

    # -----------
    # Summaries
    # -----------
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)

    # Save cross-validation results
    save_cv_path = f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validation.txt"
    with open(save_cv_path, 'w') as f:
        np.set_printoptions(precision=4, suppress=True)
        f.write('Dice\n')
        for i, d in enumerate(Dice_list):
            f.write(f"Fold {i}: {d}\n")
        f.write(f"Each Class Dice Avg: {np.mean(total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(total_Dice, axis=0)}\n")
        f.write(f"All classes Dice Avg: {total_Dice.mean()}\n")
        f.write(f"All classes Dice Std: {np.mean(total_Dice, axis=1).std()}\n")

        f.write("\nHD\n")
        for i, h in enumerate(HD_list):
            f.write(f"Fold {i}: {h}\n")
        f.write(f"Each Class HD Avg: {np.mean(total_HD, axis=0)}\n")
        f.write(f"Each Class HD Std: {np.std(total_HD, axis=0)}\n")
        f.write(f"All classes HD Avg: {total_HD.mean()}\n")
        f.write(f"All classes HD Std: {np.mean(total_HD, axis=1).std()}\n")

        f.write("\nASD\n")
        for i, a in enumerate(ASD_list):
            f.write(f"Fold {i}: {a}\n")
        f.write(f"Each Class ASD Avg: {np.mean(total_ASD, axis=0)}\n")
        f.write(f"Each Class ASD Std: {np.std(total_ASD, axis=0)}\n")
        f.write(f"All classes ASD Avg: {total_ASD.mean()}\n")
        f.write(f"All classes ASD Std: {np.mean(total_ASD, axis=1).std()}\n")

    print(f'All {len(folds)} folds done, cross-validation results saved to {save_cv_path}')
    sys.exit(0)


def get_parser():
    parser = argparse.ArgumentParser(description='DDP training script with multi-channel features')
    parser.add_argument('--dataset', type=str, default='acdc', help='Dataset name')
    parser.add_argument('--model', type=str, default='unet', help='Model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d or 3d model')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--torch_compile', action='store_true')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load', type=str, default=False)
    parser.add_argument('--cp_path', type=str, default='./exp/')
    parser.add_argument('--log_path', type=str, default='./log/')
    parser.add_argument('--unique_name', type=str, default='test')
    parser.add_argument('--use_k_fold', action='store_true')
    parser.add_argument('--k_fold', type=int, default=5, help='Number of folds for cross-val')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')

    # Copied from your single-GPU script
    parser.add_argument('--all_train', action='store_true')
    parser.add_argument('--crop_on_tumor', action='store_true')
    parser.add_argument('--multi_ch_tumor', action='store_true')
    parser.add_argument('--multi_ch_tumor_data_root', type=str, default='')
    parser.add_argument('--multi_ch_tumor_classes', type=int, default=61)
    parser.add_argument('--debug_val', action='store_true')

    # Dist setup
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:3456', type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--world_size', default=-1, type=int, help='Number of total processes')
    parser.add_argument('--rank', default=-1, type=int, help='Node rank for distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training')

    # Possibly from your config file:
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_alpha', type=float, default=0.99)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--iter_per_epoch', type=int, default=250)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--aux_weight', nargs='+', type=float, default=[1.0, 0.5])
    parser.add_argument('--aug_device', type=str, default='cpu')
    parser.add_argument('--proc_idx', type=int, default=0)

    parser.add_argument('--reproduce_seed', type=int, default=None)

    args = parser.parse_args()

    # Load extra config YAML
    config_path = f"config/{args.dataset}/{args.model}_{args.dimension}.yaml"
    if not os.path.exists(config_path):
        raise ValueError(f"The specified configuration doesn't exist: {config_path}")
    print(f"Loading configurations from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    for key, value in config.items():
        setattr(args, key, value)

    # If multi-ch tumor dataset is used, override classes & data_root
    if args.multi_ch_tumor:
        args.classes = args.multi_ch_tumor_classes
        args.data_root = args.multi_ch_tumor_data_root
        print('Using multi-channel tumor dataset')
        print(f'Overwriting classes to {args.classes}')
        print(f'Overwriting data_root to {args.data_root}')

    return args


if __name__ == '__main__':
    main()