import builtins
import logging
import os
import random
import time
import training.losses_foundation as lf
from collections import OrderedDict

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
#mp.set_sharing_strategy('file_system')
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from training.dataset.dim3.sampler import ChunkedSampler
from training.dataset.dim3.sampler_clip import one_organ_per_batch_sampler
from training.dataset.dim3.sampler_one_patient_per_batch import SamplerPerPatientLongitudinal

import HungarianAlgorithm as HA
import gc


from training.utils import update_ema_variables
from training.validation import validation_ddp as validation
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result, 
    get_optimizer, 
    filter_validation_results,
    unwrap_model_checkpoint,
)
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings
import matplotlib.pyplot as plt
import copy

from model.dim3.medformer import RTSuper
import copy



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

counter_mg = 0


def train_net(net, trainset, testset, args, ema_net=None, fold_idx=0):
    
    ########################################################################################
    # Dataloader Creation
    #train_sampler = DistributedSampler(trainset) if args.distributed else None
    try:
        leng = len(trainset.img_list)
    except:
        assert trainset.gigantic_length==False, 'You must set gigantic_length to False in the dataset if you want to use the dataloader with a sampler'
        leng = trainset.__len__()
    
    if args.clip_pretrain:
        train_sampler = one_organ_per_batch_sampler(
            dataset_size=leng,#real size of the dataset
            samples_per_epoch=args.iter_per_epoch*args.batch_size*args.ngpus_per_node,
            shuffle=True,
            seed=42,
            rank=dist.get_rank() if args.distributed else 0,
            world_size=dist.get_world_size() if args.distributed else 1,
            dataset = trainset,
            batch_size=args.batch_size_global)
           
        trainLoader = data.DataLoader(
            trainset, 
            batch_sampler=train_sampler,
            pin_memory=(args.aug_device != 'gpu'),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers>0),
        )
    elif args.model_genesis_pretrain:
        train_sampler = DistributedSampler(trainset) if args.distributed else None
        trainLoader = data.DataLoader(
            trainset, 
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=(args.aug_device != 'gpu'),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers>0),
        )
    elif args.time_points>1:
        train_sampler = SamplerPerPatientLongitudinal(
            dataset=trainset,
            batch_size_total=args.batch_size_global,   # TOTAL batch across all GPUs
            samples_per_epoch=args.iter_per_epoch,     # number of anchor-batches per epoch
            shuffle=True,
            seed=42,
            rank=dist.get_rank() if args.distributed else 0,
            world_size=dist.get_world_size() if args.distributed else 1,
            metadata=args.reports,
        )

        trainLoader = data.DataLoader(
            trainset,
            batch_sampler=train_sampler,               # <-- key point
            pin_memory=(args.aug_device != 'gpu'),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
        )   
    else:
        train_sampler = ChunkedSampler(
            dataset_size=leng,#real size of the dataset
            samples_per_epoch=args.iter_per_epoch*args.batch_size*args.ngpus_per_node,
            shuffle=True,
            seed=42,
            rank=dist.get_rank() if args.distributed else 0,
            world_size=dist.get_world_size() if args.distributed else 1)
        
        trainLoader = data.DataLoader(
            trainset, 
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=(args.aug_device != 'gpu'),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers>0),
        )
    
    test_sampler = DistributedSampler(testset) if args.distributed else None
    testLoader = data.DataLoader(
        testset,
        batch_size=1,  # has to be 1 sample per gpu, as the input size of 3D input is different
        shuffle=(test_sampler is None), 
        sampler=test_sampler,
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    logging.info(f"Created Dataset and DataLoader")

    ########################################################################################
    # Initialize tensorboard, optimizer, amp scaler and etc.
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}") if is_master(args) else None

    optimizer = get_optimizer(args, net)
    
    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args, net)

    #criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda().float())
    #criterion = nn.BCEWithLogitsLoss()
    #criterion_dl = DiceLossMultiClass()

    matcher=None
        
    # no scaler for bf16 training. f16 is UNSTABLE, do not use it!

    ########################################################################################
    # Start training
    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000
    
    updated_grad_stage_1 = False
    updated_grad_stage_2 = False
    
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)#this shuffles the dataset
        if hasattr(trainLoader.dataset, 'shuffle_atlas'):
            trainLoader.dataset.shuffle_atlas()

        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        #exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=args.warmup, max_epoch=args.epochs)
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, epoch=epoch, warmup_epoch=args.warmup, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")
        
        if  (epoch < args.train_full_network_epoch) and (not updated_grad_stage_1):
            if args.distributed:
                net.module.train_first_layer_only()
                net = DistributedDataParallel(net.module, device_ids=[args.proc_idx], find_unused_parameters=args.ablate_longitudinal_attention)
            else:
                net.train_first_layer_only()
            updated_grad_stage_1 = True
        if (epoch >= args.train_full_network_epoch) and (not updated_grad_stage_2):
            if args.distributed:
                net.module.restore_stage_trainability()
                net = DistributedDataParallel(net.module, device_ids=[args.proc_idx], find_unused_parameters=args.ablate_longitudinal_attention)
            else:
                net.restore_stage_trainability()
            updated_grad_stage_2 = True
                
       
        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, args,
                    matcher=matcher, loss_wrapper=None,
                    mtl_balancer=None)
        
        ##################################################################################
        # Evaluation, save checkpoint and log training info
        
        
        if is_master(args):
            # save the latest checkpoint, including net, ema_net, and optimizer
            net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net_state_dict,
                'ema_model_state_dict': ema_net_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_latest.pth")

            if (epoch+1) % 25 == 0:
                # save the model
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': net_state_dict,
                    'ema_model_state_dict': ema_net_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_epoch_{epoch+1}.pth")

        #if False:
        if (epoch+1) % args.val_freq == 0 and (not args.clip_pretrain):
            net_for_eval = ema_net if args.ema else net

            dice_list_test, ASD_list_test, HD_list_test = validation(net_for_eval, testLoader, args, matcher=matcher)
            if is_master(args):
                dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(dice_list_test, ASD_list_test, HD_list_test, args) # filter results for some dataset, e.g. amos_mr
                log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)
            
                if dice_list_test.mean() >= best_Dice.mean():
                    best_Dice = dice_list_test
                    best_HD = HD_list_test
                    best_ASD = ASD_list_test

                    # Save the checkpoint with best performance
                    net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': net_state_dict,
                        'ema_model_state_dict': ema_net_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")

                logging.info("Evaluation Done")
                logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")

                writer.add_scalar('LR', exp_scheduler, epoch+1)

        

    return best_Dice, best_HD, best_ASD

import torch
from typing import Optional, Dict, Any, List

@torch.no_grad()
def debug_grad_flow(
    model: torch.nn.Module,
    *,
    prefix: str = "",
    max_names_none: int = 50,
    max_names_zero: int = 50,
    only_rank0: bool = True,
    require_distributed: bool = False,
    print_summary: bool = True,
    # If tol is None -> check exact zeros only.
    # If tol is not None -> treat grads with max(abs(grad)) <= tol as "near-zero".
    tol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call AFTER backward().

    Reports:
      - grad=None params (no gradient flowed)
      - grad present but all-zero (or near-zero if tol is set)

    Returns a dict with counts and sampled names.
    """

    # Decide whether to run on this rank
    rank = 0
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    if require_distributed and not is_dist:
        return {"skipped": True, "reason": "distributed_not_initialized"}

    if is_dist:
        rank = torch.distributed.get_rank()
        if only_rank0 and rank != 0:
            return {"skipped": True, "reason": f"not_rank0 (rank={rank})"}

    n_params_total = 0
    n_params_requires = 0
    n_params_got = 0
    n_params_none = 0

    n_elems_requires = 0
    n_elems_got = 0
    n_elems_none = 0

    n_params_allzero = 0
    n_elems_allzero = 0

    none_names: List[str] = []
    zero_names: List[str] = []

    # Helper for "zero" check
    def is_all_zero(g: torch.Tensor) -> bool:
        if g.numel() == 0:
            return True
        if tol is None:
            # exact zero
            return torch.count_nonzero(g) == 0
        # near-zero under tolerance
        return g.abs().max().item() <= tol

    for name, p in model.named_parameters():
        n_params_total += 1
        if not p.requires_grad:
            continue

        n_params_requires += 1
        n = p.numel()
        n_elems_requires += n

        g = p.grad
        if g is None:
            n_params_none += 1
            n_elems_none += n
            if len(none_names) < max_names_none:
                none_names.append(name)
            continue

        n_params_got += 1
        n_elems_got += n

        # all-zero / near-zero grad values
        if is_all_zero(g):
            n_params_allzero += 1
            n_elems_allzero += n
            if len(zero_names) < max_names_zero:
                zero_names.append(name)

    header = f"[grad-debug]{' ' + prefix if prefix else ''}"
    if print_summary:
        mode = "exact-zero" if tol is None else f"near-zero<= {tol:g}"
        print(
            f"{header} rank={rank} | "
            f"params(require_grad)={n_params_requires} | "
            f"got_grad={n_params_got} | "
            f"no_grad={n_params_none} | "
            f"all_{mode}={n_params_allzero} | "
            f"elems(require_grad)={n_elems_requires} | "
            f"elems_got_grad={n_elems_got} | "
            f"elems_no_grad={n_elems_none} | "
            f"elems_all_{mode}={n_elems_allzero}",
            flush=True,
        )

    if n_params_none > 0:
        print(f"{header} first {len(none_names)} params with grad=None:", flush=True)
        for n in none_names:
            print(f"  - {n}", flush=True)

    if n_params_allzero > 0:
        mode = "exactly zero" if tol is None else f"near-zero (<= {tol:g})"
        print(f"{header} first {len(zero_names)} params with grad {mode}:", flush=True)
        for n in zero_names:
            print(f"  - {n}", flush=True)

    return {
        "rank": rank,
        "n_params_total": n_params_total,
        "n_params_requires_grad": n_params_requires,
        "n_params_got_grad": n_params_got,
        "n_params_no_grad": n_params_none,
        "n_params_allzero": n_params_allzero,
        "n_elems_requires_grad": n_elems_requires,
        "n_elems_got_grad": n_elems_got,
        "n_elems_no_grad": n_elems_none,
        "n_elems_allzero": n_elems_allzero,
        "no_grad_names": none_names,
        "allzero_names": zero_names,
        "tol": tol,
    }

def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, args, matcher=None, loss_wrapper=None, mtl_balancer=None):
    if mtl_balancer is not None and loss_wrapper is not None:
        raise ValueError("You cannot use both mtl and loss_wrapper at the same time")
    gc.collect()
    elapsed_time_meter = AverageMeter("Elapsed Time", ":6.2f")
    
    net.train()
    start_epoch_time = time.time()  # Track epoch start time
    loss_meters = OrderedDict()
    progress=None
    iter_num_per_epoch = 0
    for i, inputs in enumerate(trainLoader):
        patient_id = inputs['patient_id']
        date = inputs['date']
        age = inputs['age']
        name = inputs['name']
        #print(net.module)
        #layer = net.module.tumor_classifier.model.head.weight[0]
        #layer = net.module.down1.conv_blocks[0].conv1.conv.weight[0,0,0]
        #print('Weight:',net.module.tumor_classifier.model[1].head.weight[0])
        
        #if net.module.tumor_classifier.model.head.weight.grad is not None:
         #   print('grad:',net.module.tumor_classifier.model[1].head.weight.grad[0])
            
        contrast=None
        report_embeddings=None
        if 'ufo' in args.dataset:
            img = inputs["image"]
            label = inputs["label"]
            unk_voxels = inputs["unk_channels"].float()
            tumor_volumes_in_crop = inputs["volumes"].float()
            chosen_segment_mask = inputs["mask"].float()
            tumor_diameters = inputs["diameters"].float()
            if "weights" in inputs:
                class_weights = inputs["weights"].float()
                if torch.equal(class_weights, torch.ones_like(class_weights)):
                    class_weights = None
            else:
                class_weights = None
            if 'attenuation' in inputs.keys():
                tumor_attenuaton = inputs["attenuation"].float()
            else:
                tumor_attenuaton = None
            if 'diameters_per_voxel' in inputs.keys():
                tumor_volumes_in_crop_per_voxel = inputs['voumes_per_voxel']
                tumor_diameters_per_voxel = inputs['diameters_per_voxel']
            else:
                tumor_volumes_in_crop_per_voxel = None
                tumor_diameters_per_voxel = None
            if args.clip_pretrain:
                report_embeddings = inputs['clip_embedding'].float()
                report_embeddings = report_embeddings.cuda(non_blocking=True)
            if not args.model_genesis_pretrain:
                label = label.long()
            try:
                names = inputs['name']
            except KeyError:
                names = None
                
            if "slices_cropped_dict" in inputs.keys() and args.slice_loss:
                slices_cropped_dict = inputs["slices_cropped_dict"]
            else:
                slices_cropped_dict = None
                
            if "sizes_slices" in inputs.keys() and args.slice_loss:
                sizes_slices = inputs["sizes_slices"]
            else:
                sizes_slices = None
                
            if "sizes_malignancy" in inputs.keys() and args.malignancy_classification:
                sizes_malignancy = inputs["sizes_malignancy"]
            else:
                sizes_malignancy = None
                
            if "malignancy_per_voxel" in inputs.keys() and args.malignancy_classification:
                malignancy_per_voxel = inputs["malignancy_per_voxel"]
            else:
                malignancy_per_voxel = None
                
            if "contrast" in inputs.keys():
                contrast = inputs["contrast"]
            else:
                contrast = None
            
            if "annotated_per_voxel" in inputs.keys():
                annotated_per_voxel = inputs["annotated_per_voxel"]
            else:
                annotated_per_voxel = "unknown"
                
            if "age" in inputs.keys():
                age = inputs["age"]
            else:
                age = None
            if "sex" in inputs.keys():
                sex = inputs["sex"]
            else:
                sex = None

            if 'subsegment_cropped_in' in inputs.keys():
                subsegment_cropped_in = inputs['subsegment_cropped_in']
            else:
                subsegment_cropped_in = None
                
            if 'date' in inputs.keys():
                date = inputs['date']
            else:                
                dates = None
                
            if 'patient_id' in inputs.keys():
                patient_id = inputs['patient_id']
            else:                
                patient_id = None
                
            if 'organ_cropped_cannon' in inputs.keys():
                organ_cropped_cannon = inputs['organ_cropped_cannon']
            else:
                organ_cropped_cannon = None
                
            #print(f"Epoch {epoch+1}, Iteration {i+1}, BDMAP_ID: {name}; Patient ID: {patient_id}, Date: {date}, Age: {age}, Organ cropped: {subsegment_cropped_in}", flush=True, file=sys.stderr)
        
            #print('Tumor volumes in crop returned:', tumor_volumes_in_crop, flush=True, file=sys.stderr)
        elif 'jhh' in args.dataset and not args.epai_stage_2:
            img, label, unk_voxels = inputs[0], inputs[1], inputs[2].float()
            if not args.model_genesis_pretrain:
                label = label.long()
            tumor_diameters, tumor_volumes_in_crop, chosen_segment_mask, tumor_attenuaton = None, None, None, None
            tumor_volumes_in_crop_per_voxel = None
            tumor_diameters_per_voxel = None
            names=None
        elif args.epai_stage_2:
            img, label = inputs[0], inputs[1]
            unk_voxels, tumor_volumes_in_crop, chosen_segment_mask, tumor_diameters, tumor_attenuaton = None, None, None, None, None
            tumor_volumes_in_crop_per_voxel = None
            tumor_diameters_per_voxel = None
            names=None
        else:
            img, label, class_weights = inputs[0], inputs[1], inputs[2].float()
            if not args.model_genesis_pretrain:
                label = label.long()
            unk_voxels, tumor_volumes_in_crop, chosen_segment_mask, tumor_diameters, tumor_attenuaton = None, None, None, None, None
            tumor_volumes_in_crop_per_voxel = None
            tumor_diameters_per_voxel = None
            names=None
            slices_cropped_dict=None
            sizes_slices = None
            sizes_malignancy = None
            malignancy_per_voxel = None
            
        tumor_info = inputs['tumor_info_input']
            
            
        if args.epai_stage_2:
            stage_1 = label[:,-1].float().unsqueeze(1)
            #print('binary lesion label in train_ddp:', torch.unique(stage_1))
            label = label[:,:-1].long()
        else:
            stage_1 = None
            
        if args.model_genesis_pretrain:
            #moved to dataset
            #print(f'Shape of img: {img.shape}')
            #img, label = mg.generate_one_pair(img.cpu().numpy())
            #print(f'Shape of img after model genesis: {img.shape}, label: {label.shape}')
            #img=torch.from_numpy(img).cuda(non_blocking=True)
            #label=torch.from_numpy(label).cuda(non_blocking=True)
            #print('generated pair for model genesis')
            assert img.shape == label.shape, 'Image and label must have the same shape, do you apply model genesis in your dataset?'
            global counter_mg
            if counter_mg<10:
                counter_mg+=1
                #save samples for debugging
                os.makedirs('debug_model_genesis/',exist_ok=True)
                lf.save_tensor_as_nifti(img[0,0],f'debug_model_genesis/{counter_mg}_x.nii.gz')
                lf.save_tensor_as_nifti(label[0,0],f'debug_model_genesis/{counter_mg}_y.nii.gz')
            #we sustitute the image and label by the pair generated by model genesis
        
        #print('Label max and shape:', label.max(), label.shape)
        if args.aug_device != 'gpu':
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            if unk_voxels is not None:
                unk_voxels = unk_voxels.cuda(non_blocking=True)
            if tumor_volumes_in_crop is not None:
                tumor_volumes_in_crop = tumor_volumes_in_crop.cuda(non_blocking=True)
                if not args.use_all_data:
                    assert (tumor_volumes_in_crop.sum(dim=-1)>=0).all(), 'There are samples without tumor volume in the batch, use --use_all_data to accept tumors of unknown size (marked as -999999 volume)'
            if chosen_segment_mask is not None:
                chosen_segment_mask = chosen_segment_mask.cuda(non_blocking=True)
            if tumor_diameters is not None:
                tumor_diameters = tumor_diameters.cuda(non_blocking=True)
       
        step = i + epoch * len(trainLoader) # global steps
        
        optimizer.zero_grad(set_to_none=True)
        assert not torch.isnan(img).any(), 'Input is nan'
        assert torch.max(img)<=100, f'Input is bigger than 100: {torch.max(img)}'
        assert torch.min(img)>=-100, f'Input is smaller than -100: {torch.min(img)}'

        #if contrast is None:
        #    raise ValueError('No contrast loaded')
        
        if mtl_balancer is not None:
            raise ValueError('Not very usefu, not supported anymore')
        
        # get time
        start_timer = time.time()
        
        lesion_class_names = trainLoader.dataset.lesion_class_names if hasattr(trainLoader.dataset, 'lesion_class_names') else None
        classes = trainLoader.dataset.classes
        class_weights=class_weights if 'class_weights' in locals() else None

        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): #do not use float16, unstable
                print('Using amp bfloat16')
                result = net(img, tumor_info=tumor_info, stage_1_out = stage_1, labels = label, cut_attenuation_grad = (epoch < (args.warmup*4)),name=names,
                             annotated_per_voxel=annotated_per_voxel,no_mask_training=args.no_mask,
                             age=age,sex=sex,
                             dates = date, patient_ids = patient_id, cropped_organs = subsegment_cropped_in, dist = dist,
                             organ_cropped_cannon = organ_cropped_cannon) 
                
                if args.use_time_consistency_loss:
                    other_time_point=result['student'].pop('other_time_point')
                    is_paired=result['student'].pop('is_paired')
                    registration_field = result['student'].pop('registration_field_other_to_current')
                    spacing_registration = result['student'].pop('spacing_registration')
                else:
                    other_time_point = None
                    is_paired = None
                    registration_field = None
                    spacing_registration = None
                
                loss, loss_all = lf.calculate_loss_teacher_student(
                    result=result, 
                    label=label, 
                    unk_voxels=unk_voxels, 
                    args=args,
                    matcher=matcher,
                    chosen_segment_mask=chosen_segment_mask,
                    tumor_volumes_in_crop=tumor_volumes_in_crop, 
                    tumor_diameters=tumor_diameters,
                    classes=classes,
                    img=img,
                    class_weights=class_weights,
                    report_embeddings=report_embeddings, 
                    dist=dist,
                    tumor_attenuaton=tumor_attenuaton,
                    lesion_class_names=lesion_class_names,
                    tumor_volumes_in_crop_per_voxel=tumor_volumes_in_crop_per_voxel,
                    tumor_diameters_per_voxel=tumor_diameters_per_voxel,
                    names=names,
                    slices_cropped_dict=slices_cropped_dict, 
                    sizes_slices=sizes_slices,
                    sizes_malignancy=sizes_malignancy, 
                    malignancy_per_voxel=malignancy_per_voxel,
                    contrast=contrast,
                    other_time_point=other_time_point, 
                    is_paired=is_paired,
                    registration_field = registration_field, 
                    spacing_registration = spacing_registration,
                    organ_cropped_cannon = organ_cropped_cannon,
                    )
        else:
            result = net(img, tumor_info=tumor_info, stage_1_out = stage_1, labels = label, cut_attenuation_grad = (epoch < (args.warmup*4)),name=names,
                        annotated_per_voxel=annotated_per_voxel,no_mask_training=args.no_mask,
                        age=age,sex=sex,
                        dates = date, patient_ids = patient_id, cropped_organs = subsegment_cropped_in, dist = dist,
                        organ_cropped_cannon = organ_cropped_cannon) 
            
            if args.use_time_consistency_loss:
                other_time_point=result['student'].pop('other_time_point') if 'student' in result and 'other_time_point' in result['student'] else None
                is_paired=result['student'].pop('is_paired') if 'student' in result and 'is_paired' in result['student'] else None
                registration_field = result['student'].pop('registration_field_other_to_current') if 'student' in result and 'registration_field_other_to_current' in result['student'] else None
                spacing_registration = result['student'].pop('spacing_registration') if 'student' in result and 'spacing_registration' in result['student'] else None
            else:
                other_time_point = None
                is_paired = None
                registration_field = None
                spacing_registration = None
            
            loss, loss_all = lf.calculate_loss_teacher_student(
                    result=result, 
                    label=label, 
                    unk_voxels=unk_voxels, 
                    args=args,
                    matcher=matcher,
                    chosen_segment_mask=chosen_segment_mask,
                    tumor_volumes_in_crop=tumor_volumes_in_crop, 
                    tumor_diameters=tumor_diameters,
                    classes=classes,
                    img=img,
                    class_weights=class_weights,
                    report_embeddings=report_embeddings, 
                    dist=dist,
                    tumor_attenuaton=tumor_attenuaton,
                    lesion_class_names=lesion_class_names,
                    tumor_volumes_in_crop_per_voxel=tumor_volumes_in_crop_per_voxel,
                    tumor_diameters_per_voxel=tumor_diameters_per_voxel,
                    names=names,
                    slices_cropped_dict=slices_cropped_dict, 
                    sizes_slices=sizes_slices,
                    sizes_malignancy=sizes_malignancy, 
                    malignancy_per_voxel=malignancy_per_voxel,
                    contrast=contrast,
                    other_time_point=other_time_point, 
                    is_paired=is_paired,
                    registration_field = registration_field, 
                    spacing_registration = spacing_registration,
                    organ_cropped_cannon = organ_cropped_cannon,
                    )
            
        #print(f'Forward time: {time.time()-start_timer:.4f}s', flush=True, file=sys.stderr)


    
        start_bkg_timer = time.time()
        
        try:
            loss.backward()
        except:
            debug_grad_flow(net, prefix=f"e{epoch} i{i}", max_names_none=40, max_names_zero=40, tol=0.0)
            loss.backward()
        # Clip gradients before stepping the optimizer.
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
            
        #print(f'Backward time: {time.time()-start_bkg_timer:.4f}s', flush=True, file=sys.stderr)
            
        ema_log_time = time.time()
        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)
            
        if args.train_mode == 'teacher_student_ema':
            update_ema_variables(net.module.student, net.module.teacher, args.ema_alpha, step)
            

        if len(loss_meters) == 0:
            loss_meters = {k: AverageMeter(k, ":6.4f") for k in loss_all.keys()}
            loss_meters['Elapsed Time'] = AverageMeter("Elapsed Time", ":6.2f")

        for k, v in loss_all.items():
            loss_meters[k].update(v.detach().item(), img.shape[0])

        elapsed_time = time.time() - start_epoch_time

        loss_meters['Elapsed Time'].update(elapsed_time, n=1)

        if progress is None:
            progress = ProgressMeter(
                                    len(trainLoader) if args.dimension == '2d' else args.iter_per_epoch,
                                    list(loss_meters.values()),
                                    prefix=f"{args.unique_name} epoch: [{epoch + 1}]",
                                )

        if i % args.print_freq == 0:
            progress.display(i)
        
        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break

        #torch.cuda.empty_cache()
        #print(f'Last operations time: {time.time()-ema_log_time:.4f}s', flush=True, file=sys.stderr)

    if is_master(args):
        for key, meter in loss_meters.items():
            writer.add_scalar(f"Train/{key}", meter.avg, epoch+1)


def compare(net, checkpoint, max_print=200, strg=''):
    # 1) get checkpoint state_dict
    sd = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint doesn't look like a state_dict or contain one under 'state_dict'/'model'.")

    # 2) normalize keys: strip "module." if present
    def strip_module(d):
        if any(k.startswith("module.") for k in d.keys()):
            return {k[len("module."):]: v for k, v in d.items()}
        return d

    net_sd = strip_module(net.state_dict())
    ckpt_sd = strip_module(sd)

    # 3) key diffs
    net_keys = set(net_sd.keys())
    ckpt_keys = set(ckpt_sd.keys())

    missing = sorted(net_keys - ckpt_keys)
    extra   = sorted(ckpt_keys - net_keys)

    # 4) value/shape diffs on common keys
    mismatched = []
    for k in sorted(net_keys & ckpt_keys):
        a = net_sd[k].detach().cpu()
        b = ckpt_sd[k].detach().cpu()

        if a.shape != b.shape:
            mismatched.append((k, "shape", tuple(a.shape), tuple(b.shape), None))
            continue

        # compare in float32 (works fine for fp16/bf16 too)
        diff = (a.float() - b.float()).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        if max_abs != 0.0:
            mismatched.append((k, "value", tuple(a.shape), tuple(b.shape), max_abs))

    print(f"Missing in ckpt: {len(missing)} | Extra in ckpt: {len(extra)} | Mismatched: {len(mismatched)}")

    if missing:
        print("\n-- missing (net has, ckpt doesn't) --")
        for k in missing[:max_print]:
            print(k)

    if extra:
        print("\n-- extra (ckpt has, net doesn't) --")
        for k in extra[:max_print]:
            print(k)

    if mismatched:
        print("\n-- mismatched --")
        for k, kind, sh_a, sh_b, max_abs in mismatched[:max_print]:
            if kind == "shape":
                print(f"{k}: SHAPE net={sh_a} ckpt={sh_b}")
            else:
                print(f"{k}: VALUE shape={sh_a} max|diff|={max_abs:.6g}")
           
    if mismatched+missing+extra == 0:
        print("All parameters match!")
    else:
        raise ValueError(f'Parameter mismatch found: {len(missing)} missing, {len(extra)} extra, {len(mismatched)} mismatched.'+strg)

    return {"missing": missing, "extra": extra, "mismatched": mismatched}

def get_parser():
    parser = argparse.ArgumentParser(description='CBIM Meidcal Image Segmentation')
    parser.add_argument('--dataset', type=str, default='acdc', help='dataset name')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile to accelerate training, only supported by pytorch2.0')

    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='the path to save checkpoint and logging info')
    parser.add_argument('--log_path', type=str, default='./log/', help='the path to save tensorboard log')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    parser.add_argument('--use_k_fold', action='store_true', help='uses k fold cross validation')#just don't.... Use OOD evaluation instead.
    parser.add_argument('--all_train', action='store_true', help='Uses all dataset in training')
    parser.add_argument('--crop_on_tumor', action='store_true', help='Uses all dataset in training')#use this!
    parser.add_argument('--multi_ch_tumor', action='store_true', help='Use when predicting tumor instances, uses Hungarian algorithm for matching predictions')
    parser.add_argument('--multi_ch_tumor_data_root', type=str, default='/projects/bodymaps/Pedro/data/atlas_300_medformer_multi_ch_tumor_npy/', help='data root for multi channel tumor dataset')
    parser.add_argument('--multi_ch_tumor_classes', type=int, default=61, help='number of classes for multi channel tumor dataset') 
    parser.add_argument('--debug_val',  action='store_true', help='Runs validation before training')
    parser.add_argument('--workers', type=int, default=None, help='overwrites number of workers in config file') 
    parser.add_argument('--load_augmented',  action='store_true', help='Loads pre-saved crops for training. Should speed up things.')  
    parser.add_argument('--save_destination', type=str, default=None, help='destination to save augmented data or to load it')  
    parser.add_argument('--save_augmented', action='store_true', help='Saves after agumentation.')   
    parser.add_argument('--learnable_loss_weights', action='store_true', help='Allows learnable loss weigths (https://arxiv.org/pdf/1705.07115).')  
    parser.add_argument('--data_root', type=str, default=None, help='data root for dataset')
    parser.add_argument('--UFO_root', type=str, default=None, help='data root for UFO dataset')
    parser.add_argument('--jhh_root', type=str, default=None, help='data root for JHH dataset')
    parser.add_argument('--ucsf_ids', type=str, default=None, help='location of a csv file with the UFO IDs to use for training')
    parser.add_argument('--test_ids_exclude', type=str, default=None, help='location of a csv file with the test ids to exclude from training')
    parser.add_argument('--atlas_ids', type=str, default=None, help='location of a csv file with the atlas training ids to include in training')

    # NEW DDP arguments
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes for multi-node training')
    parser.add_argument('--rank', type=int, default=0, help='node rank for multi-node training')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:8001', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    
    #report_volume_loss_basic
    parser.add_argument('--report_volume_loss_basic', type=float, default=1, help='weight for the volume loss basic')
    parser.add_argument('--seg_loss', type=float, default=1, help='weight for the volume loss basic')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path') 
    parser.add_argument('--warmup', type=int, default=5, help='number of warmup epochs') 
    parser.add_argument('--loss', type=str, default='l2_entropy', help='type of loss function to use in reports') 
    parser.add_argument('--classification_branch', action='store_true', help='adds a classification branch to the model bottleneck')
    parser.add_argument('--cls_gate', action='store_true', help='multiplies the segmentation sigmoid output by the classification sigmoid output--gate')
    parser.add_argument('--cls_gate_norm', action='store_true', help='before applying the the cls gate, the segmentation output is normalized, making its maximum value above 0.5 become 1')
    parser.add_argument('--cls_on_output', action='store_true', help='if true, the classification branch is on the output of the model, otherwise it is on the bottleneck')

    #use the arguments below to load a pre-trained model and fine-tune it with a different class list. It uses output neuron keeping, which preserves weights for common classes across the old and new class lists.
    parser.add_argument('--update_output_layer', action='store_true', help='update the output layer to have the same number of classes as the number of classes in the class_list')
    parser.add_argument('--old_classes', type=str, default=None, help='old classes, we will keep weights/kernels of the old classes. This parameter should be a location of a yaml file with the old classes, we will sort them!')
    
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--classes_number', type=int, default=None, help='number of classes')
    parser.add_argument('--ball_bce_weight', type=float, default=1, help='weight for the BCE loss of the ball loss')
    parser.add_argument('--ball_dice_weight', type=float, default=1, help='weight for the Dice loss of the ball loss')
    parser.add_argument('--stardard_ce_ball', action='store_true', help='use standard cross entropy averaging inside the ball loss. Otherwise, we take the average loss for forground and background pixels independently and sum them, giving more weight to avoiding FN.')
    parser.add_argument('--lr', type=float, default=0.0006, help='learning rate')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--mtl', type=str, default=None, help='multi-task learning method. If None, no MTL. Uses method from https://github.com/SamsungLabs/MTL/')
    parser.add_argument('--balanced_cropper', action='store_true', help='use the new balanced cropper')
    parser.add_argument('--balance_pos_neg', action='store_true', help='balance healthy and disease cts')    
    parser.add_argument('--class_weights', action='store_true', help='balance classes by their frequency in the dataset. This will use the inverse frequency of each class to weight the loss function.')
    
    parser.add_argument('--epai_stage_2', action='store_true', help='uses epai stage 2 training')
    parser.add_argument('--stage_1_path', default=None, type=str, help='path to save folder of epai stage 1 results')
    parser.add_argument('--aggregator_mode', type=str, default='concat', help='mode for the aggregator')
    
    parser.add_argument('--clip_pretrain', action='store_true', help='pretrains with the clip loss')
    parser.add_argument('--clip_source', type=str, default='/projects/bodymaps/Pedro/data/report_embeddings_clinical_longformer/', help='pretrains with the clip loss')
    
    
    parser.add_argument('--no_mask', action='store_true', help='uses no segmentation mask for training, only reports')
    
    #pretrain model genesis
    parser.add_argument('--model_genesis_pretrain', action='store_true', help='skips ALL other losses, just uses model-genesis pre-training')

    parser.add_argument('--pancreas_only', action='store_true', help='trains only on the pancreas')
    parser.add_argument('--kidney_only', action='store_true', help='trains only on the kidney')
    parser.add_argument('--UFO_only', action='store_true', help='trains only on the pancreas')
    parser.add_argument('--Atlas_only', action='store_true', help='trains only on the kidney')
    parser.add_argument('--no_pancreas_subseg', action='store_true', help='blances positives and negatives')
    parser.add_argument('--ball_volume_margin', type=float, default=0.25, help='Margin of tolerance for tumor volume and diameter in the ball loss')
    parser.add_argument('--volume_loss_tolerance', type=float, default=0.25, help='Margin of tolerance for tumor volume and diameter in the ball loss')
    
    #extra classifiers on top of the segmentation output
    parser.add_argument('--attenuation_classifier', type=str, default='none')
    parser.add_argument('--train_att_MLP_on_mask_only', action='store_true', help='if true, the attenuation classifier MLP is trained only on the mask (segmentation) output. Otherwise, it is trained on mask and model outputs.')
    parser.add_argument('--att_weight', type=float, default=0.01, help='weight for the tumor attenuation loss')
    parser.add_argument('--tumor_classifier', action='store_true', help='if true, adds a tumor classifier on top of the segmentation output. The classifier classifies tumor number and diameters.')
    parser.add_argument('--cls_weight', type=float, default=0.01, help='weight for the tumor classifier loss')
    parser.add_argument('--tumor_classes',nargs='+',default=None,help="List of tumor types to process")
    parser.add_argument('--reports',default=None,help="Path to csv with reports")
    parser.add_argument('--attenuation_classifier_venous', action='store_true', help='runs attenuation classifier only on venous phase')
    

    #using slices
    parser.add_argument('--slice_loss', action='store_true', help='use the slice loss for training')
    parser.add_argument('--train_on_slices_only', action='store_true', help='use the slice loss for training')
    parser.add_argument('--sanity_path_debug', default='./DatasetSanityMultiTumorOnePerTumor/')
    parser.add_argument('--use_all_data', action='store_true', help='uses all data, including reports w/o tumor size, for training')
    parser.add_argument('--use_sample_weigths', action='store_true', help='uses weights to give more power to reports with more information (tumor slice, tumor size), using weights per sample. These weights are updated according to the number of each type of report (better reports usually are fewer).')
    parser.add_argument('--balance_supervision_report_quality', action='store_true', help='balances the supervision according to report quality, so that reports with more tumor information are seen more often. Quality tiers: tumor size and slice, tumor size, no tumor size')
    
    parser.add_argument('--atlas_meta', type=str, default=None, help='path to atlas metadata (per voxel dataset)')
    parser.add_argument('--exclude_ids', type=str, default=None, help='these ids will be excluded from the training set')
    
    parser.add_argument('--malignancy_classification', action='store_true', help='will train to differentiate between benign and malignant tumors, adds benign and malignant classes beyond the lesion classes')
    parser.add_argument('--triangle_consistency', action='store_true', help='Uses an auxiliary loss to enforce triangle coherence: lesion = benign + malignant')
    parser.add_argument('--benign_maligant_only', action='store_true', help='loads only data confirmed as benign or malignant, excluding uncertain cases')
    parser.add_argument('--malignant_col', type=str, default='pathology_and_radiology_malignant', help='for a less strict malignancy definition, set as malignancy (radiology based)')
    parser.add_argument('--benign_col', type=str, default='radiology_benign_ICD_pathology_ok', help='column indicating benign cases')
    parser.add_argument('--load_malignancy', action='store_true', help='loads information about malignancy of lesions')
    parser.add_argument('--include_ball_loss_malignancy', action='store_true', help='includes the ball loss label refinement for the malignancy classification, instead of just using distillation')
    parser.add_argument('--upsample_malig_benign', action='store_true', help='upsamples malignant and benign cases to half of dataset; incompatible with --benign_maligant_only')
    parser.add_argument('--relaxed_malignancy_col', default=None, help='set to malignancy if you want to use radiology-based malignancy in cases without pathology, instead of strict pathology-based malignancy only. PS: radiology-based malignancy do not get prioritized in data loading anyway (see clean_ufo)')
    
    
    
    parser.add_argument('--WD', type=float, default=None, help='weight decay, defaults to 0.05, like MedFormer')
    parser.add_argument('--cls_on_segmentation', action='store_true', help='if true, the classification branch is on the segmentation output')
    parser.add_argument('--binarize_cls_on_segmentation', action='store_true', help='if true, the classification branch on the segmentation output receives binary inputs (straight through trick)')
    parser.add_argument('--organ_mask_dilation', type=int, default=51, help='dilation to compensate for organ mask inaccuracies')


    #distillation
    parser.add_argument('--train_mode', type=str, default='both', help='train mode for RTSuper: student, teacher, both, or teacher_student_ema')
    parser.add_argument('--num_inpt_ch', type=int, default=21, help='number of input channels for RTSuper')
    parser.add_argument('--teacher_report_info_prob', type=float, default=1.0, help='probability of providing report information to the teacher in RTSuper (0-1 scale)')
    parser.add_argument('--student_report_info_prob', type=float, default=0.0, help='probability of providing report information to the student in RTSuper (0-1 scale)')
    parser.add_argument('--ball_distill_loss_weight', type=float, default=1.0, help='weight of the ball distillation loss')
    parser.add_argument('--standard_distill_loss_weight', type=float, default=0.1, help='weight of the standard distillation loss (BCE + Dice on outputs)')
    parser.add_argument('--feature_distill_loss_weight', type=float, default=0.01, help='weight of the feature distillation loss (L1 on features)')
    parser.add_argument('--ema_alpha', type=float, default=0.999, help='alpha for EMA teach')
    parser.add_argument('--remove_transformer_decoder', action='store_true', help='removes the transformer decoder in RTSuper teacher_decoder mode')
    parser.add_argument('--remove_dynamic_conv', action='store_true', help='removes the dynamic conv layers in RTSuper teacher_decoder mode')
    parser.add_argument('--ablate_dynamic_decoder', action='store_true', help='Ablation: replace the report-informed decoder with a static-conv decoder (drops both 3x3x3 KernelSelectorMLP and 1x1x1 transformer-generated kernels). teacher_decoder mode only.')
    parser.add_argument('--remove_size', action='store_true', help='Ablation: mask all tumor sizes in the report metadata (sets Tumor Size (mm) to "u" after sample selection). Affects both model input and loss supervision.')
    parser.add_argument('--remove_attenuation', action='store_true', help='Ablation: mask all tumor attenuations in the report metadata (sets Standardized Attenuation to "" after sample selection). Affects both model input and loss supervision.')
    parser.add_argument('--ablate_longitudinal_attention', action='store_true', help='Ablation: drop the inter-image (other-time) cross-attention in the teacher transformer decoder while keeping --time_points 2 and the time-consistency loss. Requires DDP find_unused_parameters=True; we toggle it on automatically when this flag is set.')
    #give_tumor_size_input
    parser.add_argument('--give_tumor_size_input', action='store_true', help='gives tumor size as input to the model (RTSuper). PS: if mode is teacher_decoder, this has no effect, size is given anyway.')
    parser.add_argument('--train_full_network_epoch', type=int, default=7, help='number of epochs before we start training the full network')
    parser.add_argument('--MLP_in_dim', type=int, default=23, help='dimension of the MLP in RT Super')
    parser.add_argument('--pretrained_teacher_student', type=str, default=None, help='pretrained model path for RTSuper teacher-student model')
    parser.add_argument('--never_give_size_decoder', action='store_true', help='never gives size information to the transformer decoder in RTSuper teacher_decoder mode')
    parser.add_argument('--age_and_sex_provided', action='store_true', help='provides patient age and sex to the classifier on top of the segmenter')
    parser.add_argument('--maximum_loaded_per_patient', type=int, default=2, help='if loading per patient, maximum number of samples to load per patient')
    #use_transformer_conv3
    parser.add_argument('--use_transformer_conv3', action='store_true', help='uses the transformer decoder to select the kernels used in the 3x3x3 convolutions in the report-informed decoder')
    parser.add_argument('--time_points', type=int, default=1, help='number of time points that the model will see as input')
    parser.add_argument('--time_fusion', type=str, default='transformer_decoder', help='type of time fusion: early or transformer_decoder')
    parser.add_argument('--concat_time_info', action='store_true', help='if false, the information from the other time point is only given to the transformer decoder. Otherwise, it is also concatenated in the convolutional branch.')
    
    #time-consistency loss
    parser.add_argument('--use_time_consistency_loss', action='store_true', help='uses a time consistency loss to enforce consistent predictions across time points')
    parser.add_argument('--initialization_registration', type=str, default='unigradicon', help='weights to use for initialization of the registration network')
    parser.add_argument('--registration_mode', type=str, default='image', help='deprecated')
    parser.add_argument('--registration_loss_weight', type=float, default=0.1, help='weight for the registration loss in the time consistency loss')
    parser.add_argument('--time_consistency_loss_weight', type=float, default=1.0, help='weight for the time consistency loss')
    parser.add_argument('--intersection_volume_loss_weight', type=float, default=1.0, help='weight for the time volume loss')
    parser.add_argument('--intersection_loss_weight', type=float, default=1.0, help='weight for the intersection loss in the time consistency loss')
    parser.add_argument('--restrict_volume_ball_loss_in_time', action='store_true', help='restricts the ball loss volume loss to only consider the tumor volume that is consistent across time points, to avoid penalizing true changes in tumor volume across time')
    parser.add_argument('--train_registration', action='store_true', help='train the registration network in the time consistency loss, instead of keeping it fixed')
    parser.add_argument('--pretrained_teacher_student_longitudinal', type=str, default=None, help='loads a pretrained longitudinal model, must a string pointing to the pretrained model')
    args = parser.parse_args()
    
    
    
    ema_alpha=args.ema_alpha
    atlas_meta=args.atlas_meta
    reports = args.reports
    dr = args.data_root
    epochs = args.epochs
    ufo_root = args.UFO_root
    jhh_root = args.jhh_root
    w = args.workers
    lr = args.lr
    WD = args.WD
    classes_number = args.classes_number
    
    args.clip_loss = False
    args.load_clip = False

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if w is not None:
        args.num_workers = w
        print(f'Overwriting number of workers to {w}')
    if dr is not None:
        args.data_root = dr
    if epochs is not None:
        args.epochs = epochs
    if ufo_root is not None:
        args.UFO_root = ufo_root
    if jhh_root is not None:
        args.jhh_root = jhh_root
    if classes_number is not None:
        args.classes = classes_number
    if lr is not None:
        args.base_lr = lr
        print(f'Overwriting learning rate to {lr}')
    if atlas_meta is not None:
        args.atlas_meta = atlas_meta
        print(f'Overwriting atlas meta to {atlas_meta}')
    if WD is not None:
        args.weight_decay = WD
        print(f'Overwriting weight_decay to {WD}')
            
    if reports is not None:
        args.reports = reports


    if args.epai_stage_2:
        args.update_output_layer = True
        args.classes_number = 4
        args.classification_branch = True
        args.cls_on_output = True
        
    if args.model_genesis_pretrain:
        #disable deep supervision
        args.aux_loss = False
        args.classes = 1
        args.classes_number = 1
        
    if args.clip_pretrain:
        #disable deep supervision
        args.clip_loss = True
        args.load_clip = True
        
    args.batch_size_global = args.batch_size
    
    if ema_alpha is not None:
        args.ema_alpha = ema_alpha
        
    if args.use_time_consistency_loss:
        assert args.time_points > 1, 'Time consistency loss requires time_points > 1'

    args.ema=False
        
    return args
    
    

def init_network(args,classes=None,old_classes=None):
    if args.model_genesis_pretrain:
        c = old_classes
        classes = ['model_genesis']
        print('set classes as model_genesis')
    elif args.update_output_layer and ('epai_stage_2' not in args.pretrained or args.pretrained is None):
        c = old_classes # we must load the checkpoint with the old classes
    else:
        c = classes
        
    c = sorted(c) #added sort since we sort in loss function. notice that malignancy classes will be added after sorting
        
        
    print('Old classes:', old_classes)
    
    net = get_model(args, pretrain=args.pretrain,classes=c)
    
    checkpoint = torch.load(args.pretrained)['model_state_dict']
    #compare(net, checkpoint, strg='; just after get_model') correct
        
    if args.update_output_layer or args.malignancy_classification or args.age_and_sex_provided:
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
        
        net=update_output_layer_onk(net, original_classes=old_classes, new_classes=new_classes, copy_pancreas=args.no_mask,
                                    binarize_cls_on_segmentation=args.binarize_cls_on_segmentation,age_and_sex=args.age_and_sex_provided)
        if args.pretrained and 'malig' in args.pretrained:
            #this is not really needed, we already loaded the model in net = get_model(args, pretrain=args.pretrain,classes=c)
            try:
                net.load_state_dict(torch.load(args.pretrained)['model_state_dict'], strict=False)  
            except:
                print(f'Your pretrained model has some parameter mismatch with the current model, maybe because of the output layer update.')
                net=update_output_layer_onk(net, original_classes=old_classes, new_classes=new_classes, copy_pancreas=args.no_mask,
                                    binarize_cls_on_segmentation=args.binarize_cls_on_segmentation,age_and_sex=False)
                net.load_state_dict(torch.load(args.pretrained)['model_state_dict'], strict=False)
                net=update_output_layer_onk(net, original_classes=old_classes, new_classes=new_classes, copy_pancreas=args.no_mask,
                                    binarize_cls_on_segmentation=args.binarize_cls_on_segmentation,age_and_sex=args.age_and_sex_provided)
                
        
            
    #checkpoint = torch.load(args.pretrained)['model_state_dict']
    #compare(net, checkpoint, strg='; just after update output layer')
        
    teacher_student = RTSuper(net,train_mode = args.train_mode, num_inpt_ch = args.num_inpt_ch,
            teacher_report_info_prob=args.teacher_report_info_prob, student_report_info_prob=args.student_report_info_prob,
            EMA_net = copy.deepcopy(net),
            use_transformer_decoder=not(args.remove_transformer_decoder),
            use_dynamic_conv=not(args.remove_dynamic_conv),
            MLP_in_dim=args.MLP_in_dim,
            age_and_sex_provided=args.age_and_sex_provided,
            use_transformer_conv3=args.use_transformer_conv3,
            concat_time_info=args.concat_time_info,
            ablate_dynamic_decoder=args.ablate_dynamic_decoder,
            ablate_longitudinal_attention=args.ablate_longitudinal_attention,
    )
    
    #compare(teacher_student.teacher, checkpoint, strg='; just after teacher student, testing teacher')
    #compare(teacher_student.student, checkpoint, strg='; just after teacher student, testing student')
    
    
    if args.ema:
        raise ValueError('No need for ema')
        ema_net = copy.deepcopy(teacher_student)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None
        
    if (not args.resume) and (args.pretrained_teacher_student is not None):
        checkpoint = torch.load(args.pretrained_teacher_student)
        try:
            teacher_student.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except:
            teacher_student = RTSuper(net,train_mode = args.train_mode, num_inpt_ch = args.num_inpt_ch,
                    teacher_report_info_prob=args.teacher_report_info_prob, student_report_info_prob=args.student_report_info_prob,
                    EMA_net = copy.deepcopy(net),
                    use_transformer_decoder=not(args.remove_transformer_decoder),
                    use_dynamic_conv=not(args.remove_dynamic_conv),
                    MLP_in_dim=args.MLP_in_dim,
                    age_and_sex_provided=args.age_and_sex_provided,
                    use_transformer_conv3=False,
                    concat_time_info=False,#we make longitudinal later, so we can load from a non-longitudinal model
                    ablate_dynamic_decoder=args.ablate_dynamic_decoder,
                    ablate_longitudinal_attention=args.ablate_longitudinal_attention,
            )
            teacher_student.load_state_dict(checkpoint['model_state_dict'], strict=False)
        teacher_student.teacher_report_info_prob = args.teacher_report_info_prob
        teacher_student.student_report_info_prob = args.student_report_info_prob
        teacher_student.restore_stage_trainability()
        print(f'Loaded RTSuper teacher-student pretrained model from {args.pretrained_teacher_student}', flush=True)
        
    if args.time_points > 1:
        teacher_student.make_longitudinal(
            time_points=args.time_points,
            use_transformer_decoder=not(args.remove_transformer_decoder),
            use_dynamic_conv=not(args.remove_dynamic_conv),
            age_and_sex_provided=args.age_and_sex_provided,
            use_transformer_conv3=args.use_transformer_conv3,
            concat_time_info=args.concat_time_info,)
        
        
    if args.use_time_consistency_loss:
        teacher_student.make_registration(registration_mode=args.registration_mode,
                                          registration_input_shape=[175,175,175],
                                          initialization_registration=args.initialization_registration,
                                          train_registration=args.train_registration,)
        
    
    if args.pretrained_teacher_student_longitudinal is not None:
        checkpoint = torch.load(args.pretrained_teacher_student_longitudinal)
        teacher_student.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'Loaded pretrained longitudinal RTSuper teacher-student model from {args.pretrained_teacher_student_longitudinal}', flush=True)
    
    if args.resume:
        resume_load_model_checkpoint(teacher_student, ema_net, args)

    if args.torch_compile:
        teacher_student = torch.compile(teacher_student)

    #print(net)
    #checkpoint = torch.load(args.pretrained)['model_state_dict']
    #compare(teacher_student.student, checkpoint, strg='; end of init_network')

    return teacher_student, ema_net 





def main_worker(proc_idx, ngpus_per_node, fold_idx, args, result_dict=None, trainset=None, testset=None):
    # seed each process
    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # set process specific info
    args.proc_idx = proc_idx
    args.ngpus_per_node = ngpus_per_node

    # suppress printing if not master
    if args.multiprocessing_distributed and args.proc_idx != 0:
        def print_pass(*args, **kwargs):
            pass

        #builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + proc_idx
        
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.proc_idx)

        # adjust data settings according to multi-processing
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = int((args.num_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)


    args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
    os.makedirs(args.cp_dir, exist_ok=True)
    configure_logger(args.rank, args.cp_dir+f"/fold_{fold_idx}.txt")
    save_configure(args)

    logging.info(
        f"\nDataset: {args.dataset},\n"
        + f"Model: {args.model},\n"
        + f"Dimension: {args.dimension}"
    )
    
    if args.old_classes is not None:
        with open(args.old_classes, 'r') as f:
            old_classes = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
        args.old_classes = sorted(old_classes)
        old_classes=args.old_classes
    else:
        if args.epai_stage_2:
            raise ValueError('You must provide the old classes for epai stage 2')
        old_classes = None
    net, ema_net = init_network(args,classes=trainset.classes,old_classes=old_classes)
      
    
    net.to('cuda')
    if args.ema:
        ema_net.to('cuda')
    if args.distributed:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DistributedDataParallel(net, device_ids=[args.proc_idx], find_unused_parameters=args.ablate_longitudinal_attention)
        # set find_unused_parameters to True if some of the parameters is not used in forward

        if args.ema:
            ema_net = nn.SyncBatchNorm.convert_sync_batchnorm(ema_net)
            ema_net = DistributedDataParallel(ema_net, device_ids=[args.proc_idx], find_unused_parameters=args.ablate_longitudinal_attention)
            
            for p in ema_net.parameters():
                p.requires_grad_(False)


    logging.info(f"Created Model")
    best_Dice, best_HD, best_ASD = train_net(net, trainset, testset, args, ema_net, fold_idx=fold_idx)
    
    logging.info(f"Training and evaluation on Fold {fold_idx} is done")
    
    if args.distributed:
        if is_master(args):
            # collect results from the master process
            result_dict['best_Dice'] = best_Dice
            result_dict['best_HD'] = best_HD
            result_dict['best_ASD'] = best_ASD
    else:
        return best_Dice, best_HD, best_ASD
        

        



if __name__ == '__main__':
    # parse the arguments
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_start_method('spawn')
    args.log_path = args.log_path + '%s/'%args.dataset

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed


    ngpus_per_node = torch.cuda.device_count()
    
    
    Dice_list, HD_list, ASD_list = [], [], []
    if args.use_k_fold:
        the_folds=range(args.k_fold)
    else:
        the_folds=[0]
    for fold_idx in the_folds:
        if args.multiprocessing_distributed:
            with mp.Manager() as manager:
            # use the Manager to gather results from the processes
                result_dict = manager.dict()
                    
                # Since we have ngpus_per_node processes per node, the total world_size
                # needs to be adjusted accordingly
                args.world_size = ngpus_per_node * args.world_size
                trainset = get_dataset(args, mode='train', fold_idx=fold_idx, all_train=args.all_train, crop_on_tumor=args.crop_on_tumor,
                           load_augmented=args.load_augmented, save_destination=args.save_destination,
                           save_augmented=args.save_augmented) 
                #x = trainset.__getitem__(0,BDMAP_ID='BDMAP_00055037')
                #raise ValueError(f'Debug: loaded item from trainset: \n volumes: \n {x["volumes"]}\n Diameters:\n {x["diameters"]}\n Attenuation:\n {x["attenuation"]}')
                testset = get_dataset(args, mode='test', fold_idx=fold_idx)
                # Use torch.multiprocessing.spawn to launch distributed processes:
                # the main_worker process function
                mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, fold_idx, args, result_dict, trainset, testset))
                best_Dice = result_dict['best_Dice']
                best_HD = result_dict['best_HD']
                best_ASD = result_dict['best_ASD']
            args.world_size = 1
        else:
            trainset = get_dataset(args, mode='train', fold_idx=fold_idx, all_train=args.all_train, crop_on_tumor=args.crop_on_tumor,
                           load_augmented=args.load_augmented, save_destination=args.save_destination)  
            
            #x = trainset.__getitem__(0,BDMAP_ID='BDMAP_00055037')
            #raise ValueError(f'Debug: loaded item from trainset: \n volumes: \n {x["volumes"]}\n Diameters:\n {x["diameters"]}\n Attenuation:\n {x["attenuation"]}')
            testset = get_dataset(args, mode='test', fold_idx=fold_idx)
            # Simply call main_worker function
            best_Dice, best_HD, best_ASD = main_worker(0, ngpus_per_node, fold_idx, args, trainset=trainset, testset=testset)



        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)
    
    #############################################################################################
    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)
    

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validation.txt",  'w') as f:
        np.set_printoptions(precision=4, suppress=True) 
        f.write('Dice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {Dice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(total_Dice, axis=0)}\n")
        f.write(f"All classes Dice Avg: {total_Dice.mean()}\n")
        f.write(f"All classes Dice Std: {np.mean(total_Dice, axis=1).std()}\n")

        f.write("\n")

        f.write("HD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {HD_list[i]}\n")
        f.write(f"Each Class HD Avg: {np.mean(total_HD, axis=0)}\n")
        f.write(f"Each Class HD Std: {np.std(total_HD, axis=0)}\n")
        f.write(f"All classes HD Avg: {total_HD.mean()}\n")
        f.write(f"All classes HD Std: {np.mean(total_HD, axis=1).std()}\n")

        f.write("\n")

        f.write("ASD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {ASD_list[i]}\n")
        f.write(f"Each Class ASD Avg: {np.mean(total_ASD, axis=0)}\n")
        f.write(f"Each Class ASD Std: {np.std(total_ASD, axis=0)}\n")
        f.write(f"All classes ASD Avg: {total_ASD.mean()}\n")
        f.write(f"All classes ASD Std: {np.mean(total_ASD, axis=1).std()}\n")



        
    print(f'All {args.k_fold} folds done.')

    sys.exit(0)