import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import yaml
import nibabel as nib
import math
from . import info_nce as nce
import random
from functools import reduce
import copy
import io
from collections import defaultdict
from typing import List, Optional, Dict, Iterable
import torch
import time

def clean_name(name):
    name = name.replace('_left','').replace('_right','').replace('_head','').replace('_tail','').replace('_body','')
    if 'adrenal' in name and '_gland' not in name:
        name = name.replace('adrenal','adrenal_gland')
    if 'gallbladder' in name:
        name = name.replace('gallbladder','gall_bladder')
    if 'pancreatic' in name:
        name = name.replace('pancreatic','pancreas')
    if 'uterus' in name:
        name = 'prostate'
    for i in list(range(1,9)):
        name = name.replace(f'_segment_{str(i)}','')
    return name

def organ_name_to_idx(class_list_seg, organ_cropped_cannon):
    assert isinstance(organ_cropped_cannon, list), f'organ_cropped_cannon must be a list of strings (one per batch item), got {type(organ_cropped_cannon)}'
    B = len(organ_cropped_cannon)
    all_indices = []
    for b in range(B):
        #find indices that match organ_cropped_cannon
        cropped_name = organ_cropped_cannon[b]
        found_indices = []
        for class_idx, class_name in enumerate(class_list_seg):
            if (class_name == cropped_name) or (clean_name(class_name) == clean_name(cropped_name)):
                found_indices.append(class_idx)
        if len(found_indices) == 0:
            raise ValueError(f'Organ {cropped_name} not found in class_list_seg: {class_list_seg}')
        all_indices.append(found_indices)
    return all_indices

def get_tumor_organ_mask(organ_cropped_cannon,labels_paired,labels_other_time_paired,class_list_seg):
    organ_cropped_idx = organ_name_to_idx(class_list_seg,organ_cropped_cannon)
    label_cropped_organ = []
    label_cropped_organ_other_time = []
    for i,idx_list in enumerate(organ_cropped_idx,0):
        l = labels_paired[i, idx_list]
        l_ot = labels_other_time_paired[i, idx_list]
        if len(l.shape) == 4:
            l = l.sum(0) #sum across organ channels if multiple match the cropped organ
            l_ot = l_ot.sum(0)
        assert l.dim() == 3, f'After summing across organ channels, label should be [D,H,W], got {l.shape}'
        assert l_ot.dim() == 3, f'After summing across organ channels, label should be [D,H,W], got {l_ot.shape}'
        label_cropped_organ.append(l)
        label_cropped_organ_other_time.append(l_ot)
    label_cropped_organ = torch.stack(label_cropped_organ, dim=0).unsqueeze(1)  # [B, 1, D, H, W]
    label_cropped_organ_other_time = torch.stack(label_cropped_organ_other_time, dim=0).unsqueeze(1)  # [B, 1, D, H, W]
    return label_cropped_organ, label_cropped_organ_other_time

debug_intersection_counter=0
def intersection_loss(output,ball_mask, ball_mask_other_time_registered, sigmoid_already_applied = False,
                      debug=True,
                      debug_dicts=None, main_process = None, image= None, image_other_time=None, images_other_time_registered = None, 
                      ball_mask_other_time_non_registered = None,
                      dist=None):
    """
    This loss is the core of the time_consistency_loss.
    """
    
    
    if isinstance(output, torch.Tensor):
        outputs = [output]
    else:
        outputs = output
    
    mask1 = ball_mask
    mask2 = ball_mask_other_time_registered
    assert mask1.shape == mask2.shape, f"Masks must have the same shape, got {mask1.shape} and {mask2.shape}"
    for output in outputs:
        assert mask1.shape == output.shape, f"Batch size of masks and outputs must match, got {mask1.shape} and {output.shape}"
    assert len(mask1.shape) == 5 and mask1.shape[1] == 1, f"Masks must have 1 channel (B, 1, H, W, D); apply this to the tumor channel only. Other channels are already penalized by other losses. Shape of mask1: {mask1.shape}"
    #assert all 
    #assert binary
    assert torch.equal(mask1.bool().float(),mask1.float()), "Mask1 must be binary (0 or 1)"
    assert torch.equal(mask2.bool().float(),mask2.float()), "Mask2 must be binary (0 or 1)"
    
    joint_ball_mask = mask1 + mask2
    intersection = (joint_ball_mask == 2).float() #intersection is where both masks are 1
    union_not_intersection = (joint_ball_mask == 1).float()
    non_empty_ball_mask_other_time_registered = (ball_mask_other_time_registered.sum(dim=(-1,-2,-3),keepdim=True)>0).float().type_as(ball_mask)
    non_empty_reduced = non_empty_ball_mask_other_time_registered.squeeze(-1).squeeze(-1).squeeze(-1) #shape B,1
    
    
    ambiguous = union_not_intersection
    
    bces = []
    dices = []
    written = False
    for output in outputs:#allows for deep supervision
        #assert shape
        assert output.shape == intersection.shape, f"Output shape {output.shape} must match mask shape {intersection.shape}"
        if not sigmoid_already_applied:
            bce = F.binary_cross_entropy_with_logits(output, intersection.detach(), reduction='none')
        else:
            bce = F.binary_cross_entropy(output, intersection.detach(), reduction='none')
        bce = (1-ambiguous)*bce #do not optimize ambiguous areas
        global debug_intersection_counter
        if debug and debug_intersection_counter<30 and main_process is not None and main_process and (not written):#debug
            written = True
            #save here: bce, ambiguous, intersection, mask1, mask2, joint_ball_mask
            os.makedirs('SanityIntersectionLoss',exist_ok=True)
            os.makedirs('SanityIntersectionLoss/'+str(debug_intersection_counter),exist_ok=True)
            save_tensor_as_nifti(bce[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/bce.nii.gz')
            save_tensor_as_nifti(ambiguous[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/ambiguous.nii.gz')
            save_tensor_as_nifti(intersection[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/intersection.nii.gz')
            save_tensor_as_nifti(mask1[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/mask1.nii.gz')
            save_tensor_as_nifti(mask2[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/mask2.nii.gz')
            save_tensor_as_nifti(joint_ball_mask[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/joint_ball_mask.nii.gz')
            if images_other_time_registered is not None and len(images_other_time_registered)>0:
                save_tensor_as_nifti(images_other_time_registered[0].squeeze(0).squeeze(0),'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/image_other_time_registered.nii.gz',dist)
                save_tensor_as_nifti(image_other_time[0].squeeze(0).squeeze(0),'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/image_other_time.nii.gz',dist)
                save_tensor_as_nifti(image[0].squeeze(0).squeeze(0),'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/image.nii.gz',dist)
                save_tensor_as_nifti(ball_mask_other_time_registered[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/ball_mask_other_time_registered.nii.gz',dist)
                if ball_mask_other_time_non_registered is not None:
                    save_tensor_as_nifti(ball_mask_other_time_non_registered[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/ball_mask_other_time_non_registered.nii.gz',dist)
                save_tensor_as_nifti(ball_mask[0],'SanityIntersectionLoss/'+str(debug_intersection_counter)+'/ball_mask.nii.gz',dist)
            if debug_dicts is not None and len(debug_dicts)>0:
                with open('SanityIntersectionLoss/'+str(debug_intersection_counter)+'/debug_dict.yaml', 'w') as f:
                    yaml.dump(debug_dicts[0], f)
            
            debug_intersection_counter+=1
        assert len(bce.shape) == len(non_empty_ball_mask_other_time_registered.shape), f"BCE shape {bce.shape} must match non_empty_ball_mask_other_time_registered shape {non_empty_ball_mask_other_time_registered.shape}"
        bce =  bce * non_empty_ball_mask_other_time_registered
        bce = bce.mean(dim=(-1,-2,-3)) #mean over spatial dims, shape [B, 1]
        assert bce.shape == (output.shape[0], 1), f"Expected shape [B, 1], got {bce.shape}"
        dice = DiceLossMultiClass(preds=output, targets=intersection.detach(), 
                                known_voxels=(1-ambiguous),
                                sigmoid = (not sigmoid_already_applied),
                                reduce = False)
        assert len(dice.shape) == len(non_empty_reduced.shape), f"Dice shape {dice.shape} must match non_empty_reduced shape {non_empty_reduced.shape}"
        dice = dice * non_empty_reduced
        assert len(dice.shape) == 2 and dice.shape[1] == 1, f"We expect a 2D shape, got {dice.shape}"
        bces.append(bce)
        dices.append(dice)
    bce = sum(bces)/len(bces)
    dice = sum(dices)/len(dices)
    
    loss = {'intersection_per_voxel_bce': bce, 'intersection_per_voxel_dice': dice}
    print(f'Intersection loss - BCE: {bce}, Dice: {dice}', flush=True)
    return loss


def intersection_volume_loss(output, output_other_time_registered, mask_tumor_other_time, all_tumors_smaller_or_equal,slices_cropped_dict,args,
                             chosen_segment_mask,tumor_volumes_in_crop,labels,dist):
    if isinstance(output, torch.Tensor):
        outputs = [output]
    else:
        outputs = output
    
    assert chosen_segment_mask.shape == output_other_time_registered.shape
    assert chosen_segment_mask.shape == outputs[0].shape
    assert chosen_segment_mask.shape == labels.shape
    assert mask_tumor_other_time.shape == outputs[0].shape
    assert chosen_segment_mask.shape[1] == 1, "here, all our inputs should only have 1 channel, the lesion channel; reduce before"
    assert outputs[0].shape[1] == 1, f"here, all our inputs should only have 1 channel, the lesion channel; reduce before"
    
    output_other_time_registered = torch.sigmoid(output_other_time_registered)
    
    #mask_tumor_other_time is the ball mask for the other time point, dilated. So, it can compensate for registration errors.
    #here, we can dilate it even more, for safety. Previously, we dilated 2cm for the intersection loss. 
    #now, we use this binary mask as a shift to the soft output_other_time_registered
    #get output_other_time_registered max per batch
    max_per_batch = output_other_time_registered.view(output_other_time_registered.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
    output_other_time_registered = (output_other_time_registered + mask_tumor_other_time * max_per_batch)/2 
    #we sum and recalibrate the probs, so the maximum of output_other_time_registered does not change
    
    losses = []
    for b in range(outputs[0].shape[0]):
        #check if the other time mask is empty
        if mask_tumor_other_time[b].sum().item() == 0:
            losses.append(torch.tensor(0.0).type_as(output[0]))
            continue
        if isinstance(all_tumors_smaller_or_equal, torch.Tensor):
            cond = bool(all_tumors_smaller_or_equal[b].item())
        else:
            cond = bool(all_tumors_smaller_or_equal[b])
        if not cond:
            losses.append(torch.tensor(0.0).type_as(output[0]))
            continue
            #we only apply this loss if all tumors in the current time are smaller or equal to all tumors in the other time
            #why? because our target is: the intersection volume should match the current time tumor volume from report
        l_sup = []
        for supervision_level in range(len(outputs)):
            out = outputs[supervision_level]
            out = select_on_batch_and_expand(out, b)
            out_other = select_on_batch_and_expand(output_other_time_registered, b)
            assert out.shape == out_other.shape, f"Output shape {out.shape} must match registered other time point shape {out_other.shape}"
            soft_intersection = torch.sigmoid(out) * out_other#already applied sigmoid to output_other_time_registered
            organ_mask = select_on_batch_and_expand(chosen_segment_mask, b)
            tumor_volumes_report = select_on_batch_and_expand(tumor_volumes_in_crop,b)
            label = select_on_batch_and_expand(labels,b)
            su = slices_cropped_dict['slices_used']
            if torch.is_tensor(su):
                su_b = su[b:b+1]
            else:
                su_b = [su[b]]
            slices = {
            "binary_slices_mask": select_on_batch_and_expand(slices_cropped_dict["binary_slices_mask"], b),
            "slices_used": su_b,
            }
            assert label.sum().item() == 0, f'We do not expect tumors annotated per voxel here, and here labels should only have the tumor channel (1 ch)'
            l = volume_loss_basic(soft_intersection, organ_mask, tumor_volumes_report,
                                  label,
                                  classes = ['organ_lesion'], #we are feeding here a single-class output
                                  unk_voxels=torch.zeros_like(label),
                                  loss_function=args.loss,
                                  sigmoid=False,tolerance=args.volume_loss_tolerance,
                                  slices_cropped_dict=slices,
                                  dist=dist)
            l_sup.append(l['dice_volume_loss'])
        l_sup = torch.stack(l_sup).mean()
        losses.append(l_sup)
        
    loss = torch.stack(losses,dim=0)
    print(f'Intersection volume loss: {loss}', flush=True)
    return loss
    

def all_gather_no_grad(x: torch.Tensor, dist=None) -> torch.Tensor:
    """
    All-gather without autograd graph.
    Returns concatenated tensor across ranks along dim=0.
    """
    if dist is None or (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
        return x

    world = dist.get_world_size()
    x = x.contiguous()
    with torch.no_grad():
        gathered = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(gathered, x)
        out = torch.cat(gathered, dim=0)
    return out

def take_pairs_from_global(
    x_global: torch.Tensor,           # [G, ...]
    pair_gidx_local: torch.Tensor,    # [B] int64, -1 = no pair
) -> torch.Tensor:
    """
    Returns x_pair_local: [B, ...]
    Unpaired rows are filled with zeros.
    """
    assert pair_gidx_local.dtype == torch.int64
    B = pair_gidx_local.numel()
    device = x_global.device

    # safe indices
    idx = pair_gidx_local.clamp_min(0)  # replace -1 -> 0
    x_pair = x_global.index_select(0, idx)  # [B, ...]

    # zero out unpaired
    has_pair = (pair_gidx_local >= 0)
    if not has_pair.all():
        # broadcast mask to x_pair shape
        view = (B,) + (1,) * (x_pair.dim() - 1)
        x_pair = x_pair * has_pair.view(*view).to(x_pair.dtype)

    return x_pair

import scipy
def measure_tumor_volumes(tensor_3d, ns=None,th=None,erode=0, ct=None):
    #erode: any tumor that disapears afer the binary erosion is ignored.
    #print('Received segments:',segments)
    assert len(tensor_3d.shape)==3, 'Input tensor must be 3D (H, W, L)'
    array_3d = tensor_3d.float().cpu().numpy()
    
    from math import atan2, degrees
    
    # Define the connectivity (3x3x3 for 26-connectivity)
    structure = np.ones((3, 3, 3), dtype=int)
    
    # Label the connected components
    labeled_array, num_features = scipy.ndimage.label(array_3d, structure=structure)
    
    if num_features == 0:
        return torch.zeros(10).float()-1 #if no tumors, return -1 for all 10 volumes
    
    # Calculate the size of each connected component
    sizes = scipy.ndimage.sum(array_3d, labeled_array, range(1, num_features + 1))
    sorted_indices = np.argsort(sizes)[::-1]
    
    volumes = []
    if ns is None:
        ns=list(range(1,len(sorted_indices)+1))
    for n in ns:
        if n > num_features:
            continue
        # Find the label of the n-th largest connected component
        nth_largest_label = sorted_indices[n-1] + 1

        # Get a boolean mask of the n-th largest connected component
        nth_largest_component_mask = (labeled_array == nth_largest_label)
        
        if erode>0:
            #ignores tumors that disappear upon binary erosion
            eroded=scipy.ndimage.binary_erosion(nth_largest_component_mask,
                                                structure=np.ones((erode,erode,erode)),
                                                iterations=1)
            if eroded.sum()==0:
                continue
        
        # Measure the volume (number of voxels in the n-th largest component)
        volume = nth_largest_component_mask.sum()
        volumes.append(volume)
        
    #sort by descending volume
    volumes = sorted(volumes, reverse=True)
    
    #pad with 0 until length 10
    if len(volumes) < 10:
        volumes = volumes + [0]*(10-len(volumes))
    else:
        volumes = volumes[:10]
        
    volumes = torch.tensor(volumes).float()

    return volumes

def select_on_batch_and_expand(tensor, batch_indices):
    tensor_new = tensor[batch_indices]  # Select the desired batch items
    if tensor_new.numel() == 0:
        raise ValueError(f"No items selected. Check batch_indices: {batch_indices}")
    if len(tensor_new.shape) < len(tensor.shape):
        tensor_new = tensor_new.unsqueeze(0)
    return tensor_new
    
debug_registration_counter = 0
def time_consistency_loss(list_of_outputs,ball_mask,tumor_volumes_in_crop,label,chosen_segment_mask,
                          dist, other_time_point, is_paired, registration_field, spacing_registration, classes, 
                          sigmoid_already_applied = False,
                          organ_dsc_threshold = 0.7,
                          slices_cropped_dict=None,
                          args=None, skip_loss=False,
                          image=None,
                          organ_cropped_cannon=None,
                          class_list_seg=None,
                          names=None,
                          debug=True):
    from icon_registration.mermaidlite import compute_warped_image_multiNC
    """
    This loss calculates the intersection between the ball refined pseudo-mask for 2 time points or
    2 contrast phases (aligned with registration). Areas that intersect are deemed high-confidence 
    for the tumor. So, they are further enforced to become 1. 
    Areas that do not intersect are deemed ambiguous. So, we do not optimize them here 
    (they can be optimized in other losses though). Areas where both time points/phases are zero 
    are deemed high-confidence for the background. So, they are further enforced to become 0.
    """
    total_loss = None
    list_of_outputs = list_of_outputs[:] #make a copy
    batch_size = list_of_outputs[0].shape[0]
    
    rank = dist.get_rank() if dist and dist.is_initialized() else -1
    #def info(t):
    #    return (tuple(t.shape), str(t.dtype), str(t.device))
    #print(f"[rank {rank}] pre-gather "
    #    f"ball={info(ball_mask)} "
    #    f"vol={info(tumor_volumes_in_crop)} "
    #    f"label={info(label)} "
    #    f"chosen={info(chosen_segment_mask)} "
    #    f"out0={info(list_of_outputs[0] if isinstance(list_of_outputs, list) else list_of_outputs)} "
    #    f"img={(info(image) if isinstance(image, torch.Tensor) else type(image))}",
    #    flush=True)
    
    #print('Gathering started for time consistency loss', flush=True)
    #start with the all gathers to grab the other time point for the ball refined mask,
    #tumor volume in crop, and tumor volume per voxel
    #fist we need to gather the ball mask for this time point and the other time point:
    ball_mask_all = all_gather_no_grad(ball_mask, dist=dist)
    #print('Gathering done for ball_mask_all', flush=True)
    
    #now also gather tumor_volumes_in_crop, this is a torch tensor, so we can easily all gather in ddp
    tumor_volumes_in_crop_all = all_gather_no_grad(tumor_volumes_in_crop, dist=dist)
    #print('Gathering done for tumor_volumes_in_crop_all', flush=True)
    
    #now the labels, this is important when the other time point is annotated per voxel
    label_all = all_gather_no_grad(label, dist=dist)
    #print('Gathering done for label_all', flush=True)
    
    #now the chosen_segment_mask, this is mainly important here to define which is the tumor channel. 
    #The losses here will operate in this channels, as others (background) are already penalized by other losses
    #also, the losses here shall not be applied to cases annotated per voxel
    chosen_segment_mask_all = all_gather_no_grad(chosen_segment_mask, dist=dist)
    #print('Gathering done for chosen_segment_mask_all', flush=True)
    
    teacher_last_layer = list_of_outputs[0]
    teacher_last_layer_all = all_gather_no_grad(teacher_last_layer, dist=dist)
    #print('Gathering done for teacher_last_layer_all', flush=True)
    
    image_other_time_all = all_gather_no_grad(image, dist=dist)
    #print('Gathering done for image_other_time_all', flush=True)
    
    names_all = all_gather_object_no_grad(names, dist=dist)
    names_global = []
    for obj in names_all:
        if obj is None:
            continue
        if isinstance(obj, (list, tuple)):
            names_global.extend(list(obj))
        else:
            names_global.append(obj)
    
    if is_paired is None or (not is_paired.any()):
        #we do this after all_gather so we run the all_gather on all DDP processes (necessary?)
        return_dict = {}
        if args.intersection_loss_weight > 0:
            return_dict['intersection_per_voxel_bce'] = torch.tensor(0.0).type_as(ball_mask)
            return_dict['intersection_per_voxel_dice'] = torch.tensor(0.0).type_as(ball_mask)
        if args.intersection_volume_loss_weight > 0:
            return_dict['intersection_volume_loss'] = torch.tensor(0.0).type_as(ball_mask)
        if skip_loss:
            return None, None, None, None
        return torch.tensor(0.0).type_as(ball_mask), return_dict
    
    batch_size = ball_mask.shape[0]
    assert registration_field.shape[0] == batch_size, "Registration field batch size must match ball mask batch size"
        
    #take the pairs for the other time point:
    ball_mask_other_time = take_pairs_from_global(ball_mask_all, other_time_point)
    tumor_volumes_in_crop_other_time = take_pairs_from_global(tumor_volumes_in_crop_all, other_time_point)
    label_other_time = take_pairs_from_global(label_all, other_time_point)
    chosen_segment_mask_other_time = take_pairs_from_global(chosen_segment_mask_all, other_time_point)
    teacher_last_layer_other_time = take_pairs_from_global(teacher_last_layer_all, other_time_point)
    image_other_time = select_on_batch_and_expand(image_other_time_all, other_time_point)
    
    ball_mask_other_time = select_on_batch_and_expand(ball_mask_other_time, is_paired)
    tumor_volumes_in_crop_other_time = select_on_batch_and_expand(tumor_volumes_in_crop_other_time, is_paired)
    label_other_time = select_on_batch_and_expand(label_other_time, is_paired)
    chosen_segment_mask_other_time = select_on_batch_and_expand(chosen_segment_mask_other_time, is_paired)
    teacher_last_layer_other_time = select_on_batch_and_expand(teacher_last_layer_other_time, is_paired)
    image_other_time = select_on_batch_and_expand(image_other_time, is_paired)
    
    #also take only paired samples from the current time point:
    image = select_on_batch_and_expand(image, is_paired)
    ball_mask = select_on_batch_and_expand(ball_mask, is_paired)
    tumor_volumes_in_crop = select_on_batch_and_expand(tumor_volumes_in_crop, is_paired)
    label = select_on_batch_and_expand(label, is_paired)
    chosen_segment_mask = select_on_batch_and_expand(chosen_segment_mask, is_paired)
    for i in range(len(list_of_outputs)):
        list_of_outputs[i] = select_on_batch_and_expand(list_of_outputs[i], is_paired)
    teacher_last_layer = select_on_batch_and_expand(teacher_last_layer, is_paired)
    registration_field = select_on_batch_and_expand(registration_field, is_paired)
    
    label_cropped_organ, label_cropped_organ_other_time = get_tumor_organ_mask(organ_cropped_cannon, label, label_other_time, class_list_seg)
    
    
    #now we have only the paired samples, pairing between the current time and the other time.
    #let's get lesion channels, we do not care about organs here
    #ball mask is already 1 ch
    if len(ball_mask.shape) == 5:
        assert ball_mask.shape[1] == 1, "ball_mask should have 1 channel"
    else:
        assert len(ball_mask.shape) == 4, "ball_mask should have shape [B, H, W, D]"
        ball_mask = ball_mask.unsqueeze(1) #add channel dim
        ball_mask_other_time = ball_mask_other_time.unsqueeze(1)
    label = get_lesion_channels(label, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes,assertion=False)
    for i in range(len(list_of_outputs)):
        list_of_outputs[i] = get_lesion_channels(list_of_outputs[i], classes)
    label_other_time = get_lesion_channels(label_other_time, classes)
    chosen_segment_mask_other_time = get_lesion_channels(chosen_segment_mask_other_time, classes,assertion=False)
    teacher_last_layer = get_lesion_channels(teacher_last_layer, classes)
    teacher_last_layer_other_time = get_lesion_channels(teacher_last_layer_other_time, classes)
    
    #find cases annotated per report
    assert len(chosen_segment_mask.shape) == 5, "label should have shape [B, C, H, W, D]"
    report_annotated = (chosen_segment_mask.sum(dim=(-1,-2,-3,-4))>0) #shape [B], True if annotated per report (any class), 0 if not
    # keep_local maps local sample i -> True iff it passed both is_paired AND report_annotated.
    # Used by callers (e.g., restrict_volume_ball_loss_in_time) to align tensors that live in
    # the unfiltered B_local space (slice_mask, chosen_segment_mask) with tensors filtered
    # down to B_filtered inside this function.
    keep_local = torch.zeros(batch_size, dtype=torch.bool, device=ball_mask.device)
    paired_idx_local = torch.where(is_paired)[0]
    keep_local[paired_idx_local] = report_annotated.bool()
    if not report_annotated.any():
        #no report-annotated tumor here, we can skip these cases
        return_dict = {}
        if args.intersection_loss_weight > 0:
            return_dict['intersection_per_voxel_bce'] = torch.tensor(0.0).type_as(ball_mask)
            return_dict['intersection_per_voxel_dice'] = torch.tensor(0.0).type_as(ball_mask)
        if args.intersection_volume_loss_weight > 0:
            return_dict['intersection_volume_loss'] = torch.tensor(0.0).type_as(ball_mask)
        if skip_loss:
            return None, None, None, None
        return torch.tensor(0.0).type_as(ball_mask), return_dict
    
    ball_mask = select_on_batch_and_expand(ball_mask, report_annotated)
    ball_mask_other_time = select_on_batch_and_expand(ball_mask_other_time, report_annotated)
    tumor_volumes_in_crop = select_on_batch_and_expand(tumor_volumes_in_crop, report_annotated)
    tumor_volumes_in_crop_other_time = select_on_batch_and_expand(tumor_volumes_in_crop_other_time, report_annotated)
    label = select_on_batch_and_expand(label, report_annotated)
    label_other_time = select_on_batch_and_expand(label_other_time, report_annotated)
    chosen_segment_mask = select_on_batch_and_expand(chosen_segment_mask, report_annotated)
    chosen_segment_mask_other_time = select_on_batch_and_expand(chosen_segment_mask_other_time, report_annotated)
    teacher_last_layer = select_on_batch_and_expand(teacher_last_layer, report_annotated)
    teacher_last_layer_other_time = select_on_batch_and_expand(teacher_last_layer_other_time, report_annotated)
    for i in range(len(list_of_outputs)):
        list_of_outputs[i] = select_on_batch_and_expand(list_of_outputs[i], report_annotated)
    registration_field = select_on_batch_and_expand(registration_field, report_annotated)
    # after you compute report_annotated and filter other tensors...
    label_cropped_organ = select_on_batch_and_expand(label_cropped_organ, report_annotated)
    label_cropped_organ_other_time = select_on_batch_and_expand(label_cropped_organ_other_time, report_annotated)
    # if organ_cropped_cannon is a python list, filter it too
    if isinstance(organ_cropped_cannon, list):
        organ_cropped_cannon = [o for o, keep in zip(organ_cropped_cannon, report_annotated.tolist()) if keep]
        
        
    #now get the tumor channel, defined by the channel where chosen_segment_mask is >0
    tumor_channel = (chosen_segment_mask.sum(dim=(-1,-2,-3),keepdim=True)>0).float() #shape [B, C, 1, 1, 1], 1 if this channel is the tumor channel
    assert tumor_channel.sum(dim=(-1,-2,-3,-4)).max() == 1, "There should be only one tumor channel per case"
    label = (label * tumor_channel).sum(dim=1,keepdim=True) #shape [B, 1, H, W, D], only the tumor channel remains
    label_other_time = (label_other_time * tumor_channel).sum(dim=1,keepdim=True) #shape [B, 1, H, W, D], only the tumor channel remains
    assert label.shape == ball_mask.shape, "label and ball_mask should have the same shape now that we reduced to 1 channel"
    teacher_last_layer = (teacher_last_layer * tumor_channel).sum(dim=1,keepdim=True) #shape [B, H, W, D], only the tumor channel remains
    teacher_last_layer_other_time = (teacher_last_layer_other_time * tumor_channel).sum(dim=1,keepdim=True) #shape [B, H, W, D], only the tumor channel remains
    chosen_segment_mask = (chosen_segment_mask * tumor_channel).sum(dim=1,keepdim=True) #shape [B, 1, H, W, D], only the tumor channel remains
    chosen_segment_mask_other_time = (chosen_segment_mask_other_time * tumor_channel).sum(dim=1,keepdim=True) #shape [B, 1, H, W, D], only the tumor channel remains
    for i in range(len(list_of_outputs)):
        list_of_outputs[i] = (list_of_outputs[i] * tumor_channel).sum(dim=1,keepdim=True)
        assert list_of_outputs[i].shape == ball_mask.shape, "Each output and ball_mask should have the same shape now that we reduced to 1 channel"
        
    #is our other time annotated per voxel? or by report?
    #if per voxel: substitute label in ball_mask_other_time, and fill tumor_volumes_in_crop with volumes from the mask
    assert len(label_other_time.shape) == 5, "label_other_time should have shape [B, 1, H, W, D]"
    other_time_annotated_per_voxel = (label_other_time.sum(dim=(-1,-2,-3,-4),keepdim=True)>0).float()
    #expand to the same shape as ball_mask_other_time
    other_time_annotated_per_voxel = other_time_annotated_per_voxel.expand_as(ball_mask_other_time)
    mask_tumor_other_time = torch.where(other_time_annotated_per_voxel>0, label_other_time, ball_mask_other_time) #if annotated per voxel, take the label, otherwise take the ball mask
    #now we take tumor volumes from report or measure from mask
    tumor_volumes_other_time = []
    for b in range(mask_tumor_other_time.shape[0]):
        if other_time_annotated_per_voxel[b].sum().item()>0:
            tumor_volumes_other_time.append(measure_tumor_volumes(mask_tumor_other_time[b].squeeze(0).squeeze(0)).to(tumor_volumes_in_crop.device).type_as(tumor_volumes_in_crop))
        else:
            tumor_volumes_other_time.append(tumor_volumes_in_crop_other_time[b])
    
    #now we compare the other time tumor sizes to this time. Are the current tumors smaller?
    all_smaller = []
    all_smaller_or_equal = []
    for b in range(mask_tumor_other_time.shape[0]):
        tumor_volume_this_time = tumor_volumes_in_crop[b]
        tumor_volume_other_time = tumor_volumes_other_time[b]
        #check if any of the 2 time points have negative volumes, which indicates tumors of unknown size
        if (tumor_volume_this_time<0).any() or (tumor_volume_other_time<0).any():
            all_smaller.append(False)
            all_smaller_or_equal.append(False)
            continue
        smaller = []
        smaller_equal = []
        if tumor_volume_this_time.sum().item() == 0 and tumor_volume_other_time.sum().item() == 0:
            all_smaller.append(False)
            all_smaller_or_equal.append(False)
            continue
        for i in range(tumor_volume_this_time.shape[0]):
            if (tumor_volume_this_time[i] == 0) and (tumor_volume_other_time[i] == 0): #padding!
                smaller.append(True)
                smaller_equal.append(True)
            elif tumor_volume_this_time[i] < tumor_volume_other_time[i]:
                smaller.append(True)
                smaller_equal.append(True)
            elif tumor_volume_this_time[i] == tumor_volume_other_time[i]:
                smaller.append(False)
                smaller_equal.append(True)
            else:
                smaller.append(False)
                smaller_equal.append(False)
        smaller = torch.tensor(smaller, device=ball_mask.device, dtype=torch.bool).all()
        smaller_or_equal = torch.tensor(smaller_equal, device=ball_mask.device, dtype=torch.bool).all()
        all_smaller.append(smaller)
        all_smaller_or_equal.append(smaller_or_equal)
    all_smaller = torch.tensor(all_smaller, device=ball_mask.device, dtype=torch.bool)
    all_smaller_or_equal = torch.tensor(all_smaller_or_equal, device=ball_mask.device, dtype=torch.bool)
        
    #register
    #print('Starting registration of the other time point masks', flush=True)
    mask_tumor_other_time_pre_register = mask_tumor_other_time
    mask_tumor_other_time = compute_warped_image_multiNC(
            mask_tumor_other_time.float(),
            registration_field,
            spacing_registration,
            spline_order=0,
            zero_boundary=True
        )
    #print('Registration done for mask_tumor_other_time', flush=True)
    mask_tumor_other_time = (mask_tumor_other_time > 0.5).float()
    mask_tumor_other_time = dilate_volume(mask_tumor_other_time,kernel_size=21, full_pass_radius=2)#quite big dilation (2cm) to compensate for registration errors
    non_empty_other_time_tumor_mask = (mask_tumor_other_time.sum(dim=(-1,-2,-3),keepdim=True)>0)
    label_cropped_organ_other_time_registered = compute_warped_image_multiNC(
            label_cropped_organ_other_time.float(),
            registration_field,
            spacing_registration,
            spline_order=0,
            zero_boundary=True
        )
    #print('Registration done for label_cropped_organ_other_time_registered', flush=True)
    label_cropped_organ_other_time_registered = (label_cropped_organ_other_time_registered > 0.5).float()
    #label_cropped_organ_other_time_registered = dilate_volume(label_cropped_organ_other_time_registered,kernel_size=5, full_pass_radius=2)
    if args.intersection_volume_loss_weight > 0:
        teacher_last_layer_other_time = compute_warped_image_multiNC(
                teacher_last_layer_other_time.detach().float(),
                registration_field,
                spacing_registration,
                spline_order=1,
                zero_boundary=True,
            )
    #print('Registration done for teacher_last_layer_other_time', flush=True)
        
    #reinforce binary
    ball_mask = (ball_mask > 0.5).float()
    label_cropped_organ_other_time_registered = (label_cropped_organ_other_time_registered > 0.5).float()
    #register output from other time
    
    #check the dice between label_cropped_organ and label_cropped_organ_other_time_registered, if it is very low, it means the registration likely failed, so we skip the loss in this case
    assert label_cropped_organ.shape == label_cropped_organ_other_time_registered.shape, f"B mismatch: organ={label_cropped_organ.shape} tumor_mask={label_cropped_organ_other_time_registered.shape}"
    organ_dsc = dice4d(label_cropped_organ, label_cropped_organ_other_time_registered)
    main_process = None
    debug_dict = None
    images_other_time_registered = []
    debug_dicts = []
    global debug_registration_counter
    if debug_registration_counter<20:
        for b in range(mask_tumor_other_time.shape[0]):
            #warp the image too
            image_other_time_b = image_other_time[b]
            image_other_time_registered = compute_warped_image_multiNC(
                    image_other_time_b.unsqueeze(0).float(),
                    registration_field[b:b+1],
                    spacing_registration,
                    spline_order=1,
                    zero_boundary=True
                )
            images_other_time_registered.append(image_other_time_registered)
            #print('Registration done for image_other_time_registered', flush=True)
            image_b = image[b]
            #create SanityRegistration if it does not exist
            #run only on the main DDP process to avoid multiple processes writing at the same time
            main_process = (dist is None) or (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            if main_process and debug:
                os.makedirs('SanityRegistration',exist_ok=True)
                os.makedirs('SanityRegistration/'+str(debug_registration_counter),exist_ok=True)
                save_tensor_as_nifti(label_cropped_organ_other_time[b],'SanityRegistration/'+str(debug_registration_counter)+'/label_other_time.nii.gz',dist)
                save_tensor_as_nifti(label_cropped_organ_other_time_registered[b],'SanityRegistration/'+str(debug_registration_counter)+'/label_cropped_organ_other_time_registered.nii.gz',dist)
                save_tensor_as_nifti(label_cropped_organ[b],'SanityRegistration/'+str(debug_registration_counter)+'/label_cropped_organ.nii.gz',dist)
                save_tensor_as_nifti(image_other_time_registered.squeeze(0).squeeze(0),'SanityRegistration/'+str(debug_registration_counter)+'/image_other_time_registered.nii.gz',dist)
                save_tensor_as_nifti(image_other_time_b.squeeze(0).squeeze(0),'SanityRegistration/'+str(debug_registration_counter)+'/image_other_time.nii.gz',dist)
                save_tensor_as_nifti(image_b.squeeze(0).squeeze(0),'SanityRegistration/'+str(debug_registration_counter)+'/image.nii.gz',dist)
                save_tensor_as_nifti(mask_tumor_other_time[b],'SanityRegistration/'+str(debug_registration_counter)+'/mask_tumor_other_time.nii.gz',dist)
                save_tensor_as_nifti(mask_tumor_other_time_pre_register[b],'SanityRegistration/'+str(debug_registration_counter)+'/mask_tumor_other_time_pre_register.nii.gz',dist)
                save_tensor_as_nifti(ball_mask[b],'SanityRegistration/'+str(debug_registration_counter)+'/ball_mask.nii.gz',dist)
                
                debug_dict = {
                    'name_this_gpu': names,
                    'rank': dist.get_rank() if dist and dist.is_initialized() else -1,
                    'names_all': names_global,
                    'other_time_annotated_per_voxel_sum': other_time_annotated_per_voxel[b].sum().item(),
                    'other_time_ball_mask_sum': ball_mask_other_time[b].sum().item(),
                    'label_cropped_organ_other_time_sum': label_cropped_organ_other_time[b].sum().item(),
                    'label_cropped_organ_sum': label_cropped_organ[b].sum().item(),
                    'tumor_volumes_in_crop': tumor_volumes_in_crop[b].sum().item(),
                    'tumor_volumes_other_time': tumor_volumes_other_time[b].sum().item(),
                    'all_smaller': bool(all_smaller[b].item()),
                    'all_smaller_or_equal': bool(all_smaller_or_equal[b].item()),
                    'organ': organ_cropped_cannon[b],
                    'dsc': organ_dsc[b].item(),
                }
                debug_dicts.append(debug_dict)
                
                #save debug_dict as yaml
                with open('SanityRegistration/'+str(debug_registration_counter)+'/debug_dict.yaml', 'w') as f:
                    yaml.dump(debug_dict, f)
                
                
                debug_registration_counter+=1
    
    print(f'Organ DSC after registration: {organ_dsc}; Organ volume target/registered/before registration = {label_cropped_organ.sum(dim=(-1,-2,-3))}/{label_cropped_organ_other_time_registered.sum(dim=(-1,-2,-3))}/{chosen_segment_mask_other_time.sum(dim=(-1,-2,-3))}', flush=True)
                                                      
    gate_organ_dsc = (organ_dsc>=organ_dsc_threshold).float() #shape [B], 1 if good registration, 0 if bad registration
    
    if skip_loss:
        return mask_tumor_other_time, all_smaller, gate_organ_dsc, keep_local
    
    #calculate the intersection loss on the registered masks:
    loss_dict = {}
    if args.intersection_loss_weight > 0:
        loss_dict = intersection_loss(list_of_outputs, ball_mask, mask_tumor_other_time, sigmoid_already_applied=sigmoid_already_applied,
                                      debug_dicts=debug_dicts, main_process = main_process, image= image, image_other_time=image_other_time, 
                                      images_other_time_registered = images_other_time_registered, dist=dist, 
                                      ball_mask_other_time_non_registered = mask_tumor_other_time_pre_register, )
        
        for k,v in loss_dict.items():
            loss_dict[k] = v * args.intersection_loss_weight
    
    print(f'VVV Is smaller: {all_smaller}, Tumor volumes this time: {tumor_volumes_in_crop}, Tumor volumes other time: {torch.stack(tumor_volumes_other_time)}', flush=True)
    if (args.intersection_volume_loss_weight > 0) and all_smaller_or_equal.any():
        #print(f'Calculating intersection volume loss', flush=True)
        intersect_loss_vol = intersection_volume_loss(list_of_outputs, 
                                 output_other_time_registered=teacher_last_layer_other_time.detach(), 
                                 mask_tumor_other_time = mask_tumor_other_time,
                                 all_tumors_smaller_or_equal=all_smaller_or_equal,
                                 slices_cropped_dict=slices_cropped_dict,
                                 args=args,
                                 chosen_segment_mask=chosen_segment_mask,
                                 tumor_volumes_in_crop=tumor_volumes_in_crop,
                                 labels=label,dist=dist,)
        loss_dict['intersection_volume_loss'] = intersect_loss_vol * args.intersection_volume_loss_weight
        print(f"Intersection volume loss: {loss_dict['intersection_volume_loss']}", flush=True)
    elif (args.intersection_volume_loss_weight > 0) and not all_smaller.any():
        loss_dict['intersection_volume_loss'] = torch.zeros((ball_mask.shape[0],), device=ball_mask.device, dtype=ball_mask.dtype)
                
    non_empty_other_time_tumor_mask = non_empty_other_time_tumor_mask.float().squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)
    gate_organ_dsc = gate_organ_dsc.squeeze(-1)
    assert gate_organ_dsc.shape == non_empty_other_time_tumor_mask.shape, f"gate_organ_dsc and non_empty_other_time_tumor_mask should have the same shape, got {gate_organ_dsc.shape} and {non_empty_other_time_tumor_mask.shape}"
    assert len(gate_organ_dsc.shape) == 1, f"gate_organ_dsc should be 1D after squeezing, got shape {gate_organ_dsc.shape}"
    gate = gate_organ_dsc*non_empty_other_time_tumor_mask
    for k,v in loss_dict.items():
        g = gate
        # normalize v to [B]
        while v.dim() > 1 and v.shape[-1] == 1:
            v = v.squeeze(-1)
        assert len(v.shape) == 1, f"Loss {k} should be 1D after squeezing, got shape {v.shape}, key is {k}"
        #g losses with organ dsc
        while len(g.shape) < len(v.shape):
            g = g.unsqueeze(-1)
        assert v.shape[0] == g.shape[0], f"Loss batch size {v} does not match g batch size {g}"
        assert len(g.shape) == len(v.shape), f"g has more dimensions than loss, cannot broadcast: g shape {g.shape}, loss shape {v.shape}"
        loss_dict[k] = (g * v).mean()
        
    
    total_loss = sum(loss_dict.values())    
    
    
    #print(f'Finished time consistency loss. Total loss: {total_loss}, Details: {loss_dict}', flush=True)
    
    return total_loss, loss_dict
    
import yaml

def all_gather_object_no_grad(obj, dist=None):
    """All-gather python objects (e.g., strings, lists) across ranks."""
    if dist is None or (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
        return [obj]
    world = dist.get_world_size()
    gathered = [None for _ in range(world)]
    with torch.no_grad():
        dist.all_gather_object(gathered, obj)
    return gathered

def safe_sum(x: torch.Tensor) -> float:
    """Deterministic scalar sum for tensors."""
    return float(x.detach().float().sum().item())

def to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if torch.is_tensor(x):
        return bool(x.item())
    return bool(x)
    

def dice4d(a: torch.Tensor, b: torch.Tensor, thr: float = 0.5, eps: float = 1e-5) -> float:
    """
    a, b: torch tensors, any shape but must match. Typically [1,1,D,H,W] or [D,H,W].
    Returns scalar dice as python float.
    """
    assert len(a.shape) == 5, f"Expected 5D tensors, got {len(a.shape)}D"
    assert a.shape == b.shape, f"Expected matching shapes, got {a.shape} and {b.shape}"
    a = (a > thr)
    b = (b > thr)

    inter = (a & b).sum(dim = (-1,-2,-3)).float()
    sa = a.sum(dim = (-1,-2,-3)).float()
    sb = b.sum(dim = (-1,-2,-3)).float()
    return (2.0 * inter / (sa + sb + eps))
    
     
        
    
    
    
    
    

def calculate_loss_teacher_student(result, label, unk_voxels, args,
                            matcher,chosen_segment_mask,tumor_volumes_in_crop, 
                            tumor_diameters,
                            classes,img,
                            class_weights,report_embeddings, dist,
                            tumor_attenuaton,
                            lesion_class_names,
                            tumor_volumes_in_crop_per_voxel,
                            tumor_diameters_per_voxel,
                            names,
                            slices_cropped_dict, sizes_slices,
                            sizes_malignancy, malignancy_per_voxel,
                            contrast,
                            other_time_point=None, is_paired=None,
                            registration_field = None, 
                            spacing_registration = None,
                            organ_cropped_cannon=None,):
    rank = dist.get_rank() if dist and dist.is_initialized() else -1
    #print(f'Calculating loss for teacher and student, rank {rank}', flush=True)
    chosen_segment_mask_contricted = None
    if args.use_time_consistency_loss:
        # print shapes of teacher and student outputs for debugging
        outputs_to_optimize = []
        #we remove malignant and benign (not in classes)
        if isinstance(result['teacher']['segmentation'], tuple) or isinstance(result['teacher']['segmentation'], list):
            outputs_to_optimize.append(result['teacher']['segmentation'][0][:,:len(classes)]) #last layer output
            outputs_to_optimize.append(result['teacher']['segmentation'][1][:,:len(classes)]) #deep supervision
        else:
            outputs_to_optimize.append(result['teacher']['segmentation'][:,:len(classes)])
        if isinstance(result['student']['segmentation'], tuple) or isinstance(result['student']['segmentation'], list):
            outputs_to_optimize.append(result['student']['segmentation'][0][:,:len(classes)])
            outputs_to_optimize.append(result['student']['segmentation'][1][:,:len(classes)])
        else:
            outputs_to_optimize.append(result['student']['segmentation'][:,:len(classes)])
    
    start_loss_timer = time.time()
    loss_all = None
    loss = None
    if args.train_mode in ['teacher','both','teacher_decoder']:
        returned = calculate_loss(
                            model_output=result['teacher'], label=label, 
                            unk_voxels=unk_voxels, args=args,
                            matcher=matcher,chosen_segment_mask=chosen_segment_mask,
                            tumor_volumes_report=tumor_volumes_in_crop, 
                            tumor_diameters=tumor_diameters,
                            classes=classes,loss_wrapper=None,input_tensor=img,
                            class_weights=class_weights,
                            model_genesis=args.model_genesis_pretrain,
                            clip_only = args.clip_pretrain, report_embeddings=report_embeddings, dist=dist,
                            tumor_attenuation_label=tumor_attenuaton,attenuation_classifier=args.attenuation_classifier,
                            lesion_classes=lesion_class_names,
                            tumor_volumes_in_crop_per_voxel=tumor_volumes_in_crop_per_voxel,
                            tumor_diameters_per_voxel=tumor_diameters_per_voxel,
                            names = names,
                            slices_cropped_dict=slices_cropped_dict, sizes_slices=sizes_slices,
                            dynamic_sample_weights=args.use_sample_weigths,
                            no_mask=args.no_mask,
                            sizes_malignancy=sizes_malignancy, malignancy_per_voxel=malignancy_per_voxel,
                            contrast=contrast,
                            subseg_dilation=args.organ_mask_dilation,
                            return_refined_mask=args.use_time_consistency_loss) # pass class_weights if available, otherwise None
        #print(f'Finished  calculating teacher loss, rank {rank}', flush=True)
        
        if not args.use_time_consistency_loss:
            loss_all_teacher,_ = returned
        else:
            loss_all_teacher,_, refined_mask_teacher = returned
            if args.restrict_volume_ball_loss_in_time:
                #print(f'starting the mask_tumor_other_time branch, rank {rank}', flush=True)
                #we register the refined_mask_teacher of the other time into this time, and we use it to constrict the organ mask 
                #to regions that will have tumor and we run the ball and volume loss (calculate_loss) again
                mask_tumor_other_time, all_smaller, gate_organ_dsc, keep_local = time_consistency_loss( skip_loss=True,
                                                                                            list_of_outputs=outputs_to_optimize,
                                                                                            ball_mask=refined_mask_teacher,
                                                                                            tumor_volumes_in_crop=tumor_volumes_in_crop,
                                                                                            label=label,
                                                                                            chosen_segment_mask=chosen_segment_mask,
                                                                                            dist=dist, other_time_point=other_time_point,
                                                                                            is_paired=is_paired,
                                                                                            registration_field = registration_field,
                                                                                            spacing_registration = spacing_registration,
                                                                                            classes = classes,
                                                                                            sigmoid_already_applied = False,
                                                                                            slices_cropped_dict = slices_cropped_dict,
                                                                                            args=args, image = img,
                                                                                            organ_cropped_cannon = organ_cropped_cannon,
                                                                                            class_list_seg = classes,
                                                                                            names = names,
                                                                                            debug = True)
                #print(f'Finished  calculating time_consistency_loss loss in mask_tumor_other_time, rank {rank}', flush=True)
                if mask_tumor_other_time is not None and (all_smaller.bool() & gate_organ_dsc.squeeze(-1).bool()).any():
                    #the code found paired cases annotated with report and registered the other time point ball mask to this time point
                    #at least one batch sample has only smaller tumors wrt the other time
                    #the registration is good for at least one batch sample based on organ dsc
                    acceptable_cases = (all_smaller.bool() & gate_organ_dsc.squeeze(-1).bool())  # [B_filtered]

                    #check for non-empty mask_tumor_other_time
                    mask_tumor_other_time_non_empty = (mask_tumor_other_time.sum(dim=(-1,-2,-3))>0).squeeze(-1)  # [B_filtered]
                    acceptable_cases = acceptable_cases & mask_tumor_other_time_non_empty  # [B_filtered]

                    #slice_mask comes from the dataloader at B_local; mask_tumor_other_time and acceptable_cases
                    #live at B_filtered (after is_paired + report_annotated filters inside time_consistency_loss).
                    #Use keep_local to align: do the per-voxel work at B_filtered, scatter back to B_local for the
                    #final multiplication so non-kept samples are left unchanged.
                    slice_mask_full = slices_cropped_dict['binary_slices_mask']  # [B_local, 1, D, H, W], may be on CPU
                    keep_local_idx = keep_local.to(slice_mask_full.device)
                    slice_mask = slice_mask_full[keep_local_idx]                # [B_filtered, 1, D, H, W]

                    acceptable_cases_3d = acceptable_cases.view(-1,1,1,1,1).expand_as(slice_mask)
                    acceptable_cases_3d = acceptable_cases_3d.float() * slice_mask.float().type_as(acceptable_cases_3d)

                    #reject zeros: not acceptable
                    acceptable_case = acceptable_cases_3d.sum(dim=(-1,-2,-3))>10 #shape [B_filtered, 1]; kept for parity, currently unused

                    acceptable_cases_3d = acceptable_cases.view(-1,1,1,1,1).expand_as(slice_mask).type_as(mask_tumor_other_time)

                    #dilate
                    mask_tumor_other_time = dilate_volume(mask_tumor_other_time,kernel_size=5, full_pass_radius=3)

                    mask_tumor_other_time_multiplier = mask_tumor_other_time + (1-acceptable_cases_3d) #where acceptable: we add nothing. Where not acceptable: we add all ones
                    mask_tumor_other_time_multiplier = torch.clamp(mask_tumor_other_time_multiplier, 0, 1) #drop 2 we generated above
                    # mask_tumor_other_time_multiplier is [B_filtered, 1, D, H, W]

                    #scatter to B_local: identity (1) outside the keep set so chosen_segment_mask is preserved there.
                    full_multiplier = torch.ones(
                        chosen_segment_mask.shape[0], 1, *chosen_segment_mask.shape[-3:],
                        device=mask_tumor_other_time_multiplier.device,
                        dtype=mask_tumor_other_time_multiplier.dtype,
                    )
                    full_multiplier[keep_local] = mask_tumor_other_time_multiplier

                    chosen_segment_mask_contricted = chosen_segment_mask * full_multiplier #we constrict the chosen_segment_mask with the registered mask from the other time point

                    #B_local-aligned acceptable_cases_3d for the assertion (zeros outside keep set, so 1-x = 1 there).
                    full_acceptable_cases_3d = torch.zeros_like(full_multiplier)
                    full_acceptable_cases_3d[keep_local] = acceptable_cases_3d

                    #assert that where acceptable_cases is 0, we did not change the mask. Where it is one, we changed:
                    assert torch.allclose((1 - full_acceptable_cases_3d) * chosen_segment_mask, (1 - full_acceptable_cases_3d) * chosen_segment_mask_contricted), "Where not acceptable, the mask should be the same"
                    #assert not torch.allclose(chosen_segment_mask_contricted, chosen_segment_mask)
                    
                    #now, re-do the loss, use the constricted mask
                    returned = calculate_loss(
                            model_output=result['teacher'], label=label, 
                            unk_voxels=unk_voxels, args=args,
                            matcher=matcher,chosen_segment_mask=chosen_segment_mask,
                            chosen_segment_mask_contricted = chosen_segment_mask_contricted,
                            tumor_volumes_report=tumor_volumes_in_crop, 
                            tumor_diameters=tumor_diameters,
                            classes=classes,loss_wrapper=None,input_tensor=img,
                            class_weights=class_weights,
                            model_genesis=args.model_genesis_pretrain,
                            clip_only = args.clip_pretrain, report_embeddings=report_embeddings, dist=dist,
                            tumor_attenuation_label=tumor_attenuaton,attenuation_classifier=args.attenuation_classifier,
                            lesion_classes=lesion_class_names,
                            tumor_volumes_in_crop_per_voxel=tumor_volumes_in_crop_per_voxel,
                            tumor_diameters_per_voxel=tumor_diameters_per_voxel,
                            names = names,
                            slices_cropped_dict=slices_cropped_dict, sizes_slices=sizes_slices,
                            dynamic_sample_weights=args.use_sample_weigths,
                            no_mask=args.no_mask,
                            sizes_malignancy=sizes_malignancy, malignancy_per_voxel=malignancy_per_voxel,
                            contrast=contrast,
                            subseg_dilation=args.organ_mask_dilation,
                            return_refined_mask=args.use_time_consistency_loss) # pass class_weights if available, otherwise None
                    loss_all_teacher,_, refined_mask_teacher = returned
                    #print(f'Finished  re-calculating calculate_loss in mask_tumor_other_time, rank {rank}', flush=True)
        
        
        loss_teacher=loss_all_teacher['overall']
        loss = loss_teacher
        
        # sanity: overall must equal sum of components (excluding 'overall')
        comp_sum = sum(v for k, v in loss_all_teacher.items() if k != "overall")
        diff = (loss_all_teacher["overall"] - comp_sum).abs().item()
        if diff > 1e-4:
            raise RuntimeError(f"Loss mismatch: overall-sum={diff}")

        # debug: show biggest contributors
        #items = sorted([(k, float(v.detach().cpu())) for k, v in loss_all_teacher.items() if k != "overall"],
        #            key=lambda x: -abs(x[1]))
        #print("LOSS BREAKDOWN:", items, flush=True)
        #print("LOSS OVERALL:", float(loss_all_teacher["overall"].detach().cpu()), flush=True)
        
        loss_all = {'teacher_'+k:v for k,v in loss_all_teacher.items()}
        #print(f'Loss calculation time (teacher): {time.time()-start_loss_timer:.4f}s', flush=True, file=sys.stderr)
        start_loss_timer = time.time()
        

        
        
    if args.train_mode in ['student','both','teacher_student_ema','teacher_decoder']:
        #print(f'starting student branck, rank {rank}', flush=True)
        loss_all_student,sample_weights=calculate_loss(model_output=result['student'], label=label, unk_voxels=unk_voxels, args=args,
                            matcher=matcher,chosen_segment_mask=chosen_segment_mask,
                            chosen_segment_mask_contricted = chosen_segment_mask_contricted,
                            tumor_volumes_report=tumor_volumes_in_crop, 
                            tumor_diameters=tumor_diameters,
                            classes=classes,loss_wrapper=None,input_tensor=img,
                            class_weights=class_weights,
                            model_genesis=args.model_genesis_pretrain,
                            clip_only = args.clip_pretrain, report_embeddings=report_embeddings, dist=dist,
                            tumor_attenuation_label=tumor_attenuaton,attenuation_classifier=args.attenuation_classifier,
                            lesion_classes=lesion_class_names,
                            tumor_volumes_in_crop_per_voxel=tumor_volumes_in_crop_per_voxel,
                            tumor_diameters_per_voxel=tumor_diameters_per_voxel,
                            names = names,
                            slices_cropped_dict=slices_cropped_dict, sizes_slices=sizes_slices,
                            dynamic_sample_weights=args.use_sample_weigths,
                            no_mask=args.no_mask,
                            sizes_malignancy=sizes_malignancy, malignancy_per_voxel=malignancy_per_voxel,
                            contrast=contrast,
                            subseg_dilation=args.organ_mask_dilation) # pass class_weights if available, otherwise None
        #print(f'finished calculate_loss in student branch, rank {rank}', flush=True)
        loss_all_update = {'student_'+k:v for k,v in loss_all_student.items()}
        if args.ball_distill_loss_weight > 0:
            #remove ball 
            loss_all_update = {k:v for k,v in loss_all_update.items() if 'ball' not in k}
        if loss_all is not None:
            loss_all.update(loss_all_update)
        else:
            loss_all = loss_all_update
            
            
        loss_student = 0
        for key in loss_all_update.keys():
            #print('loss key:', key)
            if 'overall' not in key:
                loss_student = loss_student + loss_all_update[key]
        
        #print(f'Loss calculation time (student): {time.time()-start_loss_timer:.4f}s', flush=True, file=sys.stderr)
        start_loss_timer = time.time()
        
        
        #now the ball distillation loss
        #check if the case is not annotated per voxel:
        
        if isinstance(result['teacher']['segmentation'], tuple) or isinstance(result['teacher']['segmentation'], list):
            r = result['teacher']['segmentation'][0] #last layer output
            r_s = result['student']['segmentation'][0]
        else:
            r = result['teacher']['segmentation']
            r_s = result['student']['segmentation']
            
        if args.malignancy_classification:
            model_output_teacher, malig_benign_teacher = split_outputs_malignancy(result['teacher'],classes)
            model_output_student, malig_benign_student = split_outputs_malignancy(result['student'],classes)
            
            if isinstance(malig_benign_teacher, tuple) or isinstance(malig_benign_teacher, list): #multiple heads
                malig_benign_teacher = malig_benign_teacher[0]
                malig_benign_student = malig_benign_student[0]
                r = model_output_teacher['segmentation'][0]
                r_s = model_output_student['segmentation'][0]
            
            lesion_classes = [c for c in sorted(classes) if 'lesion' in c]
            malignants = [c.replace('lesion', 'malignant') for c in lesion_classes]
            malign_out_teacher = malig_benign_teacher[:,:len(malignants)].detach()
            benign_out_teacher = malig_benign_teacher[:,len(malignants):].detach()
            malign_out_student = malig_benign_student[:,:len(malignants)]
            benign_out_student = malig_benign_student[:,len(malignants):]
        else:
            benign_out_teacher = None
            malign_out_teacher = None
            benign_out_student = None
            malign_out_student = None
            
        distill_loss = 0.0
        if args.ball_distill_loss_weight > 0:
            if slices_cropped_dict is not None:
                ct_z_spacing_original = slices_cropped_dict["z_spacing"]
                slices_mask=slices_cropped_dict['slices_mask']
                max_slice = slices_cropped_dict['max_slice']
            else:
                ct_z_spacing_original = None
                slices_mask=None
                max_slice = None
            distill_loss = ball_loss(out=r.detach(), 
                                        labels=label, unk_voxels=unk_voxels, 
                                        chosen_segment_mask=chosen_segment_mask_contricted if chosen_segment_mask_contricted is not None else chosen_segment_mask, 
                                        tumor_volumes=tumor_volumes_in_crop, 
                                        tumor_diameters=tumor_diameters, classes=classes,
                                        apply_dice_loss=('dice' in args.loss), input_tensor=None,
                                        sigmoid=True,
                                        standard_ce=args.stardard_ce_ball, 
                                        class_weights = class_weights,
                                        single_class= args.epai_stage_2,
                                        diameter_margin=args.ball_volume_margin, volume_margin=args.ball_volume_margin,
                                        sizes_slices=sizes_slices, ct_z_spacing_original=ct_z_spacing_original,
                                        slices_mask=slices_mask, max_slice = max_slice,
                                        sample_weights=sample_weights,
                                        malignant_benign_loss=args.malignancy_classification,
                                        benign_out=benign_out_teacher,
                                        malignant_out=malign_out_teacher,
                                        sizes_malignancy = sizes_malignancy,
                                        student_out=r_s,
                                        student_benign_out=benign_out_student,
                                        student_malignant_out=malign_out_student
                                        )
            distill_loss = {'ball_distill_'+k:v for k,v in distill_loss.items() if 'student' in k} #keep only student losses
            loss_all.update(distill_loss)
            #sum all distillation losses
            distill_loss = sum(distill_loss.values())
            distill_loss = distill_loss * args.ball_distill_loss_weight
            #print(f'finished distill ball loss in student branch, rank {rank}', flush=True)
            
            #print(f'Loss calculation time (ball distillation): {time.time()-start_loss_timer:.4f}s', flush=True, file=sys.stderr)
            start_loss_timer = time.time()
        
        #standard distillation loss
        if args.standard_distill_loss_weight > 0:
            bce_standard = F.binary_cross_entropy_with_logits(r_s, torch.sigmoid(r).detach(), reduction='none', weight=class_weights)
            dice_standard = DiceLossMultiClass(r_s, torch.sigmoid(r).detach(), torch.ones_like(r_s), 
                                                    sigmoid=True,class_weights=class_weights)
            standard_distill_loss = bce_standard.mean() + dice_standard.mean()
            if args.malignancy_classification:
                bce_malig = F.binary_cross_entropy_with_logits(malign_out_student, torch.sigmoid(malign_out_teacher).detach(), 
                                                                reduction='mean', weight=class_weights)
                dice_malig = DiceLossMultiClass(malign_out_student, torch.sigmoid(malign_out_teacher).detach(), torch.ones_like(malign_out_student), 
                                                    sigmoid=True,class_weights=class_weights)
                bce_benign = F.binary_cross_entropy_with_logits(benign_out_student, torch.sigmoid(benign_out_teacher).detach(), 
                                                                reduction='mean', weight=class_weights)
                dice_benign = DiceLossMultiClass(benign_out_student, torch.sigmoid(benign_out_teacher).detach(), torch.ones_like(benign_out_student),
                                                    sigmoid=True,class_weights=class_weights)
                standard_distill_loss = bce_standard.mean() + dice_standard.mean() + bce_malig.mean() + dice_malig.mean() + bce_benign.mean() + dice_benign.mean()
            distill_loss += standard_distill_loss * args.standard_distill_loss_weight
            loss_all.update({'standard_distill_loss': standard_distill_loss})
            #print(f'finished standard distill loss in student branch, rank {rank}', flush=True)
            
            #print(f'Loss calculation time (standard distillation): {time.time()-start_loss_timer:.4f}s', flush=True, file=sys.stderr)
            start_loss_timer = time.time()
        
        if args.feature_distill_loss_weight > 0:
            feature_distill_losses = []
            for feat in ['bottleneck_features','mid_decoder_features','out_features']:
                teacher_feat = result['teacher'][feat].detach()
                student_feat = result['student'][feat]
                assert student_feat.shape == teacher_feat.shape, f"{feat}: {student_feat.shape} vs {teacher_feat.shape}"
                geo=1e-6 + (torch.mean(torch.abs(student_feat),dim=(-1,-2,-3,-4),keepdim=True)**0.5)*\
                            (torch.mean(torch.abs(teacher_feat),dim=(-1,-2,-3,-4),keepdim=True)**0.5)
                teacher_feat = teacher_feat/geo
                student_feat = student_feat/geo
                loss_feat = F.l1_loss(student_feat,teacher_feat.detach().type_as(student_feat),reduction='none')
                feature_distill_losses.append(loss_feat.mean())
            feature_distill_loss = sum(feature_distill_losses)/len(feature_distill_losses)
            distill_loss += feature_distill_loss * args.feature_distill_loss_weight
            loss_all.update({'feature_distill_loss': feature_distill_loss})
            #print(f'finished feature distill loss in student branch, rank {rank}', flush=True)
            
            #print(f'Loss calculation time (feature distillation): {time.time()-start_loss_timer:.4f}s', flush=True, file=sys.stderr)
            start_loss_timer = time.time()
    
        if loss is not None:
            loss += loss_student + distill_loss
        else:
            loss = loss_student + distill_loss
    
        if args.use_time_consistency_loss:
            #print(f'starting the use_time_consistency_loss branch', flush=True)
            #args.use_time_consistency_loss:
            assert args.train_mode == 'teacher_decoder', "Time consistency loss is only implemented for teacher_decoder mode for now"
            #train the registration module
            registration_loss = result['student']['registration_loss']
            loss = loss + registration_loss * args.registration_loss_weight
            loss_all['registration_loss'] = registration_loss     
            
            #now, the time consistency loss:
            #we use the ball refined mask of the teacher last layer as the guide (refined_mask_teacher)
            #but we can optimize all segmentation outputs of the teacher and the student
            
            time_consistency_l,time_consistency_loss_dict = time_consistency_loss(
                                list_of_outputs=outputs_to_optimize,
                                ball_mask=refined_mask_teacher,
                                tumor_volumes_in_crop=tumor_volumes_in_crop,
                                label=label,
                                chosen_segment_mask=chosen_segment_mask,
                                dist=dist, other_time_point=other_time_point, 
                                is_paired=is_paired,
                                registration_field = registration_field, 
                                spacing_registration = spacing_registration, 
                                classes = classes, 
                                sigmoid_already_applied = False,
                                slices_cropped_dict = slices_cropped_dict,
                                args=args, image = img,
                                organ_cropped_cannon = organ_cropped_cannon,
                                class_list_seg = classes,
                                names = names,
                                debug = True) 

            loss = loss + (time_consistency_l * args.time_consistency_loss_weight)
            loss_all.update(time_consistency_loss_dict)
            #print(f'finished time_consistency_loss_dict, rank {rank}', flush=True)

    #print(f'Finished teacher and student loss calculation, rank {rank}', flush=True)
    return loss, loss_all


def dilate_volume(volume, kernel_size, full_pass_radius=3):
    # ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # for small kernels, just do one pass
    if kernel_size <= (2*full_pass_radius+1):
        return dilate_volume_conv(volume, kernel_size)

    # compute how many "radius‑3" (kernel=7) passes we need
    # radius = (kernel_size‑1)//2  (an integer number of voxels)
    radius = (kernel_size - 1) // 2

    num_full = radius // full_pass_radius  # integer division
    remainder = radius % full_pass_radius  # 0, 1, or 2

    # apply all full radius‑3 passes
    for _ in range(num_full):
        volume = dilate_volume_conv(volume, 2*full_pass_radius + 1)

    # handle the leftover radius if any (1→kernel=3, 2→kernel=5)
    if remainder > 0:
        volume = dilate_volume_conv(volume, 2*remainder + 1)

    return volume



def dilate_volume_conv(volume, kernel_size):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        kernel_size+=1



    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    #we can use a maxpool or a ball convolution to dilate the volume. Maxpool should be faster, but it uses a cube kernel, while the ball kernel is more accurate.
    #dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)
    ball_kernel = create_ball_kernel(kernel_size).type_as(volume).unsqueeze(0).unsqueeze(0).repeat(volume.shape[1],1, 1, 1, 1)

    # Calculate padding such that the output size is the same as the input size.
    kernel_size = ball_kernel.shape[-1]
    padding = kernel_size // 2

    dilated = F.conv3d(volume, ball_kernel, padding=padding, groups=volume.shape[1])
    #binarize
    dilated = (dilated > 0).float()

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated

def dilate_volume_maxpool(volume, kernel_size):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    kernel_size = max(1,int(kernel_size/(2**(0.5))))#compensates for the fact that maxpool is not a round kernel
    if kernel_size%2==0:
        kernel_size+=1

    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number for proper alignment.")

    # Calculate padding such that the output size is the same as the input size.
    padding = kernel_size // 2


    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated

counter = 0

def get_known_voxels(y: torch.Tensor, unk_voxels: torch.Tensor, dilation=5,sanity=False, classes = None):
    """
    We cannot calculate the BCE loss for voxels we do not know the ground-truth for.
    This function will output a per-voxel masks showing the known voxels. You can use it to mask the loss (or the output and label).
    Args:
        y (torch.Tensor): Tensor of shape (B, C, H, W, D).
        unk_voxels (torch.Tensor): Tensor of shape (B, C, H, W, D) indicating the regions that have tumors not annotated per voxel for each class. I.e., in this tensor, 1 represents voxels we do not know the per-voxel ground-truth. 
        Zero representas voxels we do know the per-voxel ground-truth.
        dilation (int): Size of the cubic structuring element for dilation. Default is 5.
    """
    unk_voxels=unk_voxels.float()
    assert torch.equal(unk_voxels.bool().float(),unk_voxels), 'unk_voxels must be binary'

    if dilation>0:
        #dilate unk voxels: adds a margin around the unknown voxels
        unk_voxels = dilate_volume(unk_voxels, dilation)

    #print("unk_voxels unique values:", torch.unique(unk_voxels), flush=True)
    #print("unk_voxels sum:", unk_voxels.sum(), flush=True)
    one = torch.ones(unk_voxels.shape).type_as(unk_voxels)
    known_voxels = one-unk_voxels
    known_voxels = known_voxels.type_as(y).float()
    assert torch.equal(known_voxels + unk_voxels,one)

    #print('Sum of known voxels:',known_voxels.sum())
    #print('Sum of unknown voxels:',unk_voxels.sum())
    #print('Sum of all voxels:',one.sum(),'matches?',torch.equal(known_voxels + unk_voxels,one))

    if sanity:
        global counter
        if counter<10:
            debug_save_labels(y,str(counter)+'_y',label_names=classes) 
            debug_save_labels(known_voxels,str(counter)+'_known_voxels',label_names=classes)
            debug_save_labels(unk_voxels,str(counter)+'_unk_voxels',label_names=classes)
            print('Saved to '+ str(counter)+'_known_voxels')
            counter+=1



    #print number of channels with unknown voxels
    #num_unknown_channels = unk_voxels.float().sum(dim=(-1,-2,-3))>0
    #num_unknown_channels = num_unknown_channels.float().sum(-1)
    #num_unknown_channels = num_unknown_channels.mean(0)
    #print("---------Number of channels with unknown voxels: ", num_unknown_channels, flush=True, file=sys.stderr)
    #print("Number of known voxels: ", known_voxels.sum(), flush=True, file=sys.stderr)

    #with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
    #    classes = yaml.load(f, Loader=yaml.SafeLoader)

    return known_voxels



def huber_with_tolerance(x: torch.Tensor,
                                 y: torch.Tensor,
                                 tolerance: float,
                                 delta: float = 1.0,
                                 reduction: str = 'none'):
    """
    Huber-with-Tolerance using PyTorch's built-in F.huber_loss. 
    Loss is zero for |x - y| <= tolerance, and standard Huber beyond that.
    Args:
        x (Tensor): Predicted values.
        y (Tensor): Target values.
        tolerance (float): Half-width of the 'dead zone' around y.
        delta (float): Huber 'transition point' between L2 and L1. Default=1.0.
        reduction (str): Same as PyTorch's huber_loss (e.g. 'none', 'mean', 'sum').
    Returns:
        Tensor: The Huber-with-Tolerance loss. Shape depends on `reduction`.
    """
    # 1) Create a zero region inside [y - tolerance, y + tolerance].
    diff = (x - y).abs() - tolerance
    # 2) Clamp negative values to zero => effectively remove small errors
    diff = torch.clamp(diff, min=0.0)
    # 3) Apply PyTorch's Huber to 'diff' vs 0
    #    => HuberLoss( diff, 0 ), with delta controlling the L2-to-L1 transition
    return F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction=reduction)


def plot_huber_with_tolerance(huber_fn=huber_with_tolerance, tolerance=0.1, x_min=0.0, x_max=3, num_points=100):
    """
    Plots the huber_fn loss for x from x_min to x_max against y=1
    with a specified tolerance.
    
    :param huber_fn:   A function huber_fn(x, y, tolerance) -> loss tensor
    :param tolerance:  The tolerance margin around y=1
    :param x_min:      The left boundary of the x range
    :param x_max:      The right boundary of the x range
    :param num_points: How many points to sample in [x_min, x_max]
    """
    import matplotlib.pyplot as plt
    # Prepare x-values
    x_vals = np.linspace(x_min, x_max, num_points)
    y_val  = torch.tensor(1.0)  # y=1, as per your requirement
    
    # Evaluate the loss at each point
    losses = []
    for x in x_vals:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        # Our function expects x, y to have the same shape, so let's reshape if needed
        loss = huber_fn(x_tensor, y_val, tolerance)
        # If the function returns a 1D or 0D tensor, convert to float
        losses.append(loss.item() if loss.dim() == 0 else loss.mean().item())
    
    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(x_vals, losses, label=f'HWT (tolerance={tolerance})')
    plt.title('Huber-with-Tolerance Loss')
    plt.xlabel('x')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)  # losses should be >= 0
    #bound x-axis to be between x_min and x_max
    plt.xlim(x_min, x_max)
    plt.grid(True)
    plt.legend()
    plt.show()
    

import torch
import torch.nn.functional as F
import numpy as np

def l1_with_tolerance(x: torch.Tensor,
                      y: torch.Tensor,
                      tolerance: float,
                      reduction: str = 'none'):
    """
    L1-with-Tolerance:
      1) "Dead zone" of zero loss for |x - y| <= tolerance
      2) Standard L1 (|x - y|) beyond that, shifted by `tolerance`.

    Mathematically:  loss = max(|x - y| - tolerance, 0)

    Args:
        x (Tensor):       Predicted values.
        y (Tensor):       Target values.
        tolerance (float):Half-width of the 'dead zone' around y.
        reduction (str):  'none', 'mean', or 'sum'.

    Returns:
        Tensor: The L1-with-Tolerance loss. Shape depends on `reduction`.
    """
    # 1) Create a zero region inside [y - tolerance, y + tolerance].
    diff = torch.abs(x - y) - tolerance
    # 2) Clamp negative values to zero => effectively remove small errors
    diff = torch.clamp(diff, min=0.0)

    # 3) Apply a reduction
    if reduction == 'none':
        return diff
    elif reduction == 'mean':
        return diff.mean()
    elif reduction == 'sum':
        return diff.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    


def plot_l1_with_tolerance(l1_fn=l1_with_tolerance, tolerance=0.1, x_min=0.0, x_max=3.0, num_points=100):
    """
    Result: this loss made all tumor detections zero. I suspect if has too little penaliziton of the zero solution.
    Plots the L1-with-Tolerance loss for x from x_min to x_max against y=1
    with a specified tolerance.

    :param l1_fn:      The function l1_fn(x, y, tolerance) -> loss tensor
    :param tolerance:  The tolerance margin around y=1
    :param x_min:      The left boundary of the x range
    :param x_max:      The right boundary of the x range
    :param num_points: How many points to sample in [x_min, x_max]
    """
    import matplotlib.pyplot as plt

    # Prepare x-values
    x_vals = np.linspace(x_min, x_max, num_points)
    y_val  = torch.tensor(1.0)  # Fix y=1

    # Evaluate the loss at each point
    losses = []
    for x in x_vals:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        loss = l1_fn(x_tensor, y_val, tolerance)
        # Convert tensor -> float (handles scalar or 1D)
        losses.append(loss.item() if loss.dim() == 0 else loss.mean().item())

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(x_vals, losses, label=f'L1 (tol={tolerance})')
    plt.title('L1-with-Tolerance Loss')
    plt.xlabel('x')
    plt.ylabel('Loss')
    plt.xlim(x_min, x_max)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.show()


def get_lesion_channels(out, classes, assertion = False, return_class_names = False):
    #merge lesion channels if they are in the same organ. Outputs will have only lesion channels, removes organ channels.
    assert out.shape[1] == len(classes), f'Output channels {out.shape} does not match number of classes {len(classes)}'
    #print('Shapes here: ', out.shape, chosen_segment_mask.shape, flush=True, file=sys.stderr)

    lesion_out = {}

    for i,clss in enumerate(classes,0):
        #print('Class is:',clss,'Mask sum is:',chosen_segment_mask[:,i].sum())
        for suffix in ['lesion','cyst','pdac','pnet']:
            if suffix in clss:
                name = clss[:clss.index('_'+suffix)+len('_'+suffix)].replace('pancreatic','pancreas')
                if name not in lesion_out:
                    lesion_out[name] = []
                lesion_out[name].append(out[:,i])

    for key in lesion_out.keys():#this combines multi-channel outputs into a single channel
        lesion_out[key] = torch.stack(lesion_out[key],dim=0).max(dim=0).values
        

    #from dicts to tensor
    kys=list(lesion_out.keys())
    lesion_out = torch.stack([lesion_out[key] for key in kys],dim=1).type_as(out)
    
    if assertion:
        for i in range(lesion_out.shape[0]):
            # For sample i, lo has shape (num_lesion_channels, ...spatial dimensions...)
            lo = lesion_out[i]
            # Sum over all dimensions except the channel, regardless of the number of spatial dims.
            lo_sum = lo.sum(dim=(-1,-2,-3))
            # Create a boolean mask for channels with any nonzero value.
            active_mask = lo_sum > 0
            active_count = active_mask.sum().item()
            if active_count > 1:  # If more than one lesion channel is active
                # Prepare the names of the lesion channels that are active.
                active_names = [kys[j] for j in range(len(kys)) if active_mask[j]]
                raise ValueError(
                    f"Error: For sample index {i}, more than one lesion channel has active elements. "
                    f"Active lesion channels: {active_names}"
                    f"lo.sum(dim=(-1,-2,-3)): {lo.sum(dim=(-1,-2,-3))}"
                )
    if return_class_names:
        return lesion_out, kys
    else:
        return lesion_out

counter_vol=0
def volume_loss_basic(out,chosen_segment_mask,tumor_volumes, 
                      labels,unk_voxels,
                      classes='/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      dilation_segment=31, dilation_unk=7, tolerance=0.1,loss_function='selective_volume_reduction_huber',n='huber',
                      sigmoid=True, class_weights=None,
                      slices_cropped_dict=None,
                      sample_weights=None,dist=None):
    """
    Computes the basic tumor volume loss. This loss compares the total predicted tumor volume inside a subsegment with the tumor volume from the report.
    The loss is based on a relative huber loss with a margin of 0.1.

    Args:
        out (torch.Tensor): The predicted tumor volume.
        chosen_segment_mask (torch.Tensor): The mask indicating the chosen subsegment, the subsegment mask should allign with the lesion channels in the out tensor. 
        For example, if there is a lesion in the pancreas head, the pancreas lesion channel in chosen_segment_mask should show the pancreas head.
        tumor_volumes (torch.Tensor): The tumor volumes from the report.
        classes: list of class names.
        dilation: increases the tumor subsegment margins. This can compensate for errors in the sub-segment segmentation.
        slices_cropped_dict: this has slice information. If provided, we will limit the volume loss to consider only the part of the organ around the slices with the tumor.
            -keys: 'slices_mask', 'tumor_df', 'binary_slices_mask', 'max_slice', 'slices_used'
        margin: in case lesion volumes are -999, we will understand that the volume is unknown. The volume loss will use tolerance to force it to be between 0.5 cm and 12 cm
    """
    global counter_vol
    print_loss=True
    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    assert unk_voxels.shape == out.shape
    assert labels.shape == out.shape
    
    if class_weights is not None:
        assert len(class_weights.shape) == 5
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert class_weights.shape[0] == out.shape[0], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        #repeat class weights to match the batch size of out
        class_weights = class_weights.repeat(1, 1, out.shape[2], out.shape[3], out.shape[4]) # B,C,H,W,D
        
    #get only the channels with lesions---apply this in the beginning to reduce computational cost of dilation
    out = get_lesion_channels(out, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes,assertion=False)
    labels = get_lesion_channels(labels, classes)
    unk_voxels = get_lesion_channels(unk_voxels, classes)

    #activation
    if sigmoid:
        out = torch.sigmoid(out)

    #dilate the chosen segment mask
    chosen_segment_mask = dilate_volume(chosen_segment_mask,dilation_segment)
    #dilate the unk voxels
    unk_voxels = dilate_volume(unk_voxels,dilation_unk)

    #remove from this loss any channel with a tumor that is annotated per-voxel
    per_voxel_positives = (labels.sum((-1,-2,-3),keepdim=True)>0).float()#B,C, which elements we have a tumor annotated per voxel
    #labels = labels * (1-per_voxel_positives)
    out = out * (1-per_voxel_positives)
    
    #voxels we are sure have no tumor:
    #negative_voxels = 1 - ((labels + unk_voxels + chosen_segment_mask) > 0).float() #B,C

    
    if class_weights is not None:
        class_weights = get_lesion_channels(class_weights, classes)
        class_weights = class_weights.mean(dim=(-1,-2,-3)) #reduce to B,C, this will be used to weight the loss for each channel.
    

    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape
    #assert out.shape == negative_voxels.shape
    
    if slices_cropped_dict is not None:
        #we use the binary slice mask to "gate" the chosen_segment_mask, and we gate here, after dilation, as we do not want to dilate the slices
        binary_slices_mask = (slices_cropped_dict['binary_slices_mask']>0).float().to(device=chosen_segment_mask.device)
        if not torch.all(binary_slices_mask == 1.0):
            #skip if the binary_slices_mask is all ones, as this means the organ has tumors in unknown slices
            #assert chosen_segment_mask is also float
            if len(binary_slices_mask.shape) == 4:
                binary_slices_mask = binary_slices_mask.unsqueeze(1)#add channels dim if necessary
            assert len(binary_slices_mask.shape) == len(chosen_segment_mask.shape), f'binary_slices_mask shape {binary_slices_mask.shape} does not match chosen_segment_mask shape {chosen_segment_mask.shape}'
            assert binary_slices_mask.shape[0] == chosen_segment_mask.shape[0], f'binary_slices_mask shape {binary_slices_mask.shape} does not match chosen_segment_mask shape {chosen_segment_mask.shape}'
            assert binary_slices_mask.shape[-3:] == chosen_segment_mask.shape[-3:], f'binary_slices_mask shape {binary_slices_mask.shape} does not match chosen_segment_mask shape {chosen_segment_mask.shape}'
            for B in range(binary_slices_mask.shape[0]):
                if binary_slices_mask[B].sum() == 0 and slices_cropped_dict['slices_used'][B]:
                    raise ValueError(f'Binary slices mask for batch item {B} is all zeros, but slices were used')
                if  not torch.all(binary_slices_mask[B] == 1.0) and (not slices_cropped_dict['slices_used'][B]):
                    raise ValueError(f'Binary slices mask for batch item {B} is not all ones, but slices were not used')
            chosen_segment_mask = chosen_segment_mask.float()
            chosen_segment_mask_original = chosen_segment_mask.clone() #keep the original mask for debugging
            binary_slices_mask = binary_slices_mask.type_as(chosen_segment_mask)
            chosen_segment_mask = chosen_segment_mask * binary_slices_mask #here, we make the chosen_segment_mask zero in the organ slices with no tumor
            
            for B in range(chosen_segment_mask.shape[0]):
                if counter_vol<10:
                    if chosen_segment_mask_original[B].sum() == 0: #ignore cases annotated per voxel
                        continue
                    if not slices_cropped_dict['slices_used'][B]: #if you uncomment this, you will see only the cases cropped by slice
                        continue
                    counter_vol+=1
                    os.makedirs('SanityVolLoss/'+str(counter_vol), exist_ok=True)
                    save_tensor_as_nifti(chosen_segment_mask[B].sum(0),'SanityVolLoss/'+str(counter_vol)+'/chosen_segment_mask')
            
    out_in_subsegment = out * chosen_segment_mask
    #out_in_negative_voxels = out * negative_voxels
    #we do not penalize the negative voxels here, because we already penalize them in the standard segmentation losses.

    #we have 1 report volume per batch item, but to what class does it refer to? we can use chosen_segment_mask to figure that out
    report_volume = tumor_volumes.sum(-1) # shape B, we sum the multiple tumors we can have
    report_volume = report_volume.unsqueeze(-1).repeat(1,chosen_segment_mask.shape[1])#B,C
    gate=(chosen_segment_mask.sum(dim=(-1,-2,-3))>0).float()#B,C, one in the lesion channel the report volume refers to, 0 otherwise
    #assert gate.shape[-1]==3
    report_volume = report_volume * gate #B,C, only non-zero for lesion we care about in each CT patch. This is our ground-truth.

    if 'dice' in loss_function or 'dce_vol' in loss_function:
        loss=dice_based_volume_loss(out_in_subsegment,report_volume,tolerance=tolerance,E=500,cross_entropy=('entropy' in loss_function))
        #shape of loss should be B,C
        if class_weights is not None:
            #apply class weights to the loss
            loss = loss * class_weights
        loss = loss.mean(dim=-1) #mean over channels
        if sample_weights is not None:
            assert sample_weights.shape == loss.shape, f'sample_weights shape {sample_weights.shape} does not match loss shape {loss.shape}'
            loss = loss * sample_weights
        loss={'dice_volume_loss':loss.mean()}
        #print('Using dice volume loss')
        assert not torch.isnan(loss['dice_volume_loss']).any(), 'loss is nan'
        return loss
    else:
        raise ValueError('Deprecated loss function')


def dice_based_volume_loss(x,y,tolerance=0.1,E=500,cross_entropy=False):
    #assert no negative values
    #assert torch.min(y).item()>=0
    assert torch.min(x).item()>=0

    #assert no nan
    assert not torch.isnan(x).any(), 'Output is nan'
    assert not torch.isnan(y).any(), 'label is nan'

    #tolerance: return 0 if x/y is within 1+/- tolerance
    if len(x.shape)==5:
        x = x.sum((-1,-2,-3))
    assert len(x.shape)==2, f'shape of x is: {x.shape}'

    predicted_volume = x
    target_volume = y

    assert predicted_volume.shape == target_volume.shape
    
    #where the target volume is negative, it means we do not know the volume. Then, we need a dynamic target:
    #negative numbers will become 66 (volume of a 5mm diameter ball) if the predicted volume is less than 66;
    #it will become 903,000 (volume of a 120 mm diameter ball) if the predicted volume is greater than 903,000;
    #otherwise, it will be the predicted volume (0 loss).
    negatives = (target_volume < 0)
    negatives_and_small  = negatives & (predicted_volume < 66)
    negatives_and_large  = negatives & (predicted_volume > 905000)
    negatives_and_medium = negatives & ~(negatives_and_small | negatives_and_large)
    target_volume = target_volume.clone()  # avoid in-place on input
    target_volume[negatives_and_small]  = 66.0
    target_volume[negatives_and_large]  = 905000.0
    target_volume[negatives_and_medium] = predicted_volume.detach()[negatives_and_medium]

    loss=torch.abs(predicted_volume-target_volume)/(predicted_volume+target_volume+E)
    #E allows this to work when the ground-truth is zero.

    #subtract the loss at tolerance, for continuity
    v=(1-tolerance)*target_volume
    mini = target_volume.clamp(max=100)
    v    = torch.max(v, mini)
    loss_at_tolerance=torch.abs(v-target_volume)/(v+target_volume+E)

    loss=loss-(loss_at_tolerance*((~negatives).float()))

    #clamp at zero
    loss=torch.clamp(loss,min=0,max=1)

    if cross_entropy:
        #print('Using cross-entropy')
        loss = -torch.log(torch.ones(loss.shape).type_as(loss)-loss+1e-5)
    else:
        #print('Using dice volume without cross-entropy')
        pass

    return loss

import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_volume_loss_curve(
    target_volume,
    pred_range,
    *,
    tolerance=0.1,
    E=500,
    cross_entropy=False,
    title=None,
    show=True,
    save_path=None,
    ax=None,                   # if you pass an axis, derivative subplot is disabled
    show_derivative=True       # set False to hide the derivative subplot
):
    """
    Plots the loss vs predicted volume and (optionally) its numerical derivative below.

    Note: if `ax` is provided, the derivative subplot is skipped and only the top plot is drawn.
    Returns: (fig, (ax_loss, ax_deriv), preds, loss_vals, dloss) where ax_deriv and dloss may be None.
    """

    # --- build prediction grid ---
    if isinstance(pred_range, (list, np.ndarray)):
        preds = np.asarray(pred_range, dtype=np.float32)
    elif isinstance(pred_range, tuple) and len(pred_range) == 3:
        start, stop, step = pred_range
        preds = np.arange(start, stop + step, step, dtype=np.float32)
    else:
        raise ValueError("pred_range must be array-like or (start, stop, step).")

    pv = torch.tensor(preds, dtype=torch.float32).unsqueeze(1)  # (N,1)
    tv = torch.full_like(pv, float(target_volume))

    with torch.no_grad():
        loss_vals = dice_based_volume_loss(
            pv, tv, tolerance=tolerance, E=E, cross_entropy=cross_entropy
        ).squeeze(1).float().cpu().numpy()

    # --- figure/axes ---
    created_fig = False
    ax_deriv = None
    if ax is None:
        nrows = 2 if show_derivative else 1
        height = [2, 1] if show_derivative else [2]
        fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True,
                                 gridspec_kw={'height_ratios': height})
        if show_derivative:
            ax_loss, ax_deriv = axes
        else:
            ax_loss = axes
        created_fig = True
    else:
        fig = ax.figure
        ax_loss = ax

    # --- loss curve ---
    ax_loss.plot(preds, loss_vals)
    ax_loss.set_ylabel("Loss")
    xmin, xmax = float(preds.min()), float(preds.max())
    ax_loss.set_xlim(xmin, xmax)
    if title is None:
        title = f"Volume loss vs. prediction (target={target_volume}, tol={tolerance})"
    ax_loss.set_title(title)

    # zero-penalty band (clipped to visible range)
    if target_volume >= 0:
        lo = target_volume * (1 - tolerance)
        hi = target_volume * (1 + tolerance)
        lo_clip = max(lo, xmin); hi_clip = min(hi, xmax)
        if lo_clip < hi_clip:
            ax_loss.axvspan(lo_clip, hi_clip, alpha=0.15)
        if xmin <= target_volume <= xmax: ax_loss.axvline(target_volume, linestyle="--")
        if xmin <= lo <= xmax:            ax_loss.axvline(lo, linestyle=":")
        if xmin <= hi <= xmax:            ax_loss.axvline(hi, linestyle=":")
    else:
        band_lo, band_hi = 66.0, 905000.0
        lo_clip = max(band_lo, xmin); hi_clip = min(band_hi, xmax)
        if lo_clip < hi_clip:
            ax_loss.axvspan(lo_clip, hi_clip, alpha=0.15)
        if xmin <= band_lo <= xmax: ax_loss.axvline(band_lo, linestyle=":")
        if xmin <= band_hi <= xmax: ax_loss.axvline(band_hi, linestyle=":")

    ax_loss.set_xlabel("Predicted volume (mm³)")

    # --- derivative subplot (numerical) ---
    dloss = None
    if show_derivative and (ax_deriv is not None):
        # central differences with nonuniform spacing support
        dloss = np.gradient(loss_vals, preds)
        ax_deriv.plot(preds, dloss)
        ax_deriv.axhline(0.0, linestyle="--")
        ax_deriv.set_xlim(xmin, xmax)
        ax_deriv.set_xlabel("Predicted volume (mm³)")
        ax_deriv.set_ylabel("d(Loss)/d(Vol)")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show and created_fig:
        plt.show()

    return fig, (ax_loss, ax_deriv), preds, loss_vals, dloss

def plot_dice_based_volume_loss(y_value=1000, tolerance=0.1, E=500, num_points=100, x_min=0, x_max=10000,
                                cross_entropy=False):
    """
    Plots the loss for a fixed ground truth volume (y_value) as the predicted volume (x) varies.
    
    y_value : float
        The fixed ground truth volume.
    tolerance : float
        Tolerance percentage (default 0.1 means ±10%).
    E : float
        Offset constant in the denominator.
    num_points : int
        Number of points to sample for predicted volumes.
    x_min, x_max : float
        The range of predicted volumes to consider. If x_max is None, it defaults to 2*y_value.
    """
    import matplotlib.pyplot as plt
    if x_max is None:
        x_max = 2 * y_value  # Default range if not provided

    # Create a series of predicted volume values
    x_values = torch.linspace(x_min, x_max, num_points)
    
    # Create a dummy tensor "x" of shape (num_points, 1, 1, 1)
    # so that summing over the last three dims gives the predicted volume
    x_tensor = x_values.view(num_points, 1, 1, 1)
    
    # Create a target tensor "y" with the same predicted volume for each sample
    y_tensor = torch.full((num_points,), y_value)
    
    # Compute the individual loss values
    loss = dice_based_volume_loss(x_tensor, y_tensor, tolerance=tolerance, E=E, cross_entropy=cross_entropy)
    
    # Plot the loss as a function of the predicted volume
    plt.figure(figsize=(8, 6))
    plt.plot(x_values.numpy(), loss.numpy(), label='Dice-Based Volume Loss')
    plt.xlabel("Predicted Volume (x)")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Predicted Volume for Ground Truth y = {y_value}")
    plt.legend()
    plt.grid(True)
    plt.show()


def ln_with_tolerance_right_side(x: torch.Tensor,
                                 y: torch.Tensor,
                                 tolerance: float,
                                 delta: float = 1.0,
                                 n='huber'):
    """
    Huber-with-Tolerance using PyTorch's built-in F.huber_loss. 
    This loss is one-sided: it only penalizes cases where x > y + tolerance.
    Args:
        x (Tensor): Predicted values.
        y (Tensor): Target values.
        tolerance (float): Half-width of the 'dead zone' around y.
        delta (float): Huber 'transition point' between L2 and L1. Default=1.0.
    Returns:
        Tensor: The Huber-with-Tolerance loss. Shape depends on `reduction`.
    """
    #reduce in case x was not already reduced
    assert len(x.shape)==5 or len(x.shape)==2, 'x must have 5 or 2 dimensions, 2 means we already summed the spatial dimensions'
    assert len(y.shape)==2
    if len(x.shape)==5:
        x = x.sum((-1,-2,-3)) 
    
    #as we will normalize, we cannot have places where y is zero and x is not. This should already have been ensured by gating, since this is only a loss for organ segments with tumors!
    assert torch.allclose(x[y == 0], torch.zeros_like(x[y == 0]))

    #normalize
    x = x / (y + 1e-5)
    y = y / (y + 1e-5)
    diff = x - (y + tolerance)
    diff = torch.clamp(diff, min=0.0)

    if n=='huber':
        loss = F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction='none')
    elif n=='l2':
        loss = F.mse_loss(diff, torch.zeros_like(diff), reduction='none')
    elif n=='l1':
        loss = F.l1_loss(diff, torch.zeros_like(diff), reduction='none')
    else:
        raise ValueError('loss not supported')

    assert len(loss.shape)==2, 'loss shape should be B,C'

    loss = loss.mean()

    return loss

def volume_reduction_loss_selective(x,y,tolerance=0.1,n='huber',k=1.5):
    #In this loss, we want to reduce the tumor volume inside its sub-segment, to match the volume in the report.
    #however, we do not want to penalize the N voxels with the highest values. N is the volume of the tumor in the report.
    #the other voxels we will penalize.
    #we use the opposite of GWRP, without averaging, to sum the output voxels (in the subseg) that are not the top N ones.
    #We normalize this sum according to the report volume and we use a huber or l2 loss to make it be lower than the tolerance.
    #the loss is only applied if the volume in the output is greater than the report volume with the tolerance.

    B,C,H,W,D=x.shape

    pooled_x = GlobalWeightedRankPooling(x,N=y,c=0.3,inverse=True) #we ignore the top N voxels
    
    #we want to penalize only channels where the predicted tumor volume is greater than the report volume plus the tolerance
    summed_x = x.sum((-1,-2,-3))
    assert pooled_x.shape==summed_x.shape, 'shapes should be equal'
    assert len(pooled_x.shape)==2 and pooled_x.shape==y.shape, 'x shape should be B,C and match y'
    excessive_volume = (summed_x > (y * (1+tolerance)))#B,C
    #we also do not want to penalize cases where the report volume is zero here!
    pooled_x = pooled_x[excessive_volume & (y>0)]
    if pooled_x.numel() == 0:
        return torch.tensor(0).type_as(x)

    #we want to normalize the loss by the report volume
    y = y[excessive_volume & (y>0)]
    pooled_x = pooled_x / y

    #tolerance: we have a target of tolerance, not zero. Thus, we subtract the tolerance and clamp. Then our target becomes 0
    pooled_x = torch.clamp(pooled_x - tolerance, min=0.0)

    if n=='huber':
        loss = F.huber_loss(pooled_x/k, torch.zeros_like(pooled_x), delta=1.0, reduction='none')
    elif n=='l2':
        loss = F.mse_loss(pooled_x/k, torch.zeros_like(pooled_x), reduction='none')
    elif n=='l1':
        loss = F.l1_loss(pooled_x/k, torch.zeros_like(pooled_x), reduction='none')
    else:
        raise ValueError('loss not supported')
    
    loss = loss.mean()
    
    return loss


def GWRP_expansion_loss(x,y, concentrate=1, eps=1e-5):
    #this loss uses cross entropy and gwrp to enforce the model to detect a tumor of the correct size.

    #this is an expansion loss, we do not want this loss to penalize elements where y==0 then. These elements should have been already removed by gating.
    x_sum=x.sum((-1,-2,-3))
    assert torch.allclose(x_sum[y == 0], torch.zeros_like(x_sum[y == 0]))

    x=GlobalWeightedRankPooling(x,N=y,c=0.75,concentrate=concentrate)
    #we apply gwrp to x, returning a B,C tensor. The spatial dimenions are summed attributing 90% of the summation weight to the top y voxels, and 10% to the rest
    assert len(x.shape)==2, 'x shape should be B,C'

    #now we get only the x items where y is not 0
    x=x[y>0]

    #if empty, return zero
    if x.numel() == 0:
        return torch.tensor(0).type_as(x)

    #now we do cross entropy with target 1
    loss = -torch.log(x+eps)

    loss=loss.mean()

    return loss


def GWRP_background_loss(x, eps=1e-4, decision_th=30):
    #this loss uses cross entropy and gwrp to enforce no false positive tumor detection in the background. 
    #it focuses on the top "decision_th" voxels, strongly penalizing them to become 0 with cross entropy.
    #we use 30 as the decision th, since this is n acceptable decision threshold to convert a segmentation output into a binary classification label.

    #apply gwrp
    x = GlobalWeightedRankPooling(x,N=decision_th,c=0.9)

    #apply cross entropy with target 0
    loss = -torch.log((1-x)+eps)

    assert len(loss.shape)==2, 'loss shape should be B,C'

    loss = loss.mean()

    return loss


def GlobalWeightedRankPooling(x, N=1000, c=0.75, inverse=False, concentrate=1, return_weights=False,hard_cutoff=False):
    """
    Performs Global Weighted Rank Pooling (GWRP). The weights decay exponentially so that
    the top N voxels receive c% of the total weight.
    Ps: the raw weight at voxel N will be 1-c. 
    So, the inverse weight will be c.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        N (int or torch.Tensor): Number of top voxels to concentrate. If an integer, a scalar
                                 value is used; if a tensor of shape (B, C), each (B,C) pair 
                                 uses its own N.
        c (float): Fraction (e.g. 0.9 for 90%) of the total weight to be concentrated in the top N voxels.
    
    Returns:
        torch.Tensor: The pooled tensor of shape (B, C).
    """
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0).unsqueeze(0)
        reduce=True
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    B, C, H, W, D = x.shape
    L = H * W * D  # total number of voxels per (B, C)
    
    # Sort the spatial elements in descending order.
    x_sorted, sort_indices = torch.sort(x.view(B, C, L), dim=-1, descending=True)
    
    # Compute the decay factor d.
    # If N is a scalar, convert it to a tensor of shape (B, C) with that constant.
    if not torch.is_tensor(N):
        N_tensor = torch.full((B, C), N, dtype=torch.float32, device=x.device)
    else:
        N_tensor = N.to(x.device).float()
    # Ensure N is at least 1.
    N_tensor = torch.clamp(N_tensor, min=1)
    
    # Compute d elementwise: d = (1-c)^(1/N).
    d = (1 - c) ** (1.0 / N_tensor)  # shape (B, C)
    # Reshape d to (B, C, 1) so it can broadcast.
    d = d.unsqueeze(-1)
    
    # Create an index tensor of shape (1, 1, L).
    indices = torch.arange(L, dtype=torch.float32, device=x.device).view(1, 1, L)
    
    # Compute weights: each weight is d^(i), broadcasting over (B, C).
    weights_raw = d ** indices  # shape (B, C, L)
    weights = weights_raw / weights_raw.sum(dim=-1, keepdim=True)  # normalize to sum to 1

    #assert that, for a random B,C element, the sum of the first N weights is equal to c
    #rand_b=torch.randint(0,B,(1,))
    #rand_c=torch.randint(0,C,(1,))
    #summed = weights[rand_b, rand_c, :int(N_tensor[rand_b, rand_c].item())].sum()  
    #assert abs(summed.item() - c) < 0.2

    if inverse:
        # For the inverse case we want to ignore the top N voxels.
        # Create a mask that is 0 for indices < N and 1 for indices >= N.
        mask_inv = (indices >= N_tensor.unsqueeze(-1)).float()  # shape (B, C, L)
        # Use the complementary weights for the background: here we use (1 - weights_raw)
        weights = mask_inv * (1 - weights_raw)
        # Note: We do not normalize these weights to sum to 1 because the goal here is to measure
        # the background (i.e. the voxels outside the top N).
    elif concentrate!=1:
        assert concentrate>1, 'concentrate must be greater than 1'
        # Create two masks: one for the top N voxels and one for the rest.
        mask_top = (indices < N_tensor.unsqueeze(-1)).float()      # 1 for indices < N, 0 otherwise
        mask_rest = (indices >= N_tensor.unsqueeze(-1)).float()     # 1 for indices >= N, 0 otherwise
        # Leave top N voxels unchanged and scale the rest by (1/concentrate)
        new_weights = mask_top * weights + mask_rest * (weights / concentrate)
        # Renormalize the weights so they sum to 1.
        weights = new_weights / new_weights.sum(dim=-1, keepdim=True)

    if return_weights:
        if hard_cutoff:
            #make all weights after N zero and re-normalize
            mask_top = (indices < N_tensor.unsqueeze(-1)).float()
            weights = mask_top * weights
            weights = weights / weights.sum(dim=-1, keepdim=True)
        # We need to return the weights reorganized into the original spatial order.
        # sort_indices tells us, for each (B, C, i), which voxel in the unsorted order that value came from.
        # Compute the inverse permutation.
        inverse_indices = sort_indices.argsort(dim=-1)
        # unsort the weights so that they align with the original order.
        weights_unsorted = weights.gather(dim=-1, index=inverse_indices)
        # Reshape to original spatial dimensions.
        weights_unsorted = weights_unsorted.view(B, C, H, W, D)
        if reduce:
            weights_unsorted = weights_unsorted.squeeze(0).squeeze(0)
        return weights_unsorted
    
    # Compute weighted sum and normalize by the sum of weights.
    pooled = (x_sorted * weights).sum(dim=-1)

    return pooled



def DiceLossMultiClass(preds, targets, known_voxels, alpha = 0.5, beta=0.5, 
                       size_average=True, reduce=True, sigmoid=True, class_weights=None):

    if len(preds.shape)==3:
        preds=preds.unsqueeze(0).unsqueeze(0)
    if len(targets.shape)==3:
        targets=targets.unsqueeze(0).unsqueeze(0)
    if len(known_voxels.shape)==3:
        known_voxels=known_voxels.unsqueeze(0).unsqueeze(0)

    if len(preds.shape)==4:
        preds=preds.unsqueeze(0)
        targets=targets.unsqueeze(0)
        known_voxels=known_voxels.unsqueeze(0)

    assert len(preds.shape)==5
    assert (preds.shape == targets.shape) and (targets.shape == known_voxels.shape), f"Shapes do not match, pred, target and unk are: {preds.shape}, {targets.shape}, {known_voxels.shape}"

    N = preds.size(0)
    C = preds.size(1)
    
    if sigmoid:
        P = torch.sigmoid(preds)
    else:
        P = preds

    P = P * known_voxels
    targets = targets * known_voxels

    smooth = 1e-5

    class_mask = targets

    P_ = 1.0 - P
    class_mask_ = 1.0 - class_mask
    reduce_dims = (-1, -2, -3)  # D,H,W
    TP_sum = (P * class_mask).sum(dim=reduce_dims).float()      # [N,C]
    FP_sum = (P * class_mask_).sum(dim=reduce_dims).float()     # [N,C]
    FN_sum = (P_ * class_mask).sum(dim=reduce_dims).float()     # [N,C]
    FP_c = FP_sum.sum(dim=0)  # [C]
    FN_c = FN_sum.sum(dim=0)  # [C]
    alpha = FP_c / (FP_c + FN_c + smooth)  # [C]
    alpha = alpha.unsqueeze(0).expand(N, C)  # [N,C] (no new allocation like repeat)
    alpha = torch.clamp(alpha, min=0.2, max=0.8)
    beta = 1.0 - alpha
    num = TP_sum
    den = num + alpha * FP_sum + beta * FN_sum



    dice = num / (den + smooth)
    loss = 1 - dice
    if class_weights is not None:
        class_weights = class_weights.mean(dim=(-1,-2,-3))
        while len(class_weights.shape) < len(loss.shape):
            class_weights = class_weights.unsqueeze(0)
        assert class_weights.shape == loss.shape, f'Class weights shape {class_weights.shape} does not match the shape of dice loss {loss.shape}'
        # Apply class weights
        loss = loss * class_weights
    
    if not reduce:
        return loss

    if size_average:
        assert len(loss.shape) == 2, f'Loss should be 2D after reduction, but got {loss.shape}.'
        loss = loss.mean()  # Average over the batch size

    return loss

counter2=0


class MultiTaskLossWrapper(nn.Module):
    """
    Learnable loss weighting for multiple loss components.
    For loss components L_i, we learn parameters s_i and compute:
    
        WeightedLoss = sum_i [ 0.5 * exp(-s_i) * L_i + s_i ]
    """
    def __init__(self, num_losses):
        super(MultiTaskLossWrapper, self).__init__()
        # Initialize log variance parameters to 0 (i.e. initial weight exp(0)=1)
        self.s = nn.Parameter(torch.zeros(num_losses),requires_grad=True)
        
    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss = total_loss + 0.5 * torch.exp(-self.s[i]) * loss + self.s[i]
        #print the loss weights
        print('Loss weights:', [torch.exp(-s).item() for s in self.s])
        return total_loss
    

def classification_loss(cls_out, label, unk_voxels, args, chosen_segment_mask, classes, class_weights=None):
    #calculate classification loss
        
    if args.epai_stage_2:
        lesion_idx = [i for i, class_name in enumerate(classes) if (('background' in class_name) or ('pdac' in class_name) or ('pnet' in class_name) or ('cyst' in class_name))]
        lesion_labels = label[:, lesion_idx].float()
        if chosen_segment_mask is not None:
            lesion_labels += chosen_segment_mask[:, lesion_idx].float()
        #print('Lesion labels shape:', lesion_labels.shape)
        #class should be the class of the center voxel
        lesion_labels = lesion_labels[:, :, lesion_labels.shape[2]//2, lesion_labels.shape[3]//2, lesion_labels.shape[4]//2]
        #assert single label
        assert len(lesion_labels.shape)==2, f'Lesion labels shape is: {lesion_labels.shape}'
        assert lesion_labels.sum(dim=1).max()<=1, f'Lesion labels should be single label, but got {lesion_labels.sum(dim=1).max()}'
        #print('Lesion labels:', lesion_labels)
        target_idx = lesion_labels.argmax(dim=1).long() # (B,)
        #print('Lesion label:', target_idx)
        #print('cls_out out:', cls_out.shape)
        #background class? no cyst? lesion? what!?
    else:
        lesion_idx = [i for i, class_name in enumerate(classes) if ('lesion' in class_name)]
        #print(f'Classification loss for classes: {[classes[i] for i in lesion_idx]}')
        lesion_labels = label[:, lesion_idx].float()
        #multi-class
        if chosen_segment_mask is not None:
            lesion_labels += chosen_segment_mask[:, lesion_idx].float()
        lesion_labels = (lesion_labels.sum(dim=(-1,-2,-3))>0).float()
    #now check chosen_segment_mask
    assert len(cls_out.shape)==2 and cls_out.shape[0]==label.shape[0], f'Classification output shape is: {cls_out.shape}, label shape is: {label.shape}'
    if args.epai_stage_2:
        #softmax
        #for i in range(target_idx.shape[0]):
        #    print('Target idx:', target_idx[i], 'cls_out:', cls_out[i],
        #           'class:', classes[target_idx[i]])
        cls_loss   = F.cross_entropy(cls_out, target_idx,reduction='none')
    else:
        #sigmoid
        cls_loss = F.binary_cross_entropy_with_logits(cls_out, lesion_labels, reduction='none', weight=class_weights)
        #print(f'Labels: {lesion_labels[0]}')
        #print(f'cls_out: {cls_out[0]}')
        #print(f'cls_loss: {cls_loss[0]}')
    #if channels with unknown voxels are present and their label is 0, remove them from the loss (multiply by 0)
    if unk_voxels is not None:  
        unk_labels = (unk_voxels[:, lesion_idx].sum(dim=(-1,-2,-3))>0).float()
        #where unk_labels is 1 and label is 0:
        known_labels = (1-unk_labels)+lesion_labels
        known_labels = (known_labels>0).float()
        cls_loss = cls_loss * known_labels
    cls_loss = cls_loss.mean()
    #print('Classification loss:', cls_loss)
    return cls_loss


def model_genesis_loss(result,label):
    #MSE voxel-wise loss
    if isinstance(result, tuple) or isinstance(result, list):
        raise ValueError('Turn off deep supervision for model genesis pretraining')
    l = torch.nn.functional.mse_loss(result,label, reduction='mean')
    loss={'genesis_loss': l,
          'overall': l}
    return loss
    

def merge_no_overlap(d1, d2):
    overlap = d1.keys() & d2.keys()
    if overlap:
        raise KeyError(f"Cannot merge: duplicate keys found: {overlap}")
    return {**d1, **d2}




def l1_simple_attenuation_classifier(tumor_att_relative, target_att, chosen_segment_mask, class_list,
                                     contrast=None,args=None):
    """
    A L1 loss that is only activated when the tumor_att_relative (tumor_att - organ_att) has a different sign from the target attenuation.
    Ignores cases of mixed attenuation or isoattenuation or FP tumor. 
    chosen_segment_mask is used to find which is the lesion channel.
    """
    if (args is not None) and (args is not None) and args.attenuation_classifier_venous:
        tumor_att_relative=filter_by_contrast(tumor_att_relative,contrast,args)
        if tumor_att_relative is None:
            return torch.tensor(0).type_as(target_att)
        target_att=filter_by_contrast(target_att,contrast,args)
        chosen_segment_mask=filter_by_contrast(chosen_segment_mask,contrast,args)
    
    
    assert not torch.isnan(tumor_att_relative).any(), 'Attenuation out is nan'
    #first, we filter tumor_att_relative using chosen_segment_mask
    assert tumor_att_relative.shape[1] == chosen_segment_mask.shape[1], f'Output shape {tumor_att_relative.shape} does not match chosen segment mask shape {chosen_segment_mask.shape}'
    tumor_att_relative = get_lesion_channels(tumor_att_relative,class_list)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask,class_list)
    assert chosen_segment_mask.shape[1] == tumor_att_relative.shape[1], f'Chosen segment mask shape {chosen_segment_mask.shape} does not match tumor_att_relative shape {tumor_att_relative.shape}'
    chosen_segment_mask = chosen_segment_mask.sum(dim=(-1,-2,-3)) 
    chosen_segment_mask = (chosen_segment_mask > 0).float().unsqueeze(-1) #B,C, 1 if we have a lesion in this channel, 0 otherwise
    assert len(chosen_segment_mask.shape) == len(tumor_att_relative.shape), f'Chosen segment mask shape {chosen_segment_mask.shape} does not match tumor_att_relative shape {tumor_att_relative.shape}'
    tumor_att_relative = tumor_att_relative * chosen_segment_mask #B,C, we only keep the tumor channels that are in the segment mask
    assert torch.max(chosen_segment_mask.sum(-1)) <= 1, f'Chosen segment mask should have at most one lesion channel per batch item, but got {chosen_segment_mask.sum(-1)}'
    tumor_att_relative = tumor_att_relative.sum(dim=(1)) #drop channel dimension, we have only one tumor channel per CT crop
    if chosen_segment_mask.sum() == 0:
        #no lesion in this batch, return zero loss
        return 0*tumor_att_relative.sum()  # no penalization, no tumor
    
    loss = 0
    j = 0
    for b in range(target_att.shape[0]):
        tgt = target_att[b].int().tolist()
        if 999 in tgt:
            loss+=0*tumor_att_relative[b] #no penalization, iso-attenuation or unknow attenuation
            continue
        tgt = [t for t in tgt if t != 0]  # remove padding
        if len(tgt) == 0:
            loss+=0*tumor_att_relative[b] # no penalization, no tumor
            continue
        hyper = False
        hypo = False
        for t in tgt:
            if t > 0:
                hyper = True
            elif t < 0:
                hypo = True
        if (hyper and hypo):
            loss += 0*tumor_att_relative[b] #mixed attenuation, do not penalize
            
        out = tumor_att_relative[b]
        assert len(out.shape) == 1, f'Out shape should be 1D, but got {out.shape}'
        if hyper and out>0:
            loss += 0*out #correct, do not penalize
        if hypo and out<0:
            loss += 0*out #correct, do not penalize
        if hyper and out<0:
            loss += torch.abs(out) #L1 loss pushing towards zero
        if hypo and out>=0:
            loss += torch.abs(out) #L1 loss pushing towards zero
    loss = loss/target_att.shape[0]  # average over batch
    
    if torch.isnan(loss).any():
        raise ValueError('attenuation loss is nan, propagating this can destroy the network weights, STOP!')
        
    return loss


def tumor_to_organ(tumor_name):
    """
    Convert a tumor class name to an organ class name:
    - Remove '_lesion'
    - Substitutes some known patterns (like 'pancreatic' -> 'pancreas')
    - Randomly chooses a sided version for kidney, adrenal_gland, lung, or femur.
    """
    base = tumor_name.replace('_lesion', '')
    lower_base = base.lower()
    if lower_base == 'pancreatic':
        return 'pancreas'
    elif lower_base == 'kidney':
        return ['kidney_right', 'kidney_left']
    elif lower_base == 'adrenal_gland':
        return ['adrenal_gland_right', 'adrenal_gland_left']
    elif lower_base == 'adrenal':
        return ['adrenal_gland_right', 'adrenal_gland_left']
    elif lower_base == 'lung':
        return ['lung_right', 'lung_left']
    elif lower_base == 'femur':
        return ['femur_right', 'femur_left']
    elif lower_base == 'gallbladder':
        return 'gall_bladder'#added 28 apr 2025
    else:
        return base

from scipy.ndimage import label                     # 3‑D connected components
import scipy.ndimage as ndi

def attenutation_label_from_mask(
    organ_mask: torch.Tensor,
    tumor_mask: torch.Tensor,
    ct: torch.Tensor,
    c: float = 0.2,
    thresh: float = 0.5,
    max_tumours: int = 10,
) -> int:
    """
    Compare HU of up to `max_tumours` largest connected lesions against their organ.

    Returns
    -------
    1   - all tumour means  > organ_mean + organ_std * c
    0   - all tumour means  < organ_mean - organ_std * c
    2   - mixed (some larger, some smaller), none close
    999 - at least one tumour is “close” to the organ in hu value
    """

    # ----- 1. sanity checks -------------------------------------------------
    assert organ_mask.shape == tumor_mask.shape == ct.shape, f"shape mismatch, shapes: {organ_mask.shape}, {tumor_mask.shape}, {ct.shape}"
    organ_np = organ_mask.float().cpu().numpy() > thresh
    tumour_np = tumor_mask.float().cpu().numpy() > thresh
    ct_np     = ct.float().cpu().numpy()

    # ----- 2. label connected components in the tumour mask ----------------
    lbl, n_lbl = label(tumour_np, structure=np.ones((3,3,3)))  # 26‑conn.
    if n_lbl == 0:                                             # no tumour voxels
        return 999                                             # choice: "undefined"
    # voxel count per label (label 0 is background)
    sizes = np.bincount(lbl.ravel())[1:]                       # drop background
    # indices of the `max_tumours` largest components
    top_idxs = np.argsort(sizes)[::-1][:max_tumours] + 1       # +1 to skip bg

    # ----- 3. tumour mean HU values ----------------------------------------
    tumour_means = []
    for lab in top_idxs:
        vox = ct_np[lbl == lab]
        if vox.size:                                           # shouldn’t be 0
            tumour_means.append(vox.mean())

    # ----- 4. organ statistics (excluding *all* tumours) --------------------
    organ_only = organ_np & (~tumour_np)
    organ_hu   = ct_np[organ_only]
    if organ_hu.size == 0:                                     # degenerate case
        return 999
    organ_mean = organ_hu.mean()
    organ_std  = organ_hu.std()

    # ----- 5. decision logic -----------------------------------------------
    close = [
        abs(t_mean - organ_mean) < organ_std * c
        for t_mean in tumour_means
    ]
    if any(close):
        return 999

    all_gt = all(t_mean > organ_mean for t_mean in tumour_means)
    all_lt = all(t_mean < organ_mean for t_mean in tumour_means)

    if all_gt:
        return 1
    if all_lt:
        return 0
    return 2

def ce_MLP_attenuation_loss(out, target_att, chosen_segment_mask, class_list, labels=None,ct=None, lesion_classes_allowed=None,
                            contrast=None,args=None):
    """
    Here, we consider tumor attenuation as a classification problem.
    out: logits, shape B,C,4, where classes are hyper, hypo, mixed/iso attenuation
    """
    if (args is not None) and args.attenuation_classifier_venous:
        batch_before_filter = out.shape[0]
        out=filter_by_contrast(out,contrast,args)
        if out is None:
            return torch.tensor(0).type_as(target_att)
        target_att=filter_by_contrast(target_att,contrast,args)
        chosen_segment_mask=filter_by_contrast(chosen_segment_mask,contrast,args)
        if labels is not None:
            labels=filter_by_contrast(labels,contrast,args)
        batch_after_filter = out.shape[0]
        #print(f'Attenuation classifier venous-only: filtered batch size from {batch_before_filter} to {batch_after_filter}, contrast: {contrast}')
    
    assert not torch.isnan(out).any(), 'Attenuation out is nan'
    assert out.shape[1] == chosen_segment_mask.shape[1], f'Output shape {out.shape} does not match chosen segment mask shape {chosen_segment_mask.shape}'
    assert out.shape[0] == chosen_segment_mask.shape[0], f'Output shape {out.shape} does not match chosen segment mask shape {chosen_segment_mask.shape}'
    out, lesion_classes = get_lesion_channels(out,class_list,return_class_names=True)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask,class_list)
    if labels is not None:
        labels_all = labels.clone() 
        labels = get_lesion_channels(labels,class_list)
    assert chosen_segment_mask.shape[1] == out.shape[1], f'Chosen segment mask shape {chosen_segment_mask.shape} does not match out shape {out.shape}'
    chosen_segment_mask = chosen_segment_mask.sum(dim=(-1,-2,-3)) 
    chosen_segment_mask = (chosen_segment_mask > 0).float().unsqueeze(-1) #B,C, 1 if we have a lesion in this channel, 0 otherwise
    assert len(chosen_segment_mask.shape) == len(out.shape), f'Chosen segment mask shape {chosen_segment_mask.shape} does not match out shape {out.shape}'
    out = out * chosen_segment_mask #B,C, we only keep the tumor channels that are in the segment mask
    assert torch.max(chosen_segment_mask.sum(-1)) <= 1, f'Chosen segment mask should have at most one lesion channel per batch item, but got {chosen_segment_mask.sum(-1)}'
    out = out.sum(dim=(1)) #drop channel dimension, we have only one tumor channel per CT crop. Shape B,3 now
    if chosen_segment_mask.sum() == 0:
        #no lesion in this batch, return zero loss
        return 0*out.sum()  # no penalization, no tumor
    
    y = []
    out_clean = []
    #loop through target_att, skip cases of unk attenuation (999)
    for b in range(target_att.shape[0]):
        anno_by_voxel=False
        if labels is not None:
            l = labels[b]
            if l.sum()>0: # case annotated by voxel! We use it only to train the MLP (attenuation classifier)
                anno_by_voxel = True
                
        if anno_by_voxel:
            l_a = labels_all[b]
            idx = {c: i for i, c in enumerate(class_list)}  # O(1) lookup
            #inside the model, the MLP already received the label.
            #now we need to create the attenuation label, as target_att is wrong here---case without report
            #select only one tumor type
            choices = [c for c in range(l.shape[0]) if ((l[c].sum()>0) and (lesion_classes[c] in lesion_classes_allowed))]
            if len(choices) == 0:
                continue  # no valid lesions in this batch item
            mask = torch.zeros_like(l)
            choice = random.choice(choices)
            chosen_lesion = lesion_classes[choice]
            l = l[choice]
            #get organ mask
            org = tumor_to_organ(chosen_lesion)
            if org == 'uterus':
                org = 'prostate'
            if isinstance(org, list):
                if any(o not in idx for o in org):
                    raise ValueError(f'Organ {org} not found in index mapping.')
                else:
                    masks = [l_a[idx[o]] for o in org]
                    # element‑wise max across sides
                    organ_mask = reduce(torch.maximum, masks)
            else:
                if org not in idx:
                    raise ValueError(f'Organ {org} not found in index mapping.')
                else:
                    organ_mask = l_a[idx[org]]
            assert len(ct.shape)==5
            lab = attenutation_label_from_mask(organ_mask, l, ct[b][0])
            if lab == 999:
                continue
            y.append(lab)
            out_clean.append(out[b])
        else:
            tgt = target_att[b].int().tolist()
            if 999 in tgt:
                continue
            tgt = [t for t in tgt if t != 0]  # remove padding
            if len(tgt) == 0:
                continue
            
            hyper = False
            hypo = False
            for t in tgt:
                if t > 0:
                    hyper = True
                elif t < 0:
                    hypo = True
                    
            out_clean.append(out[b])
            if (hyper and hypo):
                y.append(2)
            elif hyper:
                y.append(1)
            elif hypo:
                y.append(0)
            else:
                raise ValueError(f'Invalid target attenuation class for batch item {b}: {tgt}')
    
    # If nothing usable so far
    if len(y) == 0:
        return 0 * out.sum()

    # Determine number of logits per item after your channel reduction
    K = out_clean[0].shape[-1]  # should be 1 for 'neuron', 3 for MLP
    if getattr(args, 'attenuation_classifier', None) == 'neuron':
        # --- Binary case: one logit per item ---
        if K != 1:
            raise ValueError(f"'neuron' head expects 1 logit, got K={K}")

        # drop mixed (2) and keep only 0/1
        keep = [i for i, yi in enumerate(y) if yi in (0, 1)]
        if len(keep) == 0:
            return 0 * out.sum()

        y_bin = torch.tensor([y[i] for i in keep], dtype=torch.float32, device=out.device)  # float for BCE
        logits = torch.stack([out_clean[i] for i in keep], dim=0)  # (N, 1)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # (N,)

        # optional: pos_weight to handle class imbalance (set if you want)
        # pos_weight = torch.tensor([w], device=out.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_bin)  # or with pos_weight=pos_weight

    else:
        # --- Multiclass case: three logits per item (hypo, hyper, mixed) ---
        if K != 3:
            raise ValueError(f"Multiclass head expects 3 logits, got K={K}")

        y_mc = torch.tensor(y, dtype=torch.long, device=out.device)

        # (optional) drop mixed for some configs:
        # keep = [i for i, yi in enumerate(y) if yi != 2]; ...

        logits = torch.stack(out_clean, dim=0)  # (N, 3)
        # Sanity: targets in range
        y_min, y_max = int(y_mc.min().item()), int(y_mc.max().item())
        if y_min < 0 or y_max >= 3:
            raise ValueError(f"Target out of range: y in [{y_min},{y_max}] for K=3")
        loss = torch.nn.functional.cross_entropy(logits, y_mc)

    if torch.isnan(loss).any():
        raise ValueError("attenuation loss is NaN; aborting.")

    return loss
    
import torch
from scipy.optimize import linear_sum_assignment

def match_and_reorder(pred, target, names=None):
    """
    Hungarian-match predicted tumours to the GT tumours *by relative size only*
    and return the prediction tensor reordered so that rows 0..N_gt-1 line up
    with the GT order.

    Parameters
    ----------
    pred   : Tensor shape (N_pred, 3)
             Columns: major_diam, minor_diam, objectness_logit (not used for matching)
    target : Tensor shape (N_gt, 2)
             Columns: major_diam_gt, minor_diam_gt

    Returns
    -------
    pred_reordered : Tensor shape (N_pred, 3)
        First N_gt rows correspond 1-to-1 to the GT list;
        any surplus predictions follow in their original order.
    match_idx : 1-D LongTensor of length N_gt
        `match_idx[j]` is the index in the *original* `pred` tensor that
        has been matched to GT row j (-1 if no prediction was assigned).
    N_gt: number of matched tumors / number of tumors in ground-truth
    """
    assert len(target.shape) == 2, f'Target shape must be 2D, got {target.shape}'
    
    if target.shape[1] != 2:
        target = target[:, :2]  # ignore third diameter, which is usually estimated.
    
    if target.sum() == 0:                             # no GT tumours in this image
        raise ValueError('Target must have at least one tumour with >0 diameters.')
    
    #assert no padding (zero sized tumor in gt):
    positive = target > 0
    assert positive.all(), f'Invalid target diameters, must be > 0: {target}, samples are: {names}'
    
    N_pred = pred.size(0)
    N_gt   = target.size(0)

    # build cost matrix (N_pred × N_gt)
    d_pred = pred[:, :2].unsqueeze(1)         # (N_pred, 1, 2), removes objectness scores
    d_gt   = target.unsqueeze(0)              # (1, N_gt, 2)

    rel_err = (d_pred - d_gt).abs() / (d_gt+1e-5)   # (N_pred, N_gt, 2)
    cost    = rel_err.sum(dim=2).detach().float().cpu().numpy()       # to NumPy for scipy, (N_pred, N_gt)

    row_ind, col_ind = linear_sum_assignment(cost)   # Hungarian

    # build result mapping GT → prediction
    match_idx = torch.full((N_gt,), -1, dtype=torch.long).to(pred.device)
    match_idx[col_ind] = torch.as_tensor(row_ind, dtype=torch.long).to(pred.device)

    # 1) matched predictions in GT order
    matched   = pred[match_idx[match_idx >= 0]]      # rows that got matched
    assert matched.size(0) == N_gt, f'Matched predictions count mismatch: {matched.size(0)} != {N_gt}'
    # 2) unmatched predictions in original order
    unmatched_mask = torch.ones(N_pred, dtype=torch.bool).to(pred.device)
    unmatched_mask[row_ind] = False
    unmatched = pred[unmatched_mask]

    pred_reordered = torch.cat((matched, unmatched), dim=0)

    return pred_reordered, match_idx, N_gt#number of matched cases

from typing import Any

def dissect_item(item: Any, name: str = "item", indent: int = 0) -> None:
    """
    Recursively print the structure of `item`.
      - Dict: prints its keys and recurses on each value.
      - List/Tuple: prints its length and recurses on each element.
      - torch.Tensor: prints its shape.
      - Other: prints its type (and value, if small).
    
    Args:
        item:    The object to inspect.
        name:    A label for this branch (useful at top level).
        indent:  Current indentation (for internal recursive use).
    """
    prefix = " " * indent
    # Top‐level header
    if indent == 0:
        print(f"{prefix}Inspecting '{name}' (type={type(item).__name__}):")

    # torch.Tensor
    if isinstance(item, torch.Tensor):
        print(f"{prefix}  → Tensor, shape={tuple(item.shape)}")
    
    # dict
    elif isinstance(item, dict):
        keys = list(item.keys())
        print(f"{prefix}  → Dict with keys={keys}")
        for k, v in item.items():
            print(f"{prefix}    Key '{k}':")
            dissect_item(v, name=str(k), indent=indent + 6)
    
    # list or tuple
    elif isinstance(item, (list, tuple)):
        tname = type(item).__name__
        print(f"{prefix}  → {tname}, length={len(item)}")
        for idx, v in enumerate(item):
            print(f"{prefix}    [{idx}]:")
            dissect_item(v, name=f"{tname}[{idx}]", indent=indent + 6)
    
    # other types
    else:
        # For small scalars or strings, show the value; otherwise just the type
        if isinstance(item, (int, float, str, bool)):
            print(f"{prefix}  → {type(item).__name__}, value={item}")
        else:
            print(f"{prefix}  → {type(item).__name__}")
    
def diameter_objectness_loss(diameters_out, diameters_rep, class_list, unk, chosen_mask,tumor_diameters_per_voxel,names=None):
    """
    Objectness penalization: 
    We want 0 objectness for all tumor types we know are not in the report.
    For tumor channels with unknown tumors, we do not penalize.
    For tumor channels with known tumors, we penalize the objectness of matches tumors to become one, and unmatched to become 0.
    
    Diameter penalization:
    We only penalize the diameters for the matched tumors.
    
    Arguments:
    diameters_out: output from the model, shape B,C,10,3 (batch, class number, tumors dimension, (diameter 1, diameter 2, objectness))
    diameters_report: report from the ground truth, shape B,10,3 (no C, as it is only for the chosen tumor channel, have 0 padding to ensure number of tumors is always 10, and has 3 diameters per tumor, the last is usually estimated)
    class_list: list of class names, used to get lesion channels
    unk_voxels: mask of unknown voxels, shape B,C,H,W,L, used to understand which channels have tumor. We have the tumor diameters for only one of these channels, so we do not want to penalize the channels with unknown tumor diameters. But we want to penalize objectness in channels we know have no tumors.
    chosen_segment_mask: mask of chosen segment, shape B,C,H,W,L, used to understand which channel is the chosen tumor channel, for which we have tumor diameters. We want to penalize objectness and diameters for this channel.
    """
    objectness_weight = 0.2  # weight for objectness loss
    diameter_weight = 1.0   # weight for diameter loss
    chosen_segment_mask = copy.deepcopy(chosen_mask) #avoid modifying the original 
    diameters_report = copy.deepcopy(diameters_rep) #avoid modifying the original 
    unk_voxels = copy.deepcopy(unk) #avoid modifying the original
    
    
    assert len(diameters_report.shape) == 3
    
    #labels: we must select our targets (diameters) from reports (diameters_report) or from tumor_diameters_per_voxel, depending if each batch sample was annotated by voxel or not
    tmp_target = []
    keys = list(tumor_diameters_per_voxel.keys())#lesion classes
    #print('tumor_diameters_per_voxel:')
    #print(dissect_item(tumor_diameters_per_voxel))
    #print('diameters_out:')
    #print(dissect_item(diameters_out))
    assert len(tumor_diameters_per_voxel[keys[0]]) == diameters_out.shape[0], f'We expect that, after collation, the dataloader will add a batch dimension to the tumor volumes in crop per voxel, but got {len(tumor_diameters_per_voxel[keys[0]])} and  {diameters_out.shape[0]}'
    assert len(tumor_diameters_per_voxel[keys[0]]) == diameters_report.shape[0], f'We expect that, after collation, the dataloader will add a batch dimension to the tumor volumes in crop per voxel'
    for b in list(range(tumor_diameters_per_voxel[keys[0]].shape[0])):#batch
        #random shuffle keys--We will always take the first tumor channel > 0 we see
        keys_shuffled = random.sample(keys, len(keys))
        for key in keys_shuffled:
            if tumor_diameters_per_voxel[key][b].sum() > 0:
                print('Case annotated per voxel')
                #case annotated per voxel
                diameters_report[b] = tumor_diameters_per_voxel[key][b]
                assert chosen_segment_mask[b].sum() == 0, f'Since this case is annotated per voxel, we expect the chosen segment mask to be empty!'
                chosen_segment_mask[b, class_list.index(key)] = torch.ones_like(chosen_segment_mask[b, class_list.index(key)]) #set the chosen segment mask to this channel
                break
                
                
    
    #get lesion channels
    diameters_out = get_lesion_channels(diameters_out, class_list)
    unk_voxels = get_lesion_channels(unk_voxels, class_list)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, class_list)
    
    #use unk to remove channels we do not want to penalize
    #channels that may have tumors:
    unk_ch = unk_voxels.sum(dim=(-1,-2,-3)) > 0
    unk_ch = unk_ch.float()
    to_penalize = 1 - unk_ch
    # chosen tumor type
    chosen_ch = chosen_segment_mask.sum(dim=(-1,-2,-3)) > 0
    chosen_ch = chosen_ch.float()
    # ensure the chosen channel is penalized
    to_penalize = to_penalize + chosen_ch
    to_penalize = torch.clamp(to_penalize, 0, 1)
    to_penalize = to_penalize.unsqueeze(-1)
    assert (to_penalize.shape[0] == diameters_out.shape[0]) and (to_penalize.shape[1] == diameters_out.shape[1]), f'To penalize shape {to_penalize.shape} does not match diameters_out shape {diameters_out.shape}'
    
    loss = []
    
    for b in range(diameters_out.shape[0]):#batch
        dia_out = diameters_out[b] #C, 10, 3
        dia_rep = diameters_report[b] # 10, 3
        assert dia_out.shape[1:] == dia_rep.shape, f'Diameter output shape {dia_out.shape[1:]} does not match report shape {dia_rep.shape}'
        unk = unk_voxels[b]
        chosen = chosen_segment_mask[b]
        penalize = to_penalize[b] # this is a loss mask
        
        if dia_rep.sum() < 0:
            #unknown diameter in sample, skip
            loss.append(0*dia_out.mean())
            continue
        
        if (chosen.sum()==0):
            assert dia_rep.sum() == 0, f'Chosen segment mask has no lesion, but report has tumors: {dia_rep}'
            #no lesion in this element, we just penalize objectness
            objectness = dia_out[:,:, 2]  # objectness is the last channel
            l = torch.nn.functional.binary_cross_entropy_with_logits(objectness, torch.zeros_like(objectness), reduction='none')
            assert len(penalize.shape) == len(l.shape), f'Penalize shape {penalize.shape} does not match objectness loss shape {l.shape}'
            l = l * penalize
            #only objectness loss
            loss.append(l.mean()*objectness_weight)
            continue
        
        assert dia_rep.sum() > 0, f'Chosen segment mask has lesion, but report has no tumors. dia_rep: {dia_rep}, chosen: {chosen.sum()}, sample: {names[b]}'
        
        #remove any padding in the report
        dia_rep = torch.stack([dia_rep[i] for i in range(dia_rep.shape[0]) if dia_rep[i].sum() > 0], dim=0)

        
        #get the index of the chosen channel
        chosen_ch = chosen.sum((-1,-2,-3)) > 0  # C
        chosen_ch = chosen_ch.float()
        assert chosen_ch.sum()==1, f'Chosen segment mask must have exactly one lesion channel per batch item, but got {chosen_ch.sum()}'
        max=10
        i=0
        while len(chosen_ch.shape) > 1:
            chosen_ch = chosen_ch.squeeze()
            i += 1
            if i > max:
                raise ValueError(f'Chosen channel mask has too many unreducible dimensions: {chosen_ch.shape}')
        chosen_ch = torch.nonzero(chosen_ch)  # get the index of the chosen channel
        chosen_ch = chosen_ch.squeeze().squeeze()
        if len(chosen_ch.shape) == 0:
            chosen_ch = chosen_ch.item()
        else:
            raise ValueError(f'Chosen channel mask has invalid shape: {chosen_ch.shape}')
        
        #match diameters in the output to the report
        pred_reordered, match_idx, N_gt = match_and_reorder(dia_out[chosen_ch], dia_rep, names=names[b])
        
        #objectness loss for non-chosen channels:
        objectness = dia_out[:,:, 2]  # objectness is the last channel
        assert len(penalize.shape) == len(objectness.shape), f'Penalize shape {penalize.shape} does not match objectness loss shape {objectness.shape}'
        l_obj_not_chosen = torch.nn.functional.binary_cross_entropy_with_logits(objectness, torch.zeros_like(objectness), reduction='none')
        l_obj_not_chosen = l_obj_not_chosen * penalize
        l_obj_not_chosen[chosen_ch] = 0 * l_obj_not_chosen[chosen_ch]  # zero out the chosen channel
        l_obj_not_chosen = l_obj_not_chosen.mean()
        
        #objectness loss for chosen channel (will always be 1):
        objectness = pred_reordered[:, 2]  # objectness is the last channel
        target_obj = torch.zeros_like(objectness)
        #make first N_gt elements 1, rest 0
        target_obj[:N_gt] = 1
        assert target_obj.sum() == N_gt, f'Target objectness should have {N_gt} ones, but got {target_obj.sum()}'
        l_obj_chosen = torch.nn.functional.binary_cross_entropy_with_logits(objectness, target_obj, reduction='mean')
        
        #diameter loss for matched tumors (chosen channel):
        dia_out_matched = pred_reordered[:N_gt, :2]  # only matched cases
        dia_rep_matched = dia_rep[:, :2]  # ignore third diameter, which is usually estimated.
        assert dia_out_matched.shape == dia_rep_matched.shape, f'Matched diameters shape mismatch: {dia_out_matched.shape} != {dia_rep_matched.shape}'
        l_diameter = torch.nn.functional.mse_loss(dia_out_matched, dia_rep_matched, reduction='mean')
        
        l = objectness_weight*(l_obj_not_chosen + l_obj_chosen) + diameter_weight*l_diameter
        loss.append(l)
    
    loss = torch.stack(loss, dim=0).mean()  # average over batch
    return loss

counter_seg=0

def cut_known_voxels_with_slice(known_voxels,slices_cropped_dict,chosen_segment_mask,
                                dilation_radius=16):
    """
    known voxels represents voxels we are sure of the label. Now, we modify them, making voxels inside of the chosen_segment_mask but far from the tumor slices KNOWN. Voxels outside the segment were already considered known, so no change there.
    slices_cropped_dict: this has slice information. If provided, we will consider that organ slices without tumor are knwon. Then, the segmentation loss will minimize them, as they are zero in the label.
            -keys: 'slices_mask', 'tumor_df', 'binary_slices_mask', 'max_slice', 'slices_used'
    binary_slices_mask: ones in the slices that have tumors. Not gates with organ mask, pure slice representation.
    """
    global counter_seg
    if slices_cropped_dict is None:
        raise ValueError('slices_cropped_dict must be provided to cut known voxels with slices')
    #we have many batches, some may have slices, some not
    binary_slices_mask = (slices_cropped_dict['binary_slices_mask']>0).float() # B,1,H,W,L
    for B in range(known_voxels.shape[0]):
        #in case slices were not used, masks should be all ones---this will add no new known voxel.
        if not slices_cropped_dict['slices_used'][B]:
            assert torch.all(binary_slices_mask[B] == 1.0), f'For cases where we did not use slices, binary slices mask must be all ones, but got {binary_slices_mask[B].shape} with values {binary_slices_mask[B].unique()}'
        
    #ok, now we know we can cut the knwon voxels.
    
    #1- dilation. Let's be safe. Maybe the tumor measurement is not that precise.
    binary_slices_mask = dilate_volume(binary_slices_mask,dilation_radius)
    binary_slices_mask = (binary_slices_mask > 0).float()
    
    #2- this binary_slices_mask is 1 in the UNKNOWN voxels (slices around tumors annotated by reports)
    known_vox_slices_mask = 1 - binary_slices_mask #now, known_vox_slices_mask is 1 in the known voxels
    
    #3- let's put this in the correct channel.
    gate = (chosen_segment_mask.sum(dim=(-1,-2,-3)) > 0).float()  # B,C, it is 1 only in the lesion channel
    assert len(gate.shape) == 2, f'Gate shape should be 2D, but got {gate.shape}'
    assert gate.shape[0] == known_voxels.shape[0], f'Gate shape {gate.shape} does not match known_voxels shape {known_voxels.shape}'
    assert max(gate.sum(dim=-1)) <= 1, f'Gate should have at most one lesion channel per batch item, but got {gate.sum(dim=-1)}'
    gate = gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # B,C,1,1,1
    
    
    assert len(known_vox_slices_mask.shape) == 5, f'Inverted slices mask should be 5D, but got {known_vox_slices_mask.shape}'
    assert known_vox_slices_mask.shape[1] == 1, f'Inverted slices mask shape {known_vox_slices_mask.shape} should have only one channel, but got {known_vox_slices_mask.shape[1]}'
    assert known_vox_slices_mask.shape[0] == gate.shape[0], f'Inverted slices mask shape {known_vox_slices_mask.shape} does not match gate shape {gate.shape}'
    assert gate.shape[1] == known_voxels.shape[1], f'Gate C {gate.shape} vs known_voxels {known_voxels.shape}'
    assert len(gate.shape) == len(known_vox_slices_mask.shape), f'Gate shape {gate.shape} does not match inverted slices mask shape {known_vox_slices_mask.shape}'
    
    known_vox_slices_mask = known_vox_slices_mask.float() 
    gate = gate.type_as(known_vox_slices_mask)  # ensure the same type and device, e.g. float32
    known_vox_slices_mask = known_vox_slices_mask * gate  # B,C,H,W,L --- we only keep the mask only in the lesion channel, others are 0. 
    
    #4- now, we can cut the known voxels
    assert known_vox_slices_mask.shape == known_voxels.shape, f'Inverted slices mask shape {known_vox_slices_mask.shape} does not match known voxels shape {known_voxels.shape}'
    #make the organ voxels outside of the tumor slices KNOWN:
    known_voxels = known_voxels.float() 
    known_vox_slices_mask = known_vox_slices_mask.type_as(known_voxels)  # ensure the same type and device, e.g. float32
    known_voxels = known_voxels + known_vox_slices_mask  # B,C,H,W,L, we only keep the known voxels in the lesion channel
    #threshold to 1
    known_voxels = (known_voxels > 0).float()  # B,C,H,W,L, we only keep the known voxels in the lesion channel
    
    
    gate = gate.type_as(known_voxels)  # ensure the same type and device, e.g. float32
    for B in range(known_voxels.shape[0]):
        if chosen_segment_mask[B].sum() == 0:
            continue
        if not slices_cropped_dict['slices_used'][B]:
            continue
        
        if counter_seg<10:
            counter_seg+=1
            os.makedirs('SanitySegLoss/'+str(counter_seg), exist_ok=True)
            known_voxels_tumor_channel = (known_voxels*gate)[B].sum(0)
            if chosen_segment_mask[B].sum()>0 and torch.equal(known_voxels_tumor_channel,torch.ones_like(known_voxels_tumor_channel)):
                raise ValueError(f'There is a tumor annotated per voxel, but we are saying that all voxels in the tumor channel are known and should be minimized!')
            assert known_voxels_tumor_channel.shape[-3:] == known_voxels.shape[-3:], f'Known voxels tumor channel shape {known_voxels_tumor_channel.shape} does not match known voxels shape {known_voxels.shape[-3:]}'
            save_tensor_as_nifti(known_voxels_tumor_channel,'SanitySegLoss/'+str(counter_seg)+'/known_voxels_tumor_channel')
            print(f'{counter_seg} Average known_voxels_tumor_channel: {known_voxels_tumor_channel.mean()}')
    
    return known_voxels
    
    
    
    
    
def json_to_df(maybe_json):
    if maybe_json is None:
        return None
    if isinstance(maybe_json, str):
        if maybe_json.strip() == '' or maybe_json.strip() == '[]':
            return None
        return pd.read_json(io.StringIO(maybe_json), orient="records")
    if isinstance(maybe_json, list):
        return pd.DataFrame.from_records(maybe_json)
    if isinstance(maybe_json, dict):
        # handle the rare case someone saved with a different orient
        return pd.DataFrame([maybe_json])
    # already a DataFrame?
    return maybe_json
    from collections import defaultdict
from typing import List, Optional, Dict, Iterable
import torch
from collections import defaultdict
from typing import List, Optional, Dict, Iterable
import torch

# Assumes get_lesion_channels(tensor, classes) is available in scope.

GROUP_NAMES = [
    "per_voxel",            # 0
    "report_with_slice",    # 1
    "report_with_sizes",    # 2
    "report_no_size",       # 3
    "report_no_tumor",      # 4
]

class SimpleDecayBalancer:
    """
    Global (not per-batch) weighting with exclusions and optional capping.
      - Track decayed group counts across time.
      - Convert to probabilities with a uniform prior over *active* (non-excluded) groups.
      - Per-group weight: w_g = (target_g / p_g)^tau
      - Rescale so E_p[w] = 1 over *active* groups (pre-clip).
      - Optional cap: enforce report_with_sizes <= report_with_slice,
                     and report_no_size <= report_with_sizes (fallback to slice if needed).
      - NO per-batch normalization.
      - Excluded groups: ignored in counts/probs/targets; weights = 1.0.
    """
    def __init__(self,
                 decay: float = 0.999,
                 prior_strength: float = 5.0,
                 target: Optional[Dict[str, float]] = None,
                 tau: float = 1.0,
                 min_prob: float = 1e-6,
                 max_weight: Optional[float] = 25.0,
                 exclude: Optional[Iterable[str]] = ['per_voxel'],
                 cap: bool = True,
                 verbose: bool = False):
        self.G = list(GROUP_NAMES)
        self.decay = float(decay)
        self.prior = float(prior_strength)
        self.tau = float(tau)
        self.min_prob = float(min_prob)
        self.max_weight = max_weight
        self.exclude = set(exclude or [])
        self.cap = bool(cap)
        self.verbose = bool(verbose)

        unknown = self.exclude.difference(self.G)
        assert not unknown, f"Unknown groups in exclude: {unknown}"

        # Target distribution over all groups (renormalized over *active* when used)
        if target is None:
            u = 1.0 / len(self.G)
            self.target = {g: u for g in self.G}
        else:
            s = sum(target.values())
            assert s > 0, "target must sum > 0"
            self.target = {g: float(target.get(g, 0.0)) for g in self.G}
            s = sum(self.target.values())
            self.target = {g: v / s for g, v in self.target.items()}

        # Running decayed counts for all groups
        self.counts = defaultdict(float, {g: 0.0 for g in self.G})

    # ------------ Active helpers ------------
    def _active(self) -> List[str]:
        return [g for g in self.G if g not in self.exclude]

    def _active_target(self) -> Dict[str, float]:
        active = self._active()
        if not active:
            return {}
        t = {g: self.target[g] for g in active}
        s = sum(t.values()) or 1.0
        return {g: v / s for g, v in t.items()}

    # ------------ Group inference (your rules) ------------
    @torch.no_grad()
    def _infer_groups(
        self,
        labels: torch.Tensor,               # [B, C, H, W, D]
        chosen_segment_mask: torch.Tensor,  # [B, C, H, W, D]
        sizes_slices: Optional[torch.Tensor], # [B, 10, 2] or None
        tumor_volumes: torch.Tensor,        # [B, 10]
        classes: List[str],
    ) -> torch.Tensor:
        """
        Returns group IDs in {0..4}, shape [B].
          0 per_voxel            -> get_lesion_channels(labels).sum>0
          4 report_no_tumor      -> get_lesion_channels(chosen_segment_mask).sum==0
          1 report_with_slice    -> any non-NaN in sizes_slices[b, :, 1]
          3 report_no_size       -> any (tumor_volumes[b] < 0)
          2 report_with_sizes    -> otherwise
        """
        device = labels.device
        B = labels.shape[0]

        lab_les = get_lesion_channels(labels, classes)              # [B, C', H, W, D]
        seg_les = get_lesion_channels(chosen_segment_mask, classes) # [B, C', H, W, D]

        per_voxel_mask = (lab_les.sum(dim=(1,2,3,4)) > 0)           # [B]
        no_tumor_mask = (seg_les.sum(dim=(1,2,3,4)) == 0)           # [B]

        group_ids = torch.full((B,), -1, dtype=torch.long, device=device)

        # 0) per-voxel
        group_ids[per_voxel_mask] = 0
        remain = ~per_voxel_mask

        # 4) report_no_tumor
        no_tumor = remain & no_tumor_mask
        group_ids[no_tumor] = 4
        remain = remain & (~no_tumor_mask)

        if remain.any():
            idx = torch.where(remain)[0]

            # 1) report_with_slice
            if sizes_slices is not None:
                ss = sizes_slices.to(device)[idx]        # [Br, 10, 2]
                has_slice = (~torch.isnan(ss[:, :, 1])).any(dim=1)
            else:
                has_slice = torch.zeros(idx.numel(), dtype=torch.bool, device=device)

            group_ids[idx[has_slice]] = 1
            still = idx[~has_slice]

            if still.numel() > 0:
                # 3) report_no_size
                tv = tumor_volumes.to(device)[still]     # [Bs, 10]
                has_neg = (tv < 0).any(dim=1)
                group_ids[still[has_neg]] = 3

                # 2) report_with_sizes
                group_ids[still[~has_neg]] = 2

        group_ids[group_ids < 0] = 4  # safety
        return group_ids

    # ------------ Stats -> probs -> weights ------------
    def _decay_and_add(self, group_ids: torch.Tensor):
        # decay all
        for g in self.G:
            self.counts[g] *= self.decay
        # add current batch ONLY for active groups
        for gid in group_ids.tolist():
            g = self.G[int(gid)]
            if g in self.exclude:
                continue
            self.counts[g] += 1.0

    def _current_probs_active(self) -> Dict[str, float]:
        active = self._active()
        if not active:
            return {}
        raw = {g: self.counts[g] + self.prior / len(active) for g in active}
        s = sum(raw.values()) or 1.0
        probs = {g: max(raw[g] / s, self.min_prob) for g in active}
        s2 = sum(probs.values()) or 1.0
        return {g: probs[g] / s2 for g in active}

    def _apply_caps(self, w_active: Dict[str, float]) -> Dict[str, float]:
        """Enforce: sizes <= slice; no_size <= sizes (fallback to slice if sizes not active)."""
        if not self.cap:
            return w_active
        # sizes <= slice
        if "report_with_sizes" in w_active:
            if "report_with_slice" in w_active:
                w_active["report_with_sizes"] = min(
                    w_active["report_with_sizes"], w_active["report_with_slice"]
                )
        # no_size <= sizes (fallback to slice)
        if "report_no_size" in w_active:
            cap_ref = None
            if "report_with_sizes" in w_active:
                cap_ref = w_active["report_with_sizes"]
            elif "report_with_slice" in w_active:
                cap_ref = w_active["report_with_slice"]
            if cap_ref is not None:
                w_active["report_no_size"] = min(w_active["report_no_size"], cap_ref)
        return w_active

    def current_group_weights(self) -> Dict[str, float]:
        """
        Returns per-group weights for all 5 groups.
          - Active groups: (target/p)^tau, capped (if enabled), rescaled so E_p[w]=1 (pre-clip), then clipped.
          - Excluded groups: 1.0
        """
        active = self._active()
        p = self._current_probs_active()
        t = self._active_target()

        # default: excluded -> 1.0
        w_all: Dict[str, float] = {g: 1.0 for g in self.G}

        if active:
            # raw ratios with temperature
            w = {g: (t[g] / max(p[g], self.min_prob)) ** self.tau for g in active}

            # optional caps between active groups
            w = self._apply_caps(w)

            # normalize so E_p[w] = 1 (pre-clip)
            exp_mean = sum(p[g] * w[g] for g in active) or 1.0
            w = {g: w[g] / exp_mean for g in active}

            # final clip (may slightly change E_p[w])
            if self.max_weight is not None:
                w = {g: min(w[g], float(self.max_weight)) for g in active}

            # merge back
            w_all.update(w)

        return w_all

    # ------------ Public API ------------
    @torch.no_grad()
    def define_groups(
        self,
        labels: torch.Tensor,               # [B, C, H, W, D]
        chosen_segment_mask: torch.Tensor,  # [B, C, H, W, D]
        sizes_slices: Optional[torch.Tensor], # [B, 10, 2] or None
        tumor_volumes: torch.Tensor,        # [B, 10]
        classes: List[str],
    ) -> torch.Tensor:
        """
        Infers group per sample, updates *global* decayed counts (excluding ignored groups),
        prints batch composition & current per-group weights (if verbose),
        and returns per-sample weights of shape [B], WITHOUT per-batch normalization.
        """
        device = labels.device
        group_ids = self._infer_groups(labels, chosen_segment_mask, sizes_slices, tumor_volumes, classes)

        if self.verbose:
            uniq, cnt = torch.unique(group_ids, return_counts=True)
            recv = {self.G[int(u)]: int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}
            for g in self.G:
                recv.setdefault(g, 0)
            print(f"[SimpleDecayBalancer] Batch groups: {recv}", flush=True)
            if self.exclude:
                print(f"[SimpleDecayBalancer] Excluded groups: {sorted(self.exclude)}", flush=True)

        self._decay_and_add(group_ids)

        gw = self.current_group_weights()

        if self.verbose:
            ordered = {g: float(gw[g]) for g in self.G}
            print(f"[SimpleDecayBalancer] Group weights (global, caps={'on' if self.cap else 'off'}, E_p[w]=1 pre-clip): {ordered}", flush=True)

        w_vec = torch.tensor([gw[self.G[int(g)]] for g in group_ids.tolist()],
                             dtype=torch.float32, device=device)
        return w_vec  # [B], batches can have different means
    
balancer_samples = SimpleDecayBalancer(
    decay=0.999,
    prior_strength=5.0,
    tau=1.0,                 # try 0.5 to soften extremes
    max_weight=25.0,
    exclude=['per_voxel'],  # ignore these; their weights=1
    verbose=False
)

def sanity_assert_no_lesion_mask(label, classes):
    #ensure all per-voxel labels for lesions are zero
    lesion_channels,class_names = get_lesion_channels(label, classes, return_class_names=True)
    
    if lesion_channels.sum() != 0:
        for i,c in enumerate(class_names,0):
            if lesion_channels[:,i].sum() > 0:
                raise ValueError(f'When using the argument no_mask, we expect lesion labels to be zero, but their sum was: {lesion_channels.sum()}, class with lesion: {c}, sum: {lesion_channels[:,i].sum()}')

def split_outputs_malignancy(model_output,classes):
    #raise ValueError(f'Number of classes no malig benign: {len(classes)}')
    out = model_output['segmentation']
    new={}
    for key in model_output.keys():
        if key!='segmentation':
            new[key]=model_output[key]
    if isinstance(out, (list, tuple)):
        lesion = []
        malig_benign = []
        for o in out:
            lesion.append(o[:,:len(classes)])
            malig_benign.append(o[:,len(classes):])
        new['segmentation'] = lesion
    else:
        malig_benign = out[:,len(classes):]
        new['segmentation'] = out[:,:len(classes)]
    return new, malig_benign

def split_cls_outputs_malignancy(model_output,classes):
    #construct class list:
    assert not any(['malig' in c or 'benign' in c for c in classes]), f'Classes list should not contain malignancy classes, but got: {classes}'
    assert not any(('pdac' in c) or ('pnet' in c) or ('background' in c) or ('cyst' in c) for c in classes), \
    "Malignancy classification code only prepared to handle lesion/benign/malignant lesion classes."
    lesion_classes = [c for c in sorted(classes) if 'lesion' in c]
    malignants = [c.replace('lesion', 'malignant') for c in lesion_classes]
    benigns = [c.replace('lesion', 'benign') for c in lesion_classes]
    new_classes = lesion_classes + malignants + benigns
    for key in list(model_output.keys()):
        if 'classif' in key:
            classification_out = model_output[key]
            if not isinstance(classification_out, (list, tuple)):
                classif = classification_out
                assert classif.shape[1] == len(new_classes), f'Classification output shape {classif.shape} does not match expected number of classes {len(new_classes)}'
                model_output[key+'_malig_benign_cls'] = classif[:, len(lesion_classes):]
                model_output[key] = classif[:, :len(lesion_classes)]
            else:
                tmp_lesion=[]
                tmp_benign_malig=[]
                for  classif in classification_out:
                    assert classif.shape[1] == len(new_classes), f'Classification output shape {classif.shape} does not match expected number of classes {len(new_classes)}'
                    tmp_benign_malig.append(classif[:, len(lesion_classes):])
                    tmp_lesion.append(classif[:, :len(lesion_classes)])
                model_output[key+'_malig_benign_cls'] = tmp_benign_malig
                model_output[key] = tmp_lesion
                
    #print(f'malig_benign_cls keys added to model output? {[k for k in model_output.keys() if "malig_benign_cls" in k]}')
    return model_output

counter_malig = 0


def _is_known_malignancy_value(m) -> bool:
    """
    True iff m represents a known malignancy label (0 or 1).
    Works with float/int/str. (tensor_to_pairs typically gives float or None.)
    """
    if m is None:
        return False
    # common fast path (float/int)
    if m == 0 or m == 1 or m == 0.0 or m == 1.0:
        return True
    # str path (just in case)
    if isinstance(m, str):
        return m in {"0", "1", "0.0", "1.0"}
    return False


def has_any_malignancy_label(sizes_malignancy_ct) -> bool:
    """
    True iff there is at least one *known* malignancy label (0 or 1)
    inside sizes_malignancy_ct (list of [diameter, malig]).
    """
    if not sizes_malignancy_ct:
        return False
    for _, malig in sizes_malignancy_ct:
        if _is_known_malignancy_value(malig):
            return True
    return False


def will_skip_ball_malignancy_loss_for_all_batch_items(
    tumor_volumes,
    sizes_malignancy,
    tolerance_mm: float = 3.0,
    return_debug: bool = False,
):
    """
    Returns True iff, for every batch item B, your ball_loss loop would end up with
    allow_malignancy_loss_ct = False (i.e., the malignant/benign ball-loss branch
    will be skipped for all items).

    This mirrors your ball_loss logic exactly:
      1) conflict within tolerance: mi != mj
      2) unknown size: any tumor_volume < 0
      3) no known malignancy labels at all (no 0/1)
    """
    B = tumor_volumes.shape[0] if hasattr(tumor_volumes, "shape") else len(tumor_volumes)

    per_item_allow = [] if return_debug else None

    for b in range(B):
        allow = True

        # sizes_malignancy_ct uses your exact converter (filters d<=0, non-finite; m: NaN->None)
        sizes_malignancy_ct = tensor_to_pairs(sizes_malignancy[b])

        # (1) conflicting labels among same-size tumors (within tolerance)
        # EXACT match: use `!=` directly (so None != 0.0 triggers conflict, etc.)
        for i in range(len(sizes_malignancy_ct)):
            di, mi = sizes_malignancy_ct[i]
            for j in range(i + 1, len(sizes_malignancy_ct)):
                dj, mj = sizes_malignancy_ct[j]
                if abs(di - dj) <= tolerance_mm and (mi != mj):
                    allow = False
                    break
            if not allow:
                break

        # (2) unknown size: any negative volume in that sample
        if allow:
            tv = tumor_volumes[b]
            unk_size = (tv < 0).any()
            if hasattr(unk_size, "item"):
                unk_size = bool(unk_size.item())
            if unk_size:
                allow = False

        # (3) no known malignancy label at all
        if allow and (not has_any_malignancy_label(sizes_malignancy_ct)):
            allow = False

        if return_debug:
            per_item_allow.append(allow)

        # short-circuit: if any item allows malignancy loss, then it's NOT "skip for all"
        if allow and (not return_debug):
            return False

    all_skip = True if not return_debug else all(not a for a in per_item_allow)
    return (all_skip, per_item_allow) if return_debug else all_skip



def malignant_benign_only_from_sizes_malignancy(sizes_malignancy: torch.Tensor):
    """
    sizes_malignancy: torch.Tensor of shape (B, 10, 2) with rows [diameter, malignancy].
      - diameter == 0 -> padding (ignore)
      - malignancy == 1 -> malignant
      - malignancy == 0 -> benign
      - anything else (NaN, None-like, -1, etc.) -> unknown

    Definitions (per sample):
      malignant_only = has at least one REAL entry, and ALL real entries are malignant (1.0)
      benign_only    = has at least one REAL entry, and ALL real entries are benign (0.0)

    "REAL entry" means diameter != 0
    Returns:
      malignant_only: (B,1,1,1,1) float tensor of 0/1
      benign_only:    (B,1,1,1,1) float tensor of 0/1
    """
    assert sizes_malignancy.dim() == 3 and sizes_malignancy.size(-1) == 2, \
        f"Expected (B,N,2), got {tuple(sizes_malignancy.shape)}"

    B = sizes_malignancy.size(0)
    device = sizes_malignancy.device

    malignant_only = torch.zeros(B, device=device, dtype=torch.float32)
    benign_only    = torch.zeros(B, device=device, dtype=torch.float32)

    # loop for clarity
    for b in range(B):
        has_any_real = False
        all_malignant = True
        all_benign = True

        # iterate lesions
        for i in range(sizes_malignancy.size(1)):
            d = float(sizes_malignancy[b, i, 0].detach().cpu())
            m = float(sizes_malignancy[b, i, 1].detach().cpu())

            # ignore padding / invalid diameter
            if d == 0.0:
                continue
            
            if not math.isfinite(d):
                raise ValueError(f"Diameter value is not finite: {m}; should it not be 0, negative or positive?")

            has_any_real = True

            # unknown malignancy => immediately disqualifies both "only" flags
            if (not math.isfinite(m)) or (m != 0.0 and m != 1.0):
                all_malignant = False
                all_benign = False
                break

            # update purity flags
            if m != 1.0:
                all_malignant = False
            if m != 0.0:
                all_benign = False

            # early exit if already mixed
            if (not all_malignant) and (not all_benign):
                break

        if has_any_real and all_malignant:
            malignant_only[b] = 1.0
        if has_any_real and all_benign:
            benign_only[b] = 1.0
        if has_any_real and all_malignant and all_benign:
            raise ValueError(f"bug: all malignant AND all benign?")

    # reshape to broadcast with (B,1,H,W,D)
    malignant_only = malignant_only.reshape(B, 1, 1, 1, 1)
    benign_only    = benign_only.reshape(B, 1, 1, 1, 1)
    return malignant_only, benign_only

def auto_distill_malignancy_loss(model_output,malig_benign, unk_voxels, classes, label, sizes_malignancy, malignancy_per_voxel,
                                 chosen_segment_mask, sigmoid_already_applied=False, class_weights=None,
                                 triangle_consistency=False, input_tensor=None, names=None,
                                 skip_cls = False, 
                                 include_ball_loss=False,
                                 tumor_volumes = None,
                                 tumor_diameters =None,
                                 sizes_slices = None,
                                 ct_z_spacing_original = None,
                                 slices_mask = None,
                                 max_slice = None,
                                 subseg_dilation = 31
                                 ):
    """
    We learn the malignant and benign classes by learning from the lesion class. 
    We use the labels sizes_malignancy and malignancy_per_voxel to understand if lesions are malignant or not.
    
    If no lesion present (per-voxel and label is zero): make malignant and benign zero
    If annotated per voxel: match benign or malignant to label
    If annotated by report: match benign or malignant to lesion output
    
    We create a new label, and use it with usual segmentation losses to malig_benign
    
    Where there is a lesion but you do not know if it is benign or malignant, do not penalize. Both for cases annotated by report or per voxel.
    """
    global counter_malig
    assert len(malignancy_per_voxel.shape)==2, f'Malignancy per voxel shape {malignancy_per_voxel.shape} must be 2D (B,C)'
    assert malignancy_per_voxel.shape[0]==label.shape[0], f'Malignancy per voxel batch size {malignancy_per_voxel.shape[0]} does not match label batch size {label.shape[0]}'
    assert malignancy_per_voxel.shape[1]==len(classes), f'Malignancy per voxel C {malignancy_per_voxel.shape[1]} does not match number of lesion classes {len(classes)}'
    
    
    
    
    if class_weights is not None and torch.equal(class_weights, torch.ones_like(class_weights)):
        class_weights = None
    
    out = model_output['segmentation']
    if isinstance(out, (list, tuple)):
        losses = []
        for o,mb in zip(out, malig_benign):
            tmp = {}
            for k, v in model_output.items():
                tmp[k] = v      # keep as-is (single tensor or non-scale list)
            tmp['segmentation'] = o
            l = auto_distill_malignancy_loss(model_output=tmp,malig_benign=mb,
                                            unk_voxels=unk_voxels, classes=classes, label=label,
                                            sizes_malignancy=sizes_malignancy, malignancy_per_voxel=malignancy_per_voxel,
                                            chosen_segment_mask=chosen_segment_mask,
                                            sigmoid_already_applied=sigmoid_already_applied,
                                            class_weights=class_weights,
                                            input_tensor=input_tensor,
                                            names=names, triangle_consistency=triangle_consistency,
                                            skip_cls = skip_cls, 
                                            include_ball_loss=include_ball_loss,
                                            tumor_volumes = tumor_volumes,
                                            tumor_diameters = tumor_diameters,
                                            sizes_slices = sizes_slices,
                                            ct_z_spacing_original = ct_z_spacing_original,
                                            slices_mask = slices_mask,
                                            max_slice = max_slice,
                                            )
            losses.append(l)
        #loss = torch.stack(losses, dim=0).mean()
        #list of dicts to dict
        loss = {}
        for k in losses[0]:
            loss[k] = torch.stack([losses[i][k] for i in range(len(losses))], dim=0).mean(0)
        return loss
    
    
    lesion_classes = [c for c in sorted(classes) if 'lesion' in c]
    malignants = [c.replace('lesion', 'malignant') for c in lesion_classes]
    assert malig_benign.shape[1] == 2 * len(lesion_classes)
    
    
    
    #send all to the same device
    device = out.device
    malig_benign = malig_benign.to(device)
    unk_voxels = unk_voxels.to(device)
    label = label.to(device)
    sizes_malignancy = sizes_malignancy.to(device)
    malignancy_per_voxel = malignancy_per_voxel.to(device)
    chosen_segment_mask = chosen_segment_mask.to(device)
    
    
    if include_ball_loss:
        skipped_ball_loss = will_skip_ball_malignancy_loss_for_all_batch_items(tumor_volumes=tumor_volumes,
                                                              sizes_malignancy=sizes_malignancy)
        if not skipped_ball_loss:
            malign_out = malig_benign[:,:len(malignants)]
            benign_out = malig_benign[:,len(malignants):]
            
            ball_loss_malignancy = ball_loss (
                out=out.detach(), labels=label, unk_voxels=unk_voxels, chosen_segment_mask=chosen_segment_mask, 
                tumor_volumes=tumor_volumes, tumor_diameters=tumor_diameters, classes=classes, 
                apply_dice_loss=True,
                sigmoid=(not sigmoid_already_applied), class_weights=class_weights,
                sizes_slices = sizes_slices,
                ct_z_spacing_original = ct_z_spacing_original,
                slices_mask = slices_mask,
                max_slice = max_slice,
                malignant_benign_loss=True,
                benign_out=benign_out,
                malignant_out=malign_out,
                sizes_malignancy=sizes_malignancy,
                subseg_dilation = subseg_dilation
                )
            
            #keys: 'ball_loss_benign_bce','ball_loss_malignant_bce','ball_loss_benign_dice','ball_loss_malignant_dice','malignancy_loss_applied'
            ball_applied = ball_loss_malignancy['malignancy_loss_applied']
            keep = ['ball_loss_benign_bce','ball_loss_malignant_bce',
                    'ball_loss_benign_dice','ball_loss_malignant_dice']
            ball_loss_malignancy = {k: ball_loss_malignancy[k] for k in keep}
    
    lesion_out, lesion_classes_2 = get_lesion_channels(out, classes, return_class_names=True)
    unk_voxels = get_lesion_channels(unk_voxels, classes)
    label = get_lesion_channels(label, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes)
    malignancy_per_voxel = get_lesion_channels(malignancy_per_voxel.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), classes)
    
    if class_weights is not None:
        #raise ValueError(f'Shape of class weights: {class_weights.shape}, avg value: {class_weights.mean()}')
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert len(class_weights.shape) == 5, f'Class weights should be 5D tensor, got {class_weights.shape}'
        #repeat channels to match the output shape
        class_weights = class_weights.repeat(out.shape[0], 1, out.shape[2], out.shape[3], out.shape[4])
        class_weights = get_lesion_channels(class_weights, classes)
    
    
    assert len(lesion_classes)==len(lesion_classes_2), f'Lesion classes length do not match: {lesion_classes} vs {lesion_classes_2}'
    assert all([lesion_classes[i].replace('pancreas','pancreatic')==lesion_classes_2[i].replace('pancreas','pancreatic') for i in range(len(lesion_classes))]), f'Lesion classes do not match: {lesion_classes} vs {lesion_classes_2}'
    
    malign_out = malig_benign[:,:len(malignants)]
    benign_out = malig_benign[:,len(malignants):]
    
    
    
    lesion_label = torch.zeros_like(lesion_out)
    
    #add per-voxel lesions
    lesion_label = lesion_label + label
    
    assert lesion_label.max() <= 1.0, f'Lesion label should be binary, but got max {lesion_label.max()}'
    
    #add report annotated lesions
    if not sigmoid_already_applied:
        lesion_out = torch.sigmoid(lesion_out).detach()
    else:
        lesion_out = lesion_out.detach()
        
    report_label = lesion_out * unk_voxels
    lesion_label = lesion_label + report_label
    
    
    #sanity check for values outside the 0-1 range
    tmp = (label.float() + report_label)  # this is what your sum creates
    mx = tmp.max().item()
    mn = tmp.min().item()

    if not (mn >= -1e-6 and mx <= 1.0 + 1e-6):
        # where exactly does it exceed 1?
        over = (tmp > 1.0 + 1e-6)
        # voxel-wise overlap fraction per batch (any channel)
        over_frac = over.any(dim=1).float().mean().item()
        # how big is the worst violation beyond 1?
        worst = (tmp - 1.0).max().item()

        raise AssertionError(
            "BUG: lesion supervision sum produced values outside [0,1]. "
            "This makes BCE targets invalid.\n"
            f"{(label, 'label')}\n"
            f"{(unk_voxels, 'unk_voxels')}\n"
            f"{(report_label, 'report_label')}\n"
            f"{(tmp, 'label_plus_report')}\n"
            f"over1_frac(any channel)={over_frac:.6g}, worst_excess={worst:.6g}"
        )
    
    #ensure correct range
    assert lesion_label.min() >= 0 and lesion_label.max() <= 1, f'Lesion label out os 0-1 range: {lesion_label.min()}, {lesion_label.max()}'
    #ensure unk voxels and labels are disjoint
    assert (unk_voxels+label).max() <= 1.0, f'Unknown voxels and label should be disjoint, but got max {unk_voxels+lesion_label}'
    
    #split lesions into benign, malignant and unknown malignancy
    malignant_label = torch.zeros_like(malign_out)
    benign_label = torch.zeros_like(benign_out)
    
    #1- per-voxel: use malignancy_per_voxel to separate benigns, malignants and unknown
    malignant_per_voxel = (malignancy_per_voxel.float()==1.0).float()
    benign_per_voxel = (malignancy_per_voxel.float()==0.0).float()
    assert len(malignant_per_voxel.shape)==len(lesion_label.shape), f'Malignant per voxel shape {malignant_per_voxel.shape} does not match lesion label shape {lesion_label.shape}'
    assert malignant_per_voxel.shape[0]==lesion_label.shape[0], f'Malignant per voxel batch size {malignant_per_voxel.shape[0]} does not match lesion label batch size {lesion_label.shape[0]}'
    assert malignant_per_voxel.shape[1]==lesion_label.shape[1], f'Malignant per voxel C {malignant_per_voxel.shape[1]} does not match lesion label C {lesion_label.shape[1]}'
    malignant_label_per_voxel = lesion_label * malignant_per_voxel
    benign_label_per_voxel = lesion_label * benign_per_voxel
    malignant_label = malignant_label + malignant_label_per_voxel
    benign_label = benign_label + benign_label_per_voxel
    
    #2- report annotated lesions: use sizes_malignancy to separate benigns, malignants and unknown
    #For now, we just consider images that, for each organ, have all benign or all malignant tumors. We ignore cases with both. 
    #The ball loss above takes care of mixed cases.
    #sizes_malignancy: B, 10, 2 (diameter, malignancy)
    malignant_only, benign_only = malignant_benign_only_from_sizes_malignancy(sizes_malignancy)
    
    assert len(malignant_only.shape)==len(chosen_segment_mask.shape), f'Malignant only shape {malignant_only.shape} does not match chosen_segment_mask shape {chosen_segment_mask.shape}'
    assert malignant_only.shape[0]==chosen_segment_mask.shape[0], f'Malignant only batch size {malignant_only.shape[0]} does not match chosen_segment_mask batch size {chosen_segment_mask.shape[0]}'
    
    tumor_classes_report_cropped = (chosen_segment_mask.sum(dim=(-1,-2,-3),keepdim=True) > 0).float()  # B,C, it is 1 in the tumor channel
    malignant_only = malignant_only*tumor_classes_report_cropped
    benign_only = benign_only*tumor_classes_report_cropped
    
    assert len(malignant_only.shape)==len(lesion_label.shape), f'malignant_only per voxel shape {malignant_only.shape} does not match lesion label shape {lesion_label.shape}'
    assert malignant_only.shape[0]==lesion_label.shape[0], f'malignant_only per voxel batch size {malignant_only.shape[0]} does not match lesion label batch size {lesion_label.shape[0]}'
    assert malignant_only.shape[1]==lesion_label.shape[1], f'malignant_only per voxel C {malignant_only.shape[1]} does not match lesion label C {lesion_label.shape[1]}'
    
    malignant_label_report = lesion_label * malignant_only
    benign_label_report = lesion_label * benign_only
    
    malignant_label = malignant_label + malignant_label_report
    benign_label = benign_label + benign_label_report
    
    
    malignant_label = malignant_label.detach()
    benign_label = benign_label.detach()
    
    assert malignant_label.max() <= 1.0, f'No voxel should be annotated both with label and with reports'
    assert benign_label.max() <= 1.0, f'No voxel should be annotated both with label and with reports'
    assert (benign_label + malignant_label).max() <= 1.0, f'No voxel should be annotated both as benign and malignant'
    
    #unknown: lesion_label is not zero, but both benign and malignant are zero
    unknown_malignancy_label = (lesion_label>0).float() - (malignant_label>0).float() - (benign_label>0).float()
    unknown_malignancy_label = unknown_malignancy_label.detach()
    
    
    assert unknown_malignancy_label.max() <= 1.0, f'No voxel should be annotated both as benign and malignant'
    assert unknown_malignancy_label.min() >= 0.0, f'Unknown malignancy label should be >= 0, but got min {unknown_malignancy_label.min()}'
    
    penalize_known_malignancy = 1 - unknown_malignancy_label
    if include_ball_loss and (not skipped_ball_loss):
        #we shall not penalize again the channels already penalized by the ball loss
        assert ball_applied.shape == penalize_known_malignancy.shape, f'Ball applied shape {ball_applied.shape} does not match penalize known malignancy shape {penalize_known_malignancy.shape}'
        #ball loss is not applied to background voxels?? Yes. We can skip the whole channel here.
        penalize_known_malignancy = penalize_known_malignancy * (1.0 - ball_applied)
    
    
    #benign and malignant must be disjoint:
    assert (malignant_label + benign_label).max() <= 1.0, f'No voxel should be annotated both as benign and malignant'   
    
    #loss functions:
    if not sigmoid_already_applied:
        bce_malig = F.binary_cross_entropy_with_logits(malign_out, malignant_label.float(), reduction='none', weight=class_weights)
        bce_benign = F.binary_cross_entropy_with_logits(benign_out, benign_label.float(), reduction='none', weight=class_weights)
    else:
        bce_malig = F.binary_cross_entropy(malign_out, malignant_label.float(), reduction='none', weight=class_weights)
        bce_benign = F.binary_cross_entropy(benign_out, benign_label.float(), reduction='none', weight=class_weights)
        
    bce_malig = bce_malig * penalize_known_malignancy
    bce_benign = bce_benign * penalize_known_malignancy
    assert len(bce_malig.shape) == 5, f'BCE malign shape should be 5D, got {bce_malig.shape}'
    assert len(bce_benign.shape) == 5, f'BCE benign shape should be 5D, got {bce_benign.shape}'
    
    
    #dice
    dice_malig = DiceLossMultiClass(malign_out, malignant_label, penalize_known_malignancy, 
                                    sigmoid=(not sigmoid_already_applied),class_weights=class_weights)
    dice_benign = DiceLossMultiClass(benign_out, benign_label, penalize_known_malignancy,
                                    sigmoid=(not sigmoid_already_applied),class_weights=class_weights)
    
    
    loss_dict = {
        'loss_malig_bce': bce_malig.mean(dim=(-1,-2,-3,-4)).mean(),
        'loss_benign_bce': bce_benign.mean(dim=(-1,-2,-3,-4)).mean(),
        'loss_malig_dice': dice_malig.mean(),
        'loss_benign_dice': dice_benign.mean(),
    }
    
    if triangle_consistency:
        #logic: the sum of benign and malignant should match lesion_out
        #we only use this for where we do not know the malignancy
        if not sigmoid_already_applied:
            malign_out = torch.sigmoid(malign_out)
            benign_out = torch.sigmoid(benign_out)
        benign_or_malignant = malign_out + benign_out
        loss_triangle = torch.abs(benign_or_malignant - lesion_out.detach())
        loss_triangle = (loss_triangle * unknown_malignancy_label).mean()
        loss_dict['loss_triangle_consistency'] = loss_triangle
        
        # --- Sanity dump (similar to ball_loss) ---
    if counter_malig < 10:
        counter_malig += 1
        out_dir = os.path.join('SanityMalignancyLoss', str(counter_malig))
        os.makedirs(out_dir, exist_ok=True)

        B = lesion_out.shape[0]
        for b in range(B):
            # Input CT (if provided)
            if input_tensor is not None:
                save_tensor_as_nifti(input_tensor[b].squeeze(), os.path.join(out_dir, f'input_volume_B{b}'))

            # Targets/labels produced inside this loss
            # Aggregate across lesion channels to get a 3D map
            #mal_3d   = (malignant_label[b].sum(0) > 0).float()
            #ben_3d   = (benign_label[b].sum(0) > 0).float()
            #unk_3d   = (unknown_malignancy_label[b].sum(0) > 0).float()

            save_tensor_as_nifti(malignant_label[b].sum(0), os.path.join(out_dir, f'malignant_label_B{b}'))
            save_tensor_as_nifti(benign_label[b].sum(0), os.path.join(out_dir, f'benign_label_B{b}'))
            save_tensor_as_nifti(unknown_malignancy_label[b].sum(0), os.path.join(out_dir, f'unknown_malignancy_B{b}'))
            
            #save also the model lesion output, the model malignant and benign outputs
            if not sigmoid_already_applied:
                save_tensor_as_nifti(torch.sigmoid(malign_out[b]).sum(0), os.path.join(out_dir, f'malignant_output_B{b}'))
                save_tensor_as_nifti(torch.sigmoid(benign_out[b]).sum(0), os.path.join(out_dir, f'benign_output_B{b}'))
            else:
                save_tensor_as_nifti(malign_out[b].sum(0), os.path.join(out_dir, f'malignant_output_B{b}'))
                save_tensor_as_nifti(benign_out[b].sum(0), os.path.join(out_dir, f'benign_output_B{b}'))
            save_tensor_as_nifti(lesion_out[b].sum(0), os.path.join(out_dir, f'lesion_output_B{b}'))
            
            #save also the loss bce_malig and bce_benign
            save_tensor_as_nifti(bce_malig[b].sum(0), os.path.join(out_dir, f'bce_malignant_B{b}'))
            save_tensor_as_nifti(bce_benign[b].sum(0), os.path.join(out_dir, f'bce_benign_B{b}'))

            # Metadata: sample name, sizes_malignancy, malignancy_per_voxel (lesion channels only)
            meta = {}
            meta['name'] = (names[b] if names is not None and isinstance(names, (list, tuple)) and b < len(names) else None)
            meta['sizes_malignancy'] = sizes_malignancy[b].detach().cpu().tolist()


            # malignancy_per_voxel was expanded and mapped to lesion channels earlier:
            # shape [B, C_les, H, W, D] or [B, C_les, 1,1,1]; take the scalar per channel.
            if malignancy_per_voxel.dim() == 5:
                mpv_vec = malignancy_per_voxel[b, :, 0, 0, 0]
            elif malignancy_per_voxel.dim() == 2:
                # if caller provided [B, C_les]
                mpv_vec = malignancy_per_voxel[b]
            else:
                # last resort: flatten per channel by spatial mean
                mpv_vec = malignancy_per_voxel[b].view(malignancy_per_voxel.shape[1], -1).mean(-1)
            meta['malignancy_per_voxel'] = mpv_vec.detach().cpu().tolist()


            # Store lesion class names used in this head for reference
            meta['lesion_classes'] = list(lesion_classes)
            
            
            meta["sigmoid(lesion_out) max:"] = lesion_out[b].max().item()
            meta["report_label max:"] =  report_label[b].max().item()
            meta["lesion_label max:"] =  lesion_label[b].max().item()
            meta["benign_label max:"] =  benign_label[b].max().item()
            meta["malignant_label max:"] =  malignant_label[b].max().item()
            
            #max and mininum of unknown malignancy label
            meta["unknown_malignancy_label max:"] =  unknown_malignancy_label[b].max().item()
            meta["unknown_malignancy_label min:"] =  unknown_malignancy_label[b].min().item()
            
            #losses
            meta['Dice loss malignant:'] = dice_malig.mean().item()
            meta['Dice loss benign:'] = dice_benign.mean().item()
            meta['BCE loss malignant:'] = bce_malig.mean().item()
            meta['BCE loss benign:'] = bce_benign.mean().item()
            
            #print the maximum and minimum of the benign and malignant outputs after sigmoid'
            if not sigmoid_already_applied:
                mal_out_sigmoid = torch.sigmoid(malign_out[b])
                ben_out_sigmoid = torch.sigmoid(benign_out[b])
            else:
                mal_out_sigmoid = malign_out[b]
                ben_out_sigmoid = benign_out[b]
            meta["mal_out_sigmoid max:"] = mal_out_sigmoid.max().item()
            meta['ben_output_sigmoid max:'] = ben_out_sigmoid.max().item()
            
            

            # Write YAML
            with open(os.path.join(out_dir, f'meta_B{b}.yaml'), 'w') as fh:
                yaml.dump(meta, fh)
                
    if not skip_cls:
        #classification labels:
        cls_losses={}
        #reduce malignant labels, benign labels and penalize_known_malignancy in the spatial dimensions
        malignant_label_cls = (malignant_label.sum(dim=(-1,-2,-3))>0).float()
        benign_label_cls = (benign_label.sum(dim=(-1,-2,-3))>0).float()
        unknown_malignancy_cls = (unknown_malignancy_label.sum(dim=(-1,-2,-3))>0).float()
        penalize = 1 - unknown_malignancy_cls # If I am sure that ANY lesion in this channel is benign or malignant, I penalize the classification loss. These losses are BCE, not softmax
        for key in model_output:
            if 'malig_benign_cls' in key:
                #classification outputs for benign and malignant classes
                classif_out = model_output[key]
                if not isinstance(classif_out, (list, tuple)):
                    classif_out = [classif_out]
                cls_loss = []
                for out in classif_out:
                    #split outputs in half, first half is malignant, second half is benign
                    mal_out_cls = out[:,:len(malignants)]
                    ben_out_cls = out[:,len(malignants):]
                    assert mal_out_cls.shape[1] == len(malignants)
                    l_m = F.binary_cross_entropy_with_logits(mal_out_cls, malignant_label_cls, reduction='none')
                    mask = (penalize > 0.5) | (malignant_label_cls > 0.5)
                    l_m = (l_m * mask).sum() / mask.sum().clamp_min(1.0) # if penalize is 1, I penalize always, if penalize is 0 and label is 1, I penalize too---If I am sure that ANY lesion in this channel is benign or malignant, I penalize the classification loss. These losses are BCE, not softmax
                    l_b = F.binary_cross_entropy_with_logits(ben_out_cls, benign_label_cls, reduction='none')
                    mask = (penalize > 0.5) | (benign_label_cls > 0.5)
                    l_b = (l_b * mask).sum() / mask.sum().clamp_min(1.0)
                    l = (l_m + l_b)/2
                    cls_loss.append(l.mean())
                cls_loss = torch.stack(cls_loss, dim=0).mean(0)
                cls_losses[key] = cls_loss
                #print(f'Calculated malignancy classification loss for key {key}: {cls_loss.item()}')
            
        loss_dict.update(cls_losses)
        #print(f'Malignancy loss dict: {loss_dict.keys()}')
        
    if include_ball_loss:
        if skipped_ball_loss:
            #set all ball losses to zero
            ball_loss_malignancy = {
                'ball_loss_benign_bce': torch.tensor(0.0, device=out.device),
                'ball_loss_malignant_bce': torch.tensor(0.0, device=out.device),
                'ball_loss_benign_dice': torch.tensor(0.0, device=out.device),
                'ball_loss_malignant_dice': torch.tensor(0.0, device=out.device),
            }
        loss_dict.update(ball_loss_malignancy)
    
    #return loss
    return loss_dict
    
def filter_by_contrast(x, contrast, args):
    """
    x: Tensor of shape (B, ...)
    contrast: iterable of length B (strings like 'venous', 'arterial', etc.)
    """
    if not getattr(args, 'attenuation_classifier_venous', False) or contrast is None:
        return x

    B = x.shape[0]
    if len(contrast) != B:
        raise ValueError(f"len(contrast)={len(contrast)} but batch size={B}")

    mask = torch.tensor(
        [str(c).lower() == 'venous' for c in contrast],
        device=x.device, dtype=torch.bool
    )

    out = x[mask]          # shape: (N, ...)
    # N can be 0, 1, or more; dims are preserved for N=1
    # If you prefer to error when N==0, raise instead of returning an empty batch:
    if out.shape[0] == 0: 
        return None
    return out              # (0, ...) is valid and won’t crash downstream if handled


def calculate_loss(model_output, label, unk_voxels, args, matcher,chosen_segment_mask,tumor_volumes_report,tumor_diameters,
                   classes,loss_wrapper=None,input_tensor=None, class_weights=None, model_genesis=False,
                   clip_only=False,report_embeddings=None, dist=None,
                   tumor_attenuation_label=None, attenuation_classifier='none', lesion_classes=None,
                   tumor_volumes_in_crop_per_voxel=None,tumor_diameters_per_voxel=None,names=None,
                   slices_cropped_dict=None, sizes_slices=None, sample_weights=False,
                   dynamic_sample_weights=False, no_mask=False,
                   sizes_malignancy=None, malignancy_per_voxel=None,
                   contrast=None, subseg_dilation = 31,
                   debug_timers = False, return_refined_mask=False,
                   chosen_segment_mask_contricted = None):
    """
    slices_cropped_dict: used for the slice loss, where we cut the organ with tumor only around the tumor slice in our ball loss and volume loss.
    sizes_slices: a torch tensor of size B, 10, 2, where the last dims has the pairs of tumor dimaters and slices in the tareget organ
    dynamic_sample_weights: the idea is to give the same importance to cases annotated per voxel, cases annotated with reports with slice (few), cases annotated with report with size, cases annotated with report without size, and cases annotated with reports without tumor. 
    This will increase the power of the report with slice information (precise).
    """
    global counter2
    
    if debug_timers:
        start_time = time.time()
    
    if no_mask:
        print('No mask')
        sanity_assert_no_lesion_mask(label, classes)
    
    #print('Unk voxels:', unk_voxels)
    if  dynamic_sample_weights:
        global balancer_samples
        sample_weights = balancer_samples.define_groups(
                                            labels=label,
                                            chosen_segment_mask=chosen_segment_mask,
                                            sizes_slices=sizes_slices,      # or None
                                            tumor_volumes=tumor_volumes_report,    # [B, 10]
                                            classes=classes,
                                        )
        #print(f'Using dynamic sample weights: {sample_weights}', flush=True, file=sys.stderr)
    else:
        sample_weights = None
    
    
    if slices_cropped_dict is not None:
        ct_z_spacing_original=slices_cropped_dict["z_spacing"]
        slices_mask = slices_cropped_dict['slices_mask'] 
        max_slice = slices_cropped_dict['max_slice'] 
        assert slices_mask is not None, f'Slices mask should not be None, but got {slices_mask}'
        assert ct_z_spacing_original is not None, f'CT z spacing should not be None, but got {ct_z_spacing_original}'
        #print the shapes
        #print(f'Slices mask shape: {slices_mask.shape}, max slice: {max_slice.shape}, ct z spacing: {ct_z_spacing_original.shape}, slices_sizes {sizes_slices.shape}', flush=True, file=sys.stderr)
        #Slices mask shape: torch.Size([3, 1, 128, 128, 128]), max slice: torch.Size([3]), ct z spacing: torch.Size([3])
        assert len(slices_mask.shape) == 5, f'Slices mask should be 5D, but got {slices_mask.shape}'
        assert slices_mask.shape[0] == label.shape[0], f'Slices mask batch size {slices_mask.shape[0]} does not match label batch size {label.shape[0]}'
        
        #print(f':D:D:D:D:D:D::D:D:D::D We received a case with slices!', flush=True, file=sys.stderr)
    else:
        ct_z_spacing_original=None
        slices_mask=None
        sizes_slices=None #no slices used, so we do not need to cut the known voxels with slices
        max_slice = None
        #if unk_voxels.sum() > 0:
        #    raise ValueError('Debugging only: we received a case with unknown voxels (report anno?), but no slices cropped dict (and we aare debugging with slice!). This should not happen!')
       #print(f':( Case without slice', flush=True, file=sys.stderr)
    
    
    if hasattr(args,'malignancy_classification') and args.malignancy_classification:
        if (args.cls_gate):
            raise ValueError('Not implemented cls_gate with malignancy_classification, needs to ajust sigmoid_already_applied below and do it in 2 steps')
        model_output, malig_benign = split_outputs_malignancy(model_output,classes)
        model_output = split_cls_outputs_malignancy(model_output,classes)
        malignancy_seg_loss = auto_distill_malignancy_loss(model_output=model_output,malig_benign=malig_benign,
                                                            unk_voxels=unk_voxels, classes=classes, label=label,
                                                            sizes_malignancy=sizes_malignancy, malignancy_per_voxel=malignancy_per_voxel,
                                                            chosen_segment_mask=chosen_segment_mask_contricted if chosen_segment_mask_contricted is not None else chosen_segment_mask,
                                                            sigmoid_already_applied=False,
                                                            class_weights=class_weights,
                                                            input_tensor=input_tensor,
                                                            names=names, triangle_consistency=args.triangle_consistency,
                                                            include_ball_loss=args.include_ball_loss_malignancy,
                                                            tumor_volumes = tumor_volumes_report,
                                                            tumor_diameters = tumor_diameters,
                                                            sizes_slices = sizes_slices,
                                                            ct_z_spacing_original = ct_z_spacing_original,
                                                            slices_mask = slices_mask,
                                                            max_slice = max_slice,
                                                            subseg_dilation = subseg_dilation)
    else:
        malignancy_seg_loss = None
    
    
    if model_genesis:
        return model_genesis_loss(model_output['segmentation'],label)
    
    if clip_only:
        seg_result = model_output['segmentation']
        result_embedding = model_output['clip']
        if isinstance(seg_result, tuple) or isinstance(seg_result, list):
            tmp = 0
            for i in range(len(seg_result)):
                tmp = tmp + seg_result[i].sum()*0
        result_embedding = result_embedding+tmp*0#this is to avoid unused parameter error
        #assert same shape (all dimensions)
        assert result_embedding.shape == report_embeddings.shape, f'Result embedding shape is: {result_embedding.shape}, report embedding shape is: {report_embeddings.shape}'
        loss_ct2rep = nce.info_nce(result_embedding, report_embeddings)
        loss_rep2ct = nce.info_nce(report_embeddings, result_embedding)
        sym_loss = 0.5*(loss_ct2rep + loss_rep2ct)
        sym_loss = sym_loss*dist.get_world_size() #this compensated for the all_gather
        return {'contrastive_loss': sym_loss,
                'overall': sym_loss}
    
    if args.epai_stage_2 and (class_weights is not None):
        raise ValueError(
            "Per-sample/voxel class-weight matrices are not supported when "
            "epai_stage_2=True (single-label CrossEntropy). "
            "The standard cross-entropy loss with softmax already favors the positive class"
        )
         
    result = model_output['segmentation']
    if args.tumor_classifier:
        tumor_diameters_out = model_output['tumor diameters']
    if attenuation_classifier != 'none':
            tumor_attenuation_out = model_output['attenuation']
            
    y_class=None
    y_class_2=None 
    if args.cls_on_segmentation:
        if args.cls_on_output or args.epai_stage_2 or args.classification_branch:
            raise ValueError('Cannot use cls_on_segmentation with cls_on_output or epai_stage_2 or classification_branch')
        cls_out = model_output['classification on segmentation']
        if isinstance(cls_out, (list, tuple)):
            y_class = cls_out[0]
            y_class_2 = cls_out[1]
        else:
            y_class = cls_out
    else:
        if args.epai_stage_2 or args.cls_on_output:
            y_class_2 = model_output['classification on output']
        if args.classification_branch:
            y_class = model_output['classification']
            print('Classification shape:', y_class.shape)
            
    #raise ValueError(f"classification on output: {model_output['classification on output']}\n classification: {model_output['classification']}")
        
    att_loss=None
    assert attenuation_classifier in ['none', 'simple', 'MLP','large','neuron']
    if contrast is None and attenuation_classifier!='none':
        raise ValueError('Contrast cannot be None, must be a list of contrast phases per sample in the batch')
         
    if attenuation_classifier=='simple':
        if isinstance(tumor_attenuation_out,list):#aux loss
            att_loss = 0
            for att in tumor_attenuation_out:
                att_loss += l1_simple_attenuation_classifier(att, tumor_attenuation_label, chosen_segment_mask, classes, contrast=contrast, args=args) + 0*att.sum()
        else:
            att_loss = l1_simple_attenuation_classifier(tumor_attenuation_out, tumor_attenuation_label, chosen_segment_mask, classes, contrast=contrast, args=args) + 0*tumor_attenuation_out.sum()
    if attenuation_classifier in ['MLP','large','neuron']:
        if isinstance(tumor_attenuation_out,list):#aux loss
            att_loss = 0
            for att in tumor_attenuation_out:
                att_loss += ce_MLP_attenuation_loss(att, tumor_attenuation_label, chosen_segment_mask, classes, labels=label, ct=input_tensor, lesion_classes_allowed=lesion_classes, contrast=contrast, args=args) + 0*att.sum()
        else:
            att_loss = ce_MLP_attenuation_loss(tumor_attenuation_out, tumor_attenuation_label, chosen_segment_mask, classes, labels=label, ct=input_tensor, lesion_classes_allowed=lesion_classes, contrast=contrast, args=args) + 0*tumor_attenuation_out.sum()
    #print('Attenuation loss:', att_loss)
    
    if unk_voxels is not None:
        known_voxels = get_known_voxels(label,unk_voxels,classes=classes)#this will remove (substitute by 0) any channels we are unsure if about the label
        if slices_cropped_dict is not None:
            #here, we make the organ parts outside of the tumor slices KNOWN.
            try:
                known_voxels_sliced = cut_known_voxels_with_slice(known_voxels,slices_cropped_dict,chosen_segment_mask)
            except:
                print('Names are:',names)
                known_voxels_sliced = cut_known_voxels_with_slice(known_voxels,slices_cropped_dict,chosen_segment_mask)
            assert torch.equal((known_voxels_sliced*label).float().sum(),label.float().sum()), f'The unknown region should not cover channels where our label is different from 0---knwon. we got{(known_voxels*label).float().sum()} and {label.float().sum()}'
        else:
            known_voxels_sliced = known_voxels.clone()
        
        assert torch.equal((known_voxels*label).float().sum(),label.float().sum()), f'The unknown region should not cover channels where our label is different from 0---knwon. we got{(known_voxels*label).float().sum()} and {label.float().sum()}'
        known_voxels_original = known_voxels.clone()
        #print('Assertion successful')
    else:
        known_voxels=torch.ones(label.shape).type_as(label)
        known_voxels_sliced = known_voxels.clone()
    
    tumor_cls_loss = None
    
    if chosen_segment_mask is not None and chosen_segment_mask.sum()>0:
        for b in range(chosen_segment_mask.shape[0]):
            if unk_voxels[b].sum()==0 and chosen_segment_mask[b].sum()>0:
                raise ValueError('unk_voxels should not be all zeros if chosen_segment_mask is not all zeros')
            if tumor_volumes_report[b].sum() == 0 and chosen_segment_mask[b].sum()>0:
                raise ValueError('tumor_volumes_report should not be all zeros if chosen_segment_mask is not all zeros')
    
    #raise ValueError(f'Number of classes in classes: {len(classes)}. Number of classes in label: {label.shape[1]}. Number of classes in result: {result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1]}')
    assert len(classes) == label.shape[1], f'Number of classes in classes: {len(classes)} does not match the number of channels in label: {label.shape[1]}'
    assert len(classes) == (result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1]), \
    f'Number of classes in result: {(result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1])} does not match the number of channels in label: {label.shape[1]}'
    
    if class_weights is not None:
        if torch.equal(class_weights, torch.ones_like(class_weights)):
            class_weights = None
        
    if class_weights is not None:
        class_weights = class_weights.to(label.device) #make sure class weights are the same size as cls_out
        assert  class_weights.shape[0] == label.shape[0], f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        assert class_weights.shape[1] == label.shape[1], f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        assert len(class_weights.shape) == 2, f'Class weights should be 2D, but got {class_weights.shape}'
        
        
        
    cls_loss = None
    if (y_class is not None):
        cls_loss = classification_loss(y_class, label, unk_voxels, args, chosen_segment_mask, classes, class_weights)
    if (y_class_2 is not None):
        cls_loss_2 = classification_loss(y_class_2, label, unk_voxels, args, chosen_segment_mask, classes, class_weights)
        if (y_class is not None):
            cls_loss = cls_loss + cls_loss_2
        else:
            cls_loss = cls_loss_2
        #print('Average out:', result[-1].float().mean(dim=(-1,-2,-3)))
        #print('Average label:', label.float().mean(dim=(-1,-2,-3)))
        

    loss = 0
    loss_report = 0
    loss_segmentation = 0
    if class_weights is not None:
        class_weights = class_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        assert len(class_weights.shape)==len(label.shape), f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        
    if isinstance(result, tuple) or isinstance(result, list):
        # if use deep supervision, add all loss together---outpput: [final output, hidden output]
        for j in range(len(result)):
            r,l = result[j],label
            if j==0 and args.multi_ch_tumor:
                #hungarian algorithm---run on the final output, use same indices on hidden layer outputs
                out_ids, label_ids = matcher(r,l)
            if args.multi_ch_tumor:
                #shuffle accorind to hungarian algo output
                r=r[out_ids]
                l=l[label_ids]
                known_voxels = known_voxels_original[label_ids]
                unk_voxels = unk_voxels[label_ids]  # Inside the multi_ch_tumor block
                chosen_segment_mask = chosen_segment_mask[label_ids]
                assert r.shape == known_voxels.shape, f'Label mismatch, known voxels is: {known_voxels.shape}, r is: {r.shape}'
                assert torch.equal(known_voxels[out_ids],known_voxels[label_ids]),'Known voxels should be the same accross the label channels, which are the ones the hungarian algo is shifting around'
            
            #assert no nan in output
            assert not torch.isnan(r).any(), 'Output is nan'
            if args.report_volume_loss_basic > 0:
                if ('ball' in args.loss or 'dynamic' in args.loss or 'dll' in args.loss) and not (j!=0 and 'last' in args.loss):
                    #j!=0 and 'last' in args.loss=>applies the ball loss only to the last layer
                    #print('Using the ball loss')
                    ball_return = ball_loss (out=r, labels=l, unk_voxels=unk_voxels, 
                                             chosen_segment_mask=chosen_segment_mask_contricted if chosen_segment_mask_contricted is not None else chosen_segment_mask, 
                                        tumor_volumes=tumor_volumes_report, tumor_diameters=tumor_diameters, classes=classes, 
                                        apply_dice_loss=('dice' in args.loss), input_tensor=input_tensor,
                                        sigmoid=(not (args.cls_gate and j==0)),
                                        standard_ce=args.stardard_ce_ball, class_weights = class_weights,
                                        single_class= args.epai_stage_2,
                                        diameter_margin=args.ball_volume_margin, volume_margin=args.ball_volume_margin,
                                        sizes_slices=sizes_slices, ct_z_spacing_original=ct_z_spacing_original,
                                        slices_mask=slices_mask, max_slice = max_slice,
                                        sample_weights=sample_weights,
                                        subseg_dilation = subseg_dilation,
                                        return_refined_mask = (return_refined_mask and j==0))#we want the refined mask from the last layer of the teacher only
                    
                    if not return_refined_mask:
                        loss_r = ball_return
                    else:
                        loss_r, refined_mask = ball_return
                        
                    if 'both' in args.loss:
                        loss_r = merge_no_overlap(loss_r,volume_loss_basic(r, chosen_segment_mask_contricted if chosen_segment_mask_contricted is not None else chosen_segment_mask, 
                                                                           tumor_volumes_report, l, unk_voxels, classes, loss_function=args.loss,
                                               sigmoid=(not (args.cls_gate and j==0)), class_weights = class_weights,tolerance=args.volume_loss_tolerance,
                                               slices_cropped_dict=slices_cropped_dict,
                                               dilation_segment = subseg_dilation),
                                               sample_weights=sample_weights)
                        #print('Both')
                else:
                    loss_r = volume_loss_basic(r, chosen_segment_mask_contricted if chosen_segment_mask_contricted is not None else chosen_segment_mask, 
                                               tumor_volumes_report, l, unk_voxels, classes, loss_function=args.loss,
                                               sigmoid=(not (args.cls_gate and j==0)), class_weights = class_weights,tolerance=args.volume_loss_tolerance,
                                               slices_cropped_dict=slices_cropped_dict,
                                               sample_weights=sample_weights,
                                               dilation_segment = subseg_dilation)
                    #print('Using the volume loss')
            else:
                loss_r = torch.tensor(0).type_as(r)

            if (not (args.cls_gate and j==0)):
                #print('Using class weights:', class_weights)
                if args.epai_stage_2:
                    #softmax
                    # voxel‑wise single‑label CE (no per‑sample weights)
                    target_idx = l.argmax(dim=1).long()      # (B,H,W,D)
                    loss_seg   = F.cross_entropy(r, target_idx, reduction='none').unsqueeze(1)
                else:
                    loss_seg = F.binary_cross_entropy_with_logits(r, l.float(), reduction='none', weight=class_weights)
                    #print('Using BCE with logits in seg loss')
            else:
                #softmax already applied
                if args.epai_stage_2:
                    raise ValueError('cls_gate not implemented for epai_stage_2')
                #assert l is in the range 0-1
                assert (l>=0).all() and (l<=1).all(), f'Label is not in the range 0-1, its min is: {l.min()}, its max is: {l.max()}'
                assert (r>=0).all() and (r<=1).all(), f'Output is not in the range 0-1, its min is: {r.min()}, its max is: {r.max()}'
                #print('Using class weights:', class_weights)
                loss_seg = F.binary_cross_entropy(r, l.float(), reduction='none', weight=class_weights)

            if debug_timers:
                print(f'Time until debug save labels: {time.time() - start_time} seconds', flush=True, file=sys.stderr)
                save_time = time.time()
            if not args.epai_stage_2:
                assert loss_seg.shape == known_voxels.shape, f'Loss shape {loss_seg.shape} does not match known voxels shape {known_voxels.shape}'
            else:
                assert len(loss_seg.shape) == len(known_voxels.shape), f'Loss shape {loss_seg.shape} does not match known voxels shape {known_voxels.shape}'
            if False:#counter2<5 and j==0:
                label_names = classes
                if (not (args.cls_gate and j==0)):
                    debug_save_labels(torch.sigmoid(r),str(counter2),out_dir='SanityOutputs',label_names=label_names)
                else:
                    debug_save_labels(r,str(counter2),out_dir='SanityOutputs',label_names=label_names)
                debug_save_labels(l.float(),str(counter2),out_dir='SanityLabelsBeforeLoss',label_names=label_names)
                if args.epai_stage_2:
                    debug_save_labels(loss_seg.repeat(1,4,1,1,1),str(counter2),out_dir='SanityLossBCE',label_names=label_names)
                else:
                    debug_save_labels(loss_seg,str(counter2),out_dir='SanityLossBCE',label_names=label_names)
                    debug_save_labels(loss_seg * known_voxels,str(counter2),out_dir='SanityLossBCEAfterKnownVoxels',label_names=label_names)
                counter2+=1
                
            if debug_timers:
                print(f'Time until after debug save labels: {time.time() - save_time} seconds', flush=True, file=sys.stderr)
                time_after_save = time.time()

            loss_seg = loss_seg * known_voxels_sliced
            loss_seg = loss_seg.mean() + DiceLossMultiClass(r, l, known_voxels_sliced, sigmoid=(not (args.cls_gate and j==0)),class_weights=class_weights)
            loss_segmentation = loss_segmentation + args.aux_weight[j] * args.seg_loss * loss_seg

            if not isinstance(loss_r, dict):
                loss_report = loss_report + args.aux_weight[j] * args.report_volume_loss_basic * loss_r
            else:
                if isinstance(loss_report,int):
                    loss_report = {}
                    for key in loss_r.keys():
                        if key == 'ball_loss_bce':
                            weight = args.ball_bce_weight
                            #print(f'Using the ball bce weight: {weight}')
                        elif key == 'ball_loss_dice':
                            weight = args.ball_dice_weight
                            #print(f'Using the ball dice weight: {weight}')
                        else:
                            weight = 1
                        loss_report[key] = args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
                else:#dict
                    for key in loss_r.keys():
                        if key == 'ball_loss_bce':
                            weight = args.ball_bce_weight
                            #print(f'Using the ball bce weight: {weight}')
                        elif key == 'ball_loss_dice':
                            weight = args.ball_dice_weight
                            #print(f'Using the ball dice weight: {weight}')
                        else:
                            weight = 1
                        if key not in list(loss_report.keys()):
                            loss_report[key] = args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
                        else:
                            loss_report[key] = loss_report[key] + args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
    else:
        #raise ValueError('Result is not a tuple or list, you should be using deep supervision')
        

        if args.multi_ch_tumor:
            out_ids, label_ids = matcher(result,label)
            result=result[out_ids]
            label=label[label_ids]
            assert result.shape == known_voxels.shape
            known_voxels = known_voxels[out_ids]
            unk_voxels = unk_voxels[label_ids]  # Inside the multi_ch_tumor block
            chosen_segment_mask = chosen_segment_mask[label_ids]
            assert torch.equal(known_voxels[out_ids],known_voxels[label_ids]),'Known voxels should be the same accross the label channels, which are the ones the hungarian algo is shifting around'
        
        #assert no nan in output
        assert not torch.isnan(result).any(), 'Output is nan'
        if args.report_volume_loss_basic > 0:
            if 'ball' in args.loss or 'dynamic' in args.loss or 'dll' in args.loss:
                #j!=0 and 'last' in args.loss=>applies the ball loss only to the last layer
                loss_r = ball_loss (out=result, labels=label, unk_voxels=unk_voxels, 
                                    chosen_segment_mask=chosen_segment_mask_contricted if chosen_segment_mask_contricted is not None else chosen_segment_mask, 
                                    tumor_volumes=tumor_volumes_report, tumor_diameters=tumor_diameters, classes=classes, 
                                    apply_dice_loss=('dice' in args.loss),sigmoid=(not args.cls_gate),
                                    standard_ce=args.stardard_ce_ball,class_weights=class_weights,
                                    single_class= args.epai_stage_2,
                                    diameter_margin=args.ball_volume_margin, volume_margin=args.ball_volume_margin,
                                    sizes_slices=sizes_slices, ct_z_spacing_original=ct_z_spacing_original,
                                    slices_mask=slices_mask, max_slice = max_slice,
                                    sample_weights=sample_weights,
                                    subseg_dilation = subseg_dilation)
                if 'both' in args.loss:
                    loss_r = merge_no_overlap(loss_r,volume_loss_basic(result,
                                                                       chosen_segment_mask_contricted if chosen_segment_mask_contricted is not None else chosen_segment_mask,
                                                                       tumor_volumes_report, 
                                           label, unk_voxels, classes, loss_function=args.loss,
                                           sigmoid=(not args.cls_gate), class_weights=class_weights,tolerance=args.volume_loss_tolerance,
                                           slices_cropped_dict=slices_cropped_dict,
                                           sample_weights=sample_weights,
                                           dilation_segment = subseg_dilation))
                    #print('Both')
            else:
                loss_r = volume_loss_basic(result,chosen_segment_mask_contricted if chosen_segment_mask_contricted is not None else chosen_segment_mask,
                                           tumor_volumes_report, 
                                           label, unk_voxels, classes, loss_function=args.loss,
                                           sigmoid=(not args.cls_gate), class_weights=class_weights,tolerance=args.volume_loss_tolerance,
                                           slices_cropped_dict=slices_cropped_dict,
                                           sample_weights=sample_weights,
                                           dilation_segment = subseg_dilation)
        else:
            loss_r = torch.tensor(0).type_as(result)

        if not args.cls_gate:
            if args.epai_stage_2:
                #softmax
                target_idx = label.argmax(dim=1).long()
                loss_seg   = F.cross_entropy(result, target_idx, reduction='none') #use BCE with logits for the segmentation loss
            else:
                loss_seg = F.binary_cross_entropy_with_logits(result, label.float(), reduction='none', weight=class_weights) #use BCE with logits for the segmentation loss
        else:
            if args.epai_stage_2:
                raise ValueError('cls_gate not implemented for epai_stage_2')
            #assert l is in the range 0-1
            assert (label>=0).all() and (label<=1).all(), f'Label is not in the range 0-1, its min is: {label.min()}, its max is: {label.max()}'
            assert (result>=0).all() and (result<=1).all(), f'Output is not in the range 0-1, its min is: {result.min()}, its max is: {result.max()}'
            loss_seg = F.binary_cross_entropy(result, label.float(), reduction='none', weight=class_weights) #use BCE for the segmentation loss when cls_gate is used, this will be 0/1 for the binary classification case

        assert loss_seg.shape == known_voxels.shape

        loss_seg = loss_seg * known_voxels_sliced
        loss_seg = loss_seg.mean() + DiceLossMultiClass(result, label, known_voxels_sliced, sigmoid=(not args.cls_gate),class_weights=class_weights)
        loss_segmentation = loss_segmentation + args.seg_loss * loss_seg
        if not isinstance(loss_r, dict):
            loss_report = loss_report + args.report_volume_loss_basic * loss_r
        else:
            if isinstance(loss_report,int):
                loss_report = {}
            for key in loss_r.keys():
                if key == 'ball_loss_bce':
                    weight = args.ball_bce_weight
                    #print(f'Using the ball bce weight: {weight}')
                elif key == 'ball_loss_dice':
                    weight = args.ball_dice_weight
                    #print(f'Using the ball dice weight: {weight}')
                else:
                    weight = 1
                loss_report[key] = args.report_volume_loss_basic * weight * loss_r[key]
                
    loss={'segmentation':loss_segmentation}
    if isinstance(loss_report,dict):
        for key in loss_report.keys():
            loss[key] = loss_report[key]
    else:
        loss['report'] = loss_report
        
    if att_loss is not None:
        loss['attenuation'] = att_loss * args.att_weight
    #else:
    #    raise ValueError('Attenuation loss is None, this should not happen!')
    
    if tumor_cls_loss is not None:
        loss['tumor_classification'] = tumor_cls_loss * args.cls_weight
    #else:
    #    raise ValueError('Tumor CLS loss is None, this should not happen!')
        

    #print('Loss report is:', loss_report)
    
    if cls_loss is not None:
        loss['classification'] = cls_loss
        
    if malignancy_seg_loss is not None:
        if isinstance(malignancy_seg_loss, dict):
            for key in malignancy_seg_loss.keys():
                loss[key] = malignancy_seg_loss[key]
        else:
            loss['malignancy_segmentation'] = malignancy_seg_loss

    if loss_wrapper is None:
        loss_overall = 0
        for key in loss.keys():
            #print('loss key:', key)
            loss_overall = loss_overall + loss[key]
    else:
        #create a list of losses from the dict
        losses = [loss[key] for key in sorted(loss.keys())]
        print('Losses sent to the wrapper:',  sorted(loss.keys()))
        loss_overall = loss_wrapper(losses)
        #assert weihts are 1
        #assert args.seg_loss==1 and args.report_volume_loss_basic==1, 'We should not be weighting the losses, as this would mean we are not using the wrapper'
        #check if wrapper requires grad
        for l in loss_wrapper.parameters():
            assert l.requires_grad, 'Loss wrapper parameters should require grad'

    loss['overall']=loss_overall
    if torch.isnan(loss_overall).any():
        raise ValueError('loss is nan, propagating this can destroy the network weights, STOP!')

    #check if loss_overall requires grad
    assert loss_overall.requires_grad, 'Loss overall should require grad'
    
    if debug_timers:
        print(f'Total time in loss function: {time.time() - start_time} seconds', flush=True, file=sys.stderr)
        print(f'Time after save: {time.time() - time_after_save} seconds', flush=True, file=sys.stderr)

    if not return_refined_mask:
        return loss,sample_weights
    else:
        return loss,sample_weights,refined_mask

def debug_save_labels(labels: torch.Tensor,
                      name='',
                      label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      out_dir: str = "./LossChecking",
                      batch_idx = 0,
                      deactivate = True):
    """
    Saves each channel of the specified batch index in `labels` as a .nii.gz file.
    
    Args:
        labels (torch.Tensor): A tensor of shape (B, C, H, W, D).
        label_names_yaml (str): Path to a YAML file containing a list of label names.
                                The list will be sorted alphabetically and used
                                to name the channels.
        out_dir (str): Output directory to save the .nii.gz files. Defaults to "LossSanity".
        batch_idx (int): Which batch element to save. Defaults to 0.
    """
    if deactivate:
        return
    import nibabel as nib
    # 1. Create output folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    #raise ValueError(f'Label names is: {label_names}')
    
    # 2. Load and sort label names
    if not isinstance(label_names, list):
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)  # e.g. ["liver", "kidney", "pancreas", ...]
        
    label_names_sorted = sorted(label_names)  # sort alphabetically
    
    # 3. Basic shape check
    if len(labels.shape)==4:
        labels = labels.unsqueeze(0)

    if labels.shape[1]!=len(label_names_sorted):
        raise ValueError(f"Number of channels in labels ({labels.shape[1]}) does not match the number of label names ({len(label_names_sorted)}). Labels loaded from: {label_names}. ")
        label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_multi_ch_tumor_npy/list/label_names.yaml'
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)
        label_names_sorted = sorted(label_names)
    
    assert len(labels.shape) == 5
    B, C, H, W, D = labels.shape
    assert batch_idx < B, f"batch_idx={batch_idx} is out of range for B={B}."
    if C != len(label_names_sorted):
        label_names_sorted = [str(i) for i in list(range(C))]
    
    # 4. Extract just the batch element we want
    #    This will have shape (C, H, W, D).
    label_slice = labels[batch_idx]
    
    # 5. Loop over channels, save each one as a nii.gz
    for c in range(C):
        # Move channel c to CPU numpy for saving
        channel_data = label_slice[c].detach().float().cpu().numpy()
        
        # Build a simple identity affine; if you have real metadata, replace it
        affine = np.eye(4, dtype=np.float32)
        
        # Convert to float32 (or int16, float64, etc.)
        channel_data = channel_data.astype(np.float32)
        
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(channel_data, affine)
        
        # Derive a filename from the label name
        channel_label_name = label_names_sorted[c]
        out_path = os.path.join(out_dir, f"{name}_{channel_label_name}.nii.gz")

        #print(f'Saving: {out_path}, its sum is {channel_data.sum()}')
        
        # Save
        nib.save(nifti_img, out_path)
        
    print(f"Saved to {out_path}")



def plot_loss(ln: str = "dice", 
              y = 1,
              tolerance: float = 0.1, 
              k: float = 1.5, 
              num_points: int = 300, 
              limit: float = 3, 
              eps: float = 1e-2):
    """
    Plots a piecewise loss and the absolute value of its gradient (in log scale) as a function 
    of the predicted value x (with target y fixed to 1). The loss plotted is selected by the 
    'loss_type' parameter.
    
    Args:
        loss_type (str): Which loss to plot. Options are:
                         "huber"   -> huber_entropy,
                         "l2"      -> l2_entropy,
                         "isnet"   -> isnet_entropy.
        tolerance (float): The tolerance defining the deadzone.
        k (float): The scaling factor for the huber or l2 branch. (Ignored for "isnet".)
        num_points (int): Number of x points to sample.
        limit (float): Maximum value of x to sample.
        eps (float): Small constant for numerical stability.
    """
    import matplotlib.pyplot as plt

    # Create x_vals with requires_grad=True so that we can compute gradients.
    x_vals = torch.linspace(0, limit, num_points, requires_grad=True)


    # Set target y = 1 so that normalized target is 1.
    y_val = torch.ones_like(x_vals)*y
    
    
    # Choose the loss function based on loss_type:
    loss_label = f"Loss Type: {ln}"
    losses = ln_entropy(x_vals, y_val, tolerance=tolerance, k=k, n=ln, reduction='none', eps=eps)
    
    
    # Compute gradient of loss with respect to x_vals.
    gradients = torch.autograd.grad(outputs=losses, inputs=x_vals,
                                    grad_outputs=torch.ones_like(losses))[0]
    
    # Convert tensors to numpy arrays for plotting.
    x_np = x_vals.detach().numpy()
    losses_np = losses.detach().numpy()
    gradients_np = gradients.detach().numpy()
    
    # Create a two-panel plot.
    plt.figure(figsize=(12, 5))
    
    # Plot the loss.
    plt.subplot(1, 2, 1)
    plt.plot(x_np, losses_np, label=loss_label)
    plt.xlabel("Predicted value x")
    plt.ylabel("Loss")
    plt.title(f"{loss_label} (Target = 1)")
    plt.axvline(1-tolerance, color='r', linestyle='--', label=f"x = 1-tol")
    plt.legend()
    plt.grid(True)
    
    # Plot the absolute gradient on a log scale.
    plt.subplot(1, 2, 2)
    plt.plot(x_np, abs(gradients_np), label="|Gradient|")
    plt.xlabel("Predicted value x")
    plt.ylabel("Absolute Gradient")
    plt.title("Absolute Gradient of Loss")
    plt.yscale("log")
    plt.axvline(1-tolerance, color='r', linestyle='--', label=f"x = 1-tol")
    plt.legend()
    plt.grid(True, which="both")
    
    plt.tight_layout()
    plt.show()



def ISNetLikeNegativeLoss(x,d=0.9977,E=1, reduction='none'):
    assert len(x.shape)==5, f'Input to ISNetLikeNegativeLoss should be 5D, got {x.shape}'
    #global maxpool on spatial dimensions:
    x=GlobalWeightedRankPooling(x,d=d)
    #activation:
    x=x/(x+E)
    #cross entropy (pixel-wise):
    x=torch.clamp(x,max=1-1e-7)
    loss=-torch.log(torch.ones(x.shape).type_as(x)-x)
    #reduction:
    if reduction=='none':
        return loss
    elif reduction=='mean':
        return loss.mean()
    elif reduction=='sum':
        return loss.sum()
    else:
        raise ValueError('Reduction not supported')
    











############### BALL LOSS ####################

def create_ball_kernel(diameter, gaussian=False, gaussian_std=3, margin = 0):
    """
    Creates a 3D torch tensor (kernel) where there is a 'ball' of a given diameter.
    The diameter is first rounded up to the next odd integer. The kernel size is then
    computed to be 1.2 × (that odd diameter), rounded to the next odd integer.
    
    The ball is centered in this larger kernel. Inside the ball (hard cutoff at the
    ball boundary), values are set to 1 (or to a truncated Gaussian if `gaussian=True`).
    Outside the ball, values are 0. If `gaussian=True`, the Gaussian is centered at
    the ball center with standard deviation `gaussian_std * radius`.

    Parameters
    ----------
    diameter : float or int
        Desired diameter of the ball. Will be rounded up to the next odd integer.
    gaussian : bool, optional
        Whether to fill the ball with a Gaussian distribution, by default False.
    gaussian_std : float, optional
        Standard deviation factor (relative to the ball radius) if gaussian=True.
        For example, if the ball's radius is R and gaussian_std=1.5, the std is
        1.5*R, by default 1.5.

    Returns
    -------
    kernel : torch.FloatTensor
        A 3D tensor of shape (kernel_size, kernel_size, kernel_size) containing
        the ball (or Gaussian ball) centered in the kernel.
    """

    # --- Step 1: Round diameter to next odd integer ---
    diameter_ceil = math.ceil(diameter)
    if diameter_ceil % 2 == 0:
        diameter_ceil += 1
    diameter_odd = diameter_ceil  # The final odd diameter
    
    # --- Step 2: Compute kernel size as (1+margin) * diameter_odd, also round up to next odd. Here, we are doing localization (ball conv), so, we need no margin ---
    kernel_size_float = (1+margin) * diameter_odd
    kernel_size_ceil = math.ceil(kernel_size_float)
    if kernel_size_ceil % 2 == 0:
        kernel_size_ceil += 1
    kernel_size = kernel_size_ceil  # The final odd kernel size
    
    # Ball radius (float)
    radius = diameter_odd / 2.0

    # --- Create 1D coordinate grid from 0..(kernel_size-1), shift so center is 0 ---
    center = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords_shifted = coords - center  # center at 0
    
    # --- Compute squared distance (3D) via broadcasting ---
    distance_squared = (coords_shifted[:, None, None] ** 2
                      + coords_shifted[None, :, None] ** 2
                      + coords_shifted[None, None, :] ** 2)
    
    # --- Hard cutoff mask for the ball ---
    mask = (distance_squared <= radius**2).float()
    
    if gaussian:
        # Scale std by the ball's actual radius
        std = gaussian_std * radius
        gaussian_values = torch.exp(-distance_squared / (2.0 * std**2))
        kernel = gaussian_values * mask
        # Normalize so that sum of kernel = 1
        kernel = kernel / kernel.sum()
    else:
        kernel = mask  # Binary ball kernel

    #assert the kernel size is odd
    assert kernel.shape[0] % 2 == 1, f'Kernel size should be odd, got {kernel.shape[0]}'
    
    return kernel


def save_ball_kernel(diameter, gaussian, gaussian_std, filename):
    """
    Wrapper function that creates a ball kernel using `create_ball_kernel`,
    prints the center and border values, and saves the kernel as a .nii.gz file.
    
    Args:
        diameter (int): Diameter of the ball.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball.
        gaussian_std (float): Standard deviation of the Gaussian.
        filename (str): Path for saving the NIfTI file (should end with .nii.gz).
    """
    # Create the kernel
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std)
    
    # Determine the center index (assuming symmetric kernel)
    center_idx = diameter // 2
    center_value = kernel[center_idx, center_idx, center_idx].item()
    
    # Determine the border value as the smallest nonzero value inside the ball.
    # (This should correspond roughly to the values at the edge.)
    border_value = kernel[kernel > 0].min().item()
    
    print(f"Center value: {center_value}")
    print(f"Border value: {border_value}")
    
    # Convert to numpy array (nibabel works with numpy)
    kernel_np = kernel.float().cpu().numpy()
    
    # Create a default affine (identity) matrix
    affine = np.eye(4)
    
    # Create and save the NIfTI image
    nii_img = nib.Nifti1Image(kernel_np, affine)
    nib.save(nii_img, filename)
    print(f"Saved ball kernel to {filename}")

def ball_convolution(x,diameter,gaussian, gaussian_std):
    """
    Performs a 3D convolution on the input tensor `x` using a ball kernel of diameter `diameter`.
    Optionally, the values inside the ball can follow a Gaussian distribution with standard deviation `gaussian_std`.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball.
        gaussian_std (float): Standard deviation of the Gaussian.
    
    Returns:
        torch.Tensor: Convolved tensor of shape (B, C, H, W, D).
    """
    #if diameter is not odd, add 1:
    if diameter%2==0:
        diameter+=1

    # Create the ball kernel
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    
    # Convert kernel to 5D tensor (B=1, C=1, H, W, D)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Perform the 3D convolution
    out = F.conv3d(x, kernel, padding=kernel.shape[-1]//2)

    assert out.shape == x.shape, f'Output shape should be the same as input shape, got {out.shape} and {x.shape}'
    return out

def insert_ball_old(out_spatial,best_center,diameter,margin):
    # Use a binary (non-Gaussian) ball kernel.
    binary_ball_kernel = create_ball_kernel(diameter*(1+margin), gaussian=False)
    #we add the margin only here, we do not use the margin in the convolution, for better detection.
    
    # Create an empty volume for the ball mask with the same spatial shape as x.
    masked_volume = torch.zeros_like(out_spatial)
    H, W, D = masked_volume.shape
    d_half = binary_ball_kernel.shape[-1] // 2
    cx, cy, cz = best_center

    # For each dimension, compute the overlapping indices between the input volume and the ball kernel.
    # X-dimension:
    vol_x_min = max(0, cx - d_half)
    vol_x_max = min(H, cx + d_half + 1)
    mask_x_min = 0 if cx - d_half >= 0 else -(cx - d_half)
    mask_x_max = mask_x_min + (vol_x_max - vol_x_min)

    # Y-dimension:
    vol_y_min = max(0, cy - d_half)
    vol_y_max = min(W, cy + d_half + 1)
    mask_y_min = 0 if cy - d_half >= 0 else -(cy - d_half)
    mask_y_max = mask_y_min + (vol_y_max - vol_y_min)

    # Z-dimension:
    vol_z_min = max(0, cz - d_half)
    vol_z_max = min(D, cz + d_half + 1)
    mask_z_min = 0 if cz - d_half >= 0 else -(cz - d_half)
    mask_z_max = mask_z_min + (vol_z_max - vol_z_min)

    # Place the binary ball kernel into the masked_volume at the computed overlapping region.
    masked_volume[vol_x_min:vol_x_max, vol_y_min:vol_y_max, vol_z_min:vol_z_max] = \
        binary_ball_kernel[mask_x_min:mask_x_max, mask_y_min:mask_y_max, mask_z_min:mask_z_max]
    return masked_volume

def insert_ball(out_spatial, best_center, diameter, margin):
    """
    Places a 'ball' of size diameter * (1 + margin) into out_spatial at the 3D coordinate best_center.
    The 3D ordering is assumed to be (z, y, x).
    """
    # 1) Build the ball kernel for insertion
    binary_ball_kernel = create_ball_kernel(diameter*(1+margin), gaussian=False)

    # 2) Prepare an empty volume with same shape as out_spatial
    masked_volume = torch.zeros_like(out_spatial)
    
    # 3) Extract shape in (z, y, x) order
    Z, Y, X = masked_volume.shape
    
    # 4) The kernel half-width
    d_half = binary_ball_kernel.shape[-1] // 2
    
    # 5) Unpack best_center as (cz, cy, cx)
    cz, cy, cx = best_center
    
    # 6) Compute overlap in Z dimension
    vol_z_min = max(0, cz - d_half)
    vol_z_max = min(Z, cz + d_half + 1)
    mask_z_min = 0 if cz - d_half >= 0 else -(cz - d_half)
    mask_z_max = mask_z_min + (vol_z_max - vol_z_min)

    # 7) Compute overlap in Y dimension
    vol_y_min = max(0, cy - d_half)
    vol_y_max = min(Y, cy + d_half + 1)
    mask_y_min = 0 if cy - d_half >= 0 else -(cy - d_half)
    mask_y_max = mask_y_min + (vol_y_max - vol_y_min)

    # 8) Compute overlap in X dimension
    vol_x_min = max(0, cx - d_half)
    vol_x_max = min(X, cx + d_half + 1)
    mask_x_min = 0 if cx - d_half >= 0 else -(cx - d_half)
    mask_x_max = mask_x_min + (vol_x_max - vol_x_min)

    # 9) Place the kernel region into masked_volume
    masked_volume[
        vol_z_min:vol_z_max,
        vol_y_min:vol_y_max,
        vol_x_min:vol_x_max
    ] = binary_ball_kernel[
        mask_z_min:mask_z_max,
        mask_y_min:mask_y_max,
        mask_x_min:mask_x_max
    ]

    return masked_volume



def binarize_membership(x: torch.Tensor, values, *, out_dtype=None):
    xi   = x.round().to(torch.int32)                   # enforce ints
    vals = torch.as_tensor(values, device=xi.device, dtype=xi.dtype)
    mask = torch.isin(xi, vals)
    return mask.to(out_dtype or x.dtype)

    
def cut_tumor_segment_with_slices(tumor_segment_mask, slices_mask, tumor_slice, original_spacing, tumor_diameter, max_slice,
                                  radius_tolerance=2):
    """
    We use the tumor slices for the chosen organ, 
    with z-axis mirroring (we are unsure if the radiologist counted slices from top to bottom or bottom to top),
    to "cut" the segment mask. Also consider tumor sizes when cutting. 
    To crop the slice mask itself, we use a trick: we just concatenate it to the labels, and separate afterwards.
    """
    if isinstance(tumor_diameter,torch.Tensor):
        tumor_diameter = tumor_diameter.item()
    
    try:
        tumor_slice = float(tumor_slice)
    except Exception:
        print(f"tumor_slice is not numeric: {tumor_slice}", flush=True, file=sys.stderr)
        return None
    if math.isnan(tumor_slice) or math.isinf(tumor_slice):
        print(f"tumor_slice is NaN/Inf: {tumor_slice}", flush=True, file=sys.stderr)
        return None
    if max_slice==0:
        print(f"max_slice is 0, cannot use tumor_slice: {tumor_slice}", flush=True, file=sys.stderr)
        return None
    if (tumor_slice<0) or (tumor_slice>max_slice):
        print(f"tumor_slice is out of bounds: {tumor_slice}, slices_mask max is {max_slice}", flush=True, file=sys.stderr)
        return None
    
    #mirror slices
    slice_mirror = max_slice - tumor_slice
    slices = [tumor_slice, slice_mirror]
    #check if any negative values in 'Mirror Image'
    if (slice_mirror < 0):
        raise ValueError(f'Negative slice values found in {slice_mirror}, this should not happen. Max slice is {max_slice}, and the slices are {tumor_slice}')
    
    allowed_slices = []
    for slice_value in slices:
        radius = max(round((tumor_diameter+2) / 2),1)*radius_tolerance  # at least 1 slice radius
        radius = int(round(radius / original_spacing))  # number of the ORIGINAL slices this radius (in mm) corresponds to
        #create a range of allowed slices. Remember:  the network works at 1×1×1 mm and slices_mask stores original indices. THIS WILL BREAK IF YOU ARE NOT WORKING AT 1x1x1, needs update in this case
        allowed_slices.extend(range(int(slice_value - radius), int(slice_value + radius + 1)))
        #print(f'Slices allowed mirror: from {int(slice_value - radius)} to {int(slice_value + radius + 1)}, slice is {slice_value}, size is {tumor_diameter}, max slice is {max_slice}', flush=True, file=sys.stderr)
        
    allowed_slices= list(set(allowed_slices))  # remove duplicates
    allowed_slices = [s for s in allowed_slices if (s >= 0 and s <= max_slice)]
    
    #use the allowed slices to cut the tumor segment mask, pick up only voxels whose VALUES (not position) are in allowed_slices
    binary_slices_mask = binarize_membership(slices_mask, allowed_slices)
    
    if binary_slices_mask.sum() == 0:
        #slice does not intersect organ
        #raise ValueError(f'Slice not in crop!')\
        return None
    #now we can cut the tumor segment mask with the slices mask
    assert tumor_segment_mask.shape == binary_slices_mask.shape, f'Tumor segment mask shape {tumor_segment_mask.shape} should match binary_slices_mask shape {binary_slices_mask.shape}'
    tumor_segment_mask_new = tumor_segment_mask.float() * binary_slices_mask.float().to(device=tumor_segment_mask.device)
    
    if tumor_segment_mask_new.sum() == 0:
        print(f'ATTENTION: Slice does not intersect organ!', flush=True, file=sys.stderr)
        return None
    
    return binary_slices_mask #this is correct, we only calculate tumor_segment_mask_new here to avoid slices out of organ
    
    

def isolate_tumor(x, diameter, gaussian, gaussian_std, tumor_volume,
                  diameter_margin=0.5,volume_margin=0.5,
                  tumor_slice=None, ct_z_spacing_original = None,
                  slices_mask=None, tumor_segment_mask=None,
                  max_slice=None, denoise = True):
    """
    Uses a ball convolution over x and applies a maximum operation to find the best
    fitting ball center. Then, it multiplies the input by a volume with the same size
    as the input, but with a binary ball placed at the given object center coordinate.
    Finally, after the multiplication, we find the top N voxels inside the remaining volume.
    N is the tumor volume.

    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball (for convolution).
        gaussian_std (float): Standard deviation of the Gaussian.
        tumor_volume (int): Number of voxels to select as the tumor volume.
        slices_mask (tensor): mask where each voxel VALUE indicated the original z slice index in the tensor (before preprocessing and augmentation)
        tumor_segment_mask (tensor): mask of the tumor segment, where each voxel is 1 if it is part of the tumor segment, 0 otherwise.
    Returns:masked_volume should be within 
        torch.Tensor: A binary tumor mask of shape (H, W, D) with 1's in the top N voxels.
    """
    
    
    reduce=False
    if len(x.shape)==3:
        reduce=True
        x = x.unsqueeze(0).unsqueeze(0)
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    #round diameter
    diameter = np.round(diameter).astype(int)
    #round tumor volume
    tumor_volume = np.round(tumor_volume).astype(int)

    # Ensure the diameter is odd.
    if diameter % 2 == 0:
        diameter += 1

    # Create the ball kernel for convolution.
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    # Convert kernel to a 5D tensor (shape: 1, 1, H, W, D).
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    #assert volume is not larger than the number of voxels in the ball
    if tumor_volume > 100000:
        assert tumor_volume <= (kernel>0).sum()*1.2, f'Tumor volume should be smaller than the number of voxels in the ball, got {tumor_volume} and {(kernel>0).sum()}'
    #ball_vol = int((kernel > 0).sum().item())
    #if tumor_volume > ball_vol:
        #clamp
    #    tumor_volume = max(ball_vol - 1, 1)

    if tumor_slice is not None:
        if len(slices_mask.shape) == 4:
            slices_mask = slices_mask.squeeze(0)
        assert len(slices_mask.shape)==3, f'Slices mask should be 3D, got {slices_mask.shape}'
        #we use the tumor slice to zero x 
        binary_slices_mask = cut_tumor_segment_with_slices(tumor_segment_mask=tumor_segment_mask, slices_mask=slices_mask, 
                                                           tumor_slice=tumor_slice, original_spacing=ct_z_spacing_original, tumor_diameter=diameter,
                                                           max_slice=max_slice)
        if binary_slices_mask is None:
            binary_slices_mask = torch.ones_like(x)  # if no slices mask, use all slices
        else:
            binary_slices_mask=binary_slices_mask.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
            assert len(binary_slices_mask.shape) == 5, f'Slices mask should be 5D, got {binary_slices_mask.shape}'
    else:
        #if no tumor slice, we use all slices
        binary_slices_mask = torch.ones_like(x) 
        
    assert x.shape == binary_slices_mask.shape, f'Input tensor shape {x.shape} should match slices mask shape {binary_slices_mask.shape}'
    # Perform 3D convolution.
    out = F.conv3d(x.float()*binary_slices_mask.float(), kernel, padding=kernel.shape[-1] // 2)
    #slice usage: we zero x where binary_slices_mask is 0, so that the convolution maximum (best ball position) will be in the tumor slice

    assert out.shape == x.shape, f"Output shape should match input shape, got {out.shape} vs {x.shape}"

    # --- Step 1: Find the best fitting ball center ---
    # Assume x is of shape (1, 1, H, W, D); take the spatial part.
    out_spatial = out[0, 0]  # shape: (H, W, D)
    max_idx = torch.argmax(out_spatial)
    best_center = np.unravel_index(max_idx.item(), out_spatial.shape)  # (cx, cy, cz)
    
    # --- Step 2: Create a binary ball mask at the best center ---
    masked_volume = insert_ball(out_spatial,best_center,diameter,diameter_margin)
    new_dim = diameter
    while masked_volume.sum() < tumor_volume:
        #if the ball is in the border of the image, its volume may be less than the tumor volume, We increase the size of the ball until we reach the tumor volume.
        old_dim = new_dim
        new_dim = int(np.round(new_dim * 1.1))
        print(f'Increasing ball size to {new_dim}, current volume is {masked_volume.sum()}, tumor volume is {tumor_volume}')
        if old_dim == new_dim:
            new_dim += 1
        if new_dim % 2 == 0:
            new_dim += 1
        if new_dim >= max(x.shape[-1], x.shape[-2], x.shape[-3]):
            break
        masked_volume = insert_ball(out_spatial,best_center,new_dim,diameter_margin)
    if tumor_volume < (50**3):
        assert (masked_volume.sum() > tumor_volume*0.5), f'masked_volume should be within 50% of the tumor volume! got {masked_volume.sum()} and {tumor_volume}'
    if tumor_volume > (6**3):
        assert (masked_volume.sum() < (diameter*(1+diameter_margin))**3), f'masked_volume should not surpass a cube with the tumor diameter! got {masked_volume.sum()} and {tumor_volume} and diameter {diameter}'

    # --- Step 3: Multiply the input by the binary ball mask ---
    # x has shape (B, C, H, W, D); expand masked_volume to match.
    #assert no negative value in x
    assert (x >= 0).all(), f'Input tensor should not have negative values, got {x.min()}'
    masked_x = (x * masked_volume.unsqueeze(0).unsqueeze(0))

    # --- Step 4: Find the top N voxels in the masked volume ---
    # Remove batch and channel dimensions.
    masked_x_vol = masked_x[0, 0]
    flattened = masked_x_vol.view(-1)
    # Get indices of the top N voxel values.
    t=min(flattened.shape[-1]-1, tumor_volume)
    margin_small = min(0.5,volume_margin)
    t_small = int(t*(1-margin_small))
    t_small =  max(t_small, min(100,tumor_volume))  # Ensure at 4mm tumor
    t_big = min(flattened.shape[-1]-1,int(tumor_volume*(1+volume_margin)))
    topN_values, topN_indices = torch.topk(flattened, t)
    topN_values_small, topN_indices_small = torch.topk(flattened, t_small)
    topN_values_big, topN_indices_big = torch.topk(flattened, t_big)
    #how many indices? Assert this matches the tumor volume
    assert len(topN_indices) == t, f'Expected {tumor_volume} indices, got {len(topN_indices)}'
    # Create a binary volume: set top N positions to 1, rest to 0.
    tumor_mask_flat = torch.zeros_like(flattened)
    tumor_mask_flat[topN_indices] = 1
    tumor_mask_flat_small = torch.zeros_like(flattened)
    tumor_mask_flat_small[topN_indices_small] = 1
    tumor_mask_flat_big = torch.zeros_like(flattened)
    tumor_mask_flat_big[topN_indices_big] = 1
    
    # Reshape to original spatial dimensions.
    tumor_mask = tumor_mask_flat.view_as(masked_x_vol)
    tumor_mask_small = tumor_mask_flat_small.view_as(masked_x_vol)
    tumor_mask_big = tumor_mask_flat_big.view_as(masked_x_vol)
    # Assert the sum here still matches the tumor volume.
    assert tumor_mask.sum() == t, f'Tumor mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    
    #ensure no tumor_max value is outside the ball
    tumor_mask = tumor_mask * masked_volume
    tumor_mask_small = tumor_mask_small * masked_volume
    tumor_mask_big = tumor_mask_big * masked_volume

    if reduce:
        tumor_mask = tumor_mask.squeeze(0).squeeze(0)

    iters = 0
    while tumor_volume < (50**3) and tumor_mask.sum() < tumor_volume*0.7:
        #zero values inside the ball may not be chosen as the top N voxels. In such cases, we dilate the mask
        print(f'dilating tumor mask, iteration {iters}, current volume is {tumor_mask.sum()}, tumor volume is {tumor_volume}')
        if iters >5:
            return tumor_mask, tumor_mask_small, tumor_mask_big, binary_slices_mask, x.float()*binary_slices_mask.float()
        #dilate the mask
        tumor_mask = dilate_volume(tumor_mask, 7)*masked_volume
        tumor_mask_small = dilate_volume(tumor_mask_small, 7)*masked_volume
        tumor_mask_big = dilate_volume(tumor_mask_big, 7)*masked_volume
        iters+=1

    if tumor_volume < (50**3):
        assert (tumor_mask.sum() > tumor_volume*0.5), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'
    if tumor_volume > (5**3):
        assert (tumor_mask.sum() < tumor_volume*((1+volume_margin)**3)*3), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    #assert it is binary
    assert (tumor_mask == 0).sum() + (tumor_mask == 1).sum() == tumor_mask.numel(), f'Tumor mask should be binary, got {tumor_mask.sum()}'
    
    if denoise:
        #this is important to avoid grid artifacts!
        tumor_mask = smooth_mask_avg(tumor_mask)
        tumor_mask_small = smooth_mask_avg(tumor_mask_small)
        tumor_mask_big = smooth_mask_avg(tumor_mask_big)

    return tumor_mask, tumor_mask_small, tumor_mask_big, binary_slices_mask, x.float()*binary_slices_mask.float()


def smooth_mask_avg(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Smooth a binary-ish mask with avg_pool3d + threshold, BUT:
    if smoothing would erase a non-empty mask (turn it all zeros) for a given batch item,
    keep the original mask for that batch item.

    Expects mask shape: [B, 1, D, H, W]
    Returns float mask in {0,1} with same shape/device.
    """
    original_dim = mask.shape
    if len(mask.shape) < 5:
        mask = mask.unsqueeze(0)
    if len(mask.shape) < 5:
        mask = mask.unsqueeze(0)
    if len(mask.shape) < 5:
        raise ValueError(f'Expected mask with at least 5 dimensions, got {mask.shape}')

    orig = (mask > 0.5).float()

    sm = F.max_pool3d(orig, kernel_size=k, stride=1, padding=k // 2)
    sm = (sm > 0.5).float()#conservative. if any neighbor voxel is 1, we keep it

    # sums per batch item
    orig_sum = orig.sum(dim=(1, 2, 3, 4))  # [B]
    sm_sum   = sm.sum(dim=(1, 2, 3, 4))    # [B]

    # fallback if original non-empty but smoothed becomes empty
    fallback = (orig_sum > 0) & (sm_sum == 0)  # [B] bool

    if fallback.any():
        fb = fallback.view(-1, 1, 1, 1, 1)
        sm = torch.where(fb, orig, sm)
        
    if len(sm.shape) > len(original_dim):
        sm = sm.squeeze(0)
    if len(sm.shape) > len(original_dim):
        sm = sm.squeeze(0)
    if sm.shape != original_dim:
        raise ValueError(f'Smoothed mask shape {sm.shape} does not match original shape {original_dim}')

    return sm

def tensor_to_pairs(t):
    """
    t: torch.Tensor of shape (N, 2) = [diameter, slice]
    Filters out non-finite / <=0 diameters; converts slice NaNs to None.
    Returns a Python list of [diameter(float), slice(float|None)].
    """
    arr = t.detach().cpu().float().numpy()
    pairs = []
    for d, s in arr:
        d = float(d)
        if not math.isfinite(d) or d <= 0:
            continue
        s = None if not math.isfinite(float(s)) else float(s)
        pairs.append([d, s])
    return pairs


def tensor_to_pairs_negatives(t):
    """
    t: torch.Tensor of shape (N, 2) = [diameter, slice]
    Gets only pair for unknown diameter lesions (negative diameter here); converts slice NaNs to None.
    Returns a Python list of [diameter(float), slice(float|None)].
    """
    arr = t.detach().cpu().float().numpy()
    pairs = []
    for d, s in arr:
        d = float(d)
        if d>= 0:
            continue
        s = None if not math.isfinite(float(s)) else float(s)
        pairs.append([d, s])
    return pairs

def pop_first_by_diameter(pairs, diameter, tol=0.3, fallback_to_nearest=False, tolerance_increment=0, tolerance_increment_max=5):
    """
    pairs: list of [diameter, slice]
    diameter: target diameter (int|float|str ok)
    tol: absolute tolerance for matching, 1mm, for some rounding error
    Returns [matched_d, matched_slice] or None. Removes the matched item from pairs.
    """
    base_tol = tol
    tol = tol+tolerance_increment
    
    if len(pairs)==0:
        raise ValueError(f'No pairs provided for matching diameter {diameter}. Pairs: {pairs}. Maybe you did already pop all of them.')
    d = float(diameter)
    # 1) first tolerant match (preserves original order)
    for i, (dj, sj) in enumerate(pairs):
        if abs(dj - d) <= tol:
            return pairs.pop(i)
    # 2) optional nearest fallback if no match in tol
    if fallback_to_nearest and pairs:
        i = int(np.argmin([abs(dj - d) for dj, _ in pairs]))
        return pairs.pop(i)
    
    if tolerance_increment < tolerance_increment_max:
        #try again, with larger tolerance. Increment one by one until 5mm
        return pop_first_by_diameter(
            pairs, diameter, tol=base_tol,
            fallback_to_nearest=fallback_to_nearest,
            tolerance_increment=tolerance_increment + 1,
            tolerance_increment_max=tolerance_increment_max
        )
        
    raise ValueError(f'No match found in size-diameter pairs for diameter {d} with tolerance {tol}. Pairs: {pairs}')

counter3=0

cases_with_unk_sizes = 0


def has_any_malignancy_label(sizes_malignancy_ct):
    """
    True iff there is at least one *known* malignancy label (0 or 1)
    inside sizes_malignancy_ct (list of (diameter, malig)).
    malig can be 0/1, or NaN/None for unknown.
    """
    if len(sizes_malignancy_ct) == 0:
        return False

    for _, malig in sizes_malignancy_ct:
        # skip None
        if malig == 0 or malig == 1 or malig == 0.0 or malig == 1.0 or malig == '0' or malig == '1' or malig == '0.0' or malig == '1.0':
            return True

    return False

def ball_loss(out, labels, unk_voxels, chosen_segment_mask, tumor_volumes, tumor_diameters, classes, apply_dice_loss,
              diameter_margin=0.2, volume_margin=0.2, gaussian=True, 
              gaussian_std=3, gwrp=True, gwrp_concentration=0.5, dilation_for_background=7,
              subseg_dilation=31,input_tensor=None, unk_dilation=1,
              sigmoid=True, standard_ce=False, class_weights=None,
              single_class=False, use_small_pseudo_mask=True,
              sizes_slices = None,
              ct_z_spacing_original = None,
              slices_mask = None,
              max_slice = None,
              sample_weights = None,
              malignant_benign_loss=False,
              benign_out=None,
              malignant_out=None,
              sizes_malignancy = None,
              student_out=None,
              student_benign_out=None,
              student_malignant_out=None,
              return_refined_mask=False,
              ):
    """
    
    This funciton first uses a ball convolution to isolate the tumor. Then, it selects the top N voxels inside the ball as a "pseudo-label" and applies BCE loss per-voxel.
    Optionally, we can average the per-voxel BCE loss using GWRP weights calculated for the isolated tumor voxels. This will give more emphasis in increasing high confidence voxels.
    Args:
    x is the model output
    tumor_diameter is a tensor of size B,T,3, batch, number of tumors in the crop, and 3 diameters
    diameter_margin: how much much we want the ball diameter to be bigger than the maximum tumor diameter
    gaussian: if a gaussian kernel is used in the ball convolution for better centering on the tumor
    gaussian_std: the higher, the smaller the difference between the ball kernel center and border values
    gwrp: wether to use GRWP to average each BCE loss. If so, more weight is given to increasing high confidence voxels.
    sigmoid: wether to apply sigmoid to the output.
    dilation_for_background: we apply a dilation kernel of this size to the tumor pseudo-mask, and define everything outside this mask as background, and use BCE loss to make the backgropund 0
    subseg_dilation: how much we dilate the tumor subsegment. Radiologists/AI may not be super precise when defining the subsegment, and tumors may grow out of organs, so we add a generous margin here.
    standard_ce: if True, we use a standard averaging for the BCE loss. Otherwise, we acerage the foreground and background voxel losses separately, and then sum the two losses.
    class_weights: optional 5D tensor to apply class weights. This is useful when dealing with imbalanced positives and negatives per class or datasets with many classes.
    sizes_slices: a torch tensor of size B,10,2, where the last dim represents pairs of tumor diameters in the crop, and their slices (unknown is nan)
    slices_mask: a tensor of shape (B, 1, H, W, D) where each voxel value indicates the original z slice index in the tensor (before preprocessing and augmentation). This is used to cut the tumor segment mask to the slices where the tumor is present.
    max_slice: slice count of original CT scan
    Important: this loss assumes the output resolution is 1x1x1 mm, and that diammeters are in mm and volumes in mm^3. If the resolution is different, you should adjust the diameters and volumes accordingly or introduce a scaling factor.
    negative volumes: negative volumes indicate lesions of unknown size, in such cases, we ensure at least a tumor of 5mm in diameter is segmented. 
    
    Variation of the ball loss for tumors where we want to classify between benign and malignant. Uses tumor
    sizes and malignancy labels to create a benign pseudo mask and a malignant pseudo mask. Masks are created
    from a reference (lesion output or gt mask), and applied to the model benign and malignant outputs.
    sizes_malignancy: a torch tensor of size B,10,2, where the last dim represents pairs of tumor diameters in the crop, and their malignancy (0=benign, 1=malignant, unknown is nan)
    benign_out: model output for benign tumors
    malignant_out: model output for malignant tumors
    student_out: optional model output from a student model, for knowledge distillation
    student_benign_out: optional model output from a student model for benign tumors, for knowledge distillation
    student_malignant_out: optional model output from a student model for malignant tumors, for knowledge distillation
    """
    #start timer
    start = time.time()
    
    
    global counter3
    global cases_with_unk_sizes
    
    if (not malignant_benign_loss) and ((benign_out is not None) or (malignant_out is not None) or (sizes_malignancy is not None)):
        raise ValueError('you sent benign/malignant segmentation outputs, but set malignant_benign_loss to False, please set it to True to use these outputs.')
    if malignant_benign_loss:
        if (benign_out is None) or (malignant_out is None) or (sizes_malignancy is None):
            raise ValueError('you set malignant_benign_loss to True, but did not send benign/malignant segmentation outputs, please send them to use this loss.')
    
    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    assert unk_voxels.shape == out.shape
    assert labels.shape == out.shape
    if class_weights is not None:
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert len(class_weights.shape) == 5, f'Class weights should be 5D tensor, got {class_weights.shape}'
        #repeat channels to match the output shape
        class_weights = class_weights.repeat(out.shape[0], 1, out.shape[2], out.shape[3], out.shape[4])

    # separately handle batch elements with negative volumes (unknown sizes)
    neg_batch_mask = (tumor_volumes < 0).any(dim=-1)      # [B]
    if use_small_pseudo_mask:
        diam = int((diameter_margin+1)*5+1)
    else:
        diam = 6
    volume_neg_sample = torch.tensor([int((4/3)*math.pi*((diam/2)**3))] + [0]*9, 
                                    device=tumor_volumes.device, dtype=tumor_volumes.dtype)  # [T=10]
    volume_neg_sample=volume_neg_sample.unsqueeze(0).repeat(tumor_volumes.shape[0],1) # [B,T]
    diameters_neg_sample = torch.tensor([[diam,diam,diam]] + [[0,0,0] for _ in range(9)],
                                        device=tumor_diameters.device, dtype=tumor_diameters.dtype) # [T=10, 3]
    diameters_neg_sample=diameters_neg_sample.unsqueeze(0).repeat(tumor_diameters.shape[0],1,1) # [B,T,3]
    assert diameters_neg_sample.shape == tumor_diameters.shape, f'Diameters neg sample shape {diameters_neg_sample.shape} should match tumor diameters shape {tumor_diameters.shape}'
    assert volume_neg_sample.shape == tumor_volumes.shape, f'Volume neg sample shape {volume_neg_sample.shape} should match tumor volumes shape {tumor_volumes.shape}'
    assert len(neg_batch_mask.unsqueeze(-1).shape) == len(tumor_volumes.shape), f'Neg batch mask shape {neg_batch_mask.unsqueeze(-1).shape} should match tumor volumes shape {tumor_volumes.shape}'
    assert len(neg_batch_mask.unsqueeze(-1).unsqueeze(-1).shape) == len(tumor_diameters.shape), f'Neg batch mask shape {neg_batch_mask.unsqueeze(-1).unsqueeze(-1).shape} should match tumor diameters shape {tumor_diameters.shape}'
    tumor_volumes   = torch.where(neg_batch_mask.unsqueeze(-1).to(tumor_volumes.device), volume_neg_sample.to(tumor_volumes.device),   tumor_volumes)   # [B,T]
    tumor_diameters = torch.where(neg_batch_mask.unsqueeze(-1).unsqueeze(-1).to(tumor_diameters.device), diameters_neg_sample.to(tumor_diameters.device), tumor_diameters)  # [B,T,3]
    if sizes_slices is not None:
        #shape of sizes_slices: B,10,2
        size_slice_unk = torch.tensor(
            [[float(diam), float('nan')]] + [[0.0, float('nan')] for _ in range(9)],
            device=sizes_slices.device, dtype=sizes_slices.dtype
        )                             # [T=10, 2]
        size_slice_unk = size_slice_unk.unsqueeze(0).repeat(sizes_slices.shape[0], 1, 1)  # [B,10,2]
        assert size_slice_unk.shape == sizes_slices.shape, f'Size slice unk shape {size_slice_unk.shape} should match sizes_slices shape {sizes_slices.shape}'
        assert len(neg_batch_mask.unsqueeze(-1).unsqueeze(-1).shape) == len(sizes_slices.shape), f'Neg batch mask shape {neg_batch_mask.unsqueeze(-1).unsqueeze(-1).shape} should match sizes_slices shape {sizes_slices.shape}'
        sizes_slices = torch.where(neg_batch_mask.unsqueeze(-1).unsqueeze(-1).to(sizes_slices.device), size_slice_unk.to(sizes_slices.device), sizes_slices)
    #cases_with_unk_sizes = cases_with_unk_sizes + neg_batch_mask.float().sum()
    #print(f'-- Cases with unk sizes so far: {cases_with_unk_sizes}, current batch has {neg_batch_mask.float().sum().item()} cases with unk sizes', flush=True)

    #get only the channels with lesions
    out = get_lesion_channels(out, classes)
    if student_out is not None:
        student_out = get_lesion_channels(student_out, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes, assertion=False)
    unk_voxels = get_lesion_channels(unk_voxels, classes)
    labels = get_lesion_channels(labels, classes)
    if class_weights is not None:
        class_weights = get_lesion_channels(class_weights, classes)
        
    if malignant_benign_loss:
        assert benign_out.shape == out.shape, f'Benign output shape {benign_out.shape} should match output shape {out.shape} after getting lesion channels'
        assert malignant_out.shape == out.shape, f'Malignant output shape {malignant_out.shape} should match output shape {out.shape} after getting lesion channels'

    chosen_segment_mask = dilate_volume(chosen_segment_mask,subseg_dilation)
    #dilate the unk voxels
    unk_voxels = dilate_volume(unk_voxels,unk_dilation)
    to_penalize = torch.ones_like(out)
    #remove the unk voxels from the penalization
    to_penalize = to_penalize * (1 - unk_voxels)
    #also remove the knwon labels
    to_penalize = to_penalize * (1 - labels)
    #but add back the chosen segment mask
    to_penalize = to_penalize + chosen_segment_mask
    #binarize
    to_penalize = (to_penalize > 0).float()


    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape

    losses = []
    losses_dice = []
    
    if malignant_benign_loss:
        losses_malignant_bce = []
        losses_benign_bce = []
        losses_malignant_dice = []
        losses_benign_dice = []
        #make a zeros object of the same shape as out
        malignancy_loss_applied = torch.zeros_like(out)
        applied_malignancy_losses = []
    
    if student_out is not None:
        losses_student = []
        losses_dice_student = []
        if malignant_benign_loss:
            losses_malignant_bce_student = []
            losses_benign_bce_student = []
            losses_malignant_dice_student = []
            losses_benign_dice_student = []
            
    refined_masks = []

    for B in range(out.shape[0]):#batch itens
        #assert diameters and violumes make sense
        assert torch.equal(tumor_diameters[B].sum(-1)>0, tumor_volumes[B]>0), f'Tumor diameters and volumes should be consistent, got {tumor_diameters[B]} and {tumor_volumes[B]}'
        if sizes_slices is not None:
            sizes_slices_ct = tensor_to_pairs(sizes_slices[B])
            
        if malignant_benign_loss:
            allow_malignancy_loss_ct = True
            #unk size: diameter is negative
            #1 (malignant), 0 (benign), NaN (unknown or empty channel)
            #shape: 10,2
            unk_sizes_malignancy_ct = tensor_to_pairs_negatives(sizes_malignancy[B])
            sizes_malignancy_ct = tensor_to_pairs(sizes_malignancy[B])
            x_benign = benign_out[B]
            x_malignant = malignant_out[B]
            if student_benign_out is not None and student_malignant_out is not None:
                student_x_benign = student_benign_out[B]
                student_x_malignant = student_malignant_out[B]
            #check sizes_malignancy_ct, group tumors that have the same diameter in the CT. If they have different malignancy labels, skip the loss (allow_malignancy_loss_ct=False)
            for i in range(len(sizes_malignancy_ct)):
                di = sizes_malignancy_ct[i][0]
                malignancy_i = sizes_malignancy_ct[i][1]
                for j in range(i+1, len(sizes_malignancy_ct)):
                    dj = sizes_malignancy_ct[j][0]
                    malignancy_j = sizes_malignancy_ct[j][1]
                    if abs(di - dj) <= 3:#3 mm tolerance
                        if malignancy_i != malignancy_j:
                            allow_malignancy_loss_ct = False
            
        #get correct batch and class
        x = out[B]
        if student_out is not None:
            x_student = student_out[B]
        tumor_seg = chosen_segment_mask[B]
        #current_x is still 4 D, with one class per tumor type. Assert at most one of these channels is non-zero (due to the chosen_segment_mask):
        assert (tumor_seg.sum((-1,-2,-3))>0).float().sum()<=1, f'Only one channel should be non-zero, got {tumor_seg.sum((-1,-2,-3))}'
        unknown_size = neg_batch_mask[B].item()
        if malignant_benign_loss and unknown_size:
            allow_malignancy_loss_ct = False #samples with unk tumor size should be processed with the standard malignt/benign auto-distill loss
            
        if malignant_benign_loss and (not has_any_malignancy_label(sizes_malignancy_ct)):
            allow_malignancy_loss_ct = False
        
        # if no tumor in this batch, create a zero pseudo label
        # this is not really necessary, as batches w/o tumors are already being penalized in the segmentation loss, outside this fuction. You should be able to remove this part.
        if tumor_seg.sum()==0 or tumor_volumes[B].sum()==0:
            # no tumor in this batch, create a zero pseudo label
            pseudo_mask = torch.zeros_like(x)
            refined_masks.append(pseudo_mask[0])#3D
            if sigmoid:
                if not single_class:
                    #standard, use sigmoid
                    loss = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        loss_benign_bce = F.binary_cross_entropy_with_logits(x_benign, pseudo_mask, reduction='none')
                        loss_malignant_bce = F.binary_cross_entropy_with_logits(x_malignant, pseudo_mask, reduction='none')
                    if student_out is not None:
                        loss_student = F.binary_cross_entropy_with_logits(x_student, pseudo_mask, reduction='none')
                        if malignant_benign_loss and allow_malignancy_loss_ct:
                            loss_benign_bce_student = F.binary_cross_entropy_with_logits(student_x_benign, pseudo_mask, reduction='none')
                            loss_malignant_bce_student = F.binary_cross_entropy_with_logits(student_x_malignant, pseudo_mask, reduction='none')
                else:
                    #use softmax
                    loss = F.cross_entropy(x, pseudo_mask, reduction='none')
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        loss_benign_bce = F.cross_entropy(x_benign, pseudo_mask, reduction='none')
                        loss_malignant_bce = F.cross_entropy(x_malignant, pseudo_mask, reduction='none')
                    if student_out is not None:
                        loss_student = F.cross_entropy(x_student, pseudo_mask, reduction='none')
                        if malignant_benign_loss and allow_malignancy_loss_ct:
                            loss_benign_bce_student = F.cross_entropy(student_x_benign, pseudo_mask, reduction='none')
                            loss_malignant_bce_student = F.cross_entropy(student_x_malignant, pseudo_mask, reduction='none')
                #print('ball loss uses BCE with logits')
            else:
                if not single_class:
                    #assert x is in the range 0-1
                    assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                    #assert pseudo_mask is in the range 0-1
                    assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                    loss = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        loss_benign_bce = F.binary_cross_entropy(x_benign, pseudo_mask, reduction='none')
                        loss_malignant_bce = F.binary_cross_entropy(x_malignant, pseudo_mask, reduction='none')
                    if student_out is not None:
                        loss_student = F.binary_cross_entropy(x_student, pseudo_mask, reduction='none')
                        if malignant_benign_loss and allow_malignancy_loss_ct:
                            loss_benign_bce_student = F.binary_cross_entropy(student_x_benign, pseudo_mask, reduction='none')
                            loss_malignant_bce_student = F.binary_cross_entropy(student_x_malignant, pseudo_mask, reduction='none')
                else:
                    #single class, but consider that softmax was already applied. Thus, use nll loss
                    #from one-hot to class indices: argmax
                    loss = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        loss_benign_bce = F.nll_loss(x_benign, pseudo_mask.argmax(dim=1), reduction='none')
                        loss_malignant_bce = F.nll_loss(x_malignant, pseudo_mask.argmax(dim=1), reduction='none')
                    if student_out is not None:
                        loss_student = F.nll_loss(x_student, pseudo_mask.argmax(dim=1), reduction='none')
                        if malignant_benign_loss and allow_malignancy_loss_ct:
                            loss_benign_bce_student = F.nll_loss(student_x_benign, pseudo_mask.argmax(dim=1), reduction='none')
                            loss_malignant_bce_student = F.nll_loss(student_x_malignant, pseudo_mask.argmax(dim=1), reduction='none')
            assert loss.shape == tumor_seg.shape
            loss = loss * to_penalize[B]
            if malignant_benign_loss and allow_malignancy_loss_ct:
                loss_benign_bce = loss_benign_bce * to_penalize[B]
                loss_malignant_bce = loss_malignant_bce * to_penalize[B]
            if student_out is not None:
                loss_student = loss_student * to_penalize[B]
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    loss_benign_bce_student = loss_benign_bce_student * to_penalize[B]
                    loss_malignant_bce_student = loss_malignant_bce_student * to_penalize[B]
            if class_weights is not None:
                # apply class weights if provided
                loss = loss * class_weights[B]
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    loss_benign_bce = loss_benign_bce * class_weights[B]
                    loss_malignant_bce = loss_malignant_bce * class_weights[B]
                if student_out is not None:
                    loss_student = loss_student * class_weights[B]
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        loss_benign_bce_student = loss_benign_bce_student * class_weights[B]
                        loss_malignant_bce_student = loss_malignant_bce_student * class_weights[B]
            loss = loss.mean()
            if malignant_benign_loss and allow_malignancy_loss_ct:
                loss_benign_bce = loss_benign_bce.mean()
                loss_malignant_bce = loss_malignant_bce.mean()
            if student_out is not None:
                loss_student = loss_student.mean()
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    loss_malignant_bce_student = loss_malignant_bce_student.mean()
                    loss_benign_bce_student = loss_benign_bce_student.mean()
            if apply_dice_loss:
                if class_weights is not None:
                    w = class_weights[B]
                else:
                    w = None
                dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                losses_dice.append(dice_loss)
                
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    loss_benign_dice = DiceLossMultiClass(preds=x_benign, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                    loss_malignant_dice = DiceLossMultiClass(preds=x_malignant, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                    losses_benign_dice.append(loss_benign_dice)
                    losses_malignant_dice.append(loss_malignant_dice)
                    
                if student_out is not None:
                    dice_loss_student = DiceLossMultiClass(preds=x_student, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                    losses_dice_student.append(dice_loss_student)
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        loss_benign_dice_student = DiceLossMultiClass(preds=student_x_benign, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                        loss_malignant_dice_student = DiceLossMultiClass(preds=student_x_malignant, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                        losses_benign_dice_student.append(loss_benign_dice_student)
                        losses_malignant_dice_student.append(loss_malignant_dice_student)
                
            losses.append(loss.mean())
            if malignant_benign_loss and allow_malignancy_loss_ct:
                losses_benign_bce.append(loss_benign_bce.mean())
                losses_malignant_bce.append(loss_malignant_bce.mean())
                malignancy_loss_applied[B,:]=to_penalize[B]
            if student_out is not None:
                losses_student.append(loss_student.mean())
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    losses_benign_bce_student.append(loss_benign_bce_student.mean())
                    losses_malignant_bce_student.append(loss_malignant_bce_student.mean())
                    
            if malignant_benign_loss and allow_malignancy_loss_ct:
                applied_malignancy_losses.append(True)
            if malignant_benign_loss and not allow_malignancy_loss_ct:
                applied_malignancy_losses.append(False)
                
            continue
        
        #get tumor class
        for c in range(x.shape[0]):
            if tumor_seg[c].sum()>0:
                #this is the tumor class, c
                x = x[c]# we can do this here because the channels w/o tumor are already being penalized in the segmentation loss, outside this fuction
                penalize = to_penalize[B][c]
                if class_weights is not None:
                    c_weight = class_weights[B][c] #get the class weights for this batch and class
                else:
                    c_weight = None
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    x_benign = x_benign[c]
                    x_malignant = x_malignant[c]
                if student_out is not None:
                    x_student = x_student[c]
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        student_x_benign = student_x_benign[c]
                        student_x_malignant = student_x_malignant[c]
                tumor_channel_sel = c
                break
        tumor_seg = tumor_seg.sum(0)
        current_tumor_diameters = tumor_diameters[B]
        current_tumor_volumes = tumor_volumes[B]

        # Get the sort indices for tumor_volumes in descending order (sort tumor by tumor volume)
        sorted_indices = torch.argsort(current_tumor_volumes, descending=True)

        # Filter indices to keep only those with volume > 0
        sorted_indices = sorted_indices[current_tumor_volumes[sorted_indices] > 0]
        #print('--------Sorted indices:', sorted_indices)
        #print('--------SORTED VOLUMES:', current_tumor_volumes[sorted_indices])
        #print('--------UNSORTED VOLUMES:', current_tumor_volumes)

        #Create the pseudo-mask
        pseudo_masks = []
        pseudo_masks_small = []
        pseudo_masks_big = []
        
        if malignant_benign_loss and allow_malignancy_loss_ct:
            pseudo_masks_benign = []
            pseudo_masks_malignant = []
            #small
            pseudo_masks_small_benign = []
            pseudo_masks_small_malignant = []
            #big
            pseudo_masks_big_benign = []
            pseudo_masks_big_malignant = []
            #unk malignancy
            pseudo_masks_unk_malignancy = []
        
        #update x for the next tumor: remove pseudo_mask, so that this tumor is not selected again.
        if sigmoid:
            x_iter = torch.sigmoid(x)*tumor_seg
        else:
            x_iter = x*tumor_seg
            
        #if slices_cropped_dict is provided, we must match each tumor idx with a tumor in the dataframe, matching by size
        
            
        
        for tumor_idx in sorted_indices:
            #iterate tumor by tumor, largest to smallest
            vol=current_tumor_volumes[tumor_idx].item()
            dia=current_tumor_diameters[tumor_idx]
            
            
            
            #get the maximum diameter
            max_diameter = torch.max(dia).item()
            max_diameter_original = max_diameter
            assert max_diameter>0, f'Tumor diameter should be larger than 0, got {max_diameter}'
            assert vol>0, f'Tumor volume should be larger than 0, got {vol}'
            if vol==0 or max_diameter == 0:
                print('Found 0 tumor where it should not be')
                continue
            if max_diameter <= 1:
                print('Found 1mm diameter, increasing to 3')
                max_diameter = 3
            if vol <= 1:
                print('Found 1mm volume, increasing to 9')
                vol = 9
                
            if malignant_benign_loss and allow_malignancy_loss_ct:
                _,malig = pop_first_by_diameter(sizes_malignancy_ct,max_diameter_original)
                if malig == 1:
                    malignancy_tumor = 'malignant'
                elif malig == 0:
                    malignancy_tumor = 'benign'
                elif malig is None:
                    malignancy_tumor = 'unknown'
                else:
                    raise ValueError(f'Unexpected malignancy value: {malig}')
                
            if sizes_slices is not None:
                #get the tumor slice from its size
                _,tumor_slice = pop_first_by_diameter(sizes_slices_ct,max_diameter_original)
                slices_mask_B = slices_mask[B].squeeze(0)
                ct_z_spacing_original_item = ct_z_spacing_original[B].item()
                max_slice_item = max_slice[B].item()
            else:
                tumor_slice = None
                slices_mask_B = None
                ct_z_spacing_original_item = None
                max_slice_item = None
            
            #assert it is not zero
            #ball convolution: use isolate_tumor to get the top 'tumor_volume' voxels in the outpus, inside the best fitting ball position
            pseudo_mask,pseudo_mask_small,pseudo_mask_big, binary_slices_mask, slices_gated_probabilities = isolate_tumor(x_iter.detach(), 
                                                                                                                          diameter=max_diameter, 
                                                                          gaussian=gaussian, gaussian_std=gaussian_std, tumor_volume=vol,
                                                                          diameter_margin=diameter_margin,volume_margin=volume_margin,
                                                                          tumor_slice=tumor_slice, ct_z_spacing_original = ct_z_spacing_original_item,
                                                                          slices_mask=slices_mask_B, tumor_segment_mask=tumor_seg,
                                                                          max_slice=max_slice_item)
            pseudo_masks.append(pseudo_mask)
            pseudo_masks_small.append(pseudo_mask_small)
            if unknown_size:
                #this batch sample has a tumor of unknown size. In the ball loss, we used a single, small tumor. But we do not know if this is actually a big tumor or many tumors.
                #thus, we consider pseudo_masks_big to be the entire organ/subsegment, so that the loss will not push the output to zero outside the small tumor.
                pseudo_mask_big = tumor_seg
            pseudo_masks_big.append(pseudo_mask_big)
            
            if malignant_benign_loss and allow_malignancy_loss_ct:
                zero = pseudo_mask*0
                if malignancy_tumor == 'benign':
                    pseudo_masks_benign.append(pseudo_mask)
                    pseudo_masks_small_benign.append(pseudo_mask_small)
                    pseudo_masks_big_benign.append(pseudo_mask_big)
                    pseudo_masks_malignant.append(zero)
                    pseudo_masks_small_malignant.append(zero)
                    pseudo_masks_big_malignant.append(zero)
                    pseudo_masks_unk_malignancy.append(zero)
                elif malignancy_tumor == 'malignant':
                    pseudo_masks_benign.append(zero)
                    pseudo_masks_small_benign.append(zero)
                    pseudo_masks_big_benign.append(zero)
                    pseudo_masks_malignant.append(pseudo_mask)
                    pseudo_masks_small_malignant.append(pseudo_mask_small)
                    pseudo_masks_big_malignant.append(pseudo_mask_big)
                    pseudo_masks_unk_malignancy.append(zero)
                elif malignancy_tumor == 'unknown':
                    pseudo_masks_benign.append(zero)
                    pseudo_masks_small_benign.append(zero)
                    pseudo_masks_big_benign.append(zero)
                    pseudo_masks_malignant.append(zero)
                    pseudo_masks_small_malignant.append(zero)
                    pseudo_masks_big_malignant.append(zero)
                    pseudo_masks_unk_malignancy.append(pseudo_mask_big)
                else:
                    raise ValueError(f'Unexpected malignancy_tumor value: {malignancy_tumor}')
            
            
            x_iter = x_iter * (1 - pseudo_mask) #remove the pseudo mask from the output, so that it is not selected again
        
        #stack the pseudo masks
        refined_mask = torch.stack(pseudo_masks).sum(0)
        refined_mask = (refined_mask > 0).float()
        refined_masks.append(refined_mask)
        if use_small_pseudo_mask:
            pseudo_mask = torch.stack(pseudo_masks_small).sum(0)
            if malignant_benign_loss and allow_malignancy_loss_ct:
                pseudo_mask_benign = torch.stack(pseudo_masks_small_benign).sum(0)
                pseudo_mask_malignant = torch.stack(pseudo_masks_small_malignant).sum(0)
        else:
            pseudo_mask = torch.stack(pseudo_masks).sum(0)
            if malignant_benign_loss and allow_malignancy_loss_ct:
                pseudo_mask_benign = torch.stack(pseudo_masks_benign).sum(0)
                pseudo_mask_malignant = torch.stack(pseudo_masks_malignant).sum(0)
        if malignant_benign_loss and allow_malignancy_loss_ct:
            pseudo_mask_unk_malignancy = torch.stack(pseudo_masks_unk_malignancy).sum(0)
            pseudo_mask_benign = (pseudo_mask_benign > 0).float()
            pseudo_mask_malignant = (pseudo_mask_malignant > 0).float()
            pseudo_mask_unk_malignancy = (pseudo_mask_unk_malignancy > 0).float()
        pseudo_mask = (pseudo_mask > 0).float()
        
        dilated_pseudo_mask = torch.stack(pseudo_masks_big).sum(0)
        dilated_pseudo_mask = (dilated_pseudo_mask > 0).float()
        if malignant_benign_loss and allow_malignancy_loss_ct:
            dilated_pseudo_mask_benign = torch.stack(pseudo_masks_big_benign).sum(0)
            dilated_pseudo_mask_benign = (dilated_pseudo_mask_benign > 0).float()
            dilated_pseudo_mask_malignant = torch.stack(pseudo_masks_big_malignant).sum(0)
            dilated_pseudo_mask_malignant = (dilated_pseudo_mask_malignant > 0).float()

        #we can add a tolerance margin around the pseudo mask, where we do not penalize the outputs for not being zero
        if dilation_for_background>0:
            dilated_pseudo_mask=dilate_volume(dilated_pseudo_mask, dilation_for_background)
            if malignant_benign_loss and allow_malignancy_loss_ct:
                dilated_pseudo_mask_benign=dilate_volume(dilated_pseudo_mask_benign, dilation_for_background)
                dilated_pseudo_mask_malignant=dilate_volume(dilated_pseudo_mask_malignant, dilation_for_background)
                pseudo_mask_unk_malignancy=dilate_volume(pseudo_mask_unk_malignancy, dilation_for_background)
            
        penalize_base = penalize
        
        if malignant_benign_loss and allow_malignancy_loss_ct:
            border_benign = dilated_pseudo_mask_benign - pseudo_mask_benign
            border_benign = (border_benign > 0).float()
            penalize_benign = penalize_base * (1 - border_benign) 
            penalize_benign = penalize_benign - pseudo_mask_unk_malignancy #remove unk malignancy region from benign penalization
            #threshold between 0 and 1
            penalize_benign = (penalize_benign > 0).float()
            border_malignant = dilated_pseudo_mask_malignant - pseudo_mask_malignant
            border_malignant = (border_malignant > 0).float()
            penalize_malignant = penalize_base * (1 - border_malignant)
            penalize_malignant = penalize_malignant - pseudo_mask_unk_malignancy #remove unk malignancy region from malignant penalization
            #threshold between 0 and 1
            penalize_malignant = (penalize_malignant > 0).float()
            
        border = dilated_pseudo_mask - pseudo_mask
        #threshold at 0
        border = (border > 0).float()
        penalize=penalize * (1 - border)
        #penalize is a tensor with the voxels where we want to apply our losses to here


        #BCE loss with mask
        if sigmoid:
            if not single_class:
                BCE = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    BCE_benign = F.binary_cross_entropy_with_logits(x_benign, pseudo_mask_benign, reduction='none')
                    BCE_malignant = F.binary_cross_entropy_with_logits(x_malignant, pseudo_mask_malignant, reduction='none')
                if student_out is not None:
                    BCE_student = F.binary_cross_entropy_with_logits(x_student, pseudo_mask, reduction='none')
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        BCE_benign_student = F.binary_cross_entropy_with_logits(student_x_benign, pseudo_mask_benign, reduction='none')
                        BCE_malignant_student = F.binary_cross_entropy_with_logits(student_x_malignant, pseudo_mask_malignant, reduction='none')
            else:
                #single class
                BCE = F.cross_entropy(x, pseudo_mask, reduction='none')
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    BCE_benign = F.cross_entropy(x_benign, pseudo_mask_benign, reduction='none')
                    BCE_malignant = F.cross_entropy(x_malignant, pseudo_mask_malignant, reduction='none')
                if student_out is not None:
                    BCE_student = F.cross_entropy(x_student, pseudo_mask, reduction='none')
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        BCE_benign_student = F.cross_entropy(student_x_benign, pseudo_mask_benign, reduction='none')
                        BCE_malignant_student = F.cross_entropy(student_x_malignant, pseudo_mask_malignant, reduction='none')
        else:
            if not single_class:
                #assert x is in the range 0-1
                assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                #assert pseudo_mask is in the range 0-1
                assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                BCE = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    BCE_benign = F.binary_cross_entropy(x_benign, pseudo_mask_benign, reduction='none')
                    BCE_malignant = F.binary_cross_entropy(x_malignant, pseudo_mask_malignant, reduction='none')
                if student_out is not None:
                    BCE_student = F.binary_cross_entropy(x_student, pseudo_mask, reduction='none')
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        BCE_benign_student = F.binary_cross_entropy(student_x_benign, pseudo_mask_benign, reduction='none')
                        BCE_malignant_student = F.binary_cross_entropy(student_x_malignant, pseudo_mask_malignant, reduction='none')
            else:
                #single class, but consider that softmax was already applied. Thus, use nll loss
                #from one-hot to class indices: argmax
                BCE = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    BCE_benign = F.nll_loss(x_benign, pseudo_mask_benign.argmax(dim=1), reduction='none')
                    BCE_malignant = F.nll_loss(x_malignant, pseudo_mask_malignant.argmax(dim=1), reduction='none')
                if student_out is not None:
                    BCE_student = F.nll_loss(x_student, pseudo_mask.argmax(dim=1), reduction='none')
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        BCE_benign_student = F.nll_loss(student_x_benign, pseudo_mask_benign.argmax(dim=1), reduction='none')
                        BCE_malignant_student = F.nll_loss(student_x_malignant, pseudo_mask_malignant.argmax(dim=1), reduction='none')
        assert (penalize.shape==BCE.shape), f'To penalize and BCE should have the same shape, got {penalize.shape} and {BCE.shape}'
        BCE = BCE * penalize #cut the loss gradient in the border. Remember that unk voxels were already removed from x
        if malignant_benign_loss and allow_malignancy_loss_ct:
            assert (penalize_benign.shape==BCE_benign.shape), f'To penalize benign and BCE benign should have the same shape, got {penalize_benign.shape} and {BCE_benign.shape}'
            BCE_benign = BCE_benign * penalize_benign
            assert (penalize_malignant.shape==BCE_malignant.shape), f'To penalize malignant and BCE malignant should have the same shape, got {penalize_malignant.shape} and {BCE_malignant.shape}'
            BCE_malignant = BCE_malignant * penalize_malignant
        if student_out is not None:
            assert (penalize.shape==BCE_student.shape), f'To penalize and BCE student should have the same shape, got {penalize.shape} and {BCE_student.shape}'
            BCE_student = BCE_student * penalize
            if malignant_benign_loss and allow_malignancy_loss_ct:
                assert (penalize_benign.shape==BCE_benign_student.shape), f'To penalize benign and BCE benign student should have the same shape, got {penalize_benign.shape} and {BCE_benign_student.shape}'
                BCE_benign_student = BCE_benign_student * penalize_benign
                assert (penalize_malignant.shape==BCE_malignant_student.shape), f'To penalize malignant and BCE malignant student should have the same shape, got {penalize_malignant.shape} and {BCE_malignant_student.shape}'
                BCE_malignant_student = BCE_malignant_student * penalize_malignant

        #dice loss
        if apply_dice_loss:
            #remove tumor surroundings, to avoid penalizing them: we are not super sure if this region is tumor or not.
            dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=penalize,sigmoid=sigmoid,class_weights=c_weight)
            if malignant_benign_loss and allow_malignancy_loss_ct:
                dice_loss_benign = DiceLossMultiClass(preds=x_benign, targets=pseudo_mask_benign, known_voxels=penalize_benign,sigmoid=sigmoid,class_weights=c_weight)
                dice_loss_malignant = DiceLossMultiClass(preds=x_malignant, targets=pseudo_mask_malignant, known_voxels=penalize_malignant,sigmoid=sigmoid,class_weights=c_weight)
            if student_out is not None:
                dice_loss_student = DiceLossMultiClass(preds=x_student, targets=pseudo_mask, known_voxels=penalize,sigmoid=sigmoid,class_weights=c_weight)
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    dice_loss_benign_student = DiceLossMultiClass(preds=student_x_benign, targets=pseudo_mask_benign, known_voxels=penalize_benign,sigmoid=sigmoid,class_weights=c_weight)
                    dice_loss_malignant_student = DiceLossMultiClass(preds=student_x_malignant, targets=pseudo_mask_malignant, known_voxels=penalize_malignant,sigmoid=sigmoid,class_weights=c_weight)
            
        if not standard_ce:
            #we separate foreground and background, calculate the average per-voxel loss for them separatelly, than sum it. We can use GRWP in the foreg. or not.
            if gwrp:
                #we do BCE for the entire channel, but we do not simply average it. We can use GWRP to average the tumor values (positive GT)
                #we add the pseudo-mask to boost its voxels values and concentrate GWRP there.
                assert pseudo_mask.sum() > 0, f'Pseudo mask should have at least one voxel, got {pseudo_mask.sum()}, volume is {vol} and diameter is {max_diameter}'
                if sigmoid:
                    foreg_weights = GlobalWeightedRankPooling(torch.sigmoid(x)*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                else:
                    foreg_weights = GlobalWeightedRankPooling(x*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    if pseudo_mask_benign.sum() == 0:
                        foreg_weights_benign = 0 * foreg_weights
                    else:
                        if sigmoid:
                            foreg_weights_benign = GlobalWeightedRankPooling(torch.sigmoid(x_benign)*pseudo_mask_benign+pseudo_mask_benign, N=pseudo_mask_benign.sum(), c=gwrp_concentration,return_weights=True,
                                                                        hard_cutoff=True)
                        else:
                            foreg_weights_benign = GlobalWeightedRankPooling(x_benign*pseudo_mask_benign+pseudo_mask_benign, N=pseudo_mask_benign.sum(), c=gwrp_concentration,return_weights=True,
                                                                        hard_cutoff=True)
                    if pseudo_mask_malignant.sum() == 0:
                        foreg_weights_malignant = 0 * foreg_weights
                    else:
                        if sigmoid:
                            foreg_weights_malignant = GlobalWeightedRankPooling(torch.sigmoid(x_malignant)*pseudo_mask_malignant+pseudo_mask_malignant, N=pseudo_mask_malignant.sum(), c=gwrp_concentration,return_weights=True,
                                                                        hard_cutoff=True)
                        else:
                            foreg_weights_malignant = GlobalWeightedRankPooling(x_malignant*pseudo_mask_malignant+pseudo_mask_malignant, N=pseudo_mask_malignant.sum(), c=gwrp_concentration,return_weights=True,
                                                                        hard_cutoff=True)
                #print highest and lowest non-zero values in foreg_weights
                assert foreg_weights.sum() > 0.95 and foreg_weights.sum() < 1.05, f'GWRP weights should be normalized to 1, got {foreg_weights.sum()}'
                #renormlize gwrp weights so they sum to pseudo_mask.sum()
                foreg_weights = foreg_weights * pseudo_mask.sum()
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    if pseudo_mask_benign.sum() > 0:
                        assert foreg_weights_benign.sum() > 0.95 and foreg_weights_benign.sum() < 1.05, f'GWRP benign weights should be normalized to 1, got {foreg_weights_benign.sum()}'
                        foreg_weights_benign = foreg_weights_benign * pseudo_mask_benign.sum()
                    if pseudo_mask_malignant.sum() > 0:
                        assert foreg_weights_malignant.sum() > 0.95 and foreg_weights_malignant.sum() < 1.05, f'GWRP malignant weights should be normalized to 1, got {foreg_weights_malignant.sum()}'
                        foreg_weights_malignant = foreg_weights_malignant * pseudo_mask_malignant.sum()
                #print('GWRP Foreg weights range:', foreg_weights[foreg_weights>0].max(), foreg_weights[foreg_weights>0].min())
                #assert sum of foreg_weights is close to 1
                foreg_weights = foreg_weights*pseudo_mask
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    foreg_weights_benign = foreg_weights_benign*pseudo_mask_benign
                    foreg_weights_malignant = foreg_weights_malignant*pseudo_mask_malignant
                assert BCE.shape == foreg_weights.shape, f'BCE and GWRP weights should have the same shape, got {BCE.shape} and {foreg_weights.shape}'
                loss_foreground = (BCE*foreg_weights)#.mean() #we can use mean here because 
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    assert BCE_benign.shape == foreg_weights_benign.shape, f'BCE benign and GWRP benign weights should have the same shape, got {BCE_benign.shape} and {foreg_weights_benign.shape}'
                    loss_foreground_benign = (BCE_benign*foreg_weights_benign)
                    assert BCE_malignant.shape == foreg_weights_malignant.shape, f'BCE malignant and GWRP malignant weights should have the same shape, got {BCE_malignant.shape} and {foreg_weights_malignant.shape}'
                    loss_foreground_malignant = (BCE_malignant*foreg_weights_malignant)
                if student_out is not None:
                    assert BCE_student.shape == foreg_weights.shape, f'BCE student and GWRP weights should have the same shape, got {BCE_student.shape} and {foreg_weights.shape}'
                    loss_foreground_student = (BCE_student*foreg_weights)
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        assert BCE_benign_student.shape == foreg_weights_benign.shape, f'BCE benign student and GWRP benign weights should have the same shape, got {BCE_benign_student.shape} and {foreg_weights_benign.shape}'
                        loss_foreground_benign_student = (BCE_benign_student*foreg_weights_benign)
                        assert BCE_malignant_student.shape == foreg_weights_malignant.shape, f'BCE malignant student and GWRP malignant weights should have the same shape, got {BCE_malignant_student.shape} and {foreg_weights_malignant.shape}'
                        loss_foreground_malignant_student = (BCE_malignant_student*foreg_weights_malignant)
            else:
                #print('Using simple mean for BCE loss')
                loss_foreground = (BCE*pseudo_mask)#.mean()
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    loss_foreground_benign = (BCE_benign*pseudo_mask_benign)
                    loss_foreground_malignant = (BCE_malignant*pseudo_mask_malignant)
                if student_out is not None:
                    loss_foreground_student = (BCE_student*pseudo_mask)
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        loss_foreground_benign_student = (BCE_benign_student*pseudo_mask_benign)
                        loss_foreground_malignant_student = (BCE_malignant_student*pseudo_mask_malignant)
            
            #Background:
            bkg_weights = 1 - dilated_pseudo_mask
            loss_background = (BCE*bkg_weights)#.mean()
            if malignant_benign_loss and allow_malignancy_loss_ct:
                bkg_weights_benign = 1 - dilated_pseudo_mask_benign
                loss_background_benign = (BCE_benign*bkg_weights_benign)
                bkg_weights_malignant = 1 - dilated_pseudo_mask_malignant
                loss_background_malignant = (BCE_malignant*bkg_weights_malignant)
            if student_out is not None:
                loss_background_student = (BCE_student*bkg_weights)
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    loss_background_benign_student = (BCE_benign_student*bkg_weights_benign)
                    loss_background_malignant_student = (BCE_malignant_student*bkg_weights_malignant)
            
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(loss_background.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == loss_background.shape[0], f'Class weights {class_weights[B].shape} do not match loss_background shape {loss_background.shape}'
                loss_foreground = loss_foreground * c_weight
                loss_background = loss_background * c_weight
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    loss_foreground_benign = loss_foreground_benign * c_weight
                    loss_background_benign = loss_background_benign * c_weight
                    loss_foreground_malignant = loss_foreground_malignant * c_weight
                    loss_background_malignant = loss_background_malignant * c_weight
                if student_out is not None:
                    loss_foreground_student = loss_foreground_student * c_weight
                    loss_background_student = loss_background_student * c_weight
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        loss_foreground_benign_student = loss_foreground_benign_student * c_weight
                        loss_background_benign_student = loss_background_benign_student * c_weight
                        loss_foreground_malignant_student = loss_foreground_malignant_student * c_weight
                        loss_background_malignant_student = loss_background_malignant_student * c_weight
            loss_foreground = loss_foreground.mean()
            loss_background = loss_background.mean()
            loss = loss_foreground + loss_background
            losses.append(loss)#BCE loss
            if malignant_benign_loss and allow_malignancy_loss_ct:
                loss_foreground_benign = loss_foreground_benign.mean()
                loss_background_benign = loss_background_benign.mean()
                loss_foreground_malignant = loss_foreground_malignant.mean()
                loss_background_malignant = loss_background_malignant.mean()
                loss_benign = loss_foreground_benign + loss_background_benign
                loss_malignant = loss_foreground_malignant + loss_background_malignant
                losses_benign_bce.append(loss_benign)
                losses_malignant_bce.append(loss_malignant)
            if student_out is not None:
                loss_foreground_student = loss_foreground_student.mean()
                loss_background_student = loss_background_student.mean()
                loss_student = loss_foreground_student + loss_background_student
                losses_student.append(loss_student)
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    loss_foreground_benign_student = loss_foreground_benign_student.mean()
                    loss_background_benign_student = loss_background_benign_student.mean()
                    loss_benign_student = loss_foreground_benign_student + loss_background_benign_student
                    losses_benign_bce_student.append(loss_benign_student)
                    loss_foreground_malignant_student = loss_foreground_malignant_student.mean()
                    loss_background_malignant_student = loss_background_malignant_student.mean()
                    loss_malignant_student = loss_foreground_malignant_student + loss_background_malignant_student
                    losses_malignant_bce_student.append(loss_malignant_student)
        else:
            #print('Using standard CE for BCE loss')
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(BCE.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == BCE.shape[0], f'Class weights {c_weight.shape} do not match BCE shape {BCE.shape}'
                BCE = BCE * c_weight
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    BCE_benign = BCE_benign * c_weight
                    BCE_malignant = BCE_malignant * c_weight
                if student_out is not None:
                    BCE_student = BCE_student * c_weight
                    if malignant_benign_loss and allow_malignancy_loss_ct:
                        BCE_benign_student = BCE_benign_student * c_weight
                        BCE_malignant_student = BCE_malignant_student * c_weight
            BCE = BCE.mean()
            losses.append(BCE)#simple mean.
            if malignant_benign_loss and allow_malignancy_loss_ct:
                BCE_benign = BCE_benign.mean()
                BCE_malignant = BCE_malignant.mean()
                losses_benign_bce.append(BCE_benign)
                losses_malignant_bce.append(BCE_malignant)
            if student_out is not None:
                BCE_student = BCE_student.mean()
                losses_student.append(BCE_student)
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    BCE_benign_student = BCE_benign_student.mean()
                    BCE_malignant_student = BCE_malignant_student.mean()
                    losses_benign_bce_student.append(BCE_benign_student)
                    losses_malignant_bce_student.append(BCE_malignant_student)

        if apply_dice_loss:
            losses_dice.append(dice_loss.mean())
            if malignant_benign_loss and allow_malignancy_loss_ct:
                losses_benign_dice.append(dice_loss_benign.mean())
                losses_malignant_dice.append(dice_loss_malignant.mean())
            if student_out is not None:
                losses_dice_student.append(dice_loss_student.mean())
                if malignant_benign_loss and allow_malignancy_loss_ct:
                    losses_benign_dice_student.append(dice_loss_benign_student.mean())
                    losses_malignant_dice_student.append(dice_loss_malignant_student.mean())

        #print(f'Ball loss time until saving debug: {time.time()-start:.2f} seconds')
        start2 = time.time()
        if counter3<10:

            counter3+=1
            os.makedirs('SanityBallLoss/'+str(counter3), exist_ok=True)
            if sigmoid:
                save_tensor_as_nifti(torch.sigmoid(x),'SanityBallLoss/'+str(counter3)+'/x')
            else:
                save_tensor_as_nifti(x,'SanityBallLoss/'+str(counter3)+'/x')
            save_tensor_as_nifti(pseudo_mask,'SanityBallLoss/'+str(counter3)+'/pseudo_mask')
            save_tensor_as_nifti(border,'SanityBallLoss/'+str(counter3)+'/border')
            save_tensor_as_nifti(tumor_seg,'SanityBallLoss/'+str(counter3)+'/tumor_segment')
            save_tensor_as_nifti(penalize.float(),'SanityBallLoss/'+str(counter3)+'/to_penalize')
            if input_tensor is not None:
                save_tensor_as_nifti(input_tensor[B].squeeze(),'SanityBallLoss/'+str(counter3)+'/input_volume')
            if sizes_slices is not None:
                #binary_slices_mask, slices_gated_probabilities sanity
                save_tensor_as_nifti(binary_slices_mask.squeeze(0).squeeze(0),'SanityBallLoss/'+str(counter3)+'/binary_slices_mask')
                save_tensor_as_nifti(slices_gated_probabilities.squeeze(0).squeeze(0),'SanityBallLoss/'+str(counter3)+'/slices_gated_probabilities')

            #save tumor volumes and diameters as yaml
            with open('SanityBallLoss/'+str(counter3)+'/tumor_volumes.yaml', 'w') as file:
                yaml.dump(tumor_volumes.tolist(), file)
            with open('SanityBallLoss/'+str(counter3)+'/tumor_diameters.yaml', 'w') as file:
                yaml.dump(tumor_diameters.tolist(), file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter3)+'/known_voxels')
            l=losses[-1].item()
            if apply_dice_loss:
                l+=losses_dice[-1].item()
            if sigmoid:
                info=f'Volume in output: {torch.sigmoid(x).sum().item()}, Volume in report: {vol}, Loss: {l}'
            else:
                info=f'Volume in output: {x.sum().item()}, Volume in report: {vol}, Loss: {l}'
            #print(info)
            #save the loss as yaml
            with open('SanityBallLoss/'+str(counter3)+'/loss.yaml', 'w') as file:
                yaml.dump(l, file)
            #save the info as yaml
            with open('SanityBallLoss/'+str(counter3)+'/info.yaml', 'w') as file:
                yaml.dump(info, file)
                
                
            # --- malignant/benign dumps (only when the malignant/benign ball loss branch is active) ---
            if malignant_benign_loss and allow_malignancy_loss_ct:
                # outputs
                if sigmoid:
                    save_tensor_as_nifti(torch.sigmoid(x_benign), os.path.join(f'SanityBallLoss/{str(counter3)}/', 'x_benign'))
                    save_tensor_as_nifti(torch.sigmoid(x_malignant), os.path.join(f'SanityBallLoss/{str(counter3)}/', 'x_malignant'))
                else:
                    save_tensor_as_nifti(x_benign, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'x_benign'))
                    save_tensor_as_nifti(x_malignant, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'x_malignant'))

                # labels (pseudo masks)
                save_tensor_as_nifti(pseudo_mask_benign, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'pseudo_mask_benign'))
                save_tensor_as_nifti(pseudo_mask_malignant, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'pseudo_mask_malignant'))

                # unknown malignancy region (if created)
                if 'pseudo_mask_unk_malignancy' in locals():
                    save_tensor_as_nifti(pseudo_mask_unk_malignancy, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'pseudo_mask_unk_malignancy'))

                # borders + penalize masks for each head
                if 'border_benign' in locals():
                    save_tensor_as_nifti(border_benign, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'border_benign'))
                if 'border_malignant' in locals():
                    save_tensor_as_nifti(border_malignant, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'border_malignant'))
                if 'penalize_benign' in locals():
                    save_tensor_as_nifti(penalize_benign.float(), os.path.join(f'SanityBallLoss/{str(counter3)}/', 'to_penalize_benign'))
                if 'penalize_malignant' in locals():
                    save_tensor_as_nifti(penalize_malignant.float(), os.path.join(f'SanityBallLoss/{str(counter3)}/', 'to_penalize_malignant'))

                # dilated masks (helpful to sanity-check background handling)
                if 'dilated_pseudo_mask_benign' in locals():
                    save_tensor_as_nifti(dilated_pseudo_mask_benign, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'dilated_pseudo_mask_benign'))
                if 'dilated_pseudo_mask_malignant' in locals():
                    save_tensor_as_nifti(dilated_pseudo_mask_malignant, os.path.join(f'SanityBallLoss/{str(counter3)}/', 'dilated_pseudo_mask_malignant'))

                # --- save sizes_malignancy info for THIS batch sample ---
                if sizes_malignancy is not None:
                    with open(os.path.join(f'SanityBallLoss/{str(counter3)}/', f'sizes_malignancy.yaml'), 'w') as file:
                        yaml.dump(sizes_malignancy[B].detach().cpu().tolist(), file)
                
            print('Saved to '+ 'SanityBallLoss/'+ str(counter3)+'/loss.yaml')
        #print(f'Ball loss time for debug saving: {time.time()-start2:.2f} seconds')
            
        if malignant_benign_loss and allow_malignancy_loss_ct:
            malignancy_loss_applied[B,tumor_channel_sel]=penalize
            applied_malignancy_losses.append(True)
        if malignant_benign_loss and not allow_malignancy_loss_ct:
            applied_malignancy_losses.append(False)
            
    refined_masks = torch.stack(refined_masks,dim=0)
    assert refined_masks.shape == out[:,0].shape, f'Refined masks shape {refined_masks.shape} does not match the lesion output shape {out.shape}; We expect the pseudo-mask to be of shape B,H,W,D, remember we penalize only one tumor type per sample'
    
    ball_loss_bce = torch.stack(losses)
    ball_loss_dice = torch.stack(losses_dice) if apply_dice_loss else torch.zeros_like(torch.stack(losses))
    if sample_weights is not None:
        assert sample_weights.shape[0] == ball_loss_bce.shape[0], f'Sample weights length {sample_weights.shape} does not match number of losses {ball_loss_bce.shape}'
        ball_loss_bce = ball_loss_bce * sample_weights
        ball_loss_dice = ball_loss_dice * sample_weights
        
    return_dict = {'ball_loss_bce':ball_loss_bce.mean(),
                    'ball_loss_dice':ball_loss_dice.mean()}
    
    if malignant_benign_loss:
        if sample_weights is not None:
            filtered_weights = [sample_weights[i] for i in range(sample_weights.shape[0]) if applied_malignancy_losses[i]]
            if len(filtered_weights)>0:
                filtered_weights = torch.stack(filtered_weights)
        
        if len(losses_benign_bce)>0:
            ball_loss_benign_bce = torch.stack(losses_benign_bce)
            if sample_weights is not None:
                ball_loss_benign_bce = ball_loss_benign_bce * filtered_weights
        else:
            ball_loss_benign_bce = torch.zeros_like(torch.stack(losses))
        if len(losses_malignant_bce)>0:
            ball_loss_malignant_bce = torch.stack(losses_malignant_bce)
            if sample_weights is not None:
                ball_loss_malignant_bce = ball_loss_malignant_bce * filtered_weights
        else:
            ball_loss_malignant_bce = torch.zeros_like(torch.stack(losses))
        if len(losses_benign_dice)>0:
            ball_loss_benign_dice = torch.stack(losses_benign_dice)
            if sample_weights is not None:
                ball_loss_benign_dice = ball_loss_benign_dice * filtered_weights
        else: 
            ball_loss_benign_dice = torch.zeros_like(torch.stack(losses))
        if len(losses_malignant_dice)>0:
            ball_loss_malignant_dice = torch.stack(losses_malignant_dice)
            if sample_weights is not None:
                ball_loss_malignant_dice = ball_loss_malignant_dice * filtered_weights
        else:
            ball_loss_malignant_dice = torch.zeros_like(torch.stack(losses))
            
        
            
        additional = {
                'ball_loss_benign_bce':ball_loss_benign_bce.mean(),
                'ball_loss_malignant_bce':ball_loss_malignant_bce.mean(),
                'ball_loss_benign_dice':ball_loss_benign_dice.mean(),
                'ball_loss_malignant_dice':ball_loss_malignant_dice.mean(),
                'malignancy_loss_applied':malignancy_loss_applied}
        
        return_dict.update(additional)
        
    if student_out is not None:
        ball_loss_bce_student = torch.stack(losses_student)
        ball_loss_dice_student = torch.stack(losses_dice_student) if apply_dice_loss else torch.zeros_like(torch.stack(losses_student))
        if sample_weights is not None:
            assert sample_weights.shape[0] == ball_loss_bce_student.shape[0], f'sample_weights length {sample_weights.shape} does not match number of losses {ball_loss_bce_student.shape}'
            ball_loss_bce_student = ball_loss_bce_student * sample_weights
            ball_loss_dice_student = ball_loss_dice_student * sample_weights
        additional_student = {
                'ball_loss_bce_student':ball_loss_bce_student.mean(),
                'ball_loss_dice_student':ball_loss_dice_student.mean()}
        if malignant_benign_loss:
            if len(losses_benign_bce_student)>0:
                ball_loss_benign_bce_student = torch.stack(losses_benign_bce_student)
                if sample_weights is not None:
                    ball_loss_benign_bce_student = ball_loss_benign_bce_student * filtered_weights
            else:
                ball_loss_benign_bce_student = torch.zeros_like(torch.stack(losses))
            if len(losses_malignant_bce_student)>0:
                ball_loss_malignant_bce_student = torch.stack(losses_malignant_bce_student)
                if sample_weights is not None:
                    ball_loss_malignant_bce_student = ball_loss_malignant_bce_student * filtered_weights
            else:
                ball_loss_malignant_bce_student = torch.zeros_like(torch.stack(losses))
            if len(losses_benign_dice_student)>0:
                ball_loss_benign_dice_student = torch.stack(losses_benign_dice_student)
                if sample_weights is not None:
                    ball_loss_benign_dice_student = ball_loss_benign_dice_student * filtered_weights
            else: 
                ball_loss_benign_dice_student = torch.zeros_like(torch.stack(losses))
            if len(losses_malignant_dice_student)>0:
                ball_loss_malignant_dice_student = torch.stack(losses_malignant_dice_student)
                if sample_weights is not None:
                    ball_loss_malignant_dice_student = ball_loss_malignant_dice_student * filtered_weights
            else:
                ball_loss_malignant_dice_student = torch.zeros_like(torch.stack(losses))
                
            additional_student_2 = {
                    'ball_loss_benign_bce_student':ball_loss_benign_bce_student.mean(),
                    'ball_loss_malignant_bce_student':ball_loss_malignant_bce_student.mean(),
                    'ball_loss_benign_dice_student':ball_loss_benign_dice_student.mean(),
                    'ball_loss_malignant_dice_student':ball_loss_malignant_dice_student.mean()} 
            additional_student.update(additional_student_2)
        return_dict.update(additional_student)
        
    if return_refined_mask:
        return return_dict, refined_masks
    else:
        return return_dict
    
    


def save_tensor_as_nifti(tensor: torch.Tensor, filename: str, dist=None):
    """
    Saves a torch tensor as a NIfTI file, assuming a voxel spacing of 1x1x1 mm.

    Args:
        tensor (torch.Tensor): A torch tensor of shape (H, W, D) or (1, H, W, D).
        filename (str): The output filename (should end with .nii or .nii.gz).
    """
    main_process = (dist is None) or (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if main_process:
        
        if 'nii.gz' not in filename:
            filename += '.nii.gz'
            
        assert len(tensor.squeeze(0).shape)==3, f"Input tensor should be 3D, got {tensor.shape}"

        # Ensure tensor is on CPU and convert to numpy array.
        np_array = tensor.detach().float().cpu().numpy()
        
        # If the tensor has an extra channel dimension, squeeze it.
        if np_array.ndim == 4 and np_array.shape[0] == 1:
            np_array = np_array.squeeze(0)
        
        # Create an identity affine (voxel sizes = 1 mm in all directions).
        affine = np.eye(4)
        
        # Create the NIfTI image and save.
        nifti_img = nib.Nifti1Image(np_array, affine)
        nib.save(nifti_img, filename)
        print(f"Saved NIfTI file to {filename}")


def apply_ball_convolution_and_save(input_size=(64, 64, 64), square_size=20,
                                    ball_diameter=15, gaussian=False, gaussian_std=3.0,
                                    output_filename='ball_convolution_output.nii.gz'):
    """
    Creates an input tensor with a centered cube (i.e., a 3D "square"),
    applies the ball convolution to it, prints the center coordinates of the input,
    prints the center of mass of the output, and saves the result as a NIfTI file.
    
    Args:
        input_size (tuple): Size of the 3D input (H, W, D).
        square_size (int): Size of the cube to insert in the center.
        ball_diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting in the ball.
        gaussian_std (float): Standard deviation for the Gaussian.
        output_filename (str): Path for the output NIfTI file.
    """
    # Create a 5D input tensor (B, C, H, W, D) filled with zeros
    x = torch.zeros((1, 1, *input_size), dtype=torch.float32)
    
    # Determine the center of the input
    center = [dim // 2 for dim in input_size]
    
    # Insert a cube (all ones) at the center of the input volume.
    half_square = square_size // 2
    x[0, 0,
    center[0]-half_square : center[0]+half_square+1,
    center[1]-half_square : center[1]+half_square+1,
    center[2]-half_square : center[2]+half_square+1] = 1.0

    # Print the center coordinates of the input
    print(f"Input center coordinates: {center}")
    
    # Apply the ball convolution over the input
    output = ball_convolution(x, ball_diameter, gaussian, gaussian_std)
    
    # Remove batch and channel dimensions and convert to a NumPy array
    output_np = output.squeeze().float().cpu().numpy()
    
    # Compute the center of mass of the output
    H, W, D = output_np.shape
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    total = np.sum(output_np)
    if total == 0:
        com = (0.0, 0.0, 0.0)
    else:
        com_x = np.sum(grid_x * output_np) / total
        com_y = np.sum(grid_y * output_np) / total
        com_z = np.sum(grid_z * output_np) / total
        com = (com_x, com_y, com_z)
    
    print(f"Center of mass of output: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
    
    # Create an identity affine (customize voxel sizes if needed)
    affine = np.eye(4)
    
    # Save the convolved output as a NIfTI file
    nii_img = nib.Nifti1Image(output_np, affine)
    nib.save(nii_img, output_filename)
    
    print(f"Saved ball convolution output to {output_filename}")


def generate_input_and_process_volume(input_size=(64, 64, 64), square_size=20, square_location='center',
                                        diameter=15, gaussian=False, gaussian_std=3.0, tumor_volume=100,
                                        output_input_filename='input_volume.nii.gz', output_mask_filename='tumor_mask.nii.gz'):
    """
    Generates an input volume with a cube (square in 3D) composed of random values, places it either in the
    center or in the corner of the volume, applies isolate_tumor, and saves both the input volume and the
    resulting tumor mask as NIfTI files.
    
    Args:
        input_size (tuple): The size of the 3D input volume (H, W, D).
        square_size (int): The edge-length of the cube to insert.
        square_location (str): Where to place the cube. Options: "center" or "corner".
        diameter (int): Diameter of the ball kernel for isolate_tumor.
        gaussian (bool): Whether to use Gaussian weighting in the ball convolution.
        gaussian_std (float): Standard deviation of the Gaussian.
        tumor_volume (int): The number of voxels to select as the tumor volume.
        output_input_filename (str): File path to save the input volume (as .nii.gz).
        output_mask_filename (str): File path to save the tumor mask (as .nii.gz).
    
    Returns:
        None
    """

    # Create a 5D input tensor with shape (B, C, H, W, D)
    x = torch.zeros((1, 1, *input_size), dtype=torch.float32)
    
    # Insert a cube with random values
    if square_location.lower() == 'center':
        # Compute center and half-size
        center = [dim // 2 for dim in input_size]
        half_square = square_size // 2
        
        # Calculate starting indices so that the cube is centered
        start_x = center[0] - half_square
        start_y = center[1] - half_square
        start_z = center[2] - half_square
        
        # Make sure we get exactly square_size elements along each dimension
        x[0, 0, start_x:start_x+square_size, start_y:start_y+square_size, start_z:start_z+square_size] = \
            torch.rand((square_size, square_size, square_size))+0.5
    
    elif square_location.lower() == 'corner':
        # Place the cube at the (0,0,0) corner
        x[0, 0, 0:square_size, 0:square_size, 0:square_size] = torch.rand((square_size, square_size, square_size))
    
    else:
        raise ValueError("square_location must be either 'center' or 'corner'")
    
    # Save the input volume as a NIfTI file (save the spatial part: (H, W, D))
    input_np = x[0, 0].float().cpu().numpy()
    affine = np.eye(4)
    input_nii = nib.Nifti1Image(input_np, affine)
    nib.save(input_nii, output_input_filename)
    print(f"Saved input volume to {output_input_filename}")
    
    # Apply isolate_tumor to the input volume
    tumor_mask = isolate_tumor(x, diameter, gaussian, gaussian_std, tumor_volume)
    
    # Save the tumor mask as a NIfTI file (convert to uint8 for a binary mask)
    tumor_mask_np = tumor_mask.float().cpu().numpy().astype(np.uint8)
    tumor_mask_nii = nib.Nifti1Image(tumor_mask_np, affine)
    nib.save(tumor_mask_nii, output_mask_filename)
    print(f"Saved tumor mask to {output_mask_filename}")