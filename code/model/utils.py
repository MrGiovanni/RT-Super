import numpy as np
import torch
import torch.nn as nn
import pdb

import sys
import os
sys.path.append(os.path.abspath(".."))
from training import losses_foundation as lf
import torch.nn as nn
import torch


from typing import Mapping, Tuple, Dict
def load_state_dict_with_overlap(
    new_module: nn.Module,
    old_state_dict: Mapping[str, torch.Tensor],
    *,
    verbose: bool = False,
    tag: str = "load_state_dict_with_overlap",
) -> Tuple[nn.Module, Dict[str, int], Tuple[list[str], list[str]]]:
    """
    Transfer weights/buffers from old_state_dict into new_module as much as possible.

    Rules:
      (1) exact match: same key + same shape
      (2) 5D/4D conv weights: if spatial/kernel dims match, copy overlap in [O,I]
      (3) 2D weights (Linear/projections): copy overlap in [O,I]
      (4) 1D tensors: copy overlap in length
      else: skip

    Returns:
      (new_module, stats, (missing_keys, unexpected_keys))
    """
    new_sd = new_module.state_dict()
    load_sd: Dict[str, torch.Tensor] = {}

    copied_exact = 0
    copied_partial = 0
    skipped = 0

    for k, v_new in new_sd.items():
        v_old = old_state_dict.get(k, None)
        if v_old is None:
            skipped += 1
            continue

        # (1) exact
        if v_old.shape == v_new.shape:
            load_sd[k] = v_old
            copied_exact += 1
            continue

        # (2) conv-like (4D/5D): match kernel dims, overlap O/I channels
        if v_old.ndim in (4, 5) and v_new.ndim == v_old.ndim:
            if v_old.shape[2:] == v_new.shape[2:]:
                tmp = v_new.clone()
                o = min(v_old.shape[0], v_new.shape[0])
                i = min(v_old.shape[1], v_new.shape[1])
                tmp[:o, :i, ...] = v_old[:o, :i, ...]
                load_sd[k] = tmp
                copied_partial += 1
                continue

        # (3) 2D weights (Linear / attention projections / MLP)
        if v_old.ndim == 2 and v_new.ndim == 2:
            tmp = v_new.clone()
            o = min(v_old.shape[0], v_new.shape[0])
            i = min(v_old.shape[1], v_new.shape[1])
            tmp[:o, :i] = v_old[:o, :i]
            load_sd[k] = tmp
            copied_partial += 1
            continue

        # (4) 1D tensors: bias, norm weight/bias, running stats, etc.
        if v_old.ndim == 1 and v_new.ndim == 1:
            tmp = v_new.clone()
            n = min(v_old.shape[0], v_new.shape[0])
            tmp[:n] = v_old[:n]
            load_sd[k] = tmp
            copied_partial += 1
            continue

        skipped += 1

    missing, unexpected = new_module.load_state_dict(load_sd, strict=False)

    stats = {
        "copied_exact": copied_exact,
        "copied_partial": copied_partial,
        "skipped": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
    }

    if verbose:
        print(f"[{tag}] {stats}", flush=True)

    return new_module, stats, (missing, unexpected)

def get_model(args, pretrain=False, classes=None, classes_cls=None):
    
    if args.dimension == '2d':
        if args.model == 'unet':
            from .dim2 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
        if args.model == 'unet++':
            from .dim2 import UNetPlusPlus
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNetPlusPlus(args.in_chan, args.classes, args.base_chan)
        if args.model == 'attention_unet':
            from .dim2 import AttentionUNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return AttentionUNet(args.in_chan, args.classes, args.base_chan)

        elif args.model == 'resunet':
            from .dim2 import UNet 
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
        elif args.model == 'daunet':
            from .dim2 import DAUNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return DAUNet(args.in_chan, args.classes, args.base_chan, block=args.block)

        elif args.model in ['medformer']:
            from .dim2 import MedFormer
            if pretrain:
                raise ValueError('No pretrain model available')
            return MedFormer(args.in_chan, args.classes, args.base_chan, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, map_size=args.map_size, proj_type=args.proj_type, act=nn.ReLU, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, aux_loss=args.aux_loss)


        elif args.model == 'transunet':
            from .dim2 import VisionTransformer as ViT_seg
            from .dim2.transunet import CONFIGS as CONFIGS_ViT_seg
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = args.classes
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(args.training_size[0]/16), int(args.training_size[1]/16))
            net = ViT_seg(config_vit, img_size=args.training_size[0], num_classes=args.classes)

            if pretrain:
                net.load_from(weights=np.load(args.init_model))

            return net
        
        elif args.model == 'swinunet':
            from .dim2 import SwinUnet
            from .dim2.swin_unet import SwinUnet_config
            config = SwinUnet_config()
            net = SwinUnet(config, img_size=224, num_classes=args.classes)
            
            if pretrain:
                net.load_from(args.init_model)

            return net



    elif args.dimension == '3d':
        if args.model == 'vnet':
            from .dim3 import VNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return VNet(args.in_chan, args.classes, scale=args.downsample_scale, baseChans=args.base_chan)
        elif args.model == 'resunet':
            from .dim3 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)

        elif args.model == 'unet':
            from .dim3 import UNet
            return UNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)
        elif args.model == 'unet++':
            from .dim3 import UNetPlusPlus
            return UNetPlusPlus(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)
        elif args.model == 'attention_unet':
            from .dim3 import AttentionUNet
            return AttentionUNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)

        elif args.model == 'medformer':
            if args.time_fusion!='early':
                from .dim3 import MedFormer
                from model.dim3.medformer import update_output_layer_onk
            else:
                from model.dim3.medformer_early_fusion import MedFormer
                from model.dim3.medformer_early_fusion import update_output_layer_onk

            class_list_seg = classes
            
                
            if (classes_cls is None) and (class_list_seg is not None):
                class_list_cls = [c for c in class_list_seg if  (('background' in c) or ('lesion' in c) or ('pnet' in c) or ('cyst' in c) or ('pdac' in c))]
            else:
                class_list_cls = classes_cls
            print('Class list seg:', class_list_seg)
            print('Class list cls:', class_list_cls)

            if classes is None:
                classes = args.classes
            else:
                classes = len(classes)
                
            net = MedFormer(args.in_chan, classes, args.base_chan, map_size=args.map_size, conv_block=args.conv_block, 
            conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, 
            fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, 
            expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, proj_type=args.proj_type, 
            norm=args.norm, act=args.act, kernel_size=args.kernel_size, scale=args.down_scale, aux_loss=args.aux_loss,
            classification_branch=args.classification_branch,gate_cls=args.cls_gate,normalize_on_gate=args.cls_gate_norm,
            class_list_seg=class_list_seg,class_list_cls=class_list_cls,aggregator_mode = args.aggregator_mode,
            cls_on_output=args.cls_on_output, clip_branch=args.clip_loss,
            attenuation_cls=args.attenuation_classifier,train_att_MLP_on_mask_only=args.train_att_MLP_on_mask_only,
            tumor_classifier= args.tumor_classifier, loss_weight_att= args.att_weight, loss_weight_cls=args.cls_weight,
            cls_on_segmentation=(args.cls_on_segmentation if hasattr(args,'cls_on_segmentation') else False),
            binarize_cls_on_segmentation=(args.binarize_cls_on_segmentation if hasattr(args,'binarize_cls_on_segmentation') else False),
            report_information_in_input = (args.train_mode != 'teacher_decoder' if hasattr(args,'train_mode') else False),
            give_tumor_size_input = (args.give_tumor_size_input if hasattr(args,'give_tumor_size_input') else False),   
            use_transformer_decoder=(not args.remove_transformer_decoder), use_dynamic_conv=(not args.remove_dynamic_conv),
            never_give_size_decoder=(args.never_give_size_decoder if hasattr(args,'never_give_size_decoder') else False),
            age_and_sex_provided = (args.age_and_sex_provided if hasattr(args,'age_and_sex_provided') else False),
            use_transformer_conv3 = (args.use_transformer_conv3 if hasattr(args,'use_transformer_conv3') else False),)

            if pretrain:
                checkpoint = torch.load(args.pretrained)
                
                if hasattr(args, 'malignancy_classification') and args.malignancy_classification:
                    try:
                        try:
                            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        except:
                            #we try to load the old checkpoint when we changed the classifier design.
                            print('Could not load the checkpoint strictly, trying to skip the classifier weights...', flush=True)
                            #net.set_aggregator()
                            state = checkpoint['model_state_dict']
                            SKIP_PREFIXES = [
                                    "cls_on_segmentation",    
                                ]
                            filtered = {k: v for k, v in state.items()
                                        if not any(p in k for p in SKIP_PREFIXES)}
                            skipped = {k: v for k, v in state.items() if any(p in k for p in SKIP_PREFIXES)}
                            net.load_state_dict(filtered, strict=False)
                            #load as best as we can
                            _, stats, (missing, unexpected) = load_state_dict_with_overlap(net, skipped,verbose=True)
                            
                    except:
                        try_again=True
                        lesion_classes = [c for c in sorted(class_list_seg) if 'lesion' in c]
                        malignants = [c.replace('lesion', 'malignant') for c in lesion_classes]
                        benigns = [c.replace('lesion', 'benign') for c in lesion_classes]
                        new_classes = class_list_seg + malignants + benigns
                        net=update_output_layer_onk(net, original_classes=class_list_seg, new_classes=new_classes, 
                                                    copy_pancreas=args.no_mask,
                                                    binarize_cls_on_segmentation=args.binarize_cls_on_segmentation)     
                        net.load_state_dict(checkpoint['model_state_dict'], strict=False)  
                        print(net)
                else:
                    try:
                        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    except:
                        #we try to load the old checkpoint when we changed the classifier design. This will skip the classifier weights though
                        print('Could not load the checkpoint strictly, trying to skip the classifier weights...', flush=True)
                        #net.set_aggregator()
                        state = checkpoint['model_state_dict']
                        SKIP_PREFIXES = [
                                "cls_on_segmentation",    
                            ]
                        filtered = {k: v for k, v in state.items()
                                    if not any(p in k for p in SKIP_PREFIXES)}
                        net.load_state_dict(filtered, strict=False)
                        skipped = {k: v for k, v in state.items() if any(p in k for p in SKIP_PREFIXES)}
                        net.load_state_dict(filtered, strict=False)
                        #load as best as we can
                        _, stats, (missing, unexpected) = load_state_dict_with_overlap(net, skipped,verbose=True)
                        
                    print('Loaded checkpoint from:',args.pretrained)
                
            if args.learnable_loss_weights:
                net.loss_wrapper = lf.MultiTaskLossWrapper(num_losses=10)#we add more params than the number of losses, the last ones will be just ignored
                #we put it inside net so it gets distributed and saved in checkpoints!
                #make params in net.loss_wrapper require grad
                if args.learnable_loss_weights:
                    for p in net.loss_wrapper.parameters():
                        p.requires_grad_(True)
                print('Using learnable loss weights')
            else:
                net.loss_wrapper = None
                print('Using fixed loss weights')
                
            if args.mtl is not None:
                if args.learnable_loss_weights:
                    raise ValueError('You cannot use learnable loss weights and a custom balancer together')
                net.balancer = get_method(args.mtl)
            else:
                net.balancer = None

            return net
            
    
        elif args.model == 'unetr':
            from .dim3 import UNETR
            model = UNETR(args.in_chan, args.classes, args.training_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed='perceptron', norm_name='instance', res_block=True)
            
            return model
        elif args.model == 'vtunet':
            from .dim3 import VTUNet
            model = VTUNet(args, args.classes)

            if pretrain:
                model.load_from(args)
            return model
        elif args.model == 'swin_unetr':
            from .dim3 import SwinUNETR
            model = SwinUNETR(args.window_size, args.in_chan, args.classes, feature_size=args.base_chan)

            if args.pretrain:
                weights = torch.load('/research/cbim/vast/yg397/ConvFormer/ConvFormer/initmodel/model_swinvit.pt')
                model.load_from(weights=weights)

            return model
        elif args.model == 'nnformer':
            from .dim3 import nnFormer
            model = nnFormer(args.window_size, input_channels=args.in_chan, num_classes=args.classes, deep_supervision=args.aux_loss)

            return model
    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')

