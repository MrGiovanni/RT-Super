from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Optional, Union

from .utils import get_block, get_norm, get_act
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb
import numpy as np

from .trans_layers import TransformerBlock

from torch.utils.checkpoint import checkpoint




from typing import Mapping, Tuple, Dict
import torch
import torch.nn as nn

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    keys = list_of_dicts[0].keys()
    for d in list_of_dicts:
        if d.keys() != keys:
            raise ValueError("All dicts must have the same keys")
    return {k: [d[k] for d in list_of_dicts] for k in keys}

def as_list(v, T):
    if v is None: return [None]*T
    assert isinstance(v, list) and len(v)==T
    return v

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

def make_classifier(
    chan_num,                        # list of channels from your encoder
    dim_head,                        # list/tuple of dim‑head per level
    conv_block,
    expansion,
    attn_drop,
    proj_drop,
    map_size,
    proj_type,
    norm,
    act,
    out_class_number,
    binarize_input=False,
    num_input_ch=None,
    class_list_cls=None,
    class_list_seg=None,
):
    """Return nn.Sequential(down_after_1, aux, down_after_2, aux, down_after_3)."""

    if num_input_ch is None:
        num_input_ch = chan_num[-1]
    aux = aux_layer()
    down_after_1 = down_block(num_input_ch, chan_num[-1], 1, 0, conv_block=conv_block,
                            kernel_size=[3,3,3], down_scale=[2,2,2], heads=1,
                            dim_head=dim_head[-1], expansion=expansion, attn_drop=attn_drop,
                            proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
    down_after_2 = down_block(chan_num[-1], chan_num[-1], 1, 0, conv_block=conv_block,
                            kernel_size=[3,3,3], down_scale=[2,2,2], heads=1,
                            dim_head=dim_head[-1], expansion=expansion, attn_drop=attn_drop,
                            proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
    down_after_3 = down_block(chan_num[-1], chan_num[-1]*2, 1, 1, conv_block=conv_block,
                            kernel_size=[3,3,3], down_scale=[2,2,2], heads=4,
                            dim_head=(chan_num[-1])//4, expansion=expansion, attn_drop=attn_drop,
                            proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
    #we use a few medformer layer to reduce the dimensionality of the last deep convolutional output (before the final layer)
    extra_layer=nn.Sequential(down_after_1,aux,down_after_2, aux, down_after_3)
    # Add a classification branch after the segmentation decoder
    classifier = ClassificationBranch(in_dim=chan_num[-1]*2, 
                                            num_classes=out_class_number, reducer=False, #we have few channels, so no need to reduce them
                                            extra_layer=extra_layer, 
                                            heads=4, dim_head=16, mlp_dim=256, reduced_dim=chan_num[-1]*2,
                                            binarize_input=binarize_input,
                                            class_list_cls=class_list_cls,
                                            class_list_seg=class_list_seg,
                                            )
    classifier.num_input_ch = num_input_ch 
    return classifier

class ClassificationBranch(nn.Module):
    def __init__(self, in_dim=160, reduced_dim=64, heads=4, dim_head=16, mlp_dim=320, 
                 num_classes=3, extra_layer=None,
                 reducer=True,
                 binarize_input=False,
                 class_list_cls=None,
                 class_list_seg=None,
                 ):
        """
        For multi-tumor classification, the voxel_choice input indicates which tumor we want to classify. It is a binary tensor,
        with the same shape as the input, and one voxel is set to 1. The rest are 0. To classify multiple tumors, just run this module
        multiple times, each time with a different voxel_choice input. At inference, you can take the centers of all tumors predicted in 
        segmentation.
        """
        
        super().__init__()
        
        # Add a reducer to lower the channel dimension
        if reducer:
            self.reducer = nn.Conv3d(in_dim, reduced_dim, kernel_size=1)
        else:
            self.reducer = nn.Identity()
        # Optionally, add an extra layer if needed
        self.extra_layer = extra_layer
        # Use a transformer block with the reduced dimension
        self.transformer = TransformerBlock(
            dim=reduced_dim,         # embedding dimension is now reduced_dim
            depth=1,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim
        )
        # Classification head from the reduced dimension to num_classes
        self.head = nn.Linear(reduced_dim, num_classes)
        self.binarize_input = binarize_input
        self.class_list_cls = class_list_cls
        self.class_list_seg = class_list_seg

    def forward(self, x, segmentation_out=None, voxel_choice=None):
        """
        x: features of the segmenter
        segmentation_out: segmentation output from the segmenter
        voxel_choice: binary tensor indicating which tumor to classify
        """
        
        if segmentation_out is not None:
            #concatenate the segmentation output with the features
            x = torch.cat((x, segmentation_out), dim=1)
        if voxel_choice is not None:
            #concatenate the voxel_choice with the features
            x = torch.cat((x, voxel_choice), dim=1)

            
        if self.binarize_input:
            #during training, we randomly choose to skip binarization, or binarize with threshold 0.5, or binarize with a random threshold between 0.1 and 0.9
            #the threshold used is added as an additional channel to the input
            #assert self.class_list_cls is inside self.class_list_seg
            if self.class_list_cls is None or self.class_list_seg is None:
                raise ValueError('class_list_cls and class_list_seg must be provided when binarize_input is True.')
            assert all(c in self.class_list_seg for c in self.class_list_cls), f"All classes in class_list_cls must be in class_list_seg, got class_list_cls:{self.class_list_cls} and class_list_seg:{self.class_list_seg}"
            lesion_classes_in_seg = [i for i,c in enumerate(self.class_list_seg) if c in self.class_list_cls]
            if self.training:
                if torch.rand(1).item() > 0.5: #50% probability we simply skip binarization during training
                    x = torch.sigmoid(x/4) #sigmoid for probabilites, scaled down to avoid saturation and preserve gradients
                    #add an additional channel of zeros, meaning that you did not binarize
                    skip_channel = torch.zeros_like(x[:, :1, :, :, :], dtype=x.dtype)
                    x = torch.cat((x, skip_channel), dim=1)
                    print('Skipping binarization during training',flush=True)
                else:
                    x = torch.sigmoid(x) #no need to care about saturation during binarization
                    #binarize
                    x=x.detach()
                    #randomly select a threshold between 0.1 and 0.9
                    th_scalar = torch.rand(1).item() * 0.8 + 0.1
                    #we onlt want this thresholding to apply to lesion classes, not organs
                    #now make threshold 0.5 for non-lesion classes
                    non_lesion_mask = torch.ones((1, x.size(1), 1, 1, 1), device=x.device, dtype=x.dtype)
                    non_lesion_mask[:, lesion_classes_in_seg, :, :, :] = 0
                    threshold = th_scalar * (1 - non_lesion_mask) + 0.5 * non_lesion_mask
                    #threshold x
                    x = (x > threshold).float()
                    #add an additional channel of threshold, meaning that you binarized with this threshold
                    threshold_channel = torch.full_like(x[:, :1, :, :, :], th_scalar)
                    x = torch.cat((x, threshold_channel), dim=1)
                    #we want to tell the model what was the threshold we used!
                    print('Binarizing with threshold', th_scalar, 'during training',flush=True)
                #check which masks are empty, we can ignore the classifier for these samples
                lesion_seg = x[:, lesion_classes_in_seg, :, :, :]
                empty_masks = (lesion_seg.sum(dim=[-1,-2,-3],keepdim=False) == 0).float()
            else:
                x = torch.sigmoid(x) #no need to care about saturation during binarization
                #inference mode: we want to run over thresholded outputs, to avoid shortcuts and ensure classification is based on segmentation
                # we want to binarize with all thresholds between 0.1 and 0.9 and average the results
                #we expand on the batch dimension
                thresholds = [0.1,0.3,0.5,0.7,0.9] #you can add more thresholds for more accuracy
                thresholded_x = []
                non_lesion_mask = torch.ones((1, x.size(1), 1, 1, 1), device=x.device, dtype=x.dtype)
                non_lesion_mask[:, lesion_classes_in_seg, :, :, :] = 0
                empty_masks_list = []
                for th_scalar in thresholds:
                    th = th_scalar * (1 - non_lesion_mask) + 0.5 * non_lesion_mask
                    x_th = (x > th).float()
                    threshold_channel = torch.full_like(x_th[:, :1, :, :, :], th_scalar)
                    x_th = torch.cat((x_th, threshold_channel), dim=1)
                    thresholded_x.append(x_th)
                    lesion_seg = x_th[:, lesion_classes_in_seg, :, :, :]
                    empty_masks = (lesion_seg.sum(dim=[-1,-2,-3],keepdim=False) == 0).float()
                    empty_masks_list.append(empty_masks)
                x = torch.cat(thresholded_x, dim=0)  # new batch size is B * len(thresholds)
                empty_masks = torch.cat(empty_masks_list, dim=0)
            
            #we need to find which lesion masks are empty
            
            
                
        # x is [B, in_dim, D, H, W]
        #print('Shape of x in classification branch:', x.shape)
        if self.extra_layer is not None:
            x, tmp_map = self.extra_layer(x)
        else:
            tmp_map = torch.zeros(1, device=x.device, dtype=x.dtype)  # dummy value so gradient flows if needed


        x = self.reducer(x)  # now x becomes [B, reduced_dim, D, H, W]

        # Flatten and rearrange to [B, L, reduced_dim]
        B, C, D, H, W = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        # Pass through the transformer block
        x = self.transformer(x)  # remains [B, L, reduced_dim]
        # Global average pooling
        x = x.mean(dim=1)  # [B, reduced_dim]
        # Classification head produces output [B, num_classes]
        x = self.head(x)
        # Ensure gradient flows through tmp_map (if needed for DDP)
        x = x + 0 * tmp_map.sum()
        
        if self.binarize_input:
            #assert that number of lesion classes in seg is equal to number of classes in cls
            assert len(lesion_classes_in_seg) == x.shape[-1], f"Number of lesion classes in segmentation ({len(lesion_classes_in_seg)}) must be equal to number of classes in classification output ({x.shape[-1]})"
            #for cases where masks were empty, we ignore the output
            x = x*(1-empty_masks) + (-10)*empty_masks  # set to large negative value for empty masks (low probability after sigmoid)
            if not self.training:
                #in inference mode, we expanded the batch dimension by len(thresholds). 
                # Now, we need to average the results back to original batch size
                orig_batch_size = x.shape[0] // len(thresholds)
                #add a new dimension for thresholds
                x = x.view(orig_batch_size, len(thresholds), -1)  # [B, len(thresholds), num_classes]
                #take probabilities
                x = torch.sigmoid(x)
                #average over thresholds
                x = x.mean(dim=1)  # [B, num_classes]
                #convert back to logits
                x = torch.logit(x, eps=1e-6)
        return x
    


import torch
import torch.nn as nn

def expand_conv3d_input(orig_conv: nn.Conv3d, init_std: float = 1e-3) -> nn.Conv3d:
    """
    Return a new Conv3d with one additional input channel.
    
    The original weights are copied exactly into the first `orig_conv.in_channels` input channels of the new layer,
    preserving channel order: the new layer's first input channels correspond to the original layer's inputs.
    The extra new channel is appended AT THE END, and its weights are initialized from N(0, init_std^2)->default is a small initialization.
    If the original layer has a bias, it is copied unchanged.
    
    Args:
        orig_conv (nn.Conv3d): the existing convolution to expand.
        init_std (float): standard deviation for initializing the new-channel weights.
        
    Returns:
        nn.Conv3d: a new convolutional layer with `orig_conv.in_channels + 1` input channels,
                    original weights preserved, and one new channel initialized small.
    """
    # Extract original parameters
    old_weights = orig_conv.weight.data.clone()
    old_bias = orig_conv.bias.data.clone() if orig_conv.bias is not None else None

    old_in_ch = orig_conv.in_channels
    out_ch = orig_conv.out_channels
    kD, kH, kW = orig_conv.kernel_size
    stride = orig_conv.stride
    padding = orig_conv.padding
    dilation = orig_conv.dilation
    groups = orig_conv.groups
    has_bias = orig_conv.bias is not None

    # Create expanded conv layer
    new_conv = nn.Conv3d(
        in_channels=old_in_ch + 1,
        out_channels=out_ch,
        kernel_size=(kD, kH, kW),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=has_bias,
    )

    # Copy weights and initialize new channel
    with torch.no_grad():
        # Copy existing weights
        new_conv.weight[:, :old_in_ch, ...].copy_(old_weights)
        # Initialize new channel weights to small random values
        torch.nn.init.normal_(
            new_conv.weight[:, old_in_ch:old_in_ch+1, ...],
            mean=0.0, std=init_std
        )
        # Copy bias if present
        if has_bias:
            new_conv.bias.copy_(old_bias)

    return new_conv
class aux_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out, map_out = x
        return out + 0 * map_out.sum()
    
class MLP(nn.Module):
    def __init__(self, in_dim=22, hidden_dim=128, out_dim=16, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


import torch
from typing import Dict, List, Optional

def build_and_mask_report_inputs_by_key(
    *,
    x: torch.Tensor,
    report_info: Dict[str, torch.Tensor],
    tumor_location_mask: torch.Tensor,   # [B,1,D,H,W]
    tumor_allowed_slices: torch.Tensor,  # [B,1,D,H,W]
    known_tumor_organ: torch.Tensor,     # [B,1,D,H,W] all 0/1
    known_tumor_slices: torch.Tensor,    # [B,1,D,H,W] all 0/1
    prob_size: float,
    training: bool,
    no_mask_training: bool, #use when you train with 0 masks
    annotated_per_voxel: Optional[torch.Tensor],  # [B] bool if mask regime else None
    use_report: Optional[torch.Tensor] = None,    # [B] bool (optional)
    # canonical token order used everywhere
    token_keys_all: Optional[List[str]] = None,
    # which keys are considered "size"
    size_keys: Optional[List[str]] = None,
    # which keys to hide when "size but no location"
    location_keys_to_hide: Optional[List[str]] = None,
    # fill values
    no_size_fill: float = 0,
    hide_token_fill: float = 0,
    no_size: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Implements:

    A) no_mask_training=True (fully without mask):
       - (1-prob_size): teacher gets report WITHOUT size (location kept)
       - (prob_size): teacher gets size BUT hides location and hides tumor_organ_name
       - never block grads

    B) no_mask_training=False (mask/slices training):
       - annotated => always give size + location (no masking), no grad block
       - unannotated:
            - (1-prob_size): no size (location kept)
            - (prob_size): size but hide location + hide organ name; block grads

    Also supports per-sample use_report: if False, everything is zeroed for that sample.
    """
    device = x.device
    B = x.shape[0]

    if token_keys_all is None:
        token_keys_all = [
            "tumor_organ_name",
            "tumor_count",
            "known_tumor_count",
            "tumor_attenuation",
            "tumor_malignancy",
            "tumor_diameters",
            "tumor_volumes",
        ]
    if size_keys is None:
        size_keys = ["tumor_diameters", "tumor_volumes"]
    if location_keys_to_hide is None:
        # “size but not location”: hide these report-derived indicators of location
        # (your description: set known_tumor_organ=no, known_tumor_slices=no and remove location/slices masks)
        location_keys_to_hide = ["tumor_organ_name"]  # you explicitly said reminder: hide organ name in tumor vectors

    # --- sanity checks for keys ---
    for k in token_keys_all:
        if k not in report_info:
            raise ValueError(f"Missing key '{k}' in report_info")

    if use_report is None:
        use_report_mask = torch.ones(B, device=device, dtype=torch.bool)
    else:
        use_report_mask = use_report.to(device=device, dtype=torch.bool).view(B)

    if (annotated_per_voxel is None) and (no_mask_training == False):
        raise ValueError("annotated_per_voxel must be provided")
        
        
    # --- hard override: no_size mode ---
    # Always hide size, never hide location, never block grads.
    if no_size:
        give_size_mask = torch.zeros(B, device=device, dtype=torch.bool)   # => size keys masked below
        hide_location_mask = torch.zeros(B, device=device, dtype=torch.bool)
        block_grad_mask = torch.zeros(B, device=device, dtype=torch.bool)
    else:
        # --- sample size decisions ---
        if training:
            rand_size = (torch.rand(B, device=device) < prob_size)
        else:
            rand_size = torch.ones(B, device=device, dtype=torch.bool)

        if no_mask_training:
            give_size_mask = rand_size
            hide_location_mask = give_size_mask.clone()
            block_grad_mask = torch.zeros(B, device=device, dtype=torch.bool)
        else:
            give_size_mask = annotated_per_voxel | (~annotated_per_voxel & rand_size)
            hide_location_mask = (~annotated_per_voxel) & give_size_mask
            block_grad_mask = hide_location_mask.clone()

    # respect use_report (if report disabled => nothing given, nothing blocked)
    give_size_mask = give_size_mask & use_report_mask
    hide_location_mask = hide_location_mask & use_report_mask
    block_grad_mask = block_grad_mask & use_report_mask

    # --- mask tokens by key ---
    report_info_masked = {k: v.clone() for k, v in report_info.items()}

    # (1) if NOT giving size => set size tokens to -0.5
    no_size_mask = (~give_size_mask).view(B, 1)  # [B,1]
    for k in size_keys:
        t = report_info_masked[k]
        report_info_masked[k] = torch.where(
            no_size_mask.expand(B, t.shape[1]),
            torch.full_like(t, no_size_fill),
            t,
        )

    # (2) if hide_location => hide organ name token (and any other “location-ish” keys)
    hide_loc_mask = hide_location_mask.view(B, 1)
    for k in location_keys_to_hide:
        t = report_info_masked[k]
        report_info_masked[k] = torch.where(
            hide_loc_mask.expand(B, t.shape[1]),
            torch.full_like(t, hide_token_fill),
            t,
        )

    # (3) if use_report=False => wipe all tokens to 0 (keep old behavior)
    for k in token_keys_all:
        t = report_info_masked[k]
        report_info_masked[k] = torch.where(
            use_report_mask.view(B, 1),
            t,
            torch.zeros_like(t),
        )

    # --- build report_vector deterministically from keys ---
    # This ensures vector matches token list exactly.
    report_vector_masked = torch.cat([report_info_masked[k] for k in token_keys_all], dim=1)  # [B, sum(Dk)]

    # --- mask location tensors when hide_location ---
    hide_loc_3d = hide_location_mask.view(B, 1, 1, 1, 1).to(dtype=tumor_location_mask.dtype)
    tumor_location_mask_masked = tumor_location_mask * (1.0 - hide_loc_3d)
    tumor_allowed_slices_masked = tumor_allowed_slices * (1.0 - hide_loc_3d)
    known_tumor_organ_masked = known_tumor_organ * (1.0 - hide_loc_3d)
    known_tumor_slices_masked = known_tumor_slices * (1.0 - hide_loc_3d)

    # also wipe when use_report=False
    use_report_3d = use_report_mask.view(B, 1, 1, 1, 1).to(dtype=tumor_location_mask.dtype)
    tumor_location_mask_masked *= use_report_3d
    tumor_allowed_slices_masked *= use_report_3d
    known_tumor_organ_masked *= use_report_3d
    known_tumor_slices_masked *= use_report_3d

    # these are your 1D flags you later append to tokens/vectors
    known_tumor_organ_1d = known_tumor_organ_masked[:, 0, 0, 0, 0].unsqueeze(-1)  # [B,1]
    known_tumor_slices_1d = known_tumor_slices_masked[:, 0, 0, 0, 0].unsqueeze(-1) # [B,1]

    print(f'Give size mask: {give_size_mask}', flush=True)

    return {
        # masks
        "give_size_mask": give_size_mask,                 # [B]
        "hide_location_mask": hide_location_mask,         # [B]
        "block_grad_mask": block_grad_mask,               # [B]

        # masked “tokens” + vector
        "token_keys_all": torch.tensor([0], device=device),  # placeholder (keys are python list; returned below too)
        "token_keys_all_py": token_keys_all,              # python list
        "report_info_masked": report_info_masked,          # dict of masked tokens
        "report_vector_masked": report_vector_masked,      # [B, sum(Dk)]

        # masked spatial conditioning
        "tumor_location_mask_masked": tumor_location_mask_masked,
        "tumor_allowed_slices_masked": tumor_allowed_slices_masked,
        "known_tumor_organ_masked": known_tumor_organ_masked,
        "known_tumor_slices_masked": known_tumor_slices_masked,

        # convenient 1D flags
        "known_tumor_organ_1d": known_tumor_organ_1d,      # [B,1]
        "known_tumor_slices_1d": known_tumor_slices_1d,    # [B,1]
    }


class MedFormer(nn.Module):
    
    def __init__(self, 
        in_chan, # number of input channels PER TIME POINT
        num_classes, 
        base_chan=32, 
        map_size=[3,3,3], 
        conv_block='BasicBlock', 
        conv_num=[2,0,0,0, 0,0,2,2], 
        trans_num=[0,2,4,6, 4,2,0,0], 
        chan_num=[64, 128, 256, 320, 256, 128, 64, 32], 
        num_heads=[1,4,8,10, 8,4,1,1], 
        fusion_depth=2, 
        fusion_dim=320, 
        fusion_heads=10, 
        expansion=4, attn_drop=0., 
        proj_drop=0., 
        proj_type='depthwise', 
        norm='in', 
        act='relu', 
        kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]], 
        scale=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]], 
        aux_loss=False,
        classification_branch=False,
        gate_cls=False,normalize_on_gate=False,
        class_list_seg=None,class_list_cls=None,
        aggregator_mode = None,  # deprecated
        cls_on_output=False,  # Whether to add a classification branch after the segmentation decoder as well
        cls_on_segmentation=False,  # Whether to add a classification branch after the segmentation output (and in deep supervision)
        binarize_cls_on_segmentation=False, # Whether to binarize the input of the classifier on segmentation output
        clip_branch=False,
        clip_feats=768,
        attenuation_cls='none',
        train_att_MLP_on_mask_only=False,
        tumor_classifier=False,
        loss_weight_att = 1,
        loss_weight_cls=1,
        report_information_in_input=True,
        MLP_out_dim = 16,
        bottleneck_transformer=False,
        report_informed_decoder=False,
        report_token_dims = [1,1,1,10,10,30,10,1,1],
        prob_size = 0.3,
        use_transformer_decoder=True,
        use_dynamic_conv=True,
        give_tumor_size_input=False,
        never_give_size_decoder=False,
        age_and_sex_provided=False,
        use_transformer_conv3 = False,
        time_points = 1,
        time_fusion = 'early',
        ):
        super().__init__()
        
        self.give_tumor_size_input = give_tumor_size_input
        self.time_points = time_points
        self.time_fusion = time_fusion

        #if conv_block == 'BasicBlock':
        dim_head = [chan_num[i]//num_heads[i] for i in range(8)]
        

        
        conv_block = get_block(conv_block)
        norm = get_norm(norm)
        act = get_act(act)

        self._cls_build_cfg = dict(
            chan_num=chan_num,
            dim_head=dim_head,
            conv_block=conv_block,
            expansion=expansion,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            map_size=map_size,
            proj_type=proj_type,
            norm=norm,
            act=act,
        )
        
        if time_points>1 and time_fusion=='early':
            in_chan = (in_chan+1) * time_points #+1 because we want to add the date as channel
            out_chan = num_classes * time_points
        else:
            out_chan = num_classes
        
        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, block=conv_block, kernel_size=kernel_size[0], norm=norm, act=act)
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block=conv_block, kernel_size=kernel_size[1], down_scale=scale[0], norm=norm, act=act, map_generate=False)
        
        # down2 down3 down4 apply the B-MHA blocks
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block=conv_block, kernel_size=kernel_size[2], down_scale=scale[1], heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block=conv_block, kernel_size=kernel_size[3], down_scale=scale[2], heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)


        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block=conv_block, kernel_size=kernel_size[3], up_scale=scale[3], heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)

        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block=conv_block, kernel_size=kernel_size[2], up_scale=scale[2], heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True, no_map_out=True)

        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block=conv_block, kernel_size=kernel_size[1], up_scale=scale[1], norm=norm, act=act, map_shortcut=False)

        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block=conv_block, kernel_size=kernel_size[0], up_scale=scale[0], norm=norm, act=act, map_shortcut=False)

        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv3d(chan_num[5], out_chan, kernel_size=1)

        self.outc = nn.Conv3d(chan_num[7], out_chan, kernel_size=1)

        if classification_branch:
            self.classification_branch = ClassificationBranch(num_classes=len(class_list_cls),
                                                              extra_layer=down_block(chan_num[3], chan_num[3]//2, 0, 1, conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=4, dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True))
        else:
            self.classification_branch = None
            
        if clip_branch:
            self.clip_branch = ClassificationBranch(num_classes=clip_feats,
                                                    extra_layer=down_block(chan_num[3], chan_num[3]//2, 0, 1, conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=4, dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True))
        else:
            self.clip_branch = None

        if gate_cls:
            raise ValueError('Deprecated: gate_cls is no longer supported (it did not work).')
        else:
            self.gate_cls = None

        
        
        if cls_on_output:
            self.cls_on_output = make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                out_class_number=len(class_list_cls))
        else:
            self.cls_on_output = None
         
        self.age_and_sex_provided = age_and_sex_provided   
        if cls_on_segmentation:
            if age_and_sex_provided:
                num_input_ch = num_classes+2
            else:
                num_input_ch = num_classes
            if binarize_cls_on_segmentation:
                num_input_ch += 1
            self.cls_on_segmentation = make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                out_class_number=len(class_list_cls),num_input_ch = num_input_ch,
                                                binarize_input=binarize_cls_on_segmentation,
                                                class_list_seg=class_list_seg,class_list_cls=class_list_cls)
            #binarize_cls_on_segmentation: makes the input of the classifier binary in 50% of cases during training, and with multiple thresholds averaged during inference
        else:
            self.cls_on_segmentation = None
        
        if class_list_seg is not None:
            patterns = ('lesion', 'pdac', 'pnet', 'cyst')
            tumor_cls = [c for c in class_list_seg if any(p in c for p in patterns)]
        
        if attenuation_cls!= 'none':
            assert attenuation_cls in ['none', 'simple', 'MLP','large','neuron'], f"Invalid attenuation_cls: {attenuation_cls}. Choose from ['none', 'simple', 'MLP','neuron','large']"
            if attenuation_cls == 'MLP':
                self.att_classifier = attribute_classifier(class_list_seg,train_on_mask_only=train_att_MLP_on_mask_only, loss_weight=loss_weight_att)
            elif attenuation_cls == 'neuron':
                self.att_classifier = attribute_classifier(class_list_seg,train_on_mask_only=train_att_MLP_on_mask_only, loss_weight=loss_weight_att,
                                                           out_features=1)
            elif attenuation_cls == 'simple':
                self.att_classifier = simple_classifier(class_list_seg)
            elif attenuation_cls == 'large':
                m =  make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                       out_class_number=3*len(tumor_cls), num_input_ch = (2*len(tumor_cls)+1))
                self.att_classifier=attribute_classifier(class_list_seg,train_on_mask_only=train_att_MLP_on_mask_only,
                                                                   model=m,calculate_HU=False, loss_weight=loss_weight_att)
        else:
            self.att_classifier = None
            
        if tumor_classifier:
            raise ValueError('Deprecated: tumor_classifier is no longer supported (it did not work).')
        else:
            self.tumor_classifier = None
            
        self.report_information_in_input = report_information_in_input
        if report_information_in_input:
            self.MLP_out_dim = MLP_out_dim
            if self.give_tumor_size_input:
                in_dim=63
            else:
                in_dim = 23
            if self.age_and_sex_provided:
                in_dim += 2
            self.report_processing_MLP = MLP(in_dim=in_dim, hidden_dim=128, out_dim=MLP_out_dim, drop=0.2)
        else:
            self.report_processing_MLP = None
            
        if bottleneck_transformer:
            self.create_bottleneck_transformer()
        else:
            self.bottleneck_transformer = None
            
        self.chan_num = chan_num
        
            
        self.report_informed_decoder = report_informed_decoder
        self.report_token_dims = report_token_dims[:]
        if self.report_informed_decoder:
            self.create_report_informed_decoder(use_transformer_decoder=use_transformer_decoder,
                                                use_dynamic_conv=use_dynamic_conv,
                                                age_and_sex_provided=age_and_sex_provided,
                                                use_transformer_conv3=use_transformer_conv3)
        self.prob_size = prob_size
        self.never_give_size_decoder = never_give_size_decoder
        
    def create_bottleneck_transformer(self):
        self.bottleneck_transformer = BottleneckTransformer(self.chan_num[3])
        #nn.Conv3d(self.chan_num[3], self.chan_num[3], kernel_size=1)
        
        
    def create_report_informed_decoder(self,use_transformer_decoder=True,use_dynamic_conv=True,
                                       age_and_sex_provided=False,use_transformer_conv3=False):
        self.report_informed_decoder = True
        #num classes: read the output size for self.outc
        num_classes = self.outc.out_channels
        if age_and_sex_provided:
            token_dims = self.report_token_dims[:]+[1,1]
        else:
            token_dims = self.report_token_dims[:]
            
        report_embed_dim = [25, 25, 65, 65, 65]
            
        if self.time_points>1 and self.time_fusion=='early':
            token_dims = token_dims * self.time_points
            report_embed_dim = [dim * self.time_points for dim in report_embed_dim]
            
            
        self.report_decoder = DynamicUNetDecoder3D(num_classes = num_classes,
                                                   feature_in_channels = self.chan_num[3:], 
                                                   use_pointwise_conv1=use_transformer_decoder,
                                                   use_full_conv3=use_dynamic_conv,
                                                   age_and_sex_provided=age_and_sex_provided,
                                                   report_token_dims=token_dims,
                                                   report_embed_dim = report_embed_dim,
                                                   use_transformer_conv3=use_transformer_conv3,
                                                   time_points=self.time_points,
                                                   time_fusion=self.time_fusion,)
        
    def forward_pre_encoder(self, x, report_info, stage_1_out, use_report, name,
                    debugging,annotated_per_voxel,no_mask_training, age, sex):
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        block_grad_mask = torch.zeros(B, device=x.device, dtype=torch.bool)
            
        if self.age_and_sex_provided:
            # age, sex: [B,1]

            if self.training:
                keep_age = (torch.rand(B, 1, device=device) > 0.5).to(dtype)  # 50% keep
                keep_sex = (torch.rand(B, 1, device=device) > 0.5).to(dtype)
            else:
                keep_age = torch.ones(B, 1, device=device, dtype=dtype)  # keep all during inference
                keep_sex = torch.ones(B, 1, device=device, dtype=dtype)

            age = age.to(device=device, dtype=dtype) * keep_age
            sex = sex.to(device=device, dtype=dtype) * keep_sex
        
            
        if report_info is not None and ((not self.report_information_in_input) and (not self.report_informed_decoder)):
            raise ValueError('report_info is provided but report_information_in_input is False and report_informed_decoder is also false.')
        
        if report_info is not None:
            report_vector = report_info['tumor_info_vector_organ_count_attenuation_malignancy']
            assert report_vector.dim() == 2, f"report_vector must be 2D [B, L], got {report_vector.shape}"
            assert report_vector.shape[0] == x.shape[0], f"Batch mismatch: x has B={x.shape[0]}, report_vector has {report_vector.shape[0]}"
            tumor_location_mask = report_info['tumor_location_mask'].unsqueeze(1)
            tumor_allowed_slices = report_info['tumor_allowed_slices'].unsqueeze(1)
            assert tumor_location_mask.shape == (x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]), f'Tumor location mask shape {tumor_location_mask.shape} does not match expected shape {(x.shape[0],1,x.shape[2],x.shape[3],x.shape[4])}'
            assert tumor_allowed_slices.shape == (x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]), f'Tumor allowed slices shape {tumor_allowed_slices.shape} does not match expected shape {(x.shape[0],1,x.shape[2],x.shape[3],x.shape[4])}'
            known_tumor_organ = torch.ones_like(tumor_location_mask)
            is_all_ones = (tumor_allowed_slices > 0.999).flatten(1).all(dim=1)  # [B]
            known_tumor_slices = (~is_all_ones)[:, None, None, None, None]
            known_tumor_slices = known_tumor_slices.expand_as(tumor_allowed_slices)
            if debugging:#debug
                if name is not None:
                    print(f'Processing report for case: {name}', flush=True)
                print(f'Report vector shape: {report_vector.shape}', flush=True)
                print(f'Report vector batch 0: {report_vector[0]}', flush=True)
                print(f'Mean of tumor location mask: {tumor_location_mask.mean().item()}', flush=True)
                print(f'Mean of tumor allowed slices: {tumor_allowed_slices.mean().item()}', flush=True)
        else:
            #no information given.
            report_vector = torch.zeros([x.shape[0],63], device=x.device, dtype=x.dtype)
            tumor_location_mask = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device, dtype=x.dtype)
            tumor_allowed_slices = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device, dtype=x.dtype)
            known_tumor_organ = torch.zeros_like(tumor_location_mask)
            known_tumor_slices = torch.zeros_like(tumor_allowed_slices)
            
        if self.report_information_in_input:
            
            if report_info is not None:
                device = x.device
                B = x.shape[0]
                if self.report_informed_decoder:
                    raise ValueError(f'The masking below (build_and_mask_report_inputs_by_key) is not compatible with report_informed_decoder. It risks masking twice.')
                # annotated_per_voxel is expected to be list[bool] or tensor[bool] of shape [B]
                if isinstance(annotated_per_voxel, str):
                    raise ValueError(f"We must know whether each sample has a GT mask (annotated_per_voxel). We got the string '{annotated_per_voxel}' instead of a list.")

                if isinstance(annotated_per_voxel, (list, tuple)):
                    annotated_mask = torch.tensor(annotated_per_voxel, device=device, dtype=torch.bool)
                else:
                    annotated_mask = annotated_per_voxel.to(device=device, dtype=torch.bool)
                assert annotated_mask.shape == (B,), f"annotated_per_voxel must be shape [B], got {annotated_mask.shape}"
                if isinstance(no_mask_training, str):
                    raise ValueError(f"no_mask_training must be bool, got {no_mask_training!r}")
                if not isinstance(no_mask_training, bool):
                    raise ValueError(f"no_mask_training must be bool, got {type(no_mask_training)}")
                
                m = build_and_mask_report_inputs_by_key(
                        x=x,
                        report_info=report_info,  # dict with tumor_* keys
                        tumor_location_mask=tumor_location_mask,
                        tumor_allowed_slices=tumor_allowed_slices,
                        known_tumor_organ=known_tumor_organ,
                        known_tumor_slices=known_tumor_slices,
                        prob_size=self.prob_size,
                        training=self.training,
                        no_mask_training=no_mask_training,
                        annotated_per_voxel=annotated_mask if (not no_mask_training) else None,
                        use_report=use_report,
                        no_size = not(self.give_tumor_size_input)
                    )

                report_info_masked   = m["report_info_masked"]
                report_vector_masked = m["report_vector_masked"]
                tumor_location_mask_masked  = m["tumor_location_mask_masked"]
                tumor_allowed_slices_masked = m["tumor_allowed_slices_masked"]
                known_tumor_organ_masked    = m["known_tumor_organ_masked"]
                known_tumor_slices_masked   = m["known_tumor_slices_masked"]
                block_grad_mask      = m["block_grad_mask"]
                
            else:
                report_vector_masked = torch.zeros([x.shape[0],63], device=x.device, dtype=x.dtype)
                tumor_location_mask_masked = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device, dtype=x.dtype)
                tumor_allowed_slices_masked = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device, dtype=x.dtype)
                known_tumor_organ_masked = torch.zeros_like(tumor_location_mask)
                known_tumor_slices_masked = torch.zeros_like(tumor_allowed_slices)
                block_grad_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
                use_report = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
            
            if not self.give_tumor_size_input:
                report_vector_masked = report_vector_masked[:,:23]
                
            if self.age_and_sex_provided:
                print(f'Provided to input age {age}, sex {sex}', flush=True)
                report_vector_masked = torch.cat([report_vector_masked,age,sex],dim=1)
            
            report_embed = self.report_processing_MLP(report_vector_masked) #maybe have size, maybe not, we have masking above
            # expand to spatial dims and add to x
            B, C, D, H, W = x.shape
            report_embed = report_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 16, 1, 1, 1]
            report_embed = report_embed.expand(-1, -1, D, H, W)  # [B, 16, D, H, W]
            #print(f'Average report embed values: {report_embed.mean().item()}')
            report_info = torch.cat([
                                    tumor_location_mask_masked,
                                    known_tumor_organ_masked,
                                    tumor_allowed_slices_masked,
                                    known_tumor_slices_masked,
                                    report_embed], dim=1)  # concatenate along channel dimension
            #use_report shows which batch should use reports
            if use_report is not None:
                mask_reports = use_report.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1, 1]
                if not self.age_and_sex_provided:
                    report_info = report_info * mask_reports  # mask all channels, including the reprtt embed
                else:
                    print(f'Provided to input age {age}, sex {sex}', flush=True)
                    #never mask the report embed, because it may carry information from age and sex, which can be available in inference
                    report_info = torch.cat([
                                        tumor_location_mask_masked*mask_reports,
                                        known_tumor_organ_masked*mask_reports,
                                        tumor_allowed_slices_masked*mask_reports,
                                        known_tumor_slices_masked*mask_reports,
                                        report_embed], dim=1)#do not mask report embed, we may give the size.
                
            x = torch.cat([x, report_info], dim=1)  # concatenate along channel dimension
            
            
        return x,block_grad_mask,report_info,report_vector,use_report,known_tumor_organ,known_tumor_slices,tumor_location_mask,tumor_allowed_slices
    
    def aux_processing(self,aux_out,block_grad_mask,labels,cut_attenuation_grad,x,age,sex):
        B = x.shape[0]
        
        if self.report_information_in_input and not self.report_informed_decoder and block_grad_mask.any():
            mm = block_grad_mask.view(B,1,1,1,1).to(aux_out.dtype)
            aux_out = aux_out.detach() * mm + aux_out * (1 - mm) + aux_out * 0.0  # optional “optimizer complain” guard
                
        if self.att_classifier is not None:
            aux_att = self.att_classifier(x[:,0].unsqueeze(1),aux_out,labels,cut_attenuation_grad=cut_attenuation_grad)
        else:
            aux_att = None
        if self.cls_on_segmentation is not None:
            if self.age_and_sex_provided:
                print(f'Provided to output aux age {age}, sex {sex}', flush=True)
                aux_out_age_sex = torch.cat((aux_out, 
                                            age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:]),
                                            sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])),
                                            dim=1)
                y_class_on_seg_aux = self.cls_on_segmentation(aux_out_age_sex)
            else:
                y_class_on_seg_aux = self.cls_on_segmentation(aux_out)
        else:
            y_class_on_seg_aux = None
        return aux_out, aux_att, y_class_on_seg_aux
    
    def out_processing(self,out,block_grad_mask,labels,cut_attenuation_grad,x,age,sex):
        B = x.shape[0]
        if self.report_information_in_input and not self.report_informed_decoder and block_grad_mask.any():
            mm = block_grad_mask.view(B,1,1,1,1).to(out.dtype)
            out = out.detach() * mm + out * (1 - mm) + out * 0.0  # optional “optimizer complain” guard
        
        if self.cls_on_segmentation is not None:
            if self.age_and_sex_provided:
                print(f'Provided to output age {age}, sex {sex}', flush=True)
                out_age_sex = torch.cat((out, 
                                        age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:]),
                                        sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])),
                                        dim=1)
                y_class_on_seg = self.cls_on_segmentation(out_age_sex)
            else:
                y_class_on_seg = self.cls_on_segmentation(out)
        else:
            y_class_on_seg = None

        if self.att_classifier is not None:
            out_att = self.att_classifier(x[:,0].unsqueeze(1),out,labels,cut_attenuation_grad=cut_attenuation_grad)
        else:
            out_att = None
        return out, out_att, y_class_on_seg
            
        #tumor classifier and gate_cls removed (deprecated)
        
    def process_for_report_informed_decoder(self,x,no_mask_training,annotated_per_voxel,report_info,
                                            report_vector,use_report,known_tumor_organ,known_tumor_slices,tumor_location_mask,
                                            tumor_allowed_slices,unet_decoder_features,age, sex):
        B = x.shape[0]
        device = x.device

        # default: no one blocks grad
        block_grad_mask = torch.zeros(B, device=device, dtype=torch.bool)

        if no_mask_training or self.never_give_size_decoder:
            # nobody gets size
            give_size_mask = torch.zeros(B, device=device, dtype=torch.bool)
            print('Not giving size', flush=True)
        else:
            if not self.training:
                # everyone gets size at eval
                give_size_mask = torch.ones(B, device=device, dtype=torch.bool)
            else:
                # annotated_per_voxel is expected to be list[bool] or tensor[bool] of shape [B]
                if isinstance(annotated_per_voxel, str):
                    
                    raise ValueError("We must know whether each sample has a GT mask (annotated_per_voxel).")

                if isinstance(annotated_per_voxel, (list, tuple)):
                    annotated_mask = torch.tensor(annotated_per_voxel, device=device, dtype=torch.bool)
                else:
                    annotated_mask = annotated_per_voxel.to(device=device, dtype=torch.bool)

                assert annotated_mask.shape == (B,), f"annotated_per_voxel must be shape [B], got {annotated_mask.shape}"
                

                if isinstance(no_mask_training, str):
                    raise ValueError("We must know whether no_mask_training is enabled.")
                if not isinstance(no_mask_training, bool):
                    raise ValueError(f"no_mask_training must be bool, got {type(no_mask_training)}")
                
                # if annotated -> always give size
                # if not annotated -> give size with prob self.prob_size
                rand_mask = (torch.rand(B, device=device) < self.prob_size)  # [B]
                give_size_mask = annotated_mask | (~annotated_mask & rand_mask)

                # block grads only for samples that are NOT annotated but we decided to give size
                block_grad_mask = (~annotated_mask) & give_size_mask
                
        
        report_vector_all = []
        report_tokens_list_all = []
        TOKEN_KEYS_NOSIZE = [
        "tumor_organ_name", "tumor_count", "known_tumor_count",
        "tumor_attenuation", "tumor_malignancy",
        ]
        TOKEN_KEYS_ALL = TOKEN_KEYS_NOSIZE + ["tumor_diameters", "tumor_volumes"]
        
        #tumor_organ_name, tumor_count, known_tumor_count, tumor_attenuation, tumor_malignancy, tumor_diameters, tumor_volumes
        
        if report_info is None:
            #shapes: report_token_dims = [1,1,1,10,10,30,10,1,1]
            #keys: 'tumor_organ_name','tumor_count','known_tumor_count','tumor_attenuation','tumor_malignancy','tumor_diameters','tumor_volumes'
            report_info = {
                'tumor_organ_name': torch.zeros([x.shape[0],1], device=x.device, dtype=x.dtype),
                'tumor_count': torch.zeros([x.shape[0],1], device=x.device, dtype=x.dtype),
                'known_tumor_count': torch.zeros([x.shape[0],1], device=x.device, dtype=x.dtype),
                'tumor_attenuation': torch.zeros([x.shape[0],10], device=x.device, dtype=x.dtype),
                'tumor_malignancy': torch.zeros([x.shape[0],10], device=x.device, dtype=x.dtype),
                'tumor_diameters': torch.zeros([x.shape[0],30], device=x.device, dtype=x.dtype),
                'tumor_volumes': torch.zeros([x.shape[0],10], device=x.device, dtype=x.dtype),
            }
        #assert all keys are in report_info
        for key in TOKEN_KEYS_ALL:
            if key not in report_info:
                raise ValueError(f'Key {key} not found in report_info for report_informed_decoder.')
        if use_report is not None:
            use_report_mask = use_report.to(device=device, dtype=torch.bool)  # [B]

            # if use_report=False, we should behave as "don't give size"
            give_size_mask = give_size_mask & use_report_mask
            # also: if report not used, no need to block grads for that sample
            block_grad_mask = block_grad_mask & use_report_mask
            
            mask_reports_3d = use_report.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, 1, 1, 1]
            mask_reports_1d = use_report.unsqueeze(-1).float()  # [B, 1]
            tmp={}
            for key in TOKEN_KEYS_ALL:
                tmp[key] = report_info[key] * mask_reports_1d
            report_info = tmp
            report_vector = report_vector * mask_reports_1d
            known_tumor_organ = known_tumor_organ * mask_reports_3d
            known_tumor_slices = known_tumor_slices * mask_reports_3d
            tumor_location_mask = tumor_location_mask * mask_reports_3d
            tumor_allowed_slices = tumor_allowed_slices * mask_reports_3d
            
        # report_vector: [B, 63], where first 23 are non-size
        report_vector_masked = report_vector.clone()

        # For samples where give_size_mask==False, set ONLY the size dims to -0.5
        not_give_size_mask = (~give_size_mask).unsqueeze(1)  # [B,1] bool
        report_vector_masked[:, 23:] = torch.where(
            not_give_size_mask.expand(-1, report_vector_masked.shape[1] - 23),
            torch.full_like(report_vector_masked[:, 23:], -0.5),
            report_vector_masked[:, 23:],
        )
        
        report_info_masked = dict(report_info)  # shallow copy is fine (values are tensors)

        if "tumor_diameters" in report_info_masked:
            diam = report_info_masked["tumor_diameters"]  # [B,30]
            report_info_masked["tumor_diameters"] = torch.where(
                not_give_size_mask.expand(-1, diam.shape[1]),
                torch.full_like(diam, -0.5),
                diam,
            )

        if "tumor_volumes" in report_info_masked:
            vol = report_info_masked["tumor_volumes"]  # [B,10]
            report_info_masked["tumor_volumes"] = torch.where(
                not_give_size_mask.expand(-1, vol.shape[1]),
                torch.full_like(vol, -0.5),
                vol,
            )
            
        report_info_no_size = dict(report_info) 
        report_info_no_size['tumor_diameters'] = report_info_no_size['tumor_diameters'] * 0.0 - 0.5
        report_info_no_size['tumor_volumes'] = report_info_no_size['tumor_volumes'] * 0.0 - 0.5
            
        for i in range(len(unet_decoder_features)):
            if i < self.report_decoder.deep_supervision_level:
                #before deep supervision is applied, we never send size
                report_vector_x = report_vector[:,:23] #do not give size
                known_tumor_organ_x = known_tumor_organ[:,0,0,0,0].unsqueeze(-1)
                known_tumor_slices_x = known_tumor_slices[:,0,0,0,0].unsqueeze(-1)
                #add both to the report vector
                if not self.age_and_sex_provided:
                    report_vector_x = torch.cat([report_vector_x, known_tumor_organ_x, known_tumor_slices_x], dim=1)
                else:
                    report_vector_x = torch.cat([report_vector_x, known_tumor_organ_x, known_tumor_slices_x,
                                                age,sex], dim=1)
                report_vector_all.append(report_vector_x)
                report_tokens = [report_info_no_size[k] for k in TOKEN_KEYS_ALL]
                report_tokens += [known_tumor_organ_x, known_tumor_slices_x]
                if self.age_and_sex_provided:
                    print(f'Provided to decoder aux age {age}, sex {sex}', flush=True)
                    report_tokens += [age, sex]
                report_tokens_list_all.append(report_tokens)
            else:
                #now we can give the size info.
                report_vector_x = report_vector_masked
                known_tumor_organ_x = known_tumor_organ[:,0,0,0,0].unsqueeze(-1)
                known_tumor_slices_x = known_tumor_slices[:,0,0,0,0].unsqueeze(-1)
                if not self.age_and_sex_provided:
                    report_vector_x = torch.cat([report_vector_x, known_tumor_organ_x, known_tumor_slices_x], dim=1)
                else:
                    report_vector_x = torch.cat([report_vector_x, known_tumor_organ_x, known_tumor_slices_x,
                                                age,sex], dim=1)
                report_vector_all.append(report_vector_x)
                report_tokens = [report_info_masked[k] for k in TOKEN_KEYS_ALL]
                report_tokens += [known_tumor_organ_x, known_tumor_slices_x]
                if self.age_and_sex_provided:
                    print(f'Provided to decoder aux age {age}, sex {sex}', flush=True)
                    report_tokens += [age, sex]
                report_tokens_list_all.append(report_tokens)
        return report_vector_all, report_tokens_list_all, block_grad_mask
    
    def process_after_report_decoder(self,block_grad_mask,out_refined,aux_out_refined,x,labels,cut_attenuation_grad,age,sex):
        B = x.shape[0]
        out_report_decoder = {}
        
        out_report_decoder["refined_segmentation"] = [out_refined, aux_out_refined] if self.aux_loss else out_refined
        
        if block_grad_mask.any():
            #out is: 'refined_segmentation': [features, aux_out_refined], we need to detach only the features
            m = block_grad_mask.view(B, 1, 1, 1, 1).to(out_refined.dtype)  # broadcast
            out_refined = out_refined.detach() * m + out_refined * (1 - m) # no change in value (forward), but grad is selectively blocked per sample
            out_report_decoder["refined_segmentation"] = [out_refined, aux_out_refined]
            
        #add classifiers on top of the refined segmentation if needed
        if self.aux_loss:        
            if self.att_classifier is not None:
                aux_att_refined = self.att_classifier(x[:,0].unsqueeze(1),aux_out_refined,labels,cut_attenuation_grad=cut_attenuation_grad)
            else:
                aux_att_refined = None
            if self.cls_on_segmentation is not None:
                if self.age_and_sex_provided:
                    aux_out_age_sex_refined = torch.cat((aux_out_refined, 
                                                age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:]),
                                                sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])),
                                                dim=1)
                    y_class_on_seg_aux_refined = self.cls_on_segmentation(aux_out_age_sex_refined)
                else:
                    y_class_on_seg_aux_refined = self.cls_on_segmentation(aux_out_refined)
            else:
                y_class_on_seg_aux_refined = None
        else:
            aux_att_refined = None
            y_class_on_seg_aux_refined = None
            out_report_decoder["refined_segmentation"] = out_refined
            aux_out_refined = None  # keep local consistent too
            
        #now apply to outputs as well
        if self.cls_on_segmentation is not None:
            if self.age_and_sex_provided:
                out_age_sex_refined = torch.cat((out_refined, 
                                            age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:]),
                                            sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])),
                                            dim=1)
                y_class_on_seg_refined = self.cls_on_segmentation(out_age_sex_refined)
            else:
                y_class_on_seg_refined = self.cls_on_segmentation(out_refined)
        else:
            y_class_on_seg_refined = None
            
        if self.att_classifier is not None:
            out_att_refined = self.att_classifier(x[:,0].unsqueeze(1),out_refined,labels,cut_attenuation_grad=cut_attenuation_grad)
        else:
            out_att_refined = None
        
        #add new keys to out_report_decoder
        out_report_decoder['attenuation_refined'] = [aux_att_refined,out_att_refined] if aux_att_refined is not None else out_att_refined
        out_report_decoder['classification on segmentation_refined'] = [y_class_on_seg_aux_refined,y_class_on_seg_refined] if y_class_on_seg_aux_refined is not None else y_class_on_seg_refined
            
        assert block_grad_mask.shape == (B,)
        
        return out_report_decoder
    
    def forward(self, x, report_info=None, stage_1_out=None,labels=None, cut_attenuation_grad = False, use_report = None, name=None,
                debugging=False, bottleneck_transformer=None,annotated_per_voxel='unknown',no_mask_training='unknown',
                skip_report_informed_decoder=False, age=None, sex=None, dates = None, cropped_organs = None):
        if self.time_points>1 and self.time_fusion=='early':
            # assert all inputs are list; if None, make them list of None:
            x = as_list(x, self.time_points)
            labels = as_list(labels, self.time_points)
            use_report = as_list(use_report, self.time_points)
            age = as_list(age, self.time_points)
            sex = as_list(sex, self.time_points)
            name = as_list(name, self.time_points) if name is not None else [None]*self.time_points
            report_info = as_list(report_info, self.time_points)
            annotated_per_voxel = as_list(annotated_per_voxel, self.time_points)
            dates = as_list(dates, self.time_points)#size [time_points], B,1
            cropped_organs = as_list(cropped_organs, self.time_points)
            
            x_original = x.copy()
            
            # check if cropped_organs is the same for all time points
            # cropped_organs is a list of length time points, each element in the list is a list of strings, which must match
            assert all(co == cropped_organs[0] for co in cropped_organs), f"Cropped organs must be the same across time points. Got {cropped_organs}"
            
            preprocessed_inputs, block_grad_mask, known_tumor_organ,known_tumor_slices,tumor_location_mask,tumor_allowed_slices, report_vector = [], [], [], [], [], [], []
            for i in range(self.time_points):
                x_t,block_grad_mask_t,report_info[i],report_vector_t,use_report[i],known_tumor_organ_t,known_tumor_slices_t,tumor_location_mask_t,tumor_allowed_slices_t = self.forward_pre_encoder(
                    x=x[i], report_info=report_info[i], stage_1_out=None, use_report = use_report[i], name=name[i],
                    debugging=debugging, annotated_per_voxel=annotated_per_voxel[i],
                    no_mask_training=no_mask_training, age=age[i], sex=sex[i])
                preprocessed_inputs.append(x_t)
                block_grad_mask.append(block_grad_mask_t)
                known_tumor_organ.append(known_tumor_organ_t)
                known_tumor_slices.append(known_tumor_slices_t)
                tumor_location_mask.append(tumor_location_mask_t)
                tumor_allowed_slices.append(tumor_allowed_slices_t)
                report_vector.append(report_vector_t)
                
            mixed_inputs = []
            for i in range(len(preprocessed_inputs)):
                mixed_inputs.append(preprocessed_inputs[i])
                date = dates[i].to(device=x[i].device, dtype=x[i].dtype)
                # convert date (B,1) into a tensor matching shape of x[i]
                date = date.view(-1,1,1,1,1).expand(-1,1,*x[i].shape[-3:])
                mixed_inputs.append(date)
            x = torch.cat(mixed_inputs, dim=1)
        else:
            x,block_grad_mask,report_info,report_vector,use_report,known_tumor_organ,known_tumor_slices,tumor_location_mask,tumor_allowed_slices = self.forward_pre_encoder(
                    x=x, report_info=report_info, stage_1_out=stage_1_out, use_report = use_report, name=name,
                    debugging=debugging, annotated_per_voxel=annotated_per_voxel,
                    no_mask_training=no_mask_training, age=age, sex=sex)
        
        #run model as usual
        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3)
        if bottleneck_transformer is not None:
            x4 = bottleneck_transformer(x4)
        elif self.bottleneck_transformer is not None:
            x4 = self.bottleneck_transformer(x4)
        bottleneck_features = x4
        if self.classification_branch:
            if self.time_points>1 and self.time_fusion=='early':
                raise NotImplementedError("Classification branch not implemented for time_points > 1 and time_fusion == 'early'")
            y_class = self.classification_branch(x4)
        else:
            y_class = None
        if self.clip_branch:
            if self.time_points>1 and self.time_fusion=='early':
                raise NotImplementedError("CLIP branch not implemented for time_points > 1 and time_fusion == 'early'")
            y_clip = self.clip_branch(x4)
        else:
            y_clip = None
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        
        u1, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        u2, semantic_map = self.up2(u1, x2, semantic_map, map_list[0])
        
        mid_decoder_features = u2
        
        #deep supervision branch, run one per time point
        if self.aux_loss:
            aux_out = self.aux_out(mid_decoder_features)
            aux_out = F.interpolate(aux_out, size=x.shape[-3:], mode='trilinear', align_corners=True)
    
            #auxiliary output and loss
            if self.time_points>1 and self.time_fusion=='early':
                num_classes = self.outc.out_channels/self.time_points
                #assert num_classes is integer
                assert self.outc.out_channels % self.time_points == 0
                num_classes = self.outc.out_channels // self.time_points
                aux_att = []
                y_class_on_seg_aux = []
                aux_out = [aux_out[:, i*num_classes: (i+1)*num_classes] for i in range(self.time_points)]
                for i in range(self.time_points):
                    aux_out[i], aux_att_t, y_class_on_seg_aux_t = self.aux_processing(
                        aux_out=aux_out[i],
                        block_grad_mask=block_grad_mask[i],
                        labels=labels[i],
                        cut_attenuation_grad=cut_attenuation_grad,
                        x=x_original[i],
                        age=age[i],
                        sex=sex[i])
                    aux_att.append(aux_att_t)
                    y_class_on_seg_aux.append(y_class_on_seg_aux_t)
            else:
                aux_out, aux_att, y_class_on_seg_aux = self.aux_processing(
                        aux_out=aux_out,
                        block_grad_mask=block_grad_mask,
                        labels=labels,
                        cut_attenuation_grad=cut_attenuation_grad,
                        x=x,
                        age=age,
                        sex=sex)
        else:
            aux_out = None
            aux_att = None
            y_class_on_seg_aux = None
        
        #finalize MedFormer decoder
        u3, semantic_map = self.up3(u2, x1, semantic_map, None)
        u4, semantic_map = self.up4(u3, x0, semantic_map, None)
        out = u4
        if self.cls_on_output is not None:
            if self.time_points>1 and self.time_fusion=='early':
                raise NotImplementedError("cls_on_output branch not implemented for time_points > 1 and time_fusion == 'early'")
            #print('Shape of out is:', out.shape)
            y_class_2 = self.cls_on_output(out)
            #y_class is now [B, num_classes_cls], where num_classes_cls is the number of classes in the classification branch
        else:
            y_class_2 = None
        out_features = out
        unet_decoder_features = [x4, u1, u2, u3, u4]
        out = self.outc(out)
        
        #losses on top of output
        if self.time_points>1 and self.time_fusion=='early':
            num_classes = self.outc.out_channels/self.time_points
            #assert num_classes is integer
            assert self.outc.out_channels % self.time_points == 0
            num_classes = self.outc.out_channels // self.time_points
            out_att = []
            y_class_on_seg = []
            out = [out[:, i*num_classes: (i+1)*num_classes] for i in range(self.time_points)]
            for i in range(self.time_points):
                out[i], out_att_t, y_class_on_seg_t = self.out_processing(
                    out=out[i],
                    block_grad_mask=block_grad_mask[i],
                    labels=labels[i],
                    cut_attenuation_grad=cut_attenuation_grad,
                    x=x_original[i],
                    age=age[i],
                    sex=sex[i])
                out_att.append(out_att_t)
                y_class_on_seg.append(y_class_on_seg_t)
        else:
            out, out_att, y_class_on_seg = self.out_processing(
                    out=out,
                    block_grad_mask=block_grad_mask,
                    labels=labels,
                    cut_attenuation_grad=cut_attenuation_grad,
                    x=x,
                    age=age,
                    sex=sex)
            
        #report-informed decoder
        if self.report_informed_decoder and not skip_report_informed_decoder:
            if self.time_points>1 and self.time_fusion=='early':
                report_vector_all, report_tokens_list_all, block_grad_mask = [], [], []
                for i in range(self.time_points):
                    report_vector_all_t, report_tokens_list_all_t, block_grad_mask_t = self.process_for_report_informed_decoder(
                        x=x_original[i],no_mask_training=no_mask_training,
                        annotated_per_voxel=annotated_per_voxel[i],report_info=report_info[i],report_vector=report_vector[i],
                        use_report=use_report[i],known_tumor_organ=known_tumor_organ[i],known_tumor_slices=known_tumor_slices[i],
                        tumor_location_mask=tumor_location_mask[i],
                        tumor_allowed_slices=tumor_allowed_slices[i],
                        unet_decoder_features=unet_decoder_features,
                        age=age[i], sex=sex[i])
                    report_vector_all.append(report_vector_all_t)
                    report_tokens_list_all.append(report_tokens_list_all_t)
                    block_grad_mask.append(block_grad_mask_t)
                report_vector_concat = []
                # report_vector_all is now a list of lists, with shape time, feature_level, tensors of shape [B, feature_dim]
                for i in range(len(report_vector_all[0])):#one item per feature level
                    report_vector_concat.append(torch.cat([report_vector_all[t][i] for t in range(self.time_points)], dim=-1))
                report_vector_all = report_vector_concat
                #report tokens is now a list (time) of lists (feature level) of lists (report attributes), where each attribute is a 1D tensor
                report_tokens_concat = []
                for lvl in range(len(report_tokens_list_all[0])):  # feature levels
                    tokens_lvl = []
                    for t in range(self.time_points):
                        tokens_lvl += report_tokens_list_all[t][lvl]  # concat tokens for this level across time
                    report_tokens_concat.append(tokens_lvl)
                report_tokens_list_all = report_tokens_concat
                mask_for_decoder = torch.sigmoid(torch.cat(out,dim=1))
                tumor_location_mask_for_decoder = torch.cat(tumor_location_mask,dim=1)
                tumor_allowed_slices_for_decoder = torch.cat(tumor_allowed_slices,dim=1)
            else:
                report_vector_all, report_tokens_list_all, block_grad_mask = self.process_for_report_informed_decoder(
                        x=x,no_mask_training=no_mask_training,
                        annotated_per_voxel=annotated_per_voxel,report_info=report_info,report_vector=report_vector,
                        use_report=use_report,known_tumor_organ=known_tumor_organ,known_tumor_slices=known_tumor_slices,
                        tumor_location_mask=tumor_location_mask,
                        tumor_allowed_slices=tumor_allowed_slices,
                        unet_decoder_features=unet_decoder_features,
                        age=age, sex=sex)
                mask_for_decoder = torch.sigmoid(out)
                tumor_location_mask_for_decoder = tumor_location_mask
                tumor_allowed_slices_for_decoder = tumor_allowed_slices
                
            out_report_decoder = self.report_decoder(unet_decoder_features=unet_decoder_features,
                                        out_mask=mask_for_decoder,
                                        report_vector_all=report_vector_all,
                                        report_tokens_list_all=report_tokens_list_all,
                                        tumor_location_mask=tumor_location_mask_for_decoder,
                                        tumor_allowed_slices=tumor_allowed_slices_for_decoder,)
            
            out_refined, aux_out_refined = out_report_decoder["refined_segmentation"]
            
            #now process outputs
            if self.time_points>1 and self.time_fusion=='early':
                out_refined_t, aux_out_refined_t = [], []
                for i in range(self.time_points):
                    out_refined_t.append(out_refined[:, i*num_classes: (i+1)*num_classes])
                    aux_out_refined_t.append(aux_out_refined[:, i*num_classes: (i+1)*num_classes])
                out_refined = out_refined_t
                aux_out_refined = aux_out_refined_t
                out_report_decoder_t = []
                for i in range(self.time_points):
                    out_report_decoder_t.append(self.process_after_report_decoder(
                        block_grad_mask=block_grad_mask[i],
                        out_refined=out_refined_t[i],
                        aux_out_refined=aux_out_refined_t[i],
                        x=x_original[i],labels=labels[i],cut_attenuation_grad=cut_attenuation_grad,
                        age=age[i],sex=sex[i]))
                out_report_decoder = out_report_decoder_t
            else:
                out_report_decoder = self.process_after_report_decoder(
                        block_grad_mask=block_grad_mask,
                        out_refined=out_refined,
                        aux_out_refined=aux_out_refined,
                        x=x,labels=labels,
                        cut_attenuation_grad=cut_attenuation_grad,
                        age=age,sex=sex)
        else:
            out_report_decoder = None
        
        if self.time_points>1 and self.time_fusion=='early':
            returns = {}
            def assert_size(val):
                if val is not None:
                    assert isinstance(val, list) and len(val) == self.time_points, f"Expected list of length {self.time_points}, got {type(val)} with length {len(val) if isinstance(val, list) else 'N/A'}"
            for val in [out, aux_out, y_class, y_class_2,
                        y_clip, aux_att, out_att, y_class_on_seg_aux,
                        y_class_on_seg,out_report_decoder]:
                assert_size(val)
            
            for i in range(self.time_points):
                returns[f'time_point_{str(i)}'] = self.prepare_return(
                    out[i], aux_out=aux_out[i] if aux_out is not None else None, 
                    y_class=y_class[i] if y_class is not None else None,
                    y_class_2=y_class_2[i] if y_class_2 is not None else None,
                    y_clip=y_clip [i] if y_clip is not None else None,
                    aux_att=aux_att[i] if aux_att is not None else None,
                    out_att=out_att[i] if out_att is not None else None,
                    y_tumor=None,
                    y_class_on_seg_aux=y_class_on_seg_aux[i] if y_class_on_seg_aux is not None else None,
                    y_class_on_seg=y_class_on_seg[i] if y_class_on_seg is not None else None,
                    bottleneck_features=bottleneck_features,
                    mid_decoder_features=mid_decoder_features,
                    out_features=out_features,
                    out_report_decoder=out_report_decoder[i] if out_report_decoder is not None else None
                )
        else:
            returns = self.prepare_return(out, aux_out=aux_out, y_class=y_class, y_class_2=y_class_2,
                                    y_clip=y_clip, aux_att=aux_att, out_att=out_att, y_tumor=None,
                                    y_class_on_seg_aux=y_class_on_seg_aux, y_class_on_seg=y_class_on_seg,
                                    bottleneck_features=bottleneck_features,
                                    mid_decoder_features=mid_decoder_features,
                                    out_features=out_features,
                                    out_report_decoder=out_report_decoder)
            
        return returns
    
    def prepare_return(
        self,
        out,
        aux_out=None,
        y_class=None,
        y_class_2=None,
        y_clip=None,
        aux_att=None,
        out_att=None,
        y_tumor=None,
        y_class_on_seg_aux=None,
        y_class_on_seg=None,
        bottleneck_features=None,
        mid_decoder_features=None,
        out_features=None,
        out_report_decoder = None,
    ):
        # 1) Build the primary output exactly as before
        primary = [out, aux_out] if self.aux_loss else out
        
        retur = {'segmentation': primary}
        
        if self.classification_branch:
            retur['classification'] = y_class
        if self.cls_on_output is not None:
            retur['classification on output'] = y_class_2
        if self.clip_branch:
            retur['clip'] = y_clip
        if out_att is not None:
            retur['attenuation'] = [aux_att,out_att] if aux_att is not None else out_att
        if y_tumor is not None:
            retur['tumor diameters'] = y_tumor
        if self.cls_on_segmentation is not None:
            retur['classification on segmentation'] = [y_class_on_seg_aux,y_class_on_seg] if y_class_on_seg_aux is not None else y_class_on_seg
        if bottleneck_features is not None:
            retur['bottleneck_features'] = bottleneck_features
        if mid_decoder_features is not None:
            retur['mid_decoder_features'] = mid_decoder_features
        if out_features is not None:
            retur['out_features'] = out_features
            
        if out_report_decoder is not None:
            #update retur with all keys from out_report_decoder
            for k, v in out_report_decoder.items():
                retur[k] = v

        return retur
                    

    def forward_original(self, x, report_info=None, stage_1_out=None,labels=None, cut_attenuation_grad = False, use_report = None, name=None,
                    debugging=False, bottleneck_transformer=None,annotated_per_voxel='unknown',no_mask_training='unknown',
                    skip_report_informed_decoder=False, age=None, sex=None):
            
        if self.age_and_sex_provided:
            # age, sex: [B,1]
            B = x.shape[0]
            device = x.device
            dtype = x.dtype

            if self.training:
                keep_age = (torch.rand(B, 1, device=device) > 0.5).to(dtype)  # 50% keep
                keep_sex = (torch.rand(B, 1, device=device) > 0.5).to(dtype)
            else:
                keep_age = torch.ones(B, 1, device=device, dtype=dtype)  # keep all during inference
                keep_sex = torch.ones(B, 1, device=device, dtype=dtype)

            age = age.to(device=device, dtype=dtype) * keep_age
            sex = sex.to(device=device, dtype=dtype) * keep_sex
        
            
        if report_info is not None and ((not self.report_information_in_input) and (not self.report_informed_decoder)):
            raise ValueError('report_info is provided but report_information_in_input is False and report_informed_decoder is also false.')
        
        if report_info is not None:
            report_vector = report_info['tumor_info_vector_organ_count_attenuation_malignancy']
            assert report_vector.dim() == 2, f"report_vector must be 2D [B, L], got {report_vector.shape}"
            assert report_vector.shape[0] == x.shape[0], f"Batch mismatch: x has B={x.shape[0]}, report_vector has {report_vector.shape[0]}"
            tumor_location_mask = report_info['tumor_location_mask'].unsqueeze(1)
            tumor_allowed_slices = report_info['tumor_allowed_slices'].unsqueeze(1)
            assert tumor_location_mask.shape == (x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]), f'Tumor location mask shape {tumor_location_mask.shape} does not match expected shape {(x.shape[0],1,x.shape[2],x.shape[3],x.shape[4])}'
            assert tumor_allowed_slices.shape == (x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]), f'Tumor allowed slices shape {tumor_allowed_slices.shape} does not match expected shape {(x.shape[0],1,x.shape[2],x.shape[3],x.shape[4])}'
            known_tumor_organ = torch.ones_like(tumor_location_mask)
            is_all_ones = (tumor_allowed_slices > 0.999).flatten(1).all(dim=1)  # [B]
            known_tumor_slices = (~is_all_ones)[:, None, None, None, None]
            known_tumor_slices = known_tumor_slices.expand_as(tumor_allowed_slices)
            if debugging:#debug
                if name is not None:
                    print(f'Processing report for case: {name}', flush=True)
                print(f'Report vector shape: {report_vector.shape}', flush=True)
                print(f'Report vector batch 0: {report_vector[0]}', flush=True)
                print(f'Mean of tumor location mask: {tumor_location_mask.mean().item()}', flush=True)
                print(f'Mean of tumor allowed slices: {tumor_allowed_slices.mean().item()}', flush=True)
        else:
            #no information given.
            report_vector = torch.zeros([x.shape[0],63], device=x.device, dtype=x.dtype)
            tumor_location_mask = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device, dtype=x.dtype)
            tumor_allowed_slices = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device, dtype=x.dtype)
            known_tumor_organ = torch.zeros_like(tumor_location_mask)
            known_tumor_slices = torch.zeros_like(tumor_allowed_slices)
            
        if self.report_information_in_input:
            
            if report_info is not None:
                device = x.device
                B = x.shape[0]
                if self.report_informed_decoder:
                    raise ValueError(f'The masking below (build_and_mask_report_inputs_by_key) is not compatible with report_informed_decoder. It risks masking twice.')
                # annotated_per_voxel is expected to be list[bool] or tensor[bool] of shape [B]
                if isinstance(annotated_per_voxel, str):
                    raise ValueError(f"We must know whether each sample has a GT mask (annotated_per_voxel). We got the string '{annotated_per_voxel}' instead of a list.")

                if isinstance(annotated_per_voxel, (list, tuple)):
                    annotated_mask = torch.tensor(annotated_per_voxel, device=device, dtype=torch.bool)
                else:
                    annotated_mask = annotated_per_voxel.to(device=device, dtype=torch.bool)
                assert annotated_mask.shape == (B,), f"annotated_per_voxel must be shape [B], got {annotated_mask.shape}"
                
                m = build_and_mask_report_inputs_by_key(
                        x=x,
                        report_info=report_info,  # dict with tumor_* keys
                        tumor_location_mask=tumor_location_mask,
                        tumor_allowed_slices=tumor_allowed_slices,
                        known_tumor_organ=known_tumor_organ,
                        known_tumor_slices=known_tumor_slices,
                        prob_size=self.prob_size,
                        training=self.training,
                        no_mask_training=no_mask_training,
                        annotated_per_voxel=annotated_mask if (not no_mask_training) else None,
                        use_report=use_report,
                        no_size = not(self.give_tumor_size_input)
                    )

                report_info_masked   = m["report_info_masked"]
                report_vector_masked = m["report_vector_masked"]
                tumor_location_mask_masked  = m["tumor_location_mask_masked"]
                tumor_allowed_slices_masked = m["tumor_allowed_slices_masked"]
                known_tumor_organ_masked    = m["known_tumor_organ_masked"]
                known_tumor_slices_masked   = m["known_tumor_slices_masked"]
                block_grad_mask      = m["block_grad_mask"]
                
            else:
                report_vector_masked = torch.zeros([x.shape[0],63], device=x.device, dtype=x.dtype)
                tumor_location_mask_masked = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device, dtype=x.dtype)
                tumor_allowed_slices_masked = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device, dtype=x.dtype)
                known_tumor_organ_masked = torch.zeros_like(tumor_location_mask)
                known_tumor_slices_masked = torch.zeros_like(tumor_allowed_slices)
                block_grad_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
                use_report = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
            
            if not self.give_tumor_size_input:
                report_vector_masked = report_vector_masked[:,:23]
                
            if self.age_and_sex_provided:
                print(f'Provided to input age {age}, sex {sex}', flush=True)
                report_vector_masked = torch.cat([report_vector_masked,age,sex],dim=1)
            
            report_embed = self.report_processing_MLP(report_vector_masked) #maybe have size, maybe not, we have masking above
            # expand to spatial dims and add to x
            B, C, D, H, W = x.shape
            report_embed = report_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 16, 1, 1, 1]
            report_embed = report_embed.expand(-1, -1, D, H, W)  # [B, 16, D, H, W]
            #print(f'Average report embed values: {report_embed.mean().item()}')
            report_info = torch.cat([
                                    tumor_location_mask_masked,
                                    known_tumor_organ_masked,
                                    tumor_allowed_slices_masked,
                                    known_tumor_slices_masked,
                                    report_embed], dim=1)  # concatenate along channel dimension
            #use_report shows which batch should use reports
            if use_report is not None:
                mask_reports = use_report.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1, 1]
                if not self.age_and_sex_provided:
                    report_info = report_info * mask_reports  # mask all channels, including the reprtt embed
                else:
                    print(f'Provided to input age {age}, sex {sex}', flush=True)
                    #never mask the report embed, because it may carry information from age and sex, which can be available in inference
                    report_info = torch.cat([
                                        tumor_location_mask_masked*mask_reports,
                                        known_tumor_organ_masked*mask_reports,
                                        tumor_allowed_slices_masked*mask_reports,
                                        known_tumor_slices_masked*mask_reports,
                                        report_embed], dim=1)#do not mask report embed, we may give the size.
                
            x = torch.cat([x, report_info], dim=1)  # concatenate along channel dimension
    
    

        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3)
        
        if bottleneck_transformer is not None:
            x4 = bottleneck_transformer(x4)
        elif self.bottleneck_transformer is not None:
            x4 = self.bottleneck_transformer(x4)
            
        bottleneck_features = x4

        if self.classification_branch:
            y_class = self.classification_branch(x4)
        else:
            y_class = None
            
        if self.clip_branch:
            y_clip = self.clip_branch(x4)
        else:
            y_clip = None
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        

        u1, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        u2, semantic_map = self.up2(u1, x2, semantic_map, map_list[0])
        
        mid_decoder_features = u2

        if self.aux_loss:
            aux_out = self.aux_out(mid_decoder_features)
            aux_out = F.interpolate(aux_out, size=x.shape[-3:], mode='trilinear', align_corners=True)
            
            if self.report_information_in_input and not self.report_informed_decoder and block_grad_mask.any():
                mm = block_grad_mask.view(B,1,1,1,1).to(aux_out.dtype)
                aux_out = aux_out.detach() * mm + aux_out * (1 - mm) + aux_out * 0.0  # optional “optimizer complain” guard
                    
            if self.att_classifier is not None:
                aux_att = self.att_classifier(x[:,0].unsqueeze(1),aux_out,labels,cut_attenuation_grad=cut_attenuation_grad)
            else:
                aux_att = None
            if self.cls_on_segmentation is not None:
                if self.age_and_sex_provided:
                    print(f'Provided to output aux age {age}, sex {sex}', flush=True)
                    aux_out_age_sex = torch.cat((aux_out, 
                                                age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:]),
                                                sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])),
                                                dim=1)
                    y_class_on_seg_aux = self.cls_on_segmentation(aux_out_age_sex)
                else:
                    y_class_on_seg_aux = self.cls_on_segmentation(aux_out)
            else:
                y_class_on_seg_aux = None
        else:
            aux_out = None
            aux_att = None
            y_class_on_seg_aux = None

        u3, semantic_map = self.up3(u2, x1, semantic_map, None)
        u4, semantic_map = self.up4(u3, x0, semantic_map, None)
        out = u4
        
        if self.cls_on_output is not None:
            #print('Shape of out is:', out.shape)
            y_class_2 = self.cls_on_output(out)
            #y_class is now [B, num_classes_cls], where num_classes_cls is the number of classes in the classification branch
        else:
            y_class_2 = None
    
        out_features = out
        unet_decoder_features = [x4, u1, u2, u3, u4]
        
        out = self.outc(out)
        
        if self.report_information_in_input and not self.report_informed_decoder and block_grad_mask.any():
            mm = block_grad_mask.view(B,1,1,1,1).to(out.dtype)
            out = out.detach() * mm + out * (1 - mm) + out * 0.0  # optional “optimizer complain” guard
        
        if self.cls_on_segmentation is not None:
            if self.age_and_sex_provided:
                print(f'Provided to output age {age}, sex {sex}', flush=True)
                out_age_sex = torch.cat((out, 
                                        age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:]),
                                        sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])),
                                        dim=1)
                y_class_on_seg = self.cls_on_segmentation(out_age_sex)
            else:
                y_class_on_seg = self.cls_on_segmentation(out)
        else:
            y_class_on_seg = None

        if self.gate_cls:
            out = self.gate_cls(torch.sigmoid(out), torch.sigmoid(y_class))
            #assert out is in the range 0-1
            assert (out>=0).all() and (out<=1).all(), f'Gate out is not in the range 0-1, its min is: {out.min()}, its max is: {out.max()}'
            #remember to remove sigmoid from the dice loss and not use BCE with logits loss---we are already applying sigmoid here. Can you do gating before the sigmoid?
            #raise ValueError('Max and min of out:', out.max(), out.min())
            
        if self.att_classifier is not None:
            out_att = self.att_classifier(x[:,0].unsqueeze(1),out,labels,cut_attenuation_grad=cut_attenuation_grad)
        else:
            out_att = None
        
        if self.tumor_classifier is not None:
            y_tumor = self.tumor_classifier(x[:,0].unsqueeze(1), out, labels, cut_attenuation_grad=cut_attenuation_grad)
        else:
            y_tumor = None
            
        if self.report_informed_decoder and not skip_report_informed_decoder:
            B = x.shape[0]
            device = x.device

            
            
            # default: no one blocks grad
            block_grad_mask = torch.zeros(B, device=device, dtype=torch.bool)

            if no_mask_training or self.never_give_size_decoder:
                # nobody gets size
                give_size_mask = torch.zeros(B, device=device, dtype=torch.bool)
                print('Not giving size', flush=True)
            else:
                if not self.training:
                    # everyone gets size at eval
                    give_size_mask = torch.ones(B, device=device, dtype=torch.bool)
                else:
                    # annotated_per_voxel is expected to be list[bool] or tensor[bool] of shape [B]
                    if isinstance(annotated_per_voxel, str):
                        
                        raise ValueError("We must know whether each sample has a GT mask (annotated_per_voxel).")

                    if isinstance(annotated_per_voxel, (list, tuple)):
                        annotated_mask = torch.tensor(annotated_per_voxel, device=device, dtype=torch.bool)
                    else:
                        annotated_mask = annotated_per_voxel.to(device=device, dtype=torch.bool)

                    assert annotated_mask.shape == (B,), f"annotated_per_voxel must be shape [B], got {annotated_mask.shape}"
                    

                    if isinstance(no_mask_training, str):
                        raise ValueError("We must know whether no_mask_training is enabled.")
                    if not isinstance(no_mask_training, bool):
                        raise ValueError(f"no_mask_training must be bool, got {type(no_mask_training)}")
                    
                    # if annotated -> always give size
                    # if not annotated -> give size with prob self.prob_size
                    rand_mask = (torch.rand(B, device=device) < self.prob_size)  # [B]
                    give_size_mask = annotated_mask | (~annotated_mask & rand_mask)

                    # block grads only for samples that are NOT annotated but we decided to give size
                    block_grad_mask = (~annotated_mask) & give_size_mask
                    
            
            report_vector_all = []
            report_tokens_list_all = []
            TOKEN_KEYS_NOSIZE = [
            "tumor_organ_name", "tumor_count", "known_tumor_count",
            "tumor_attenuation", "tumor_malignancy",
            ]
            TOKEN_KEYS_ALL = TOKEN_KEYS_NOSIZE + ["tumor_diameters", "tumor_volumes"]
            
            #tumor_organ_name, tumor_count, known_tumor_count, tumor_attenuation, tumor_malignancy, tumor_diameters, tumor_volumes
            
            if report_info is None:
                #shapes: report_token_dims = [1,1,1,10,10,30,10,1,1]
                #keys: 'tumor_organ_name','tumor_count','known_tumor_count','tumor_attenuation','tumor_malignancy','tumor_diameters','tumor_volumes'
                report_info = {
                    'tumor_organ_name': torch.zeros([x.shape[0],1], device=x.device, dtype=x.dtype),
                    'tumor_count': torch.zeros([x.shape[0],1], device=x.device, dtype=x.dtype),
                    'known_tumor_count': torch.zeros([x.shape[0],1], device=x.device, dtype=x.dtype),
                    'tumor_attenuation': torch.zeros([x.shape[0],10], device=x.device, dtype=x.dtype),
                    'tumor_malignancy': torch.zeros([x.shape[0],10], device=x.device, dtype=x.dtype),
                    'tumor_diameters': torch.zeros([x.shape[0],30], device=x.device, dtype=x.dtype),
                    'tumor_volumes': torch.zeros([x.shape[0],10], device=x.device, dtype=x.dtype),
                }
            #assert all keys are in report_info
            for key in TOKEN_KEYS_ALL:
                if key not in report_info:
                    raise ValueError(f'Key {key} not found in report_info for report_informed_decoder.')
            if use_report is not None:
                use_report_mask = use_report.to(device=device, dtype=torch.bool)  # [B]

                # if use_report=False, we should behave as "don't give size"
                give_size_mask = give_size_mask & use_report_mask
                # also: if report not used, no need to block grads for that sample
                block_grad_mask = block_grad_mask & use_report_mask
                
                mask_reports_3d = use_report.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, 1, 1, 1]
                mask_reports_1d = use_report.unsqueeze(-1).float()  # [B, 1]
                tmp={}
                for key in TOKEN_KEYS_ALL:
                    tmp[key] = report_info[key] * mask_reports_1d
                report_info = tmp
                report_vector = report_vector * mask_reports_1d
                known_tumor_organ = known_tumor_organ * mask_reports_3d
                known_tumor_slices = known_tumor_slices * mask_reports_3d
                tumor_location_mask = tumor_location_mask * mask_reports_3d
                tumor_allowed_slices = tumor_allowed_slices * mask_reports_3d
                
            # report_vector: [B, 63], where first 23 are non-size
            report_vector_masked = report_vector.clone()

            # For samples where give_size_mask==False, set ONLY the size dims to -0.5
            not_give_size_mask = (~give_size_mask).unsqueeze(1)  # [B,1] bool
            report_vector_masked[:, 23:] = torch.where(
                not_give_size_mask.expand(-1, report_vector_masked.shape[1] - 23),
                torch.full_like(report_vector_masked[:, 23:], -0.5),
                report_vector_masked[:, 23:],
            )
            
            report_info_masked = dict(report_info)  # shallow copy is fine (values are tensors)

            if "tumor_diameters" in report_info_masked:
                diam = report_info_masked["tumor_diameters"]  # [B,30]
                report_info_masked["tumor_diameters"] = torch.where(
                    not_give_size_mask.expand(-1, diam.shape[1]),
                    torch.full_like(diam, -0.5),
                    diam,
                )

            if "tumor_volumes" in report_info_masked:
                vol = report_info_masked["tumor_volumes"]  # [B,10]
                report_info_masked["tumor_volumes"] = torch.where(
                    not_give_size_mask.expand(-1, vol.shape[1]),
                    torch.full_like(vol, -0.5),
                    vol,
                )
                
            report_info_no_size = dict(report_info) 
            report_info_no_size['tumor_diameters'] = report_info_no_size['tumor_diameters'] * 0.0 - 0.5
            report_info_no_size['tumor_volumes'] = report_info_no_size['tumor_volumes'] * 0.0 - 0.5
                
            for i in range(len(unet_decoder_features)):
                if i < self.report_decoder.deep_supervision_level:
                    #before deep supervision is applied, we never send size
                    report_vector_x = report_vector[:,:23] #do not give size
                    known_tumor_organ_x = known_tumor_organ[:,0,0,0,0].unsqueeze(-1)
                    known_tumor_slices_x = known_tumor_slices[:,0,0,0,0].unsqueeze(-1)
                    #add both to the report vector
                    if not self.age_and_sex_provided:
                        report_vector_x = torch.cat([report_vector_x, known_tumor_organ_x, known_tumor_slices_x], dim=1)
                    else:
                        report_vector_x = torch.cat([report_vector_x, known_tumor_organ_x, known_tumor_slices_x,
                                                    age,sex], dim=1)
                    report_vector_all.append(report_vector_x)
                    report_tokens = [report_info_no_size[k] for k in TOKEN_KEYS_ALL]
                    report_tokens += [known_tumor_organ_x, known_tumor_slices_x]
                    if self.age_and_sex_provided:
                        print(f'Provided to decoder aux age {age}, sex {sex}', flush=True)
                        report_tokens += [age, sex]
                    report_tokens_list_all.append(report_tokens)
                else:
                    #now we can give the size info.
                    report_vector_x = report_vector_masked
                    known_tumor_organ_x = known_tumor_organ[:,0,0,0,0].unsqueeze(-1)
                    known_tumor_slices_x = known_tumor_slices[:,0,0,0,0].unsqueeze(-1)
                    if not self.age_and_sex_provided:
                        report_vector_x = torch.cat([report_vector_x, known_tumor_organ_x, known_tumor_slices_x], dim=1)
                    else:
                        report_vector_x = torch.cat([report_vector_x, known_tumor_organ_x, known_tumor_slices_x,
                                                    age,sex], dim=1)
                    report_vector_all.append(report_vector_x)
                    report_tokens = [report_info_masked[k] for k in TOKEN_KEYS_ALL]
                    report_tokens += [known_tumor_organ_x, known_tumor_slices_x]
                    if self.age_and_sex_provided:
                        print(f'Provided to decoder aux age {age}, sex {sex}', flush=True)
                        report_tokens += [age, sex]
                    report_tokens_list_all.append(report_tokens)
                    
            out_report_decoder = self.report_decoder(unet_decoder_features=unet_decoder_features,
                                    out_mask=torch.sigmoid(out),
                                    report_vector_all=report_vector_all,
                                    report_tokens_list_all=report_tokens_list_all,
                                    tumor_location_mask=tumor_location_mask,
                                    tumor_allowed_slices=tumor_allowed_slices,)
            out_refined, aux_out_refined = out_report_decoder["refined_segmentation"]  # feat is a tensor [B, C, D, H, W]
            
            if block_grad_mask.any():
                #out is: 'refined_segmentation': [features, aux_out_refined], we need to detach only the features
                m = block_grad_mask.view(B, 1, 1, 1, 1).to(out_refined.dtype)  # broadcast
                out_refined = out_refined.detach() * m + out_refined * (1 - m) # no change in value (forward), but grad is selectively blocked per sample
                out_report_decoder["refined_segmentation"] = [out_refined, aux_out_refined]
                
            #add classifiers on top of the refined segmentation if needed
            if self.aux_loss:        
                if self.att_classifier is not None:
                    aux_att_refined = self.att_classifier(x[:,0].unsqueeze(1),aux_out_refined,labels,cut_attenuation_grad=cut_attenuation_grad)
                else:
                    aux_att_refined = None
                if self.cls_on_segmentation is not None:
                    if self.age_and_sex_provided:
                        aux_out_age_sex_refined = torch.cat((aux_out_refined, 
                                                    age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:]),
                                                    sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])),
                                                    dim=1)
                        y_class_on_seg_aux_refined = self.cls_on_segmentation(aux_out_age_sex_refined)
                    else:
                        y_class_on_seg_aux_refined = self.cls_on_segmentation(aux_out_refined)
                else:
                    y_class_on_seg_aux_refined = None
            else:
                aux_att_refined = None
                y_class_on_seg_aux_refined = None
                out_report_decoder["refined_segmentation"] = out_refined
                aux_out_refined = None  # keep local consistent too
                
            #now apply to outputs as well
            if self.cls_on_segmentation is not None:
                if self.age_and_sex_provided:
                    out_age_sex_refined = torch.cat((out_refined, 
                                                age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:]),
                                                sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])),
                                                dim=1)
                    y_class_on_seg_refined = self.cls_on_segmentation(out_age_sex_refined)
                else:
                    y_class_on_seg_refined = self.cls_on_segmentation(out_refined)
            else:
                y_class_on_seg_refined = None
                
            if self.att_classifier is not None:
                out_att_refined = self.att_classifier(x[:,0].unsqueeze(1),out_refined,labels,cut_attenuation_grad=cut_attenuation_grad)
            else:
                out_att_refined = None
            
            #add new keys to out_report_decoder
            out_report_decoder['attenuation_refined'] = [aux_att_refined,out_att_refined] if aux_att_refined is not None else out_att_refined
            out_report_decoder['classification on segmentation_refined'] = [y_class_on_seg_aux_refined,y_class_on_seg_refined] if y_class_on_seg_aux_refined is not None else y_class_on_seg_refined
                
            assert give_size_mask.shape == (B,)
            assert block_grad_mask.shape == (B,)
            assert (block_grad_mask <= give_size_mask).all(), "Can't block grad where we didn't give size."
        else:
            out_report_decoder = None
                
        
        return self.prepare_return(out, aux_out=aux_out, y_class=y_class, y_class_2=y_class_2,
                                y_clip=y_clip, aux_att=aux_att, out_att=out_att, y_tumor=y_tumor,
                                y_class_on_seg_aux=y_class_on_seg_aux, y_class_on_seg=y_class_on_seg,
                                bottleneck_features=bottleneck_features,
                                mid_decoder_features=mid_decoder_features,
                                out_features=out_features,
                                out_report_decoder=out_report_decoder)
        
    
        

    def rebuild_cls_on_segmentation(self, num_input_ch: int, verbose: bool = False):
        """
        Rebuild cls_on_segmentation so its first layers match the new segmentation channel count.
        Preserves pretrained weights by:
        (1) copying back any parameters/buffers whose keys+shapes match
        (2) for conv-like tensors (4D/5D) whose shapes differ only in channel dims,
            copying overlapping channels
        (3) for 1D tensors (bias/norm/running stats), copying overlapping entries
        """
        if self.cls_on_segmentation is None:
            return

        old_cls = self.cls_on_segmentation
        old_sd = old_cls.state_dict()

        out_class_number = old_cls.head.out_features
        binarize_input = getattr(old_cls, "binarize_input", False)

        cfg = self._cls_build_cfg
        new_cls = make_classifier(
            chan_num=cfg["chan_num"],
            dim_head=cfg["dim_head"],
            conv_block=cfg["conv_block"],
            expansion=cfg["expansion"],
            attn_drop=cfg["attn_drop"],
            proj_drop=cfg["proj_drop"],
            map_size=cfg["map_size"],
            proj_type=cfg["proj_type"],
            norm=cfg["norm"],
            act=cfg["act"],
            out_class_number=out_class_number,
            num_input_ch=num_input_ch,
            binarize_input=binarize_input,
            class_list_cls=getattr(old_cls, "class_list_cls", None),
            class_list_seg=getattr(old_cls, "class_list_seg", None),
        )

        new_sd = new_cls.state_dict()
        load_sd = {}

        copied_exact = 0
        copied_partial = 0
        skipped = 0

        for k, v_new in new_sd.items():
            v_old = old_sd.get(k, None)
            if v_old is None:
                skipped += 1
                continue

            # (1) exact match
            if v_old.shape == v_new.shape:
                load_sd[k] = v_old
                copied_exact += 1
                continue

            # (2) conv weights: 2D conv -> 4D [O,I,kH,kW], 3D conv -> 5D [O,I,kD,kH,kW]
            if v_old.ndim in (4, 5) and v_new.ndim == v_old.ndim:
                # require kernel sizes match
                if v_old.shape[2:] == v_new.shape[2:]:
                    tmp = v_new.clone()
                    o = min(v_old.shape[0], v_new.shape[0])
                    i = min(v_old.shape[1], v_new.shape[1])
                    tmp[:o, :i, ...] = v_old[:o, :i, ...]
                    load_sd[k] = tmp
                    copied_partial += 1
                    continue

            # (3) 1D tensors: bias, norm weight/bias, running stats
            if v_old.ndim == 1 and v_new.ndim == 1:
                tmp = v_new.clone()
                n = min(v_old.shape[0], v_new.shape[0])
                tmp[:n] = v_old[:n]
                load_sd[k] = tmp
                copied_partial += 1
                continue

            # otherwise skip
            skipped += 1

        missing, unexpected = new_cls.load_state_dict(load_sd, strict=False)

        if verbose:
            print(
                f"[rebuild_cls_on_segmentation] exact={copied_exact} partial={copied_partial} skipped={skipped} "
                f"missing_after_load={len(missing)} unexpected_after_load={len(unexpected)}",
                flush=True
            )

        self.cls_on_segmentation = new_cls



def update_output_layer_onk(model, original_classes, new_classes, copy_pancreas=False,binarize_cls_on_segmentation=False,
                            age_and_sex=False):
    """
    Update the model's final output layers so that they produce outputs for the new set of classes.
    For segmentation layers (model.outc and model.aux_out), we update them to have len(new_classes) outputs.
    For the classification branch (model.classification_branch.head) we update it only for lesion classes,
    that is, only classes whose name contains 'lesion'. Similarly, for the Gate module, we set:
        - class_list_seg = new_classes  (all segmentation classes)
        - class_list_cls = new_classes filtered to those containing 'lesion' / benign/malignant.
    
    Args:
        model (nn.Module): The pretrained model instance that has attributes outc, and possibly aux_out, classification_branch, gate_cls.
        original_classes (list of str): The original full list of class names (e.g., segmentation channels) used in the checkpoint.
        new_classes (list of str): The new full list of class names (e.g., segmentation channels).
        copy_pancreas (bool): If True, copy the weights of the pancreas class from the original model to all classes in the new model.
    Returns:
        model: The updated model.
    """
    # For classification, consider only classes with the word "lesion".
    new_class_cls = [cls for cls in new_classes if (("background" in cls) or ("lesion" in cls) or ('pdac' in cls) or ('pnet' in cls) or ('cyst' in cls))]
    old_class_cls = [cls for cls in original_classes if ("lesion" in cls)]
    new_class_no_malig_benign = [cls for cls in new_classes if ('malig' not in cls) and ('benign' not in cls)]
    malig_cls = any(('malign' in c) or ('benign' in c) for c in new_classes)
    malig_classes = [cls for cls in new_classes if (('malign' in cls) or ('benign' in cls))]
    new_class_cls = new_class_cls + malig_classes
    

    # Helper: update a Conv3d layer given an old layer and a desired new number of output channels.
    def update_conv(old_conv, new_out_channels, full_class_list):
        in_channels = old_conv.in_channels
        new_conv = nn.Conv3d(
            in_channels,
            new_out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None)
        )
        # For each new class in full_class_list, if it exists in original_classes, copy the corresponding weight.
        for new_idx, new_cls in enumerate(full_class_list):
            if (new_cls not in original_classes) and copy_pancreas:
                # Copy the pancreas class weights to all new classes.
                orig_idx = original_classes.index('pancreatic_lesion')
                new_conv.weight.data[new_idx] = old_conv.weight.data[orig_idx].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data[new_idx] = old_conv.bias.data[orig_idx].clone()
                print('Cloned the weights for class {} from index {} to new index {}'.format(new_cls, orig_idx, new_idx))
        
            if new_cls in original_classes:
                orig_idx = original_classes.index(new_cls)
                new_conv.weight.data[new_idx] = old_conv.weight.data[orig_idx].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data[new_idx] = old_conv.bias.data[orig_idx].clone()
                print('Cloned the weights for class {} from index {} to new index {}'.format(new_cls, orig_idx, new_idx))
            
            if new_cls not in original_classes and (('malignant' in new_cls) or ('benign' in new_cls)) and \
                (new_cls.replace('malignant','lesion').replace('benign','lesion') in original_classes):
                # When adding the benign/malignant classes, use the weights for the old lesion class for initialization.
                orig_idx = original_classes.index(new_cls.replace('malignant','lesion').replace('benign','lesion'))
                new_conv.weight.data[new_idx] = old_conv.weight.data[orig_idx].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data[new_idx] = old_conv.bias.data[orig_idx].clone()
                print('Cloned the weights for class {} from index {} to new index {}'.format(new_cls, orig_idx, new_idx))
                
        return new_conv

    # Update model.outc (segmentation layer) using the full new_classes.
    old_outc = model.outc
    if old_outc.out_channels != len(new_classes):
        print("Updating model.outc from {} to {} outputs".format(old_outc.out_channels, len(new_classes)))
        model.outc = update_conv(old_outc, len(new_classes), new_classes)
    else:
        print("model.outc already has {} outputs.".format(len(new_classes)))

    # Update model.aux_out if present.
    if hasattr(model, 'aux_out') and model.aux_loss:
        old_aux = model.aux_out
        if old_aux.out_channels != len(new_classes):
            print("Updating model.aux_out from {} to {} outputs".format(old_aux.out_channels, len(new_classes)))
            model.aux_out = update_conv(old_aux, len(new_classes), new_classes)
        else:
            print("model.aux_out already has {} outputs.".format(len(new_classes)))
    
    # Update classification branch head.
    if hasattr(model, 'classification_branch') and (model.classification_branch is not None):
        old_head = model.classification_branch.head
        if old_head.out_features != len(new_class_cls):
            print("Updating classification branch head from {} to {} outputs".format(old_head.out_features, len(new_class_cls)))
            in_features = old_head.in_features
            new_head = nn.Linear(in_features, len(new_class_cls))
            # Copy weights for overlapping lesion classes.
            for new_idx, new_cls in enumerate(new_class_cls):
                if (new_cls not in original_classes) and copy_pancreas:
                    orig_idx = old_class_cls.index('pancreatic_lesion')
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
                if new_cls in old_class_cls:
                    orig_idx = old_class_cls.index(new_cls)
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
            model.classification_branch.head = new_head
        else:
            print("Classification branch head already has {} outputs.".format(len(new_class_cls)))
        assert model.classification_branch.head.out_features == len(new_class_cls)
    
    
    if hasattr(model, 'cls_on_output') and model.cls_on_output is not None:
        old_head = model.cls_on_output.head
        if old_head.out_features != len(new_class_cls):
            print("Updating cls_on_output branch head from {} to {} outputs".format(old_head.out_features, len(new_class_cls)))
            in_features = old_head.in_features
            new_head = nn.Linear(in_features, len(new_class_cls))
            # Copy weights for overlapping lesion classes.
            for new_idx, new_cls in enumerate(new_class_cls):
                if (new_cls not in original_classes) and copy_pancreas:
                    orig_idx = old_class_cls.index('pancreatic_lesion')
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
                if new_cls in old_class_cls:
                    orig_idx = old_class_cls.index(new_cls)
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
            model.cls_on_output.head = new_head
        else:
            print("cls_on_output branch head already has {} outputs.".format(len(new_class_cls)))
        assert model.cls_on_output.head.out_features == len(new_class_cls)
        
    if hasattr(model, 'cls_on_segmentation') and model.cls_on_segmentation is not None:
        old_head = model.cls_on_segmentation.head
        if age_and_sex:
            expected_input_ch = len(new_classes) + 2
        else:
            expected_input_ch = len(new_classes)
        if  binarize_cls_on_segmentation:
            expected_input_ch += 1
        change_input_ch = (getattr(model.cls_on_segmentation, "num_input_ch", None) != expected_input_ch)
        if old_head.out_features != len(new_class_cls) or change_input_ch:
            model.cls_on_segmentation.class_list_seg = new_classes.copy()
            model.cls_on_segmentation.class_list_cls = new_class_cls.copy()
            if change_input_ch:
                model.rebuild_cls_on_segmentation(num_input_ch=expected_input_ch)
                print(f'Number of classes: {len(new_classes)}')
                print(f'Class list: {new_classes}')
            print("Updating cls_on_segmentation branch head from {} to {} outputs".format(old_head.out_features, len(new_class_cls)))
            in_features = old_head.in_features
            new_head = nn.Linear(in_features, len(new_class_cls))
            # Copy weights for overlapping lesion classes.
            for new_idx, new_cls in enumerate(new_class_cls):
                if (new_cls not in original_classes) and copy_pancreas:
                    orig_idx = old_class_cls.index('pancreatic_lesion')
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
                if new_cls in old_class_cls:
                    orig_idx = old_class_cls.index(new_cls)
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
            model.cls_on_segmentation.head = new_head
        else:
            print("cls_on_segmentation branch head already has {} outputs.".format(len(new_class_cls)))
        assert model.cls_on_segmentation.head.out_features == len(new_class_cls)


    
    # Update Gate module: segmentation list is new_classes; classification list is new_class_cls.
    if hasattr(model, 'gate_cls') and (model.gate_cls is not None):
        print("Updating Gate module class lists.")
        model.gate_cls.class_list_seg = new_class_no_malig_benign.copy()
        model.gate_cls.class_list_cls = new_class_cls.copy()
        # Rebuild internal mapping.
        model.gate_cls.gated_classes = {}
        for i, seg_cls in enumerate(model.gate_cls.class_list_seg):
            for j, cls_cls in enumerate(model.gate_cls.class_list_cls):
                if seg_cls == cls_cls:
                    model.gate_cls.gated_classes[seg_cls] = {'seg_idx': i, 'cls_idx': j}
                    break

    if hasattr(model,'att_classifier') and model.att_classifier is not None:
        model.att_classifier.class_list=new_class_no_malig_benign
    if hasattr(model,'tumor_classifier') and model.tumor_classifier is not None:
        model.tumor_classifier.class_list=new_class_no_malig_benign

    return model


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
    
    
class BottleneckTransformer(nn.Module):
    def __init__(self, chan_num, bottleneck_ch=256, depth=2, heads=8, dim_head=32, mlp_dim=1024):
        super().__init__()
        in_ch = chan_num[3] if isinstance(chan_num, (list, tuple)) else chan_num

        self.reducer = nn.Conv3d(in_ch, bottleneck_ch, kernel_size=1)
        self.pre_norm = nn.GroupNorm(num_groups=min(32, bottleneck_ch), num_channels=bottleneck_ch)
        # tokens will be [B, N, C]; add token LN if your TransformerBlock doesn't already
        self.token_norm = nn.LayerNorm(bottleneck_ch)

        self.transformer = TransformerBlock(
            dim=bottleneck_ch,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
        )

        self.up_ch_conv = nn.Conv3d(bottleneck_ch, in_ch, kernel_size=1)

        # Optional: start transformer as "off"
        self.gate = nn.Parameter(torch.tensor(0.0))

    def _core(self, x):
        B, C, D, H, W = x.shape

        y = self.reducer(x)
        y = self.pre_norm(y)

        y = y.flatten(2).transpose(1, 2).contiguous()  # [B, N, C']
        y = self.token_norm(y)
        y = self.transformer(y)
        y = y.transpose(1, 2).contiguous().view(B, -1, D, H, W)

        y = self.up_ch_conv(y)
        return y

    def forward(self, x):
        inpt = x
        if self.training:  # usually checkpoint only in training
            y = checkpoint(self._core, x, use_reentrant=False)
        else:
            y = self._core(x)

        return inpt + self.gate.tanh() * y

from functools import reduce
import operator

def get_mask_lesion_masks(out: torch.Tensor, class_list):
    """
    Returns:
      tumors        (B,T,D,H,W)
      tumour_organs (B,T,D,H,W)  – matched organ mask for each tumour
    """
    idx = {c: i for i, c in enumerate(class_list)}  # O(1) lookup

    tumors, organs = [], []
    indices = []

    for cls in class_list:
        if any(tag in cls for tag in ('lesion', 'pdac', 'pnet', 'cyst')):
            tumors.append(out[:, idx[cls]])          # (B,D,H,W)
            indices.append(idx[cls])  # keep track of indices for organ mask
            org = tumor_to_organ(cls)
            if isinstance(org, list):
                if any(o not in idx for o in org):
                    organ_mask = torch.ones_like(out[:, 0])
                else:
                    masks = [out[:, idx[o]] for o in org]
                    # element‑wise max across sides
                    organ_mask = reduce(torch.maximum, masks)
            else:
                if org not in idx:
                    organ_mask = torch.ones_like(out[:, 0])
                else:
                    organ_mask = out[:, idx[org]]
            organs.append(organ_mask)

    tumors = torch.stack(tumors, dim=1)         # (B,T,…
    organs = torch.stack(organs, dim=1)
    return tumors, organs, indices
     

def extract_hu(ct,
               out,
               class_list):
    """
    Extracts “soft” HU statistics (mean and standard deviation) from the CT volume,
    using mask_tumor and mask_organ as weights.
    """
    assert ct.dim() == 5, f"CT input must be 5D tensor (B, C, D, H, W), got {ct.dim()}D"
    assert out.dim() == 5, f"Mask tumor must be 5D tensor (B, C, D, H, W), got {out.dim()}D"
    
    out = torch.sigmoid(out)  # ensure mask is in [0,1]
    mask_tumor, mask_organ, indices = get_mask_lesion_masks(out, class_list)
    
    
    #standardize ct - this is needed for meand and std not to explode
    vmin = ct.amin(dim=(-1, -2, -3), keepdim=True)  # per‑volume minimum
    vmax = ct.amax(dim=(-1, -2, -3), keepdim=True)  # per‑volume maximum
    ct   = (ct - vmin) / (vmax - vmin + 1e-5)
    
    eps = 1e-5  # small epsilon to avoid division by zero

    mask_o = mask_organ * (1.0 - mask_tumor)
    mask_t = mask_tumor

    # 2) Compute weighted (soft) mean for the organ:
    #    numerator = sum(mask_o * ct)
    #    denominator = sum(mask_o)
    soft_mean_o = (mask_o * ct).sum(dim=(-1,-2,-3), keepdim=True) / (mask_o.sum(dim=(-1,-2,-3), keepdim=True) + eps)

    # 3) Compute weighted (soft) variance for the organ:
    #    var = sum[ mask_o * (ct - mean_o)^2 ] / sum(mask_o)
    
    soft_var_o = ( mask_o * (ct - soft_mean_o).pow(2) ).sum(dim=(-1,-2,-3), keepdim=True) / (mask_o.sum(dim=(-1,-2,-3), keepdim=True) + eps)
    soft_std_o = torch.sqrt(soft_var_o)

    # 4) Repeat for the tumor region:
    soft_mean_t = (mask_t * ct).sum(dim=(-1,-2,-3), keepdim=True) / (mask_t.sum(dim=(-1,-2,-3), keepdim=True) + eps)
    soft_var_t = ( mask_t * (ct - soft_mean_t).pow(2) ).sum(dim=(-1,-2,-3), keepdim=True) / (mask_t.sum(dim=(-1,-2,-3), keepdim=True) + eps)
    soft_std_t = torch.sqrt(soft_var_t)
    
    soft_mean_o = soft_mean_o.squeeze(-1).squeeze(-1).squeeze(-1)
    soft_mean_t = soft_mean_t.squeeze(-1).squeeze(-1).squeeze(-1)
    soft_std_o = soft_std_o.squeeze(-1).squeeze(-1).squeeze(-1)
    soft_std_t = soft_std_t.squeeze(-1).squeeze(-1).squeeze(-1)

    # 5) stack in the last dimension
    retur = [soft_mean_o, soft_std_o, soft_mean_t, soft_std_t, soft_mean_t - soft_mean_o, soft_std_t - soft_std_o, mask_o.mean(dim=(-1,-2,-3)), mask_t.mean(dim=(-1,-2,-3))]
    retur = torch.stack(retur, dim=-1)  # shape (B, C, 8)
    
    expected_shape = torch.Size((ct.shape[0], mask_tumor.shape[1], 8))
    assert retur.shape == expected_shape
    
    #check for nans
    if torch.isnan(retur).any():
        print(f'NaN detected in HU statistics output:, {retur}',flush=True)
        return torch.zeros_like(retur.detach()), indices
        #raise ValueError(f"NaN detected in HU statistics output:, {retur}")

    return retur, indices

def pad_channels(att,class_list, indices):
    #pad channels so that the shape of att is B,len(class_list),3
    tmp=[]
    for i in range(len(class_list)):
        if i in indices:
            tmp.append(att[:, indices.index(i)])
        else:
            tmp.append(torch.zeros_like(att[:, 0]))  # zero padding for missing classes
    att = torch.stack(tmp, dim=1)
    return att

def straight_through_trick(x,th=0.5):
    """
    This function allows gradients to flow through thresholding.
    """
    raise ValueError('This causes training instability, do not use it.')
    x= x - x.detach() + (x.detach()>th).float()
    return x


class _GradScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #print('GradScaler backward called with factor:', ctx.factor,flush=True)
        return grad_output * ctx.factor, None        # None = no grad wrt factor


class GradScaler(nn.Module):
    """ Identity layer whose backward gradient is multiplied by `factor`. """
    def __init__(self, factor: float):
        super().__init__()
        self.factor = float(factor)

    def forward(self, x):
        return _GradScaleFn.apply(x, self.factor)

import copy
class attribute_classifier(torch.nn.Module):
    """
    A MLP for attenuation values (mean and std) to classify lesions as hypo-attenuating, mixed/iso-attenuating, or hyper-attenuating.
    """
    def __init__(self, class_list, in_features=8, out_features=3,train_on_mask_only=False, model = 'MLP', calculate_HU=True,
                 mode = 'attenuation_classifier',loss_weight=1):
        super(attribute_classifier, self).__init__()
        if model == 'MLP':
            self.model = nn.Sequential(
                GradScaler(loss_weight),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, out_features),
                GradScaler(1/loss_weight)
            )
        elif model == 'neuron':
            self.model = nn.Sequential(
                GradScaler(loss_weight),
                nn.Linear(in_features, out_features,bias=False),
                GradScaler(1/loss_weight)
            )
        else:
            self.model = nn.Sequential(
                GradScaler(loss_weight),
                model,
                GradScaler(1/loss_weight)
            )
        
        self.class_list = class_list
        self.train_on_mask_only = train_on_mask_only  # if True, MLP is trained only on the mask and it is frozen when receiceiving the segmenter
        if train_on_mask_only:
            self.frozen_model = copy.deepcopy(self.model)
            #freeze
            for param in self.frozen_model.parameters():
                param.requires_grad = False
            self.frozen_model.eval()
        self.calculate_HU = calculate_HU
        self.mode = mode  # 'attenuation_classifier' or 'tumor_classifier'
        assert self.mode in ['attenuation_classifier', 'tumor_classifier'], f"Invalid mode: {self.mode}. Choose from ['attenuation_classifier', 'tumor_classifier']"
        if self.mode== 'tumor_classifier':
            if model == 'MLP':
                raise ValueError("Tumor classifier must use a different model than MLP.")
    def forward(self, ct, out, labels=None,cut_attenuation_grad=False):
        #ignore the malignant and benign classes in the class list
        out = out[:, :len(self.class_list)]  # (B, C, D, H, W)
        
        if cut_attenuation_grad:
            out = out.detach() #used to train the classifier before propagating the loss to the segmenter
        
        if labels is not None:
            #if the case is annotated by voxel (and has tumor) we train the MLP only, using mask as input
            mask_tumor, mask_organ, indices = get_mask_lesion_masks(labels, self.class_list) 
            from_mask = []
            tmp = []
            for b in list(range(mask_tumor.shape[0])):
                if mask_tumor[b].sum() != 0: #case annotated by voxel
                    tmp.append(labels[b])
                    from_mask.append(1)
                else:
                    #case not annotated by voxel, use the mask from the model
                    tmp.append(out[b])
                    from_mask.append(0)
            out = torch.stack(tmp, dim=0)  # (B, C, D, H, W)
        else:
            from_mask = torch.zeros(ct.shape[0]).int().tolist()
            
        num_lesion_classes = None
        if self.calculate_HU:
            if self.mode != 'attenuation_classifier':
                raise ValueError("calculate_HU is True but mode is not 'attenuation_classifier'.")
            inpt, indices = extract_hu(ct,out,self.class_list)
        else:
            mask_tumor, mask_organ, indices = get_mask_lesion_masks(out, self.class_list)
            num_lesion_classes = mask_tumor.shape[1]  # number of lesion classes in the mask
            if self.mode == 'attenuation_classifier':
                inpt = torch.cat((ct,mask_tumor, mask_organ), dim=1)  # (B, C, D, H, W)
            elif self.mode == 'tumor_classifier':
                inpt = torch.cat((mask_tumor, mask_organ), dim=1)  # (B, C, D, H, W)
                #no CT, we want the masks to be the bottleneck of information.
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Choose from ['attenuation_classifier', 'tumor_classifier']")
            
        if not self.train_on_mask_only:
            att = self.run_model(inpt, indices, frozen=False, num_lesion_classes=num_lesion_classes)  # run MLP on all HU features
        else:
            #if no mask was used, just use the standard MLP
            if np.mean(from_mask)==0:
                att = self.run_model(inpt, indices, frozen=False, num_lesion_classes=num_lesion_classes)
            elif np.mean(from_mask)==1: #only used images with per-voxel annotations
                att = self.run_model(inpt, indices, frozen=True, num_lesion_classes=num_lesion_classes)
                nothing = 0 * self.run_model(inpt[0].unsqueeze(0), indices, frozen=False, num_lesion_classes=num_lesion_classes).sum()
                att = att + nothing #just avoids the unused parameters problem
            else:
                #mixed case: some annotated, some not. Run both.
                #why we need this? 
                #here we want only the samples with masks to be used to train the MLP. Conversely, when training the segmenter, 
                #we only want the segmenter to be updated, not the MLP. This may avoid the segmenter to learn some shortcut solution, 
                #where it passes information to the MLP (about attenuation) without actually improving the tumor segmentation. 
                #But this may also increase the change of overfitting the MLP, as the number of masks may be small.
                frozen_out = self.run_model(inpt, indices, frozen=True, num_lesion_classes=num_lesion_classes) 
                unfrozen_out = self.run_model(inpt, indices, frozen=False, num_lesion_classes=num_lesion_classes)
                chosen = []
                for b in list(range(inpt.shape[0])):
                    if from_mask[b] == 1:
                        chosen.append(frozen_out[b])
                    elif from_mask[b] == 0:
                        chosen.append(unfrozen_out[b])
                    else:
                        raise ValueError('Invalid from_mask value: {}'.format(from_mask[b]))
                att = torch.stack(chosen, dim=0)  # (B, C, 3)
        return att
            
    def run_model(self, inpt, indices, frozen=False,num_lesion_classes=None):
        if not frozen:
            net = self.model
        else:
            for p_froz, p_live in zip(self.frozen_model.parameters(), self.model.parameters()):
                p_froz.data.copy_(p_live.data.detach())#detach may not be needed here. But it is ok to leave it.
            #keep it frozen
            for param in self.frozen_model.parameters():
                param.requires_grad = False
            net = self.frozen_model
            
        if self.calculate_HU:
            att = []
            for c in list(range(inpt.shape[1])):
                x = inpt[:, c, :]  # (B, 8)
                x = net(x)  # (B, 3)
                att.append(x)
            att = torch.stack(att, dim=1)
            att = pad_channels(att,self.class_list, indices)
            return att  # no sigmoid here; use BCEWithLogitsLoss for training
        else:
            x = net(inpt.float())  # (B, C*3)
            if self.mode == 'attenuation_classifier':
                assert num_lesion_classes == int(x.shape[1]/3), f"Number of lesion classes does not match the output of the classifier: expected {num_lesion_classes}, got {int(x.shape[1]/3)}"
                att = x.reshape((x.shape[0],num_lesion_classes,3)) #B,c',3, where c' is the number of lesion classes
                att = pad_channels(att,self.class_list, indices) #B,C,3
            elif self.mode == 'tumor_classifier':
                att = x.reshape((x.shape[0],num_lesion_classes,10,3))#B, lesion classes, tumor, (D,d,Objectness)
                att = pad_channels(att,self.class_list, indices) #B,C,3
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Choose from ['attenuation_classifier', 'tumor_classifier']")
            return att
            
        
        
        
class simple_classifier(torch.nn.Module):
    """
    Uses the difference between the mean tumor HU value of the tumor and the organ
    """
    def __init__(self, class_list, in_features=8, out_features=3):
        super(simple_classifier, self).__init__()
        self.class_list = class_list
        
    def forward(self, ct, out, labels=None,cut_attenuation_grad=False):
        x, indices = extract_hu(ct,out,self.class_list)
        # x[:, :, 4] is the difference in mean HU between tumor and organ
        diff_mean = x[:, :, 4].unsqueeze(-1)
        diff_mean = pad_channels(diff_mean,self.class_list, indices)
        return diff_mean  # shape (B, C, 1)
    
import copy
import torch.nn as nn
import torch
def get_first_conv3d(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            return name, m
    raise RuntimeError("No nn.Conv3d found")

@torch.no_grad()
def expand_conv3d_in_channels(conv: nn.Conv3d, to_add: int, mode: str = "zeros", *, rescale: bool = True):
    """
    Expand a Conv3d's input channels by `to_add`.

    mode:
      - "repeat": repeats old weights across new channels then rescales
      - "zeros": extra channels initialized to 0, avoids perturbing the pretrained model too much
      - "avg": fills new channels with the mean over old input channels
      - "repeat_zeros": repeats old weights ONLY into even input channels (0,2,4,...),
                        and sets odd channels (1,3,5,...) to 0.
                        Intended for interleaved inputs: [CT, date, CT, date, ...].

    rescale (only for repeat-like modes):
      - If True, rescales weights by (effective_old_in / effective_new_in) to keep activation magnitude similar.
    """
    old_w = conv.weight.data  # [out, in, kD, kH, kW]
    old_in = old_w.shape[1]
    new_in = old_in + to_add

    new = nn.Conv3d(
        in_channels=new_in,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    ).to(old_w.device, dtype=old_w.dtype)

    if mode == "repeat":
        new_w = old_w.repeat(1, (new_in + old_in - 1) // old_in, 1, 1, 1)[:, :new_in]
        if rescale:
            new_w *= (old_in / new_in)

    elif mode == "zeros":
        new_w = torch.zeros(
            (conv.out_channels, new_in, *old_w.shape[2:]),
            device=old_w.device, dtype=old_w.dtype
        )
        new_w[:, :old_in] = old_w

    elif mode == "avg":
        mean = old_w.mean(dim=1, keepdim=True)  # [out, 1, kD, kH, kW]
        new_w = mean.repeat(1, new_in, 1, 1, 1)

    elif mode == "repeat_zeros":
        # Start with zeros everywhere
        new_w = torch.zeros(
            (conv.out_channels, new_in, *old_w.shape[2:]),
            device=old_w.device, dtype=old_w.dtype
        )

        # Fill even channels only with repeated old weights
        even_idx = torch.arange(0, new_in, 2, device=old_w.device)
        if len(even_idx) == 0:
            raise ValueError("repeat_zeros: new_in has no even indices?")

        # Repeat old weights to cover all even positions
        rep = (len(even_idx) + old_in - 1) // old_in  # enough repeats along in-ch
        even_w = old_w.repeat(1, rep, 1, 1, 1)[:, :len(even_idx)]
        new_w.index_copy_(1, even_idx, even_w)

        # Optional rescale based on *effective* channels used (the even ones)
        if rescale:
            effective_new_in = len(even_idx)
            new_w *= (old_in / effective_new_in)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    new.weight.data.copy_(new_w)
    if conv.bias is not None:
        new.bias.data.copy_(conv.bias.data)
    return new


def update_first_conv(net, target_in, mode="zeros", *, rescale: bool = True):
    name, conv = get_first_conv3d(net)
    if conv.in_channels == target_in:
        return
    if conv.in_channels > target_in:
        raise ValueError(f"First conv has {conv.in_channels}, expected {target_in}")

    to_add = target_in - conv.in_channels

    parts = name.split(".")
    parent = net
    for p in parts[:-1]:
        parent = getattr(parent, p)

    setattr(parent, parts[-1], expand_conv3d_in_channels(conv, to_add=to_add, mode=mode, rescale=rescale))
    
import torch.nn as nn

def snapshot_requires_grad(net: nn.Module) -> dict:
    """Return a snapshot of requires_grad for every parameter (by name)."""
    return {name: p.requires_grad for name, p in net.named_parameters()}

def restore_requires_grad(net: nn.Module, snap: dict) -> None:
    """Restore requires_grad snapshot (missing keys are ignored)."""
    for name, p in net.named_parameters():
        if name in snap:
            p.requires_grad_(snap[name])

import torch.nn as nn

def _get_attr_by_path(obj: object, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(f"[freeze_trainable_except] Missing attribute '{part}' in path '{path}' (at '{type(cur).__name__}')")
        cur = getattr(cur, part)
    if cur is None:
        raise AttributeError(f"[freeze_trainable_except] Attribute path '{path}' resolved to None")
    return cur

def freeze_trainable_except(
    net: nn.Module,
    *,
    keep_first_conv3d: bool = True,
    also_train: tuple[str, ...] = ("report_processing_MLP",),
    allow_missing: bool = True,
) -> None:
    # Freeze only currently-trainable params
    for p in net.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    # Unfreeze first Conv3d
    if keep_first_conv3d:
        _, conv = get_first_conv3d(net)
        for p in conv.parameters():
            p.requires_grad_(True)

    # Unfreeze requested modules
    for path in also_train:
        try:
            mod = _get_attr_by_path(net, path)
        except AttributeError:
            if allow_missing:
                continue
            raise

        if mod is None:
            if allow_missing:
                continue
            raise AttributeError(f"[freeze_trainable_except] '{path}' is None")

        if not isinstance(mod, nn.Module):
            raise TypeError(
                f"[freeze_trainable_except] '{path}' resolved to {type(mod).__name__}, not nn.Module."
            )
        for p in mod.parameters():
            p.requires_grad_(True)
    
def expand_conv3d_out_channels_repeat(
    conv: nn.Conv3d,
    time_points: int,
    *,
    copy_bias: bool = True,
) -> nn.Conv3d:
    """
    Expand a Conv3d's OUT channels by repeating them `time_points` times.
    in_channels is unchanged.

    If original classes are [a,b,c,d] (out_channels=4),
    new out_channels = 4 * time_points, ordered as:
      [a,b,c,d, a,b,c,d, ..., a,b,c,d]  (time_points blocks)

    Weights are copied exactly (no rescaling).
    """
    if time_points <= 0:
        raise ValueError(f"time_points must be >= 1, got {time_points}")
    if time_points == 1:
        return conv

    old_w = conv.weight.data  # [out, in, kD, kH, kW]
    old_out = old_w.shape[0]
    new_out = old_out * time_points

    new = nn.Conv3d(
        in_channels=conv.in_channels,
        out_channels=new_out,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    ).to(old_w.device, dtype=old_w.dtype)

    # Repeat output filters in blocks: [old, old, old, ...]
    new_w = old_w.repeat(time_points, 1, 1, 1, 1)  # repeats along dim=0
    new.weight.data.copy_(new_w)

    if conv.bias is not None and copy_bias:
        new.bias.data.copy_(conv.bias.data.repeat(time_points))

    return new
    
class RTSuper(nn.Module):
    def __init__(self,network,train_mode = 'both', num_inpt_ch = 21, #16 for report embed, 1 for image, organ mask, slice mask, known_tumor_organ flag, and known_tumor_slice flag
                 teacher_report_info_prob=1, student_report_info_prob=0,
                 EMA_net = None,
                 use_transformer_decoder=True,
                 use_dynamic_conv=True,
                 MLP_in_dim=63,
                 MLP_out_dim=16,
                 age_and_sex_provided = False,
                 use_transformer_conv3 = False,
                 time_points = 1,
                 time_fusion = 'early',
                 ):
        """
        network: the normal network, instantiated. No need to include more input channels.
        """
        super().__init__()
        
        if train_mode!='teacher_decoder' and time_points>1:
            raise ValueError("time_points>1 is only supported in 'teacher_decoder' mode")
        
        if train_mode in ['student', 'both']:
            self.teacher = network #report-aware network
            self.student = copy.deepcopy(network) #network used for inference, report-unaware
            assert self.student.report_information_in_input
            assert self.teacher.report_information_in_input
            update_first_conv(self.teacher,num_inpt_ch)
            if student_report_info_prob>0:
                update_first_conv(self.student,num_inpt_ch)
            else:
                self.student.report_information_in_input = False
                #delete the module report_processing_MLP from the student
                self.student.report_processing_MLP = None
        elif train_mode=='teacher' or train_mode=='teacher_decoder':
            self.teacher = network
            self.student = None
            if train_mode=='teacher':
                assert self.teacher.report_information_in_input
                update_first_conv(self.teacher,num_inpt_ch)
            elif train_mode=='teacher_decoder':
                if time_points>1:
                    self.teacher.time_points = time_points
                    self.teacher.time_fusion = time_fusion
                    #rebuild first conv layer
                    assert num_inpt_ch == 1, "num_inpt_ch is the number of channels per sample; we currently support only 1 in the longitudinal mode"
                    update_first_conv(self.teacher,(num_inpt_ch+1)*time_points, mode='repeat_zeros')
                    #update output convolutions
                    if hasattr(self.teacher, "aux_out") and self.teacher.aux_out is not None:
                        self.teacher.aux_out = expand_conv3d_out_channels_repeat(self.teacher.aux_out,time_points=time_points)
                    self.teacher.outc = expand_conv3d_out_channels_repeat(self.teacher.outc,time_points=time_points)
                    
                self.teacher.create_report_informed_decoder(use_transformer_decoder=use_transformer_decoder,
                                                            use_dynamic_conv=use_dynamic_conv,
                                                            age_and_sex_provided = age_and_sex_provided,
                                                            use_transformer_conv3 = use_transformer_conv3)
                self.teacher.create_bottleneck_transformer()
                assert not self.teacher.report_information_in_input
                #assert we have the decoder set:
                assert getattr(self.teacher, 'report_informed_decoder', False), "Must set report_informed_decoder=True"
                assert hasattr(self.teacher, 'report_decoder'), "Network must have 'report_decoder' module for 'teacher_decoder' mode"
                #assert also that we have the bottleneck transformer
                assert hasattr(self.teacher, 'bottleneck_transformer'), "Network must have 'bottleneck_transformer' module for 'teacher_decoder' mode, this will be important for the longitudinal step"
                self.teacher.age_and_sex_provided = age_and_sex_provided
                    
        elif train_mode=='teacher_student_ema':
            assert EMA_net is not None, "EMA_net must be provided when train_mode is 'teacher_student_ema'"
            self.student = network
            self.teacher = EMA_net
            assert self.teacher.report_information_in_input
            assert self.student.report_information_in_input
            update_first_conv(self.teacher,num_inpt_ch)
            update_first_conv(self.student,num_inpt_ch)
            
        if train_mode!='teacher_decoder':
            if hasattr(self.teacher, 'report_processing_MLP'):
                self.teacher.report_processing_MLP = MLP(in_dim=(MLP_in_dim), hidden_dim=128, out_dim=MLP_out_dim, drop=0.2)
            if hasattr(self, 'student') and hasattr(self.student, 'report_processing_MLP'):
                self.student.report_processing_MLP = MLP(in_dim=(MLP_in_dim), hidden_dim=128, out_dim=MLP_out_dim, drop=0.2)
            
        
        self.teacher_report_info_prob = teacher_report_info_prob
        self.student_report_info_prob = student_report_info_prob
        self.train_mode = train_mode
        if train_mode == 'both':
            self.teacher.requires_grad_(True)
            self.student.requires_grad_(True)
        elif train_mode == 'student':
            self.teacher.requires_grad_(False)
            self.student.requires_grad_(True)
            self.teacher.eval()
        elif train_mode == 'teacher' or train_mode=='teacher_decoder':
            self.teacher.requires_grad_(True)
        elif train_mode == 'teacher_student_ema':
            self.teacher.requires_grad_(False)
            self.student.requires_grad_(True)
            self.teacher.eval()
        else:
            raise ValueError('Invalid train_mode: {}'.format(train_mode))
        self.snapshot_trainability()
        
    def make_longitudinal(self, time_points, time_fusion,num_inpt_ch,use_transformer_decoder,use_dynamic_conv,
                          age_and_sex_provided,use_transformer_conv3):
        if self.train_mode !='teacher_decoder':
            raise ValueError("make_longitudinal is only supported in 'teacher_decoder' mode")
        assert num_inpt_ch == 1, "num_inpt_ch is the number of channels per sample; we currently support only 1 in the longitudinal mode"
        self.teacher.age_and_sex_provided = age_and_sex_provided
        self.teacher.time_points = time_points
        self.teacher.time_fusion = time_fusion
        #rebuild first conv layer
        update_first_conv(self.teacher,(num_inpt_ch+1)*time_points, mode='repeat_zeros')
        #update output convolutions
        if hasattr(self.teacher, "aux_out") and self.teacher.aux_out is not None:
            self.teacher.aux_out = expand_conv3d_out_channels_repeat(self.teacher.aux_out,time_points=time_points)
        self.teacher.outc = expand_conv3d_out_channels_repeat(self.teacher.outc,time_points=time_points)
        #get state dict of current report-informed decoder
        state_dict = self.teacher.report_decoder.state_dict()
        #rebuild the report-informed decoder with new time points
        self.teacher.create_report_informed_decoder(use_transformer_decoder=use_transformer_decoder,
                                                use_dynamic_conv=use_dynamic_conv,
                                                age_and_sex_provided = age_and_sex_provided,
                                                use_transformer_conv3 = use_transformer_conv3,)
        #load old weights into new report-informed decoder where possible
        _, stats, (missing, unexpected) = load_state_dict_with_overlap(self.teacher.report_decoder, state_dict,
                                                                   verbose=True)
        
        
    def snapshot_trainability(self):
        """Call once after train_mode is applied (or whenever you want a restore point)."""
        self._teacher_grad_snap = snapshot_requires_grad(self.teacher) if self.teacher is not None else None
        self._student_grad_snap = snapshot_requires_grad(self.student) if self.student is not None else None

    def train_first_layer_only(self, also_train=("report_processing_MLP",), keep_first_conv3d=True):
        """
        Stage-1: within whichever nets are currently trainable (per train_mode),
        freeze everything except first conv + new modules.
        """
        if self.train_mode == 'teacher_decoder':
            #train only the decoder
            freeze_trainable_except(
                    self.teacher,
                    keep_first_conv3d=False,
                    also_train=('report_decoder','bottleneck_transformer'),
                )
        else:
            # teacher
            if self.teacher is not None:
                # only apply if teacher is trainable at all
                if any(p.requires_grad for p in self.teacher.parameters()):
                    freeze_trainable_except(
                        self.teacher,
                        keep_first_conv3d=keep_first_conv3d,
                        also_train=also_train,
                    )

            # student
            if self.student is not None:
                if any(p.requires_grad for p in self.student.parameters()):
                    freeze_trainable_except(
                        self.student,
                        keep_first_conv3d=keep_first_conv3d,
                        also_train=also_train,
                    )

    def restore_stage_trainability(self):
        """Stage-2: restore exactly what train_mode originally set. I.e., train all network."""
        if self.teacher is not None and self._teacher_grad_snap is not None:
            restore_requires_grad(self.teacher, self._teacher_grad_snap)
        if self.student is not None and self._student_grad_snap is not None:
            restore_requires_grad(self.student, self._student_grad_snap)
            
    def forward(self, x,tumor_info, annotated_per_voxel='unknown',no_mask_training='unknown', 
                age=None, sex=None, **kwargs):
        
        number_of_time_points = self.teacher.time_points
        if number_of_time_points>1:
            assert len(x)==number_of_time_points, f"Expected {number_of_time_points} time points, got {len(x)}"
            B = x[0].shape[0]
            device = x[0].device
            print(f'Number of time points in input: {number_of_time_points}', flush=True)
        else:
            B = x.shape[0]
            device = x.device
        

        send_report_teacher = (torch.rand(B, device=device) < self.teacher_report_info_prob)  # bool [B]
        send_report_student = send_report_teacher & (torch.rand(B, device=device) < self.student_report_info_prob)  # bool [B]
        if number_of_time_points>1:
            send_report_teacher = [send_report_teacher for _ in range(number_of_time_points)]
        
        if self.train_mode == 'teacher_decoder':
            #teacher and student are the same net, student is the original architecture (e.g., medformer), teacher is the output of the report-informed decoder
            out = self.teacher(x, report_info = tumor_info, 
                               use_report = send_report_teacher, 
                               annotated_per_voxel=annotated_per_voxel, 
                               no_mask_training=no_mask_training, age=age, sex=sex, **kwargs)
            if number_of_time_points>1:
                out_teacher_student = {}
                for key in out:
                    if not key.startswith("time_point_"):
                        raise ValueError(f"Expected time point information in output keys for longitudinal input. Got key: {key}")
                    out_student, out_teacher = self.split_teacher_student_out_decoder(out[key])
                    out_teacher_student[key] = {'student': out_student, 'teacher': out_teacher}
            else:
                out_student, out_teacher = self.split_teacher_student_out_decoder(out)
                out_teacher_student = {'student': out_student, 'teacher': out_teacher}
            return out_teacher_student
            
        else:
            # Teacher forward
            if self.train_mode == 'student' or self.train_mode=='teacher_student_ema':
                with torch.no_grad():
                    out_teacher = self.teacher(x, report_info = tumor_info, use_report = send_report_teacher, annotated_per_voxel=annotated_per_voxel,age=age, sex=sex, **kwargs)
            else:
                out_teacher = self.teacher(x, report_info = tumor_info, use_report = send_report_teacher, annotated_per_voxel=annotated_per_voxel,age=age, sex=sex, **kwargs)
            
            if self.train_mode != 'teacher':
                # Student forward
                if self.student_report_info_prob>0:
                    out_student = self.student(x, report_info = tumor_info, use_report = send_report_student, annotated_per_voxel=annotated_per_voxel,age=age, sex=sex, **kwargs)
                else:
                    out_student = self.student(x, report_info = None, annotated_per_voxel=annotated_per_voxel,age=age, sex=sex, **kwargs)
            else:
                out_student = None
        
            return {'student': out_student, 'teacher': out_teacher}
    
    def split_teacher_student_out_decoder(self, out):
        out_teacher = {}
        #get all keys with refined in their name
        refined = [key for key in out.keys() if 'refined' in key]
        out_teacher = {key.replace('refined_','').replace('_refined',''): out.pop(key) for key in refined}
        for key in out:
            #add the other keys so our loss works, but detach them, so we do not penalize them twice, as they are also in out_student
            if key in out_teacher.keys():
                continue
            tmp = out[key]
            if isinstance(tmp, torch.Tensor):
                out_teacher[key] = tmp.detach()
            elif isinstance(tmp, list):
                t = []
                for item in tmp:
                    if isinstance(item, torch.Tensor):
                        t.append(item.detach())
                    elif item is None:
                        t.append(None)
                    else:
                        raise ValueError('Invalid item type in list output: {}'.format(type(item)))
                out_teacher[key] = t
            elif tmp is None:
                out_teacher[key] = None
            else:
                raise ValueError('Invalid output type: {}'.format(type(tmp)))
        out_student = out
        return out_student, out_teacher
    
    def forward_teacher(self, x,tumor_info, annotated_per_voxel='unknown',no_mask_training='unknown', age=None, sex=None, **kwargs):
        
        B = x.shape[0]
        device = x.device

        send_report_teacher = (torch.rand(B, device=device) < self.teacher_report_info_prob)  # bool [B]
        
        if self.train_mode == 'teacher_decoder':
            #teacher and student are the same net, student is the original architecture (e.g., medformer), teacher is the output of the report-informed decoder
            out = self.teacher(x, report_info = tumor_info, use_report = send_report_teacher, 
                               annotated_per_voxel=annotated_per_voxel, no_mask_training=no_mask_training,age=age, sex=sex, **kwargs)
            out_teacher = {}
            out_teacher['segmentation'] = out.pop('refined_segmentation')
            for key in out:
                #add the other keys so our loss works, but detach them, so we do not penalize them twice, as they are also in out_student
                if key == 'segmentation':
                    continue
                tmp = out[key]
                if isinstance(tmp, torch.Tensor):
                    out_teacher[key] = tmp.detach()
                elif isinstance(tmp, list):
                    t = []
                    for item in tmp:
                        if isinstance(item, torch.Tensor):
                            t.append(item.detach())
                        elif item is None:
                            t.append(None)
                        else:
                            raise ValueError('Invalid item type in list output: {}'.format(type(item)))
                    out_teacher[key] = t
                elif tmp is None:
                    out_teacher[key] = None
                else:
                    raise ValueError('Invalid output type: {}'.format(type(tmp)))
            #out_student = out
            
        else:
            # Teacher forward
            if self.train_mode == 'student' or self.train_mode=='teacher_student_ema':
                with torch.no_grad():
                    out_teacher = self.teacher(x, report_info = tumor_info, use_report = send_report_teacher,age=age, sex=sex, **kwargs)
            else:
                out_teacher = self.teacher(x, report_info = tumor_info, use_report = send_report_teacher,age=age, sex=sex, **kwargs)
        
        return {'student': None, 'teacher': out_teacher}, send_report_teacher
    
    def forward_student(self, x,tumor_info, send_report_teacher,
                        annotated_per_voxel='unknown',no_mask_training='unknown',age=None,sex=None, **kwargs):
        
        
        B = x.shape[0]
        device = x.device

        send_report_student = send_report_teacher & (torch.rand(B, device=device) < self.student_report_info_prob)  # bool [B]
        
        if self.train_mode == 'teacher_decoder':
            #teacher and student are the same net, student is the original architecture (e.g., medformer), teacher is the output of the report-informed decoder
            out = self.teacher(x, report_info = tumor_info, use_report = send_report_teacher, annotated_per_voxel=annotated_per_voxel, no_mask_training=no_mask_training,
                               skip_report_informed_decoder=True,age=age, sex=sex,**kwargs)
            out_student = out
            
        else:
            if self.train_mode != 'teacher':
                # Student forward
                if self.student_report_info_prob>0:
                    out_student = self.student(x, report_info = tumor_info, use_report = send_report_student,age=age, sex=sex, **kwargs)
                else:
                    out_student = self.student(x, report_info = None, age=age, sex=sex,**kwargs)
            else:
                out_student = None
        
        return {'student': out_student, 'teacher': None}
    
    def return_inference_net(self):
        if self.train_mode in ['teacher','teacher_decoder']:
            return self.teacher
        elif self.train_mode in ['student','both','teacher_student_ema']:
            return self.student



class KernelSelectorMLP(nn.Module):
    """
    Dynamic Convolution-style kernel selection module that outputs per-sample weights/biases.
    Returns the convolution parameters.

    Inputs (batched):
      - x_features:   [B, Cin, D,H,W]
      - out_mask:     [B, Cm,  D,H,W]
      - report_embed: [B, Dr]

    Outputs (batched):
      - weight_b: [B, Cout, Cin, k,k,k]
      - bias_b:   [B, Cout]
      - gates:    [B, K]  (softmax weights over experts)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dim_report: int,
        num_mask_channels: int,
        num_experts: int = 16,
        hidden_dim: int = 256,
        num_mlp_layers: int = 2,
        use_layernorm: bool = True,
        temperature: float = 1.0,
        init_scale: float = 0.02,
        gate_dropout: float = 0.1,          # now applied to logits (not input)
        use_noisy_gating: bool = False,
        noisy_gating_std: float = 1.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel_size (e.g., 3) for 'same' padding convenience."
        assert num_experts >= 1

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.num_experts = int(num_experts)
        self.dim_report = int(dim_report)
        self.num_mask_channels = int(num_mask_channels)
        self.temperature = float(temperature)

        # Expert bank (learnable)
        self.weight_bank = nn.Parameter(
            torch.empty(num_experts, out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias_bank = nn.Parameter(torch.empty(num_experts, out_channels))
        nn.init.normal_(self.weight_bank, mean=0.0, std=init_scale)
        nn.init.zeros_(self.bias_bank)

        # Gating input: GAP(features) + GAP(mask) + report embedding
        gate_in_dim = in_channels + num_mask_channels + dim_report

        # Normalize the 3 inputs separately (single normalization each), then concatenate.
        if use_layernorm:
            self.x_norm = nn.LayerNorm(in_channels)
            self.m_norm = nn.LayerNorm(num_mask_channels)
            self.r_norm = nn.LayerNorm(dim_report)
        else:
            self.x_norm = nn.Identity()
            self.m_norm = nn.Identity()
            self.r_norm = nn.Identity()

        # MLP -> logits for K experts
        layers = []
        d = gate_in_dim
        num_mlp_layers = max(1, int(num_mlp_layers))
        for _ in range(num_mlp_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers += [nn.Linear(d, num_experts)]
        self.gate_mlp = nn.Sequential(*layers)

        # Dropout moved to logits (before softmax)
        self.logits_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else nn.Identity()

        self.use_noisy_gating = bool(use_noisy_gating)
        self.noisy_gating_std = float(noisy_gating_std)

    @staticmethod
    def gap_3d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")
        return x.mean(dim=(2, 3, 4))

    def forward(
        self,
        x_features: torch.Tensor,      # [B, Cin, D,H,W]
        out_mask: torch.Tensor,        # [B, Cm,  D,H,W]
        report_embed: torch.Tensor,    # [B, Dr]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_features.shape[1] != self.in_channels:
            raise ValueError(f"x_features channels mismatch: expected {self.in_channels}, got {x_features.shape[1]}")
        if out_mask.shape[1] != self.num_mask_channels:
            raise ValueError(f"out_mask channels mismatch: expected {self.num_mask_channels}, got {out_mask.shape[1]}")
        if report_embed.ndim != 2 or report_embed.shape[1] != self.dim_report:
            raise ValueError(f"report_embed must be [B,{self.dim_report}], got {tuple(report_embed.shape)}")

        B = x_features.shape[0]
        if out_mask.shape[0] != B or report_embed.shape[0] != B:
            raise ValueError("Batch size mismatch among x_features, out_mask, report_embed")

        x_vec = self.gap_3d(x_features)     # [B, Cin]
        m_vec = self.gap_3d(out_mask)       # [B, Cm]
        r_vec = report_embed               # [B, Dr]

        # Separate normalization (single normalization each)
        x_vec = self.x_norm(x_vec)
        m_vec = self.m_norm(m_vec)
        r_vec = self.r_norm(r_vec)

        gate_in = torch.cat([x_vec, m_vec, r_vec], dim=1)  # [B, Cin+Cm+Dr]

        logits = self.gate_mlp(gate_in)     # [B, K]

        # Dropout on logits (before softmax)
        logits = self.logits_dropout(logits)

        if self.use_noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * self.noisy_gating_std

        gates = F.softmax(logits / self.temperature, dim=1)  # [B, K]

        # Mix expert bank -> per-sample parameters
        weight_b = torch.einsum("bk,kocdhw->bocdhw", gates, self.weight_bank)
        bias_b = torch.einsum("bk,ko->bo", gates, self.bias_bank)

        return weight_b, bias_b

def _same_padding_3d(kernel_size: int) -> Tuple[int, int, int]:
    """'Same' padding for odd kernel sizes (e.g., 3 -> 1)."""
    assert kernel_size % 2 == 1, f"kernel_size must be odd for true 'same' padding, got {kernel_size}"
    p = kernel_size // 2
    return (p, p, p)


def per_sample_conv3d(
    x: torch.Tensor,                         # [B, Cin, D,H,W]
    weight_b: torch.Tensor,                  # [B, Cout, Cin/groups, k,k,k]
    bias_b: Optional[torch.Tensor] = None,   # [B, Cout] or None
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """
    Per-sample Conv3d via grouped-conv trick.

    - x:        [B, Cin, D,H,W]
    - weight_b: [B, Cout, Cin/groups, k,k,k]
    - bias_b:   [B, Cout] or None
    - groups:   conv groups PER SAMPLE (like normal conv groups).
                For depthwise conv: groups=Cin, Cout=Cin, Cin/groups=1.

    Returns:
      y: [B, Cout, D',H',W']
    """
    if x.ndim != 5:
        raise ValueError(f"x must be [B,C,D,H,W], got {tuple(x.shape)}")
    if weight_b.ndim != 6:
        raise ValueError(f"weight_b must be [B,Cout,Cin/groups,k,k,k], got {tuple(weight_b.shape)}")

    B, Cin, D, H, W = x.shape
    Bw, Cout, Cin_g, k1, k2, k3 = weight_b.shape
    if Bw != B:
        raise ValueError(f"Batch mismatch: x has B={B}, weight_b has B={Bw}")

    if Cin % groups != 0:
        raise ValueError(f"Cin={Cin} must be divisible by groups={groups}")
    if Cout % groups != 0:
        raise ValueError(f"Cout={Cout} must be divisible by groups={groups}")

    expected_cin_g = Cin // groups
    if Cin_g != expected_cin_g:
        raise ValueError(f"weight_b expects Cin/groups={Cin_g}, but Cin/groups={expected_cin_g}")

    if bias_b is not None and tuple(bias_b.shape) != (B, Cout):
        raise ValueError(f"bias_b must be [B,Cout]=({B},{Cout}), got {tuple(bias_b.shape)}")

    # [1, B*Cin, D,H,W]
    x_g = x.reshape(1, B * Cin, D, H, W)

    # [B*Cout, Cin/groups, k,k,k]
    w_g = weight_b.reshape(B * Cout, Cin_g, k1, k2, k3)

    # [B*Cout]
    b_g = bias_b.reshape(B * Cout) if bias_b is not None else None

    # groups_total = B * groups
    y = F.conv3d(
        x_g,
        w_g,
        bias=b_g,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=B * groups,
    )

    # [B, Cout, D',H',W']
    _, BCout, D2, H2, W2 = y.shape
    return y.reshape(B, Cout, D2, H2, W2)


def per_sample_depthwise_conv3d(
    x: torch.Tensor,                         # [B, Cin, D,H,W]
    weight_b: torch.Tensor,                  # [B, Cin, 1, k,k,k]
    bias_b: Optional[torch.Tensor] = None,   # [B, Cin] or None
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
) -> torch.Tensor:
    """
    Convenience wrapper: per-sample depthwise conv.
    This is just per_sample_conv3d with groups=Cin.
    """
    if x.ndim != 5:
        raise ValueError(f"x must be [B,C,D,H,W], got {tuple(x.shape)}")
    B, Cin, _, _, _ = x.shape

    if weight_b.ndim != 6 or weight_b.shape[0] != B or weight_b.shape[1] != Cin or weight_b.shape[2] != 1:
        raise ValueError(
            f"weight_b must be [B, Cin, 1, k,k,k]. Got {tuple(weight_b.shape)} for Cin={Cin}"
        )
    if bias_b is not None and tuple(bias_b.shape) != (B, Cin):
        raise ValueError(f"bias_b must be [B,Cin]=({B},{Cin}), got {tuple(bias_b.shape)}")

    return per_sample_conv3d(
        x,
        weight_b,
        bias_b=bias_b,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=Cin,
    )

from .mask2former_modeling.transformer_decoder import mask2former_transformer_decoder3d as dec


class DynamicUNetDecoder3D(nn.Module):
    """
    nnU-Net-like light decoder (functional behavior):
      - per stage:
          ConvTranspose3d upsample (learned) ->
          concat skip ->
          Conv-A (dynamic, PER-SAMPLE) -> InstanceNorm3d -> LeakyReLU ->
          Conv-B (dynamic, PER-SAMPLE) -> InstanceNorm3d -> LeakyReLU
    """

    def __init__(
        self,
        num_classes: int,
        *,
        
        # transformer decoder
        feature_in_channels = [320,256,128,64,32], #number of channels for each feature level we will cross attend to
        report_token_dims =  [1,1,1,10,10,30,10,1,1], #'tumor_organ_name','tumor_count','known_tumor_count','tumor_attenuation','tumor_malignancy','tumor_diameters','tumor_volumes', known_tumor_organ_x, known_tumor_slices_x
        max_spatial_size = 32,  # this gives you a resolution of 4mm at input size 128 and 1mm spacing. This already catches all tumors in out 130K dataset
        report_embed_dim = [25, 25, 65, 65, 65], #we do not provide size in the initial stages, before the deep supervision level
        use_depthwise_conv3: bool = False, #this convolution weigths come from DynamicConv (MLP-based selection of kernel)
        use_full_conv3: bool = True, #if True, Conv-A is full 3x3x3 conv; if False and use_depthwise_conv3=True, Conv-A is depthwise 3x3x3
        use_pointwise_conv1: bool = True, #this convolution weights come from the transformer decoder. Not implemented.
        conv_bias_transp: bool = True,
        eps: float = 1e-5,
        affine: bool = True,
        negative_slope: float = 1e-2,
        inplace: bool = True,
        mask_stage = True, #passes the masks to the transformers, in the very beginning
        deep_supervision = True,
        deep_supervision_level = 2,
        crop_size = 128,
        age_and_sex_provided = False,
        use_transformer_conv3 = False,
        conv3_num_experts = 16,
        conv3_gate_dropout = 0.1,
        conv3_temperature = 1.0,
        conv3_use_noisy_gating = False,
        conv3_noisy_gating_std = 1.0,
        time_points=1,
        time_fusion='early',
    ):
        super().__init__()
        
        if time_points>1 and time_fusion=='early':
            num_inputs_mask_stage = num_classes+(2*time_points)
            slices_location_ch = 2*time_points
            if age_and_sex_provided:
                report_embed_dim = [v+2*time_points for v in report_embed_dim]
        else:
            num_inputs_mask_stage = num_classes+2
            slices_location_ch = 2
            if age_and_sex_provided:
                report_embed_dim = [v+2 for v in report_embed_dim]
        
        
        if (not use_full_conv3) and (not use_pointwise_conv1):
            raise ValueError("Need at least one: use_full_conv3=True (3x3x3) or use_pointwise_conv1=True (1x1x1).")

        self.use_depthwise_conv3 = use_depthwise_conv3
        self.use_pointwise_conv1 = use_pointwise_conv1
        self.crop_size = crop_size

        if self.use_depthwise_conv3 and not self.use_pointwise_conv1:
            raise ValueError(
                "use_depthwise_conv3=True requires use_pointwise_conv1=True, "
                "because depthwise Conv-A preserves 2*out_ch channels and Conv-B is needed to map to out_ch."
            )

        n_stages_encoder = len(feature_in_channels)-1
        #each stage does a 2x upsampling. So, we do not need a stage for the final segmentation head
        #eg: feature_in_channels = [320,256,128,64,32] -> n_stages_encoder=4: 320->256, 256->128, 128->64, 64->32
        
        self.transpconvs = nn.ModuleList()

        # Norm+Nonlin after Conv-B (always instantiated for simplicity)
        self.norms_3x3x3 = nn.ModuleList()
        self.nonlins_3x3x3 = nn.ModuleList()
        
        
        # Norm+Nonlin after Conv-c (always instantiated for simplicity)
        self.norms_1x1x1 = nn.ModuleList()
        self.nonlins_1x1x1 = nn.ModuleList()
        
        self.skip_norms = nn.ModuleList()
        self.feat_norms = nn.ModuleList()
        self.mask_norms = nn.ModuleList()
        
        
        kernel_channels_transformer = []
        
        if use_full_conv3 and not use_transformer_conv3:
            self.kernel_selectors_MLP = nn.ModuleList()

        for s in range(n_stages_encoder):
            
            #transpose is done before concatenation.
            # 320 -> 256, 256 -> 128, 128 -> 64, 64 -> 32
            self.transpconvs.append(
                nn.ConvTranspose3d(
                    in_channels=feature_in_channels[s],
                    out_channels=feature_in_channels[s+1],
                    kernel_size=2,
                    stride=2,
                    bias=conv_bias_transp,
                )
            )
            
            self.skip_norms.append(nn.InstanceNorm3d(feature_in_channels[s+1], eps=eps, affine=affine))
            self.feat_norms.append(nn.InstanceNorm3d(feature_in_channels[s+1], eps=eps, affine=affine))
            self.mask_norms.append(nn.InstanceNorm3d(slices_location_ch, eps=eps, affine=affine)) #2 channels: organ mask and slices mask

            
            if use_full_conv3:
                if use_pointwise_conv1:
                    out_dim = 2*feature_in_channels[s+1]+slices_location_ch #+slices_location_ch is for the organ mask and the slices mask
                else:
                    out_dim = feature_in_channels[s+1]
                #after skip. output of the transpose conv was concatenated with the skip connection, creating 2*feature_in_channels[s+1]
                if not use_transformer_conv3:
                    self.kernel_selectors_MLP.append(KernelSelectorMLP(
                                                        in_channels = 2*feature_in_channels[s+1]+slices_location_ch,#+slices_location_ch is for the organ mask and the slices mask
                                                        out_channels = out_dim,
                                                        kernel_size = 3,
                                                        dim_report = report_embed_dim[s],
                                                        num_mask_channels = num_classes,))
                self.norms_3x3x3.append(nn.InstanceNorm3d(out_dim, eps=eps, affine=affine))
                self.nonlins_3x3x3.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))
            elif use_depthwise_conv3:
                raise NotImplementedError("Depthwise conv3 kernel selector not implemented in this version.")
            
            if use_pointwise_conv1:
                #after the 3x3x3 conv
                kernel_channels_transformer.append([2*feature_in_channels[s+1]+slices_location_ch, feature_in_channels[s+1]])#+slices_location_ch is for the organ mask and the slices mask
                self.norms_1x1x1.append(nn.InstanceNorm3d(feature_in_channels[s+1], eps=eps, affine=affine))
                self.nonlins_1x1x1.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))
            
        self._pad_3 = _same_padding_3d(3)
        
        #transformer decoder
        self.use_transformer_conv3 = bool(use_transformer_conv3)
        if use_pointwise_conv1:
            if mask_stage:
                feature_in_channels_t = [num_inputs_mask_stage]+[k[0] for k in kernel_channels_transformer]+[feature_in_channels[-1]+slices_location_ch] #include mask stage
                kernel_channels_t = [[num_inputs_mask_stage,num_classes]]+kernel_channels_transformer+[[feature_in_channels[-1]+slices_location_ch, num_classes]]
                generate_kernels_t = [False] + [True]*len(kernel_channels_transformer)+[True]
                generate_kernels_t_3x3 = [False] + [True]*len(kernel_channels_transformer)+[False]
            else:
                feature_in_channels_t = [k[0] for k in kernel_channels_transformer]+[feature_in_channels[-1]+slices_location_ch]
                kernel_channels_t = kernel_channels_transformer+[[feature_in_channels[-1]+slices_location_ch, num_classes]]
                generate_kernels_t = 'all'
                generate_kernels_t_3x3 = [True]*len(kernel_channels_transformer)+[False]
            self.transformer_decoder = dec.MultiScaleMaskedTransformerDecoder3d(
                num_classes_segmentation = num_classes,
                feature_in_channels = feature_in_channels_t, #the 1x1x1 conv will not change the number of channels
                kernel_channels = kernel_channels_t, #final conv: segmentation head, not used in a decoder stage
                report_token_dims = report_token_dims,
                use_report = True, 
                max_spatial_size = max_spatial_size, 
                dec_layers = 2*len(feature_in_channels_t),#2 transformer layers per level, you can reduce this for a lighter model
                generate_kernels = generate_kernels_t,
                # 3x3x3 conv 
                use_conv3_bank=use_transformer_conv3,
                conv3_num_experts=conv3_num_experts,
                conv3_gate_dropout=conv3_gate_dropout,
                conv3_temperature=conv3_temperature,
                conv3_use_noisy_gating=conv3_use_noisy_gating,
                conv3_noisy_gating_std=conv3_noisy_gating_std,
                generate_kernels_t_3x3=generate_kernels_t_3x3
            )
        else:
            self.dynamic_head = KernelSelectorMLP(
                                                in_channels = feature_in_channels[-1]+slices_location_ch,
                                                out_channels = num_classes,
                                                kernel_size = 1,
                                                dim_report = report_embed_dim[-1],
                                                num_mask_channels = num_classes)
            
        self.use_full_conv3 = use_full_conv3
        if use_full_conv3 and use_depthwise_conv3:
            raise ValueError("Choose either use_full_conv3 or use_depthwise_conv3, not both.")
        self.mask_stage = mask_stage
        
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux_head = nn.Conv3d(feature_in_channels[deep_supervision_level], 
                                      num_classes, kernel_size=1)
            self.deep_supervision_level = deep_supervision_level
        self.head_feat_norm = nn.InstanceNorm3d(feature_in_channels[-1], eps=eps, affine=affine)
        self.head_mask_norm = nn.InstanceNorm3d(slices_location_ch, eps=eps, affine=affine) #2 channels: organ mask and slices mask
        self.slices_location_ch = slices_location_ch
        

    @property
    def num_stages(self) -> int:
        return len(self.transpconvs)

    def forward_stage(
        self,
        x: torch.Tensor, #features
        skip: torch.Tensor, #skip connection from encoder
        stage_idx: int,
        out_mask: torch.Tensor,
        report_embed = None,
        report_tokens_list = None,
        queries = None,
        conv3_kernel: Optional[torch.Tensor] = None,
        conv3_bias: Optional[torch.Tensor] = None,
        conv1_kernel: Optional[torch.Tensor] = None,
        conv1_bias: Optional[torch.Tensor] = None,
        tumor_location_slices = None,
    ) -> torch.Tensor:
        """
        Run exactly ONE decoder stage, using PER-SAMPLE convolution weights.
        """
        if not (0 <= stage_idx < self.num_stages):
            raise IndexError(f"stage_idx must be in [0, {self.num_stages - 1}], got {stage_idx}")

        if x.ndim != 5 or skip.ndim != 5:
            raise ValueError(f"x and skip must be 5D [B,C,D,H,W], got x={tuple(x.shape)} skip={tuple(skip.shape)}")

        B = x.shape[0]
        if skip.shape[0] != B:
            raise ValueError(f"Batch mismatch: x has B={B}, skip has B={skip.shape[0]}")

        out_ch = skip.shape[1]

        # 1) Upsample
        x_up = self.transpconvs[stage_idx](x)  # [B, out_ch, ...]
        if x_up.shape[0] != B or x_up.shape[1] != out_ch:
            raise RuntimeError(
                f"Stage {stage_idx}: unexpected transpconv output shape {tuple(x_up.shape)}; expected [B,{out_ch},...]"
            )

        # 2) Concat
        if x_up.shape[2:] != skip.shape[2:]:
            raise RuntimeError(
                f"Spatial mismatch at stage {stage_idx}: x_up {tuple(x_up.shape)} vs skip {tuple(skip.shape)}"
            )

        #normalize x_up, skip, tumor_location_slices before concatenation
        x_up_norm = self.feat_norms[stage_idx](x_up)
        skip_norm = self.skip_norms[stage_idx](skip)
        tumor_location_slices_norm = self.mask_norms[stage_idx](tumor_location_slices)
        x_cat = torch.cat([x_up_norm, skip_norm,tumor_location_slices_norm], dim=1)  # [B, 2*out_ch+2, ...]
        in_ch_cat = x_cat.shape[1]
        expected_in_ch_cat = 2 * out_ch + self.slices_location_ch
        if in_ch_cat != expected_in_ch_cat:
            raise RuntimeError(
                f"Stage {stage_idx}: expected concat channels {expected_in_ch_cat} (=2*out_ch+2), got {in_ch_cat}"
            )
                

        # 3A) Conv-A (PER-SAMPLE)
        if self.use_depthwise_conv3:
            # conv3_kernel: [B, in_ch_cat, 1, 3,3,3]

            x_a = per_sample_depthwise_conv3d(
                x_cat,
                conv3_kernel,
                bias_b=conv3_bias,
                stride=1,
                padding=self._pad_3,
            )  # [B, in_ch_cat, ...]
            
            # 4A) Norm + Nonlin
            x_a = self.norms_3x3x3[stage_idx](x_a)
            x_a = self.nonlins_3x3x3[stage_idx](x_a)

        elif self.use_full_conv3:
            if conv3_kernel is None:
                if not self.use_transformer_conv3:
                    #use the kernel selector MLP to get the conv3_kernel and conv3_bias
                    conv3_kernel, conv3_bias = self.kernel_selectors_MLP[stage_idx](
                        x_features=x_cat,      # [B, Cin+3, D,H,W]
                        out_mask=out_mask,        # [B, Cm,  D,H,W]
                        report_embed=report_embed,    # [B, Dr]
                    )
                else:
                    print(f'Using transformer decoder to generate conv3', flush=True)
                    #use the transformer decoder to get the conv3_kernel and conv3_bias
                    if self.mask_stage:
                        t_level = stage_idx + 1
                    else:
                        t_level = stage_idx
                    queries, conv3_kernel, conv3_bias = self.transformer_decoder(
                        x = x_cat,
                        level_index = t_level,
                        report_tokens_list = report_tokens_list,
                        output_mask_avg_pool = torch.mean(out_mask, dim=(-1,-2,-3), keepdim=False),
                        previous_queries = queries,
                        layer_to_run = 'first_half',
                        generate_1x1x1_kernel=False,
                        generate_3x3x3_kernel=True,
                        )

            # full conv3_kernel: usually [B, 2*out_ch+2, in_ch_cat, 3,3,3]
            if conv3_kernel.ndim != 6 or tuple(conv3_kernel.shape[3:]) != (3, 3, 3):
                raise RuntimeError(
                    f"Stage {stage_idx}: conv3_kernel must be [B,y,x,3,3,3], got {tuple(conv3_kernel.shape)}"
                )
            if conv3_kernel.shape[0] != B:
                raise RuntimeError(
                    f"Stage {stage_idx}: expected conv3_kernel [B,y,x,3,3,3], got {tuple(conv3_kernel.shape)}"
                )

            x_a = per_sample_conv3d(
                x_cat,
                conv3_kernel,
                bias_b=conv3_bias,
                stride=1,
                padding=self._pad_3,
                groups=1,
            )  # [B, 2*out_ch, ...]
            
             # 4A) Norm + Nonlin
            x_a = self.norms_3x3x3[stage_idx](x_a)
            x_a = self.nonlins_3x3x3[stage_idx](x_a)
            
        else:
            x_a = x_cat
            #print(f'Skipping DynamicConv')

       

        # 3B) Conv-B (optional pointwise, PER-SAMPLE)
        if self.use_pointwise_conv1:
            if conv1_kernel is None:
                #use the transformer decoder to get the conv1_kernel and conv1_bias
                if self.mask_stage:
                    t_level = stage_idx + 1
                else:
                    t_level = stage_idx
                queries, conv1_kernel, conv1_bias = self.transformer_decoder(
                    x = x_a,
                    level_index = t_level,
                    report_tokens_list = report_tokens_list,
                    output_mask_avg_pool = torch.mean(out_mask, dim=(-1,-2,-3), keepdim=False),
                    previous_queries = queries,
                    layer_to_run = 'both' if not self.use_transformer_conv3 else 'second_half')

            if conv1_kernel.ndim != 6 or tuple(conv1_kernel.shape[3:]) != (1, 1, 1):
                raise RuntimeError(
                    f"Stage {stage_idx}: conv1_kernel must be [B,out_ch,in_ch,1,1,1], got {tuple(conv1_kernel.shape)}"
                )

            in_ch_1 = x_a.shape[1]
            if conv1_kernel.shape[0] != B:
                raise RuntimeError(
                    f"Stage {stage_idx}: expected conv1_kernel [B,x,x,1,1,1], got {tuple(conv1_kernel.shape)}"
                )

            x_out = per_sample_conv3d(
                x_a,
                conv1_kernel,
                bias_b=conv1_bias,
                stride=1,
                padding=0,
                groups=1,
            )  # [B, out_ch, ...]

            # 4B) Norm + Nonlin
            x_out = self.norms_1x1x1[stage_idx](x_out)
            x_out = self.nonlins_1x1x1[stage_idx](x_out)

        else:
            # only allowed when not using depthwise
            x_out = x_a
            #print(f'Skipping PointwiseConv1x1x1')

        return x_out, queries
    
    def forward_segmentation_head(self,
                                    x: torch.Tensor, #features
                                    out_mask: torch.Tensor,
                                    report_embed = None,
                                    report_tokens_list = None,
                                    queries = None,
                                    tumor_location_slices=None):
        x = self.head_feat_norm(x)
        tumor_location_slices_norm = self.head_mask_norm(tumor_location_slices)
        x =  torch.cat([x, tumor_location_slices_norm], dim=1)  # [B, Cin+2, ...]
        
        if self.use_pointwise_conv1:
            #use the transformer decoder to get the conv1_kernel and conv1_bias
            if self.mask_stage:
                level_index = self.num_stages+1
            else:
                level_index = self.num_stages
            queries, conv1_kernel, conv1_bias = self.transformer_decoder(
                level_index = level_index, #final segmentation head
                x = x,
                report_tokens_list = report_tokens_list,
                output_mask_avg_pool = torch.mean(out_mask, dim=(-1,-2,-3), keepdim=False),
                previous_queries = queries)

            x_out = per_sample_conv3d(
                x,
                conv1_kernel,
                bias_b=conv1_bias,
                stride=1,
                padding=0,
                groups=1,
            )  # [B, out_ch, ...]
        else:
            #use the dynamic head to get the conv1_kernel and conv1_bias
            conv1_kernel, conv1_bias = self.dynamic_head(
                x_features=x,      # [B, Cin, D,H,W]
                out_mask=out_mask,        # [B, Cm,  D,H,W]
                report_embed=report_embed,    # [B, Dr]
            )

            x_out = per_sample_conv3d(
                x,
                conv1_kernel,
                bias_b=conv1_bias,
                stride=1,
                padding=0,
                groups=1,
            )
        return x_out, queries
        
        
    
    def forward(self,
                unet_decoder_features, #features
                out_mask,
                report_vector_all,
                report_tokens_list_all,
                tumor_location_mask,
                tumor_allowed_slices,):
        
        features = unet_decoder_features[0] #bottleneck features
        unet_decoder_features = unet_decoder_features[1:]
        queries = None
        tumor_location_slices = torch.cat([tumor_location_mask.float(), tumor_allowed_slices.float()], dim=1)
        
        if self.mask_stage and self.use_pointwise_conv1:
            #begin by running the masks in the transformer first stage, just to refine the queries
            #features: mask, tumor_location_mask, tumor_allowed_slices
            queries = self.transformer_decoder(
                                level_index = 0,
                                x = torch.cat([out_mask.float(), tumor_location_mask.float(), tumor_allowed_slices.float()], dim=1),
                                report_tokens_list = report_tokens_list_all[0],
                                output_mask_avg_pool = torch.mean(out_mask, dim=(-1,-2,-3), keepdim=False),
                                previous_queries = None,
                                generate_1x1x1_kernel=False,
                                generate_3x3x3_kernel=False,
                                )
            
        
        for i in range(self.num_stages):
            #resize out_mask to current feature size
            tumor_location_slices_r = F.interpolate(tumor_location_slices, size=unet_decoder_features[i].shape[2:], mode='nearest')
            features,queries = self.forward_stage(
                                                x=features,
                                                skip=unet_decoder_features[i], #skip connection from encoder
                                                stage_idx=i,
                                                out_mask=out_mask,
                                                report_embed=report_vector_all[i],#report embeddings for late stages include size
                                                report_tokens_list=report_tokens_list_all[i],#report tokens for late stages include size
                                                queries=queries,
                                                tumor_location_slices = tumor_location_slices_r
                                                )
            if self.deep_supervision and i==(self.deep_supervision_level-1):
                #upsample to 128x128x128 (crop_size) if needed
                if features.shape[2:] != (self.crop_size, self.crop_size, self.crop_size):
                    aux_feature = F.interpolate(features, size=(self.crop_size, self.crop_size, self.crop_size), 
                                                mode='trilinear', align_corners=True)
                aux_out = self.aux_head(aux_feature)
        
        features, queries = self.forward_segmentation_head(
                                                x=features,
                                                out_mask=out_mask,
                                                report_embed=report_vector_all[-1],#report embeddings for late stages include size
                                                report_tokens_list=report_tokens_list_all[-1],#report tokens for late stages include size
                                                queries=queries,
                                                tumor_location_slices = tumor_location_slices
                                            )
        
        primary = [features, aux_out] if self.deep_supervision else features
        return {'refined_segmentation': primary}