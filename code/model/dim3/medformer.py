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
                    skip_channel = torch.zeros_like(x[:, :1, :, :, :])
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
            tmp_map = torch.zeros(1, device=x.device)  # dummy value so gradient flows if needed


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
    
    

@torch.no_grad()
def find_pairs_after_gather_simple(
    patient_ids_local: torch.Tensor,   # [B] int64, 0 = unpairable
    organ_ids_local: torch.Tensor,     # [B] int64, 0 = unpairable
    dist=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return for each local sample:
      pair_gidx_local: [B] int64 global index of a paired sample, or -1 if none
      has_pair_local:  [B] bool

    Pair rule:
      - pairable iff patient_id != 0 AND organ_id != 0
      - A pair is any other sample (any rank) with same patient_id AND same organ_id.
      - Deterministic choice: the first matching global index (lowest gidx) that isn't itself.
    """
    device = patient_ids_local.device
    pid_l = patient_ids_local.to(dtype=torch.int64, device=device).view(-1)
    org_l = organ_ids_local.to(dtype=torch.int64, device=device).view(-1)
    B = pid_l.numel()
    assert org_l.numel() == B, f"organ_ids_local must have {B} elements, got {org_l.numel()} with shape {organ_ids_local.shape}"
    #print(f"Finding pairs for B={B} local samples on device {device} with dist={dist}", flush=True)
    
    
    if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
        world, rank = 1, 0
    else:
        world, rank = dist.get_world_size(), dist.get_rank()
    
    if world > 1:
        # gather all B to verify equal
        b = torch.tensor([B], device=device, dtype=torch.int64)
        bs = [torch.empty_like(b) for _ in range(world)]
        dist.all_gather(bs, b)
        bs = torch.cat(bs).cpu().tolist()
        assert len(set(bs)) == 1, f"find_pairs_after_gather_simple requires equal B across ranks, got {bs}"

    # -------- gather across ranks --------
    if dist is None or (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
        pid_g = pid_l
        org_g = org_l
    else:
        pid_list = [torch.empty_like(pid_l) for _ in range(world)]
        org_list = [torch.empty_like(org_l) for _ in range(world)]
        dist.all_gather(pid_list, pid_l)
        dist.all_gather(org_list, org_l)
        pid_g = torch.cat(pid_list, dim=0)  # [G]
        org_g = torch.cat(org_list, dim=0)  # [G]
        
    #print(f'Global patient ids: {pid_g}', flush=True)
    #print(f'Global organ ids: {org_g}', flush=True)

    G = pid_g.numel()
    my_gidx_base = rank * B  # assumes equal B across ranks (same as your original)

    # -------- find partner for each local item --------
    pair_gidx_local = torch.full((B,), -1, device=device, dtype=torch.int64)

    for i in range(B):
        pid = pid_l[i].item()
        org = org_l[i].item()

        # unpairable (includes your 'random' mapped to 0)
        if pid == 0 or org == 0:
            continue

        my_gidx = my_gidx_base + i

        # 1) match patient
        same_pid = (pid_g == pid)
        # 2) match organ within patient matches
        same_both = same_pid & (org_g == org)

        # exclude self
        same_both[my_gidx] = False

        hits = torch.nonzero(same_both, as_tuple=False).flatten()
        if hits.numel() > 0:
            pair_gidx_local[i] = hits[0]  # deterministic: first global hit

    has_pair_local = pair_gidx_local >= 0
    return pair_gidx_local, has_pair_local


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


def gather_other_time(
    *,
    pair_gidx_local: torch.Tensor,                 # [B] int64, -1 = none
    out_mask_local: torch.Tensor,                  # [B, ...]
    tumor_location_mask_local: torch.Tensor,       # [B,1,D,H,W]
    tumor_allowed_slices_local: torch.Tensor,      # [B,1,D,H,W]
    report_tokens_list_all_local: List[List[torch.Tensor]],  # levels -> tokens -> [B,dim]
    dist=None,
    is_paired=None,
) -> Tuple[
    torch.Tensor,
    List[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    List[List[torch.Tensor]],
]:
    """
    Returns:
      out_mask_pair_local
      unet_decoder_features_pair_local
      tumor_location_mask_pair_local
      tumor_allowed_slices_pair_local
      report_tokens_list_all_pair_local
    """
    device = out_mask_local.device
    pair_gidx_local = pair_gidx_local.to(device=device, dtype=torch.int64)

    # ---- gather + index simple tensors ----
    out_mask_g = all_gather_no_grad(out_mask_local, dist=dist)  # [G,...]
    out_mask_pair = take_pairs_from_global(out_mask_g, pair_gidx_local)
    

    tlm_g = all_gather_no_grad(tumor_location_mask_local, dist=dist)  # [G,1,D,H,W]
    tlm_pair = take_pairs_from_global(tlm_g, pair_gidx_local)

    tas_g = all_gather_no_grad(tumor_allowed_slices_local, dist=dist)
    tas_pair = take_pairs_from_global(tas_g, pair_gidx_local)

    # ---- gather + index report_tokens_list_all (levels -> tokens) ----
    report_tokens_pair_all: List[List[torch.Tensor]] = []
    for level_tokens in report_tokens_list_all_local:
        level_pair: List[torch.Tensor] = []
        for tok in level_tokens:
            tok_g = all_gather_no_grad(tok, dist=dist)          # [G,dim]
            tok_pair = take_pairs_from_global(tok_g, pair_gidx_local)  # [B,dim]
            level_pair.append(tok_pair)
        report_tokens_pair_all.append(level_pair)

    return out_mask_pair, tlm_pair, tas_pair, report_tokens_pair_all

counter_dsc_print = 0

class MedFormer(nn.Module):
    
    def __init__(self, 
        in_chan, 
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
        aggregator_mode = None,  # depracated
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
        time_points = 1,
        use_transformer_conv3 = False,
        concat_time_info = False,
        use_time_consistency_loss = False,
        registration_input_shape = (175, 175, 175),
        registration_mode = 'image',
        initialization_registration = 'unigradicon',
        ablate_dynamic_decoder = False,
        ):
        super().__init__()

        # set early so create_report_informed_decoder can read via getattr
        self.ablate_dynamic_decoder = bool(ablate_dynamic_decoder)

        self.give_tumor_size_input = give_tumor_size_input
        self.time_points = time_points
        if time_points>1:
            assert use_transformer_decoder, "Time points > 1 requires use_transformer_decoder to be True; not implemented here for early fusion or report info in input"
        if time_points>2:
            raise NotImplementedError("Time points > 2 not implemented yet")

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
            self.aux_out = nn.Conv3d(chan_num[5], num_classes, kernel_size=1)

        self.outc = nn.Conv3d(chan_num[7], num_classes, kernel_size=1)

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
            raise ValueError('gate_cls was deprecated.')
        else:
            self.gate_cls = None

        
        if cls_on_output:
            self.cls_on_output = make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                out_class_number=len(class_list_cls))
        else:
            self.cls_on_output = None
            
        if cls_on_segmentation:
            self.age_and_sex_provided = age_and_sex_provided
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
            m =  make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                       out_class_number=10*len(tumor_cls)*3, num_input_ch = (2*len(tumor_cls)))
            self.tumor_classifier = attribute_classifier(class_list_seg,train_on_mask_only=train_att_MLP_on_mask_only,
                                                                   model=m,calculate_HU=False,mode='tumor_classifier',
                                                                   loss_weight=loss_weight_cls)
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
                                                use_transformer_conv3=use_transformer_conv3,
                                                concat_time_info=concat_time_info)
        self.prob_size = prob_size
        self.never_give_size_decoder = never_give_size_decoder
        self.age_and_sex_provided = age_and_sex_provided
        self.class_list_seg = class_list_seg
        
        self.use_time_consistency_loss = use_time_consistency_loss
        if self.use_time_consistency_loss:
            assert time_points > 1, "use_time_consistency_loss requires time_points > 1"
            self.make_registration_module(registration_module,registration_mode=registration_mode,
                                          registration_input_shape=registration_input_shape,
                                          initialization_registration=initialization_registration)
        
    def create_bottleneck_transformer(self):
        self.bottleneck_transformer = BottleneckTransformer(self.chan_num[3])
        
        
    def create_report_informed_decoder(self,use_transformer_decoder=True,use_dynamic_conv=True,
                                       age_and_sex_provided=False,use_transformer_conv3=False,
                                       concat_time_info=False):
        if getattr(self, "ablate_dynamic_decoder", False):
            # Ablation: replace the dynamic-kernel decoder with a static-conv decoder.
            # Mirrors the I/O contract of DynamicUNetDecoder3D so the rest of the
            # forward (block_grad masking, classifiers, etc.) runs unchanged.
            self.report_informed_decoder = True
            num_classes = self.outc.out_channels
            self.report_decoder = StaticUNetDecoder3D(
                num_classes=num_classes,
                feature_in_channels=self.chan_num[3:],
            )
            self.age_and_sex_provided = age_and_sex_provided
            return
        self.report_informed_decoder = True
        #num classes: read the output size for self.outc
        num_classes = self.outc.out_channels
        token_dims = self.report_token_dims[:]
        report_embed_dim = [25, 25, 65, 65, 65]
        if age_and_sex_provided:
            token_dims = token_dims+[1,1]
        if self.time_points > 1:
            token_dims = token_dims + [1]  # tokens to give date
            report_embed_dim = [d+1 for d in report_embed_dim]  # increase embed dim to accommodate date token



        self.report_decoder = DynamicUNetDecoder3D(num_classes = num_classes,feature_in_channels = self.chan_num[3:],
                                                   use_pointwise_conv1=use_transformer_decoder,
                                                   use_full_conv3=use_dynamic_conv,
                                                   age_and_sex_provided=age_and_sex_provided,
                                                   report_token_dims=token_dims,
                                                   use_transformer_conv3=use_transformer_conv3,
                                                   time_points = self.time_points,
                                                   report_embed_dim=report_embed_dim,
                                                   concat_time_info=concat_time_info)



    
    def forward(self, x, report_info=None, stage_1_out=None,labels=None, cut_attenuation_grad = False, use_report = None, name=None,
                debugging=False, bottleneck_transformer=None,annotated_per_voxel=False,no_mask_training=False,
                skip_report_informed_decoder=False, age=None, sex=None, 
                dates = None, patient_ids = None, cropped_organs = None,
                dist = None, organ_cropped_cannon = None):
        
        original_ct = x[:]
        
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
            
        #print(f'Inside forward, age and sex is: {age},{sex}, will they be provided to the model? {self.age_and_sex_provided}', flush=True)
        
        
            
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
            report_vector = torch.zeros([x.shape[0],63], device=x.device)
            tumor_location_mask = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device)
            tumor_allowed_slices = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device)
            known_tumor_organ = torch.zeros_like(tumor_location_mask)
            known_tumor_slices = torch.zeros_like(tumor_allowed_slices)
            
        if self.report_information_in_input:
            
            if report_info is not None:
                device = x.device
                B = x.shape[0]
                if self.report_informed_decoder:
                    raise(f'The masking below (build_and_mask_report_inputs_by_key) is not compatible with report_informed_decoder. It risks masking twice.')
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
                report_vector_masked = torch.zeros([x.shape[0],63], device=x.device)
                tumor_location_mask_masked = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device)
                tumor_allowed_slices_masked = torch.zeros([x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]], device=x.device)
                known_tumor_organ_masked = torch.zeros_like(tumor_location_mask)
                known_tumor_slices_masked = torch.zeros_like(tumor_allowed_slices)
                block_grad_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
                use_report = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
            
            if not self.give_tumor_size_input:
                report_vector_masked = report_vector_masked[:,:23]
                
            if self.age_and_sex_provided:
                #print(f'Provided to input age {age}, sex {sex}', flush=True)
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
                    #print(f'Provided to input age {age}, sex {sex}', flush=True)
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
                    #print(f'Provided to output aux age {age}, sex {sex}', flush=True)
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
                #print(f'Provided to output age {age}, sex {sex}', flush=True)
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
                    'tumor_organ_name': torch.zeros([x.shape[0],1], device=x.device),
                    'tumor_count': torch.zeros([x.shape[0],1], device=x.device),
                    'known_tumor_count': torch.zeros([x.shape[0],1], device=x.device),
                    'tumor_attenuation': torch.zeros([x.shape[0],10], device=x.device),
                    'tumor_malignancy': torch.zeros([x.shape[0],10], device=x.device),
                    'tumor_diameters': torch.zeros([x.shape[0],30], device=x.device),
                    'tumor_volumes': torch.zeros([x.shape[0],10], device=x.device),
                }
            #assert all keys are in report_info
            for key in TOKEN_KEYS_ALL:
                if key not in report_info:
                    raise ValueError(f'Key {key} not found in report_info for report_informed_decoder.')

            import os as _os
            if _os.environ.get('DEBUG_REPORT_INFO') == '1':
                print('=== [DEBUG_REPORT_INFO] report_informed_decoder forward ===', flush=True)
                print(f'  training={self.training}  no_mask_training={no_mask_training}  '
                      f'never_give_size_decoder={getattr(self, "never_give_size_decoder", None)}', flush=True)
                print(f'  give_size_mask = {give_size_mask.tolist()}', flush=True)
                print(f'  use_report     = {use_report.tolist() if use_report is not None else None}', flush=True)
                for _k in TOKEN_KEYS_ALL:
                    _t = report_info[_k]
                    _v = _t[0].detach().float().cpu().tolist() if _t.numel() < 60 else \
                         (_t[0].detach().float().cpu().tolist()[:20] + ['…'])
                    print(f'  {_k:<22}  shape={tuple(_t.shape)}  sum={float(_t.float().sum()):.4f}  vals[0]={_v}', flush=True)
                if 'tumor_location_mask' in (report_info or {}):
                    _t = report_info['tumor_location_mask']
                    print(f'  tumor_location_mask    shape={tuple(_t.shape)}  '
                          f'sum={float(_t.float().sum()):.4f}  '
                          f'nonzero_frac={float((_t!=0).float().mean()):.6f}', flush=True)
                if 'tumor_allowed_slices' in (report_info or {}):
                    _t = report_info['tumor_allowed_slices']
                    print(f'  tumor_allowed_slices   shape={tuple(_t.shape)}  '
                          f'sum={float(_t.float().sum()):.4f}', flush=True)

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
                    if self.time_points>1:
                        assert dates is not None, "dates must be provided when time_points > 1"
                        dates = dates.to(device=device, dtype=report_vector_x.dtype)
                        assert dates.shape == (B, 1), f"dates must be of shape [B, 1], got {dates.shape}"
                        report_vector_x = torch.cat([report_vector_x, dates], dim=1)
                    report_vector_all.append(report_vector_x)
                    report_tokens = [report_info_no_size[k] for k in TOKEN_KEYS_ALL]
                    report_tokens += [known_tumor_organ_x, known_tumor_slices_x]
                    if self.age_and_sex_provided:
                        #print(f'Provided to decoder aux age {age}, sex {sex}', flush=True)
                        report_tokens += [age, sex]
                    if self.time_points>1:
                        report_tokens += [dates]
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
                    if self.time_points>1:
                        dates = dates.to(device=device, dtype=report_vector_x.dtype)
                        assert dates.shape == (B, 1), f"dates must be of shape [B, 1], got {dates.shape}"
                        report_vector_x = torch.cat([report_vector_x, dates], dim=1)
                    report_vector_all.append(report_vector_x)
                    report_tokens = [report_info_masked[k] for k in TOKEN_KEYS_ALL]
                    report_tokens += [known_tumor_organ_x, known_tumor_slices_x]
                    if self.age_and_sex_provided:
                        #print(f'Provided to decoder aux age {age}, sex {sex}', flush=True)
                        report_tokens += [age, sex]
                    if self.time_points>1:
                        report_tokens += [dates]
                    report_tokens_list_all.append(report_tokens)
                    
            out_mask = torch.sigmoid(out)
            
            if self.time_points>1:
                other_time_point, is_paired = find_pairs_after_gather_simple(
                                            patient_ids_local = patient_ids,  
                                            organ_ids_local = cropped_organs,  
                                            dist = dist,
                                        )
                (out_mask_other_time,
                tumor_location_mask_other_time,
                tumor_allowed_slices_other_time,
                report_tokens_list_all_other_time) = gather_other_time(
                    pair_gidx_local=other_time_point,
                    out_mask_local=out_mask,
                    tumor_location_mask_local=tumor_location_mask,
                    tumor_allowed_slices_local=tumor_allowed_slices,
                    report_tokens_list_all_local=report_tokens_list_all,
                    dist=dist,
                    is_paired=is_paired
                )
            else:
                (out_mask_other_time,tumor_location_mask_other_time,tumor_allowed_slices_other_time,report_tokens_list_all_other_time) = (None,)*4
                other_time_point = None
                is_paired = torch.zeros(B, device=device, dtype=torch.bool)
            
            # Diagnostic A/B: when DEBUG_ZERO_REPORT=1, REPLACE all report
            # tokens / vectors / spatial masks with zeros before calling the
            # report decoder. If the refined output is unchanged vs the real
            # tokens, the report-informed decoder is effectively ignoring its
            # report inputs.
            import os as _os_zero
            if _os_zero.environ.get('DEBUG_ZERO_REPORT') == '1':
                report_vector_all = [torch.zeros_like(v) for v in report_vector_all]
                report_tokens_list_all = [[torch.zeros_like(t) for t in toks]
                                          for toks in report_tokens_list_all]
                tumor_location_mask = torch.zeros_like(tumor_location_mask)
                tumor_allowed_slices = torch.zeros_like(tumor_allowed_slices)
                print('[DEBUG_ZERO_REPORT] zeroed report_vector_all, '
                      'report_tokens_list_all, tumor_location_mask, '
                      'tumor_allowed_slices before report_decoder call', flush=True)

            out_report_decoder = self.report_decoder(
                                      unet_decoder_features=unet_decoder_features,
                                      out_mask=out_mask,
                                      report_vector_all=report_vector_all,
                                      report_tokens_list_all=report_tokens_list_all,
                                      tumor_location_mask=tumor_location_mask,
                                      tumor_allowed_slices=tumor_allowed_slices,
                                      out_mask_other_time=out_mask_other_time,
                                      tumor_location_mask_other_time=tumor_location_mask_other_time,
                                      tumor_allowed_slices_other_time=tumor_allowed_slices_other_time,
                                      report_tokens_list_all_other_time=report_tokens_list_all_other_time,
                                      has_other_time=is_paired,
                                      other_time_point=other_time_point,
                                      dist=dist
                                      )
            out_refined, aux_out_refined = out_report_decoder["refined_segmentation"]  # feat is a tensor [B, C, D, H, W]
            
            if self.time_points > 1: 
                # gather across ranks: [G] bool
                m_all = all_gather_no_grad(block_grad_mask.to(torch.bool), dist=dist)
                if (is_paired is not None) and is_paired.any():
                    # pull partner mask back to local: [B] bool (unpaired -> 0)
                    m_pair = take_pairs_from_global(m_all, other_time_point).to(torch.bool)
                    # union: if either blocks, both block
                    block_grad_mask = block_grad_mask.to(torch.bool) | m_pair
            
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
        else:
            out_report_decoder = None
            
        if self.use_time_consistency_loss and self.training:
            # Registration loss is a training-only construct; at inference
            # `is_paired` is not in scope (computed only inside the
            # report-informed-decoder branch, which stage-1 skips). Guard with
            # self.training to skip the whole block at eval time.
            assert self.time_points > 1, "time_points must be > 1 to use time consistency loss"
            x_other_time_all = all_gather_no_grad(original_ct, dist=dist)
            labels_all = all_gather_no_grad(labels, dist=dist)
            if (is_paired is not None) and is_paired.any():
                #gather image form other time point
                x_other_time = take_pairs_from_global(x_other_time_all, other_time_point)
                #drop batch elements not paired
                image_A = original_ct[is_paired]
                image_B = x_other_time[is_paired]
                assert (other_time_point[is_paired] >= 0).all(), "negatives indicate some non-paired samples in the paired subset, which should not happen"
                #filter to only organ masks
                labels_other_time = take_pairs_from_global(labels_all, other_time_point)
                labels_paired = labels[is_paired]
                labels_other_time_paired = labels_other_time[is_paired]
                paired_idx = torch.where(is_paired)[0].tolist()
                organ_cropped_cannon_paired = [organ_cropped_cannon[i] for i in paired_idx]
                registration_loss, registration_field_other_to_current, spacing_registration = self.registration(
                                                                                 image=image_A, image_other_time=image_B,
                                                                                 out_other_time=out_refined[is_paired],
                                                                                 dist=dist, organ_cropped_cannon = organ_cropped_cannon_paired,
                                                                                 labels_paired=labels_paired,
                                                                                 labels_other_time_paired=labels_other_time_paired)
                registration_field_other_to_current = registration_field_other_to_current.detach()
                out_paired = out[is_paired]
                #print(f'Registration loss: {registration_loss.item()}', flush=True)
                
            else:
                #touch registration net parameters to avoid DDP complaint
                registration_loss = torch.zeros((), device=x.device, dtype=x.dtype)
                #touch the registration module parameters to avoid DDP complaining about unused parameters when registration loss is not computed (no pairs)
                for p in self.registration_module.parameters():
                    if p.requires_grad:
                        registration_loss = registration_loss + p.view(-1)[0] * 0.0  # uses one element only (cheap)
                out_paired = None
                registration_field_other_to_current = None
                spacing_registration = None
                #print('No paired samples, setting registration_loss to 0 and skipping registration.', flush=True)
        else:
            registration_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            out_paired = None
            registration_field_other_to_current = None
            spacing_registration = None
            other_time_point = None
            is_paired = None
            
                
        
        return self.prepare_return(out, aux_out=aux_out, y_class=y_class, y_class_2=y_class_2,
                                   y_clip=y_clip, aux_att=aux_att, out_att=out_att, y_tumor=y_tumor,
                                   y_class_on_seg_aux=y_class_on_seg_aux, y_class_on_seg=y_class_on_seg,
                                   bottleneck_features=bottleneck_features,
                                   mid_decoder_features=mid_decoder_features,
                                   out_features=out_features,
                                   out_report_decoder=out_report_decoder,
                                   registration_loss=registration_loss,
                                   out_paired = out_paired,
                                   registration_field_other_to_current=registration_field_other_to_current,
                                   spacing_registration = spacing_registration,
                                   other_time_point=other_time_point, is_paired=is_paired)
        
    def clean_name(self,name):
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
    
    def organ_name_to_idx(self, organ_cropped_cannon):
        assert isinstance(organ_cropped_cannon, list), f'organ_cropped_cannon must be a list of strings (one per batch item), got {type(organ_cropped_cannon)}'
        B = len(organ_cropped_cannon)
        all_indices = []
        for b in range(B):
            #find indices that match organ_cropped_cannon
            cropped_name = organ_cropped_cannon[b]
            found_indices = []
            for class_idx, class_name in enumerate(self.class_list_seg):
                if (class_name == cropped_name) or (self.clean_name(class_name) == self.clean_name(cropped_name)):
                    found_indices.append(class_idx)
            if len(found_indices) == 0:
                raise ValueError(f'Organ {cropped_name} not found in class_list_seg: {self.class_list_seg}')
            all_indices.append(found_indices)
        return all_indices
            
        
    def registration(self,image, image_other_time,out_other_time, dist, organ_cropped_cannon,
                     labels_paired,labels_other_time_paired):
        from icon_registration.mermaidlite import compute_warped_image_multiNC
        #register (create deformation)
        shape = image.shape[2:]
        
        #normalize images. Unigradicon was pre-trained on images normalized to [0,1]
        #mn = image.amin(dim=(-3, -2, -1), keepdim=True)
        #mx = image.amax(dim=(-3, -2, -1), keepdim=True)
        #image = (image - mn) / (mx - mn + 1e-5)
        #mn_other = image_other_time.amin(dim=(-3, -2, -1), keepdim=True)
        #mx_other = image_other_time.amax(dim=(-3, -2, -1), keepdim=True)
        #image_other_time = (image_other_time - mn_other) / (mx_other - mn_other + 1e-5)
        
        #get the mask
        organ_cropped_idx = self.organ_name_to_idx(organ_cropped_cannon)
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
        
        #mask images
        #binarize
        label_cropped_organ = (label_cropped_organ > 0.5).float()
        label_cropped_organ_other_time = (label_cropped_organ_other_time > 0.5).float()
        assert image.shape == label_cropped_organ.shape, f'Image shape {image.shape} and label_cropped_organ shape {label_cropped_organ.shape} must match for multiplication'
        assert image_other_time.shape == label_cropped_organ_other_time.shape, f'Image other time shape {image_other_time.shape} and label_cropped_organ_other_time shape {label_cropped_organ_other_time.shape} must match for multiplication'
        image_non_masked = image.clone()
        image_other_time_non_masked = image_other_time.clone()
        image = image * label_cropped_organ
        image_other_time = image_other_time * label_cropped_organ_other_time
    
        if tuple(shape) != tuple(self.registration_input_shape):
            image = F.interpolate(image, size=self.registration_input_shape, mode='trilinear', align_corners=False)
            image_other_time = F.interpolate(image_other_time, size=self.registration_input_shape, mode='trilinear', align_corners=False)
    
    
        
        if not self.train_registration:
            with torch.no_grad():
                registration_loss = self.registration_module(image_A=image, image_B=image_other_time)
        else:
            registration_loss = self.registration_module(image_A=image, image_B=image_other_time)
        registration_loss = registration_loss.all_loss.mean()
        AB_deformation_field = self.registration_module.phi_AB_vectorfield
        BA_deformation_field = self.registration_module.phi_BA_vectorfield
        
        # Warp segmentation
        #resample field to original size:
        phi_BA_vectorfield, spacing = resample_phi_and_spacing(self.registration_module.phi_BA_vectorfield, 
                                                               self.registration_module.spacing, out_other_time.shape[2:],
                                                               align_corners=False)
        
        global counter_dsc_print
        counter_dsc_print+=1
        
        main_process = (dist is None) or (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        
            
        if counter_dsc_print<10: #degug, you can remove this
            if counter_dsc_print < 10 and main_process: 
                image_non_masked_other_time_registered = compute_warped_image_multiNC(
                                                                            image_non_masked[0].unsqueeze(0).float(),
                                                                            phi_BA_vectorfield[0].unsqueeze(0),
                                                                            spacing,
                                                                            spline_order=1,
                                                                            zero_boundary=True
                                                                        )
                mask_organ_other_time_registered = compute_warped_image_multiNC(
                                                                            label_cropped_organ_other_time[0].unsqueeze(0).float(),
                                                                            phi_BA_vectorfield[0].unsqueeze(0),
                                                                            spacing,
                                                                            spline_order=0,
                                                                            zero_boundary=True
                                                                        )
                
                os.makedirs('SanityRegistration',exist_ok=True)
                os.makedirs('SanityRegistration/medformer_'+str(counter_dsc_print),exist_ok=True)
                save_tensor_as_nifti(image[0].squeeze(0),'SanityRegistration/medformer_'+str(counter_dsc_print)+'/image_inside_medformer.nii.gz')
                save_tensor_as_nifti(image_other_time[0].squeeze(0),'SanityRegistration/medformer_'+str(counter_dsc_print)+'/image_other_time_inside_medformer.nii.gz')
                save_tensor_as_nifti(image_non_masked_other_time_registered[0].squeeze(0),'SanityRegistration/medformer_'+str(counter_dsc_print)+'/image_non_masked_other_time_registered.nii.gz')
                save_tensor_as_nifti(label_cropped_organ[0].squeeze(0),'SanityRegistration/medformer_'+str(counter_dsc_print)+'/mask_inside_medformer.nii.gz')
                save_tensor_as_nifti(label_cropped_organ_other_time[0].squeeze(0),'SanityRegistration/medformer_'+str(counter_dsc_print)+'/mask_other_time_inside_medformer.nii.gz')
                save_tensor_as_nifti(mask_organ_other_time_registered[0].squeeze(0),'SanityRegistration/medformer_'+str(counter_dsc_print)+'/mask_other_time_registered_inside_medformer.nii.gz')
                #save the masks

            #register label and compare dice
            #deform labels_other_time_paired with the same field and compare to mask
            mask_organ_other_time_registered = compute_warped_image_multiNC(
                labels_other_time_paired.float(),
                phi_BA_vectorfield,
                spacing,
                spline_order=0,
                zero_boundary=True
            )
            
            #compute dice between mask_organ_other_time_registered and labels_paired, per organ class
            dsc = dice4d(mask_organ_other_time_registered, labels_paired)
            assert len(dsc.shape) == 2, f'Expected DSC output to be 2D [B, C], got {dsc.shape}'
            
            # labels_other_time_paired: [B,C,D,H,W] (float/bool ok)
            # labels_paired:            [B,C,D,H,W]
            # mask_organ_other_time_registered: [B,C,D,H,W]
            # dsc_after = dice4d(mask_organ_other_time_registered, labels_paired)  # [B,C]

            # BEFORE registration dice (unregistered other-time mask vs current mask)
            dsc_before = dice4d(labels_other_time_paired.float(), labels_paired)  # [B,C]
            dsc_after  = dsc  # your existing after-registration dice
            delta      = dsc_after - dsc_before  # [B,C]

            # init EMA once
            if not hasattr(self, "reg_dsc_ema"):
                self.reg_dsc_ema = EMAperClassDelta(class_names=self.class_list_seg, momentum=0.95, device=None)

            # update EMA (skips empty-target cases internally)
            self.reg_dsc_ema.update(dsc_before_bc=dsc_before, dsc_after_bc=dsc_after, target_mask_bcdhw=labels_paired)

            # print per-class (instant) + EMA
            for i, c in enumerate(self.class_list_seg):
                # skip if target mask empty for this class across the paired mini-batch
                if (labels_paired[:, i] > 0.5).sum() == 0:
                    continue

                before_i = dsc_before[:, i].mean().item()
                after_i  = dsc_after[:, i].mean().item()
                delta_i  = after_i - before_i

                ema_after = self.reg_dsc_ema.ema_after[i].item() if not torch.isnan(self.reg_dsc_ema.ema_after[i]) else float("nan")
                ema_delta = self.reg_dsc_ema.ema_delta[i].item() if not torch.isnan(self.reg_dsc_ema.ema_delta[i]) else float("nan")

                print(
                    f"    --- {c} DSC: before={before_i:.4f} -> after={after_i:.4f}  Δ={delta_i:+.4f} | "
                    f"EMA after={ema_after:.4f}  EMA Δ={ema_delta:+.4f}",
                    flush=True
                )
            print("    [reg DSC EMA] " + self.reg_dsc_ema.summary_str(max_items=8), flush=True)
            avg_ema_delta = self.reg_dsc_ema.mean_ema_delta()
            print(f"    [reg DSC EMA Δ mean over classes] {avg_ema_delta:+.4f}", flush=True)
            
        return registration_loss, phi_BA_vectorfield, spacing
    
    def make_registration_module(self,registration_mode='image',
                                 registration_input_shape=[175,175,175],
                                 initialization_registration='unigradicon',
                                 train_registration=False):
        self.registration_module = init_registration_network(registration_mode=registration_mode,
                                                             initialization_registration=initialization_registration,
                                                             reg_input_shape=registration_input_shape)
        self.use_time_consistency_loss = True
        self.registration_input_shape = registration_input_shape
        self.registration_mode = registration_mode #deprecated
        if not train_registration:
            for param in self.registration_module.parameters():
                param.requires_grad = False
            self.registration_module.eval() 
        self.train_registration = train_registration
        
        
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
        registration_loss = None,
        out_paired = None,
        registration_field_other_to_current = None,
        spacing_registration = None,
        other_time_point=None, is_paired=None
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
        if registration_loss is not None:
            retur['registration_loss'] = registration_loss
        if out_paired is not None:
            retur['out_paired'] = out_paired
        if registration_field_other_to_current is not None:
            retur['registration_field_other_to_current'] = registration_field_other_to_current
        if other_time_point is not None:
            retur['other_time_point'] = other_time_point
        if is_paired is not None:
            retur['is_paired'] = is_paired
        if spacing_registration is not None:
            retur['spacing_registration'] = spacing_registration
            
        if out_report_decoder is not None:
            #update retur with all keys from out_report_decoder
            for k, v in out_report_decoder.items():
                retur[k] = v

        return retur

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


def init_registration_network(initialization_registration='unigradicon',registration_mode='image',
                              reg_input_shape=[175,175,175]):
    D, H, W = map(int, reg_input_shape)
    assert len(reg_input_shape) == 3, "reg_input_shape must be a length-3 list or tuple representing (D,H,W)"
    from unigradicon import make_network, get_unigradicon
    if registration_mode in ['image', 'mask']:
        input_ch = 1
    elif registration_mode == 'both':
        input_ch = 2
    else:
        raise ValueError('Invalid registration mode: %s'%registration_mode)
    
    reg_input_shape_single_ch = [1, 1, D, H, W]
    reg_input_shape  = [1, input_ch, D, H, W]
    if initialization_registration == 'unigradicon':
        print('Initializing registration network with unigradicon weights')
        reg_net_single_ch = get_unigradicon()
        if tuple(reg_net_single_ch.identity_map.shape[2:]) != (D, H, W):
            reg_net_single_ch.assign_identity_map(reg_input_shape_single_ch)
        reg_net_single_ch.use_label = (registration_mode=='mask')
    else:
        state_dict = torch.load(initialization_registration, map_location="cpu")
        reg_net_single_ch = make_network(reg_input_shape_single_ch, include_last_step=True, use_label=(registration_mode=='mask'))
        reg_net_single_ch.load_state_dict(state_dict, strict=True)
        
    if input_ch == 1:
        reg_net = reg_net_single_ch
    else:
        from model.dim3.medformer import load_state_dict_with_overlap
        reg_net = make_network(reg_input_shape, include_last_step=True)
        reg_net_single_ch_state_dict = reg_net_single_ch.state_dict()
        _, stats, (missing, unexpected) = load_state_dict_with_overlap(reg_net, reg_net_single_ch_state_dict,verbose=True)
    del reg_net_single_ch
    return reg_net

def resample_phi_and_spacing(phi_in: torch.Tensor, spacing_in, out_shape_zyx, align_corners=True):
    """
    phi_in:     [B,3,D,H,W] normalized [0,1] coords (z,y,x channels)
    spacing_in: length-3 (z,y,x voxel spacing used by mermaidlite at the phi_in resolution)
    out_shape_zyx: (d,h,w)
    Returns:
      phi_out:     [B,3,d,h,w]
      spacing_out: length-3 tensor
    """
    B, C, D, H, W = phi_in.shape
    d, h, w = out_shape_zyx
    assert C == 3
    #assert spacing_in is length 3
    assert len(spacing_in) == 3
    
    phi_out = F.interpolate(phi_in, size=out_shape_zyx, mode="trilinear", align_corners=align_corners)

    spacing_in = torch.as_tensor(spacing_in, dtype=phi_in.dtype, device=phi_in.device)

    in_den  = torch.tensor([max(D - 1, 1), max(H - 1, 1), max(W - 1, 1)], device=phi_in.device, dtype=phi_in.dtype)
    out_den = torch.tensor([max(d - 1, 1), max(h - 1, 1), max(w - 1, 1)], device=phi_in.device, dtype=phi_in.dtype)
    spacing_out = spacing_in * (in_den / out_den)

    return phi_out, spacing_out

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


def expand_conv3d_in_channels(conv: nn.Conv3d, to_add: int, mode: str = "zeros"):
    """
    mode:
      - "repeat": repeats old weights across new channels then rescales
      - "zeros": extra channels initialized to 0, avoids perturbing the pretrained model too much
      - "avg": fills new channels with the mean over old input channels
    """
    old_w = conv.weight.data           # [out, in, kx, ky, kz]
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

    # init weight
    if mode == "repeat":
        new_w = old_w.repeat(1, (new_in + old_in - 1)//old_in, 1, 1, 1)[:, :new_in]
        # rescale so activation magnitude stays similar
        new_w *= (old_in / new_in)
    elif mode == "zeros":
        new_w = torch.zeros((conv.out_channels, new_in, *old_w.shape[2:]),
                            device=old_w.device, dtype=old_w.dtype)
        new_w[:, :old_in] = old_w
    elif mode == "avg":
        mean = old_w.mean(dim=1, keepdim=True)  # [out, 1, kx, ky, kz]
        new_w = mean.repeat(1, new_in, 1, 1, 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    new.weight.data.copy_(new_w)
    if conv.bias is not None:
        new.bias.data.copy_(conv.bias.data)
    return new

def update_first_conv(net, target_in):
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
    setattr(parent, parts[-1], expand_conv3d_in_channels(conv, to_add=to_add,mode="zeros"))
    
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
                 concat_time_info = False,
                 train_registration = False,
                 ablate_dynamic_decoder = False,
                 ablate_longitudinal_attention = False,
                 ):
        """
        network: the normal network, instantiated. No need to include more input channels.
        """
        super().__init__()

        # Stored on self so make_longitudinal can consult it. Mirrored onto the
        # teacher network below (before create_report_informed_decoder is called)
        # so the teacher's create_report_informed_decoder branch picks it up.
        self.ablate_dynamic_decoder = bool(ablate_dynamic_decoder)
        # Drop inter-image (other-time) cross-attention in the teacher decoder
        # while keeping --time_points 2 and the time-consistency loss. We flip
        # the inner transformer decoder's `longitudinal` flag to False after the
        # decoder is built so the cross-attention modules exist (checkpoints
        # load) but their forward branches are skipped. Requires DDP
        # find_unused_parameters=True (toggled in train_ddp.py based on this flag).
        self.ablate_longitudinal_attention = bool(ablate_longitudinal_attention)

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
                self.teacher.ablate_dynamic_decoder = self.ablate_dynamic_decoder
                self.teacher.create_report_informed_decoder(use_transformer_decoder=use_transformer_decoder,
                                                            use_dynamic_conv=use_dynamic_conv,
                                                            age_and_sex_provided = age_and_sex_provided,
                                                            use_transformer_conv3 = use_transformer_conv3,
                                                            concat_time_info = concat_time_info,)
                self._apply_longitudinal_attention_ablation()
                self.teacher.age_and_sex_provided = age_and_sex_provided
                self.teacher.create_bottleneck_transformer()
                assert not self.teacher.report_information_in_input
                #assert we have the decoder set:
                assert getattr(self.teacher, 'report_informed_decoder', False), "Must set report_informed_decoder=True"
                assert hasattr(self.teacher, 'report_decoder'), "Network must have 'report_decoder' module for 'teacher_decoder' mode"
                #assert also that we have the bottleneck transformer
                assert hasattr(self.teacher, 'bottleneck_transformer'), "Network must have 'bottleneck_transformer' module for 'teacher_decoder' mode, this will be important for the longitudinal step"
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
                    
    def _apply_longitudinal_attention_ablation(self):
        """If `ablate_longitudinal_attention=True`, flip the inner transformer
        decoder's `longitudinal` flag to False so the inter-image cross-attention
        forward branches are skipped. Modules still exist (parameters load from
        checkpoint, just receive no gradients). No-op when the flag is off, when
        the teacher has no transformer decoder (static decoder ablation, single
        time point, etc.). Call after any operation that (re)builds the decoder.
        """
        if not getattr(self, 'ablate_longitudinal_attention', False):
            return
        if self.teacher is None:
            return
        rd = getattr(self.teacher, 'report_decoder', None)
        td = getattr(rd, 'transformer_decoder', None) if rd is not None else None
        if td is not None and getattr(td, 'longitudinal', False):
            td.longitudinal = False

    def make_longitudinal(self, time_points, use_transformer_decoder,use_dynamic_conv,
                          age_and_sex_provided,use_transformer_conv3,concat_time_info,):
        if self.train_mode !='teacher_decoder':
            raise ValueError("make_longitudinal is only supported in 'teacher_decoder' mode")
        if getattr(self, "ablate_dynamic_decoder", False):
            # Static decoder is time-point-agnostic (no transformer cross-attention,
            # no other-time inputs). Update bookkeeping fields and skip the rebuild.
            self.teacher.age_and_sex_provided = age_and_sex_provided
            self.teacher.time_points = time_points
            return
        self.teacher.age_and_sex_provided = age_and_sex_provided
        self.teacher.time_points = time_points
        #get state dict of current report-informed decoder
        state_dict = self.teacher.report_decoder.state_dict()
        #rebuild the report-informed decoder with new time points
        self.teacher.create_report_informed_decoder(use_transformer_decoder=use_transformer_decoder,
                                                use_dynamic_conv=use_dynamic_conv,
                                                age_and_sex_provided = age_and_sex_provided,
                                                use_transformer_conv3 = use_transformer_conv3,
                                                concat_time_info=concat_time_info,)
        #load old weights into new report-informed decoder where possible
        _, stats, (missing, unexpected) = load_state_dict_with_overlap(self.teacher.report_decoder, state_dict,verbose=True)
        # Re-apply the longitudinal-attention ablation since the decoder was rebuilt.
        self._apply_longitudinal_attention_ablation()
        
    def make_registration(self,registration_mode='image',registration_input_shape=[175,175,175],
                          initialization_registration='unigradicon',
                          train_registration=False):
        if self.train_mode !='teacher_decoder':
            raise ValueError("make_registration is only supported in 'teacher_decoder' mode")
        self.teacher.make_registration_module(registration_mode=registration_mode,registration_input_shape=registration_input_shape,
                                              initialization_registration=initialization_registration,
                                              train_registration=train_registration)

    def restore_stage_trainability(self):
        """Stage-2: restore exactly what train_mode originally set. I.e., train all network."""
        if self.teacher is not None and self._teacher_grad_snap is not None:
            restore_requires_grad(self.teacher, self._teacher_grad_snap)
        if self.student is not None and self._student_grad_snap is not None:
            restore_requires_grad(self.student, self._student_grad_snap)
            
    def forward(self, x,tumor_info, annotated_per_voxel=False,no_mask_training=False, age=None, sex=None, **kwargs):
        
        
        B = x.shape[0]
        device = x.device

        send_report_teacher = (torch.rand(B, device=device) < self.teacher_report_info_prob)  # bool [B]
        send_report_student = send_report_teacher & (torch.rand(B, device=device) < self.student_report_info_prob)  # bool [B]
        
        if self.train_mode == 'teacher_decoder':
            #teacher and student are the same net, student is the original architecture (e.g., medformer), teacher is the output of the report-informed decoder
            out = self.teacher(x, report_info = tumor_info, use_report = send_report_teacher, annotated_per_voxel=annotated_per_voxel, no_mask_training=no_mask_training, age=age, sex=sex, **kwargs)
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
    
    def forward_teacher(self, x,tumor_info, annotated_per_voxel=False,no_mask_training=False, age=None, sex=None, **kwargs):
        
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
                        annotated_per_voxel=False,no_mask_training=False,age=None,sex=None, **kwargs):
        
        
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
        use_full_conv3: bool = True, #if True, Conv-A is full 3x3x3 conv;
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
        time_points = 1,
        concat_time_info = False, #if false, the information from the other time point is only given to the transformer decoder. Otherwise, it is also concatenated in the convolutional branch.
    ):
        super().__init__()
        
        self.concat_time_info = concat_time_info
        
        if age_and_sex_provided:
            report_embed_dim = [v+2 for v in report_embed_dim]
        
        if (not use_full_conv3) and (not use_pointwise_conv1):
            raise ValueError("Need at least one: use_full_conv3=True (3x3x3) or use_pointwise_conv1=True (1x1x1).")

        self.use_pointwise_conv1 = use_pointwise_conv1
        self.crop_size = crop_size

       

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
        kernels_3x3 = []
        
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
            self.mask_norms.append(nn.InstanceNorm3d(2, eps=eps, affine=affine)) #2 channels: organ mask and slices mask

            
            if use_full_conv3:
                if use_pointwise_conv1:
                    out_dim = 2*feature_in_channels[s+1]+2 #+2 is for the organ mask and the slices mask
                else:
                    out_dim = feature_in_channels[s+1]
                in_dim = 2*feature_in_channels[s+1]+2#+2 is for the organ mask and the slices mask
                if concat_time_info and time_points>1:
                    in_dim = in_dim*time_points # we will concatenate these channels across both time points, the conv3 will reduce them back to avoid too many transformer queries
                #after skip. output of the transpose conv was concatenated with the skip connection, creating 2*feature_in_channels[s+1]
                if not use_transformer_conv3:
                    self.kernel_selectors_MLP.append(KernelSelectorMLP(
                                                        in_channels = in_dim,
                                                        out_channels = out_dim,
                                                        kernel_size = 3,
                                                        dim_report = report_embed_dim[s],
                                                        num_mask_channels = num_classes,))
                kernels_3x3.append([in_dim, out_dim])
                self.norms_3x3x3.append(nn.InstanceNorm3d(out_dim, eps=eps, affine=affine))
                self.nonlins_3x3x3.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))
            
            if use_pointwise_conv1:
                #after the 3x3x3 conv
                kernel_channels_transformer.append([2*feature_in_channels[s+1]+2, feature_in_channels[s+1]])#+2 is for the organ mask and the slices mask
                self.norms_1x1x1.append(nn.InstanceNorm3d(feature_in_channels[s+1], eps=eps, affine=affine))
                self.nonlins_1x1x1.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))
            
        self._pad_3 = _same_padding_3d(3)
        
        #transformer decoder
        self.use_transformer_conv3 = bool(use_transformer_conv3)
        if use_pointwise_conv1:
            if mask_stage:
                feature_in_channels_t = [num_classes+2]+[k[0] for k in kernel_channels_transformer]+[feature_in_channels[-1]+2] #include mask stage
                kernel_channels_t = [[num_classes+2,num_classes]]+kernel_channels_transformer+[[feature_in_channels[-1]+2, num_classes]]
                generate_kernels_t = [False] + [True]*len(kernel_channels_transformer)+[True]
                generate_kernels_t_3x3 = [False] + [True]*len(kernel_channels_transformer)+[False]
                kernels_3x3 = [None] + kernels_3x3 + [None]
            else:
                feature_in_channels_t = [k[0] for k in kernel_channels_transformer]+[feature_in_channels[-1]+2]
                kernel_channels_t = kernel_channels_transformer+[[feature_in_channels[-1]+2, num_classes]]
                generate_kernels_t = 'all'
                generate_kernels_t_3x3 = [True]*len(kernel_channels_transformer)+[False]
                kernels_3x3 = kernels_3x3 + [None]
                
            if not use_transformer_conv3:
                kernels_3x3 = None
                
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
                generate_kernels_t_3x3=generate_kernels_t_3x3,
                longitudinal=(time_points>1),
                kernels_3x3=kernels_3x3,
            )
        else:
            self.dynamic_head = KernelSelectorMLP(
                                                in_channels = feature_in_channels[-1]+2,
                                                out_channels = num_classes,
                                                kernel_size = 1,
                                                dim_report = report_embed_dim[-1],
                                                num_mask_channels = num_classes)
            
        self.use_full_conv3 = use_full_conv3
        self.mask_stage = mask_stage
        
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux_head = nn.Conv3d(feature_in_channels[deep_supervision_level], 
                                      num_classes, kernel_size=1)
            self.deep_supervision_level = deep_supervision_level
        self.head_feat_norm = nn.InstanceNorm3d(feature_in_channels[-1], eps=eps, affine=affine)
        self.head_mask_norm = nn.InstanceNorm3d(2, eps=eps, affine=affine) #2 channels: organ mask and slices mask
        
        self.longitudinal = time_points>1

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
        #longitudinal arguments:
        out_mask_other_time = None,
        report_tokens_list_other_time = None,
        other_time_point = None,
        dist=None,
        has_other_time = None
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
        if self.longitudinal:
            other_time_output_mask_avg_pool = torch.mean(out_mask_other_time, dim=(-1,-2,-3), keepdim=False)
        else:
            other_time_output_mask_avg_pool = None
            report_tokens_list_other_time = None
        tumor_location_slices_norm = self.mask_norms[stage_idx](tumor_location_slices)
        x_cat = torch.cat([x_up_norm, skip_norm,tumor_location_slices_norm], dim=1)  # [B, 2*out_ch+2, ...]
        in_ch_cat = x_cat.shape[1]
        expected_in_ch_cat = 2 * out_ch + 2
        if in_ch_cat != expected_in_ch_cat:
            raise RuntimeError(
                f"Stage {stage_idx}: expected concat channels {expected_in_ch_cat} (=2*out_ch+2), got {in_ch_cat}"
            )
            
        
                

        if self.use_full_conv3:
            if self.concat_time_info and self.longitudinal:
                if (has_other_time is not None) and (has_other_time.sum().item() > 0):
                    #gather x_cat
                    x_cat_other_time_all = all_gather_no_grad(x_cat, dist=dist)
                    x_cat_other_time = take_pairs_from_global(x_cat_other_time_all, other_time_point)
                    #concat
                    x_cat_longitudinal = torch.cat([x_cat, x_cat_other_time], dim=1)
                else:
                    #concat zeros
                    x_cat_other_time = torch.zeros_like(x_cat)
                    x_cat_longitudinal = torch.cat([x_cat, x_cat_other_time], dim=1)
            
            if conv3_kernel is None:
                if not self.use_transformer_conv3:
                    if self.concat_time_info and self.longitudinal:
                        print(f'Concatenating time info before MLP kernel selection at stage {stage_idx}', flush=True)
                        x_cat = x_cat_longitudinal
                    #use the kernel selector MLP to get the conv3_kernel and conv3_bias
                    conv3_kernel, conv3_bias = self.kernel_selectors_MLP[stage_idx](
                        x_features=x_cat,      # [B, Cin+3, D,H,W]
                        out_mask=out_mask,        # [B, Cm,  D,H,W]
                        report_embed=report_embed,    # [B, Dr]
                    )
                else:
                    #print(f'Using transformer decoder to generate conv3', flush=True)
                    #use the transformer decoder to get the conv3_kernel and conv3_bias
                    if self.mask_stage:
                        t_level = stage_idx + 1
                    else:
                        t_level = stage_idx
                    x_cat_pooled = x_cat
                    if self.longitudinal:
                        D, H, W = x_cat.shape[-3:]
                        kD = max(1, (D + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                        kH = max(1, (H + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                        kW = max(1, (W + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                        if kD > 1.2 or kH > 1.2 or kW > 1.2:
                            x_cat_pooled = F.avg_pool3d(x_cat, kernel_size=(kD,kH,kW), stride=(kD,kH,kW))
                        x_cat_other_time_all = all_gather_no_grad(x_cat_pooled, dist=dist)
                        x_cat_other_time = take_pairs_from_global(x_cat_other_time_all, other_time_point)
                    else:
                        x_cat_other_time = None
                    queries, conv3_kernel, conv3_bias = self.transformer_decoder(
                        x = x_cat_pooled,
                        level_index = t_level,
                        report_tokens_list = report_tokens_list,
                        output_mask_avg_pool = torch.mean(out_mask, dim=(-1,-2,-3), keepdim=False),
                        previous_queries = queries,
                        layer_to_run = 'first_half',
                        generate_1x1x1_kernel=False,
                        generate_3x3x3_kernel=True,
                        other_time_report_tokens_list=report_tokens_list_other_time, 
                        other_time_x=x_cat_other_time,
                        other_time_output_mask_avg_pool=other_time_output_mask_avg_pool,
                        )
                    if self.concat_time_info and self.longitudinal:
                        print(f'Concatenating time info after transformer at stage {stage_idx}', flush=True)
                        x_cat = x_cat_longitudinal #after the transformer because the transformer already attends to the other time point

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
                
                x_a_pooled = x_a
                if self.longitudinal:
                    #reduce spatial size here to make all_gather cheaper
                    D, H, W = x_a.shape[-3:]
                    kD = max(1, (D + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                    kH = max(1, (H + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                    kW = max(1, (W + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                    if kD > 1.2 or kH > 1.2 or kW > 1.2:
                        x_a_pooled = F.avg_pool3d(x_a, kernel_size=(kD,kH,kW), stride=(kD,kH,kW))
                    x_a_other_time_all = all_gather_no_grad(x_a_pooled, dist=dist)
                    x_a_other_time = take_pairs_from_global(x_a_other_time_all, other_time_point)
                else:
                    x_a_other_time = None
                    
                queries, conv1_kernel, conv1_bias = self.transformer_decoder(
                    x = x_a_pooled,
                    level_index = t_level,
                    report_tokens_list = report_tokens_list,
                    output_mask_avg_pool = torch.mean(out_mask, dim=(-1,-2,-3), keepdim=False),
                    previous_queries = queries,
                    layer_to_run = 'both' if not self.use_transformer_conv3 else 'second_half',
                    other_time_report_tokens_list=report_tokens_list_other_time, 
                    other_time_x=x_a_other_time,
                    other_time_output_mask_avg_pool=other_time_output_mask_avg_pool,
                    )

            if conv1_kernel.ndim != 6 or tuple(conv1_kernel.shape[3:]) != (1, 1, 1):
                raise RuntimeError(
                    f"Stage {stage_idx}: conv1_kernel must be [B,out_ch,in_ch,1,1,1], got {tuple(conv1_kernel.shape)}"
                )

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
    
    def forward_segmentation_head(self,x: torch.Tensor, 
                                    out_mask: torch.Tensor,
                                    report_embed = None,
                                    report_tokens_list = None,
                                    queries = None,
                                    tumor_location_slices=None,
                                    out_mask_other_time = None,
                                    report_tokens_list_other_time = None,
                                    other_time_point = None,
                                    dist=None,
                                    has_other_time=None,):
        x = self.head_feat_norm(x)
        tumor_location_slices_norm = self.head_mask_norm(tumor_location_slices)
        x =  torch.cat([x, tumor_location_slices_norm], dim=1)  # [B, Cin+2, ...]
        
        if self.use_pointwise_conv1:
            #use the transformer decoder to get the conv1_kernel and conv1_bias
            if self.mask_stage:
                level_index = self.num_stages+1
            else:
                level_index = self.num_stages
                
            x_pooled = x
            if self.longitudinal:
                D, H, W = x.shape[-3:]
                kD = max(1, (D + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                kH = max(1, (H + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                kW = max(1, (W + self.transformer_decoder.max_spatial_size - 1) // self.transformer_decoder.max_spatial_size)
                if kD > 1.2 or kH > 1.2 or kW > 1.2:
                    x_pooled = F.avg_pool3d(x, kernel_size=(kD,kH,kW), stride=(kD,kH,kW))
                x_other_time_all = all_gather_no_grad(x_pooled, dist=dist)
                x_other_time = take_pairs_from_global(x_other_time_all, other_time_point)
                other_time_output_mask_avg_pool = torch.mean(out_mask_other_time, dim=(-1,-2,-3), keepdim=False)
            else:
                x_other_time = None
                other_time_output_mask_avg_pool = None
                report_tokens_list_other_time = None
                
            queries, conv1_kernel, conv1_bias = self.transformer_decoder(
                level_index = level_index, #final segmentation head
                x = x_pooled,
                report_tokens_list = report_tokens_list,
                output_mask_avg_pool = torch.mean(out_mask, dim=(-1,-2,-3), keepdim=False),
                previous_queries = queries,
                other_time_report_tokens_list=report_tokens_list_other_time, 
                other_time_x=x_other_time,
                other_time_output_mask_avg_pool=other_time_output_mask_avg_pool,)

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
                tumor_allowed_slices,
                out_mask_other_time=None,
                tumor_location_mask_other_time=None,
                tumor_allowed_slices_other_time=None,
                report_tokens_list_all_other_time=None,
                has_other_time=None,
                other_time_point=None,
                dist=None,):
        
        features = unet_decoder_features[0] #bottleneck features
        unet_decoder_features = unet_decoder_features[1:]
        queries = None
        tumor_location_slices = torch.cat([tumor_location_mask.float(), tumor_allowed_slices.float()], dim=1)
        
        if self.mask_stage and self.use_pointwise_conv1:
            #begin by running the masks in the transformer first stage, just to refine the queries
            #features: mask, tumor_location_mask, tumor_allowed_slices
            if self.longitudinal:
                other_time_report_tokens_list = report_tokens_list_all_other_time[0]
                other_time_x = torch.cat([out_mask_other_time.float(), tumor_location_mask_other_time.float(), tumor_allowed_slices_other_time.float()], dim=1)
                other_time_output_mask_avg_pool = torch.mean(out_mask_other_time, dim=(-1,-2,-3), keepdim=False)
            else:
                other_time_report_tokens_list=None
                other_time_x = None
                other_time_output_mask_avg_pool = None
            queries = self.transformer_decoder(
                                level_index = 0,
                                x = torch.cat([out_mask.float(), tumor_location_mask.float(), tumor_allowed_slices.float()], dim=1),
                                report_tokens_list = report_tokens_list_all[0],
                                output_mask_avg_pool = torch.mean(out_mask, dim=(-1,-2,-3), keepdim=False),
                                previous_queries = None,
                                generate_1x1x1_kernel=False,
                                generate_3x3x3_kernel=False,
                                other_time_report_tokens_list=other_time_report_tokens_list, 
                                other_time_x=other_time_x,
                                other_time_output_mask_avg_pool=other_time_output_mask_avg_pool
                                )
            
        
        for i in range(self.num_stages):
            if self.longitudinal:
                report_tokens_list_other_time = report_tokens_list_all_other_time[i]
            else:
                report_tokens_list_other_time = None
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
                                                tumor_location_slices = tumor_location_slices_r,
                                                out_mask_other_time = out_mask_other_time,
                                                report_tokens_list_other_time = report_tokens_list_other_time,
                                                other_time_point = other_time_point,
                                                dist=dist,
                                                has_other_time=has_other_time
                                                )
            if self.deep_supervision and i==(self.deep_supervision_level-1):
                #upsample to 128x128x128 (crop_size) if needed
                if features.shape[2:] != (self.crop_size, self.crop_size, self.crop_size):
                    aux_feature = F.interpolate(features, size=(self.crop_size, self.crop_size, self.crop_size), 
                                                mode='trilinear', align_corners=True)
                aux_out = self.aux_head(aux_feature)
        
        if self.longitudinal:
            report_tokens_list_other_time = report_tokens_list_all_other_time[-1]
        else:
            report_tokens_list_other_time = None
        features, queries = self.forward_segmentation_head(
                                                x=features,
                                                out_mask=out_mask,
                                                report_embed=report_vector_all[-1],#report embeddings for late stages include size
                                                report_tokens_list=report_tokens_list_all[-1],#report tokens for late stages include size
                                                queries=queries,
                                                tumor_location_slices = tumor_location_slices,
                                                out_mask_other_time = out_mask_other_time,
                                                report_tokens_list_other_time = report_tokens_list_other_time,
                                                other_time_point = other_time_point,
                                                dist=dist,
                                                has_other_time=has_other_time
                                            )
        
        primary = [features, aux_out] if self.deep_supervision else features
        return {'refined_segmentation': primary}
    
    from typing import List, Optional, Dict
import torch


class StaticUNetDecoder3D(nn.Module):
    """
    Static-conv counterpart of DynamicUNetDecoder3D for ablation studies.

    Replaces both per-sample dynamic kernel mechanisms with standard convs:
      - 3x3x3 KernelSelectorMLP path  -> nn.Conv3d(k=3)
      - 1x1x1 transformer-decoder path -> nn.Conv3d(k=1)

    Mirrors the forward signature and return-dict shape of
    DynamicUNetDecoder3D so it is a drop-in replacement. All report-,
    transformer-, and longitudinal-related kwargs are accepted but ignored
    (no transformer decoder, no per-sample kernels, no cross-time
    cross-attention). The two mask channels (tumor_location_mask +
    tumor_allowed_slices) are still concatenated per stage to preserve the
    channel contract of the dynamic decoder; the user can compose with
    --teacher_report_info_prob 0 to zero them when desired.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        feature_in_channels=[320, 256, 128, 64, 32],
        deep_supervision: bool = True,
        deep_supervision_level: int = 2,
        crop_size: int = 128,
        eps: float = 1e-5,
        affine: bool = True,
        negative_slope: float = 1e-2,
        inplace: bool = True,
    ):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.deep_supervision_level = int(deep_supervision_level)
        self.crop_size = int(crop_size)
        self.feature_in_channels = list(feature_in_channels)

        n_stages = len(feature_in_channels) - 1

        self.transpconvs = nn.ModuleList()
        self.skip_norms = nn.ModuleList()
        self.feat_norms = nn.ModuleList()
        self.mask_norms = nn.ModuleList()
        self.conv3 = nn.ModuleList()
        self.norms_3x3x3 = nn.ModuleList()
        self.nonlins_3x3x3 = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.norms_1x1x1 = nn.ModuleList()
        self.nonlins_1x1x1 = nn.ModuleList()

        for s in range(n_stages):
            ch_out = feature_in_channels[s + 1]
            self.transpconvs.append(
                nn.ConvTranspose3d(
                    in_channels=feature_in_channels[s],
                    out_channels=ch_out,
                    kernel_size=2,
                    stride=2,
                    bias=True,
                )
            )
            self.skip_norms.append(nn.InstanceNorm3d(ch_out, eps=eps, affine=affine))
            self.feat_norms.append(nn.InstanceNorm3d(ch_out, eps=eps, affine=affine))
            self.mask_norms.append(nn.InstanceNorm3d(2, eps=eps, affine=affine))

            in_dim = 2 * ch_out + 2
            mid_dim = 2 * ch_out + 2
            self.conv3.append(nn.Conv3d(in_dim, mid_dim, kernel_size=3, padding=1))
            self.norms_3x3x3.append(nn.InstanceNorm3d(mid_dim, eps=eps, affine=affine))
            self.nonlins_3x3x3.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))

            self.conv1.append(nn.Conv3d(mid_dim, ch_out, kernel_size=1))
            self.norms_1x1x1.append(nn.InstanceNorm3d(ch_out, eps=eps, affine=affine))
            self.nonlins_1x1x1.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))

        # final segmentation head (mirrors DynamicUNetDecoder3D head_*_norm + final 1x1)
        self.head_feat_norm = nn.InstanceNorm3d(feature_in_channels[-1], eps=eps, affine=affine)
        self.head_mask_norm = nn.InstanceNorm3d(2, eps=eps, affine=affine)
        self.final_head = nn.Conv3d(feature_in_channels[-1] + 2, num_classes, kernel_size=1)

        if self.deep_supervision:
            self.aux_head = nn.Conv3d(
                feature_in_channels[self.deep_supervision_level], num_classes, kernel_size=1
            )

    @property
    def num_stages(self) -> int:
        return len(self.transpconvs)

    def forward(
        self,
        unet_decoder_features,
        out_mask,
        report_vector_all,
        report_tokens_list_all,
        tumor_location_mask,
        tumor_allowed_slices,
        out_mask_other_time=None,
        tumor_location_mask_other_time=None,
        tumor_allowed_slices_other_time=None,
        report_tokens_list_all_other_time=None,
        has_other_time=None,
        other_time_point=None,
        dist=None,
    ):
        # report_*, transformer-decoder-, and longitudinal-related kwargs are intentionally unused.
        features = unet_decoder_features[0]
        skips = unet_decoder_features[1:]

        tumor_location_slices = torch.cat(
            [tumor_location_mask.float(), tumor_allowed_slices.float()], dim=1
        )

        aux_out = None
        for i in range(self.num_stages):
            skip = skips[i]
            x_up = self.transpconvs[i](features)
            x_up_norm = self.feat_norms[i](x_up)
            skip_norm = self.skip_norms[i](skip)
            mask_r = F.interpolate(tumor_location_slices, size=skip.shape[2:], mode="nearest")
            mask_r_norm = self.mask_norms[i](mask_r)
            x_cat = torch.cat([x_up_norm, skip_norm, mask_r_norm], dim=1)

            x_a = self.conv3[i](x_cat)
            x_a = self.norms_3x3x3[i](x_a)
            x_a = self.nonlins_3x3x3[i](x_a)

            features = self.conv1[i](x_a)
            features = self.norms_1x1x1[i](features)
            features = self.nonlins_1x1x1[i](features)

            if self.deep_supervision and i == (self.deep_supervision_level - 1):
                if features.shape[2:] != (self.crop_size, self.crop_size, self.crop_size):
                    aux_feature = F.interpolate(
                        features,
                        size=(self.crop_size, self.crop_size, self.crop_size),
                        mode="trilinear",
                        align_corners=True,
                    )
                else:
                    aux_feature = features
                aux_out = self.aux_head(aux_feature)

        # final segmentation head: full-resolution features + full-resolution mask channels
        x = self.head_feat_norm(features)
        m = self.head_mask_norm(tumor_location_slices)
        x = torch.cat([x, m], dim=1)
        out = self.final_head(x)

        primary = [out, aux_out] if self.deep_supervision else out
        return {"refined_segmentation": primary}


class EMAperClassDelta:
    """
    EMA tracker per class for:
      - before DSC (unregistered)
      - after  DSC (registered)
      - delta = after - before

    Skips (sample,class) where TARGET mask is empty (no FG in mask_backup for that class).
    Expects:
      dsc_before_bc: [B,C]
      dsc_after_bc:  [B,C]
      target_mask_bcdhw: [B,C,D,H,W] (or [B,C,...])
    """
    def __init__(self, class_names: List[str], momentum: float = 0.95, device: Optional[torch.device] = None):
        assert 0.0 < momentum < 1.0
        self.class_names = list(class_names)
        self.m = float(momentum)
        self.device = device

        C = len(self.class_names)
        self.ema_before = torch.full((C,), float("nan"), dtype=torch.float32, device=device)
        self.ema_after  = torch.full((C,), float("nan"), dtype=torch.float32, device=device)
        self.ema_delta  = torch.full((C,), float("nan"), dtype=torch.float32, device=device)

        self.count_updates = torch.zeros((C,), dtype=torch.long, device=device)
        self.count_samples = torch.zeros((C,), dtype=torch.long, device=device)

    @torch.no_grad()
    def update(
        self,
        dsc_before_bc: torch.Tensor,
        dsc_after_bc: torch.Tensor,
        target_mask_bcdhw: torch.Tensor,
    ):
        assert dsc_before_bc.ndim == 2 and dsc_after_bc.ndim == 2, \
            f"dsc tensors must be [B,C], got {dsc_before_bc.shape} and {dsc_after_bc.shape}"
        B, C = dsc_before_bc.shape
        assert dsc_after_bc.shape == (B, C), f"shape mismatch {dsc_after_bc.shape} vs {(B,C)}"
        assert C == len(self.class_names), f"C mismatch: EMA has {len(self.class_names)} classes, got {C}"
        assert target_mask_bcdhw.shape[0] == B and target_mask_bcdhw.shape[1] == C, \
            f"target_mask must be [B,C,...], got {target_mask_bcdhw.shape}"

        # lazy device init
        if self.device is None:
            self.device = dsc_before_bc.device
            self.ema_before = self.ema_before.to(self.device)
            self.ema_after  = self.ema_after.to(self.device)
            self.ema_delta  = self.ema_delta.to(self.device)
            self.count_updates = self.count_updates.to(self.device)
            self.count_samples = self.count_samples.to(self.device)

        dsc_before_bc = dsc_before_bc.to(self.device)
        dsc_after_bc  = dsc_after_bc.to(self.device)
        target_mask_bcdhw = target_mask_bcdhw.to(self.device)

        # valid if target mask has any FG voxel
        valid_bc = (target_mask_bcdhw > 0.5).flatten(2).any(dim=2)  # [B,C]

        delta_bc = dsc_after_bc - dsc_before_bc  # [B,C]

        for c in range(C):
            v = valid_bc[:, c]
            if not v.any():
                continue

            before = dsc_before_bc[v, c].mean()
            after  = dsc_after_bc[v, c].mean()
            delt   = delta_bc[v, c].mean()

            # initialize if nan, else EMA update
            if torch.isnan(self.ema_before[c]):
                self.ema_before[c] = before
                self.ema_after[c]  = after
                self.ema_delta[c]  = delt
            else:
                m = self.m
                self.ema_before[c] = m * self.ema_before[c] + (1.0 - m) * before
                self.ema_after[c]  = m * self.ema_after[c]  + (1.0 - m) * after
                self.ema_delta[c]  = m * self.ema_delta[c]  + (1.0 - m) * delt

            self.count_updates[c] += 1
            self.count_samples[c] += int(v.sum().item())

    def summary_str(self, max_items: int = 10) -> str:
        parts = []
        for i, name in enumerate(self.class_names[:max_items]):
            a = self.ema_after[i]
            d = self.ema_delta[i]
            if torch.isnan(a) or torch.isnan(d):
                parts.append(f"{name}=nan")
            else:
                parts.append(f"{name}: after={a.item():.3f} Δ={d.item():+.3f}")
        if len(self.class_names) > max_items:
            parts.append("...")
        return " | ".join(parts)
    
    def mean_ema_delta(self) -> float:
        """
        Mean EMA delta across all classes, skipping NaNs.
        Returns nan if no class has been updated yet.
        """
        x = self.ema_delta
        valid = ~torch.isnan(x)
        if not valid.any():
            return float("nan")
        return x[valid].mean().item()
    
    
import os
import numpy as np
import nibabel as nib

def save_tensor_as_nifti(tensor: torch.Tensor, filename: str):
    """
    Saves a torch tensor as a NIfTI file, assuming a voxel spacing of 1x1x1 mm.

    Args:
        tensor (torch.Tensor): A torch tensor of shape (H, W, D) or (1, H, W, D).
        filename (str): The output filename (should end with .nii or .nii.gz).
    """
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