import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import split_idx
import pdb
import numpy as np
#from ..training.augmentation import crop_foreground_3d


def _lesion_like_name(segment_name):
    """Map an organ entry of `organs_with_tumor` to its lesion channel name.

    Mirrors `lesion_like_name` in tumor_info_builder.py (which itself mirrors
    the dataset's canonicalisation at lines ~2887-2895).

    Examples:
        'bladder'            -> 'bladder_lesion'
        'gall_bladder'       -> 'gallbladder_lesion'
        'adrenal_gland_left' -> 'adrenal_lesion'
        'pancreas'           -> 'pancreatic_lesion'
    """
    s = (segment_name
         .replace(' ', '_')
         .replace('_right', '')
         .replace('_left', '')
         .replace('_gland', '')
         .replace('gall_bladder', 'gallbladder'))
    if ('liver' in s) or ('segment' in s):
        s = 'liver'
    elif ('pancrea' in s) or ('head' in s) or ('body' in s) or ('tail' in s):
        s = 'pancreatic'
    return s + '_lesion'


def inference_whole_image(net, img, args=None):
    '''
    img: torch tensor, B, C, D, H, W
    return: prob (after softmax), B, classes, D, H, W

    Use this function to inference if whole image can be put into GPU without memory issue
    Better to be consistent with the training window size
    '''

    net.eval()

    with torch.no_grad():
        pred = net(img)

        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]

    return torch.sigmoid(pred)

def classification_to_3D(pred,D,H,W,use_transformer_decoder=False):
    if use_transformer_decoder and ('classification on segmentation_refined' in list(pred.keys())):
        pred_cls = pred['classification on segmentation_refined']
    elif 'classification on segmentation' in list(pred.keys()):
        pred_cls = pred['classification on segmentation']
    elif 'classification on output' in list(pred.keys()):
        pred_cls = pred['classification on output']
    elif 'classification' in list(pred.keys()):
        pred_cls = pred['classification']
    else:
        pred_cls = None
    if pred_cls is not None and (isinstance(pred_cls, tuple) or isinstance(pred_cls, list)):
        pred_cls = pred_cls[-1]
    if pred_cls is not None:
        #make it match the shape of pred by adding the 3 spatial dimensions (B,C) -> (B,C,D,H,W)
        pred_cls = torch.sigmoid(pred_cls)
        pred_cls = pred_cls.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, D,H,W)
    #if pred_cls is None:
    #    raise ValueError(f"No classification output found in the model output dictionary. Available keys: {list(pred.keys())}")
    return pred_cls

def inference_sliding_window(net, img, args, pancreas=None,gaussian=False,use_transformer_decoder=False,
                             age = None, sex = None, sw_roi_mask=None, keep_class_indices=None,
                             stage2_organs=None):
    # Fast-mode ROI mask: piggybacks on the existing pancreas-only window-skip
    # by acting as a pancreas argument for the SW pass. If both are provided
    # we prefer sw_roi_mask (caller's responsibility to ensure mutual
    # exclusivity with predict_pancreas_only).
    # `stage2_organs` (when not None) overrides args.organs_with_tumor for
    # the stage-2 loop — used by fast modes to skip organs whose GT mask is
    # empty for this case (the SW didn't run there, so reading stage-1's
    # output picks up leakage from neighbouring ROI organs).
    effective_pancreas = sw_roi_mask if sw_roi_mask is not None else pancreas
    _organs = stage2_organs if stage2_organs is not None else args.organs_with_tumor
    if args.inference_2_stages:
        pred_output, cls_output =  inference_2_stages(net, img, args, organs_with_tumors=_organs, class_list=args.class_list,
                                                      use_transformer_decoder=use_transformer_decoder,age=age,sex=sex,
                                                      sw_roi_mask=sw_roi_mask,
                                                      keep_class_indices=keep_class_indices)
    else:
        pred_output, cls_output = inference_sliding_window_one_pass(net, img, args, pancreas=effective_pancreas, gaussian=gaussian,
                                                                    use_transformer_decoder=use_transformer_decoder,age=age,sex=sex,
                                                                    keep_class_indices=keep_class_indices)
        # inference_sliding_window_one_pass now returns on GPU; the
        # non-stage-2 dispatch ends here so we move to CPU before
        # returning to predict_abdomenatlas_teacher_student.prediction().
        pred_output = pred_output.cpu()
        if cls_output is not None:
            cls_output = cls_output.cpu()

    if cls_output is not None and (cls_output.sum() > 0).item():
        cls_output = torch.amax(cls_output, dim=(-3, -2, -1))
        # Keep pred_output in bf16 — the downstream threshold works on
        # bf16; .float() would double the 17.9 GB tensor and OOM the GPU.
        return pred_output, cls_output.float()
    else:
        return pred_output
    
    
def inference_sliding_window_one_pass(net, img, args, pancreas=None,gaussian=False,use_transformer_decoder=False, age=None, sex=None, keep_class_indices=None):
    '''
    img: torch tensor, B, C, D, H, W
    return: prob (after softmax), B, classes, D, H, W
    pancreas: pancreas mask, used in pancreas_only_inference

    The overlap of two windows will be half the window size

    Use this function to inference if out-of-memory occurs when whole image inferencing
    Better to be consistent with the training window size
    '''
    net.eval()
    
    if pancreas is not None:
        while len(pancreas.shape) < len(img.shape):
            pancreas = pancreas.unsqueeze(0)
        assert pancreas.shape == img.shape, f"Pancreas mask shape must match image shape, got {pancreas.shape} and {img.shape}"

    B, C, D, H, W = img.shape

    win_d, win_h, win_w = args.window_size
    
    if gaussian:
        gauss_w = make_gaussian_kernel(win_d, win_h, win_w, sigma_scale=0.25).to(torch.bfloat16).cpu()

    flag = False
    if D < win_d or H < win_h or W < win_w:
        flag = True
        diff_D = max(0, win_d-D)
        diff_H = max(0, win_h-H)
        diff_W = max(0, win_w-W)

        img = F.pad(img, (0, diff_W, 0, diff_H, 0, diff_D))
        
        origin_D, origin_H, origin_W = D, H, W
        B, C, D, H, W = img.shape


    half_win_d = win_d // 2
    half_win_h = win_h // 2
    half_win_w = win_w // 2

    # `keep_class_indices` caps pred_output channels — see twin in
    # inference3d_teacher.py.
    _keep_idx_t = None
    if keep_class_indices is not None and len(keep_class_indices) > 0:
        _keep_sorted = sorted(set(int(i) for i in keep_class_indices))
        _keep_idx_t = torch.as_tensor(_keep_sorted, dtype=torch.long, device=img.device if img.is_cuda else 'cpu')
        _n_out = len(_keep_sorted)
    else:
        _n_out = args.classes

    _pred_bytes = B * _n_out * D * H * W * 2
    _free_bytes = torch.cuda.mem_get_info(img.device)[0] if img.is_cuda else 0
    # 28 GB margin (model + activations + counter + gauss; see twin in
    # inference3d_teacher.py).
    _gpu_ok = img.is_cuda and (_pred_bytes < max(_free_bytes - 28 * (1 << 30), 0))
    _acc_dev = img.device if _gpu_ok else torch.device('cpu')
    print(f'[SW] pred_output device={_acc_dev}  pred_GB={_pred_bytes/(1<<30):.1f}  free_GB={_free_bytes/(1<<30):.1f}  n_out={_n_out}/{args.classes}')
    pred_output = torch.zeros((B, _n_out, D, H, W), dtype=torch.bfloat16, device=_acc_dev)
    cls_output = None
    cls_counter = torch.zeros((B, 1, D, H, W), dtype=torch.bfloat16, device=_acc_dev)

    counter = torch.zeros((B, 1, D, H, W), dtype=torch.bfloat16, device=_acc_dev)
    one_count = torch.ones((B, 1, win_d, win_h, win_w), dtype=torch.bfloat16, device=_acc_dev)
    if gaussian:
        gauss_w = gauss_w.to(_acc_dev)

    # Phase 1: collect kept windows. Batch them in phase 2 to saturate
    # the GPU (B=1 forwards under-utilise compute).
    import os as _os_sw
    SW_BATCH = int(_os_sw.environ.get('SW_BATCH', '1'))
    kept = []
    for i in range(D // half_win_d):
        for j in range(H // half_win_h):
            for k in range(W // half_win_w):
                d0, d1 = split_idx(half_win_d, D, i)
                h0, h1 = split_idx(half_win_h, H, j)
                w0, w1 = split_idx(half_win_w, W, k)
                if pancreas is None or pancreas[:, :, d0:d1, h0:h1, w0:w1].sum() > 0:
                    kept.append((d0, d1, h0, h1, w0, w1))

    with torch.no_grad():
        for _b_start in range(0, len(kept), SW_BATCH):
            coords_batch = kept[_b_start:_b_start + SW_BATCH]
            input_batch = torch.cat(
                [img[:, :, d0:d1, h0:h1, w0:w1] for (d0, d1, h0, h1, w0, w1) in coords_batch],
                dim=0,
            )
            model_output = net(input_batch, age=age, sex=sex,
                               skip_report_informed_decoder=(not args.use_transformer_decoder))

            if isinstance(model_output, dict):
                if not use_transformer_decoder:
                    pred = model_output['segmentation']
                else:
                    pred = model_output['refined_segmentation']
            else:
                pred = model_output
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            if isinstance(pred, (tuple, list)):
                pred = pred[0]

            pred_cls_batch = None
            if isinstance(model_output, dict):
                pred_cls_batch = classification_to_3D(
                    model_output, pred.shape[-3], pred.shape[-2], pred.shape[-1],
                    use_transformer_decoder=use_transformer_decoder)

            if not args.epai_stage_2:
                pred = torch.sigmoid(pred)
            else:
                pred = F.softmax(pred, dim=1)

            # Shrink to kept channels BEFORE the device move — saves bandwidth.
            if _keep_idx_t is not None:
                _idx_dev = _keep_idx_t.to(pred.device) if _keep_idx_t.device != pred.device else _keep_idx_t
                pred = pred.index_select(dim=1, index=_idx_dev)
            pred_bf = pred.to(torch.bfloat16).to(_acc_dev)
            for b_idx, (d0, d1, h0, h1, w0, w1) in enumerate(coords_batch):
                if pred_cls_batch is not None:
                    if cls_output is None:
                        cls_output = torch.zeros((B, pred_cls_batch.shape[1], D, H, W),
                                                  dtype=torch.bfloat16, device=_acc_dev)
                    cls_output[:, :, d0:d1, h0:h1, w0:w1] += pred_cls_batch[b_idx:b_idx+1].to(torch.bfloat16).to(_acc_dev)
                    cls_counter[:, :, d0:d1, h0:h1, w0:w1] += one_count

                if not gaussian:
                    pred_output[:, :, d0:d1, h0:h1, w0:w1] += pred_bf[b_idx:b_idx+1]
                    counter[:, :, d0:d1, h0:h1, w0:w1] += one_count
                else:
                    w_patch = gauss_w[
                        :,
                        :,
                        : d1 - d0,
                        : h1 - h0,
                        : w1 - w0,
                    ]
                    w_patch_cls = w_patch.expand(B, _n_out,       *w_patch.shape[2:])
                    w_patch_cnt = w_patch.expand(B, 1,            *w_patch.shape[2:])
                    pred_output[:, :, d0:d1, h0:h1, w0:w1] += (pred_bf[b_idx:b_idx+1] * w_patch_cls)
                    counter[:, :, d0:d1, h0:h1, w0:w1] += w_patch_cnt

    pred_output /= (counter + 1e-6)
    if flag:
        pred_output = pred_output[:, :, :origin_D, :origin_H, :origin_W]

    # Caller decides where to move it (inference_2_stages keeps on GPU
    # for stage 2; inference_sliding_window moves to CPU at return).
    if cls_output is not None and (cls_output.sum() > 0).item():
        cls_output /= (cls_counter + 1e-6)
        return pred_output, cls_output

    return pred_output, None

                    
def make_gaussian_kernel(win_d, win_h, win_w, sigma_scale=0.25):
    """
    Return a tensor of shape (1, 1, win_d, win_h, win_w) whose values peak at 1
    in the centre and taper off with a 3-D Gaussian.
    sigma_scale : fraction of the window size -– 0.125 → σ ≈ win_dim / 8
    """
    z = torch.linspace(-1, 1, steps=win_d)
    y = torch.linspace(-1, 1, steps=win_h)
    x = torch.linspace(-1, 1, steps=win_w)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # choose separate σ for each axis
    sigma_d = sigma_scale * 2         # range (-1,1) ⇒ length 2
    sigma_h = sigma_scale * 2
    sigma_w = sigma_scale * 2

    g = torch.exp(
        -((zz / sigma_d) ** 2 + (yy / sigma_h) ** 2 + (xx / sigma_w) ** 2) / 2
    )
    g /= g.max()            # centre = 1.0
    return g.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)


def inference_2_stages(net, img, args, organs_with_tumors=None, class_list=None, use_transformer_decoder=False,age=None,sex=None,sw_roi_mask=None, keep_class_indices=None):
    #stage 1: coarse, run over all the CT.
    # `sw_roi_mask` (when non-None) is a binary union of GT organ masks
    # passed as the SW window-skip ROI — windows that don't intersect it
    # have their forward pass skipped (existing behaviour of the
    # `pancreas` parameter in inference_sliding_window_one_pass).
    # `keep_class_indices` shrinks pred_output to a subset of channels;
    # stage-2 reads/writes must translate full-class indices via
    # `_full_to_keep`.
    pred_output, cls_output = inference_sliding_window_one_pass(net, img, args, pancreas=sw_roi_mask,gaussian=True, use_transformer_decoder=use_transformer_decoder,age=age,sex=sex, keep_class_indices=keep_class_indices)
    if keep_class_indices is not None and len(keep_class_indices) > 0:
        _keep_sorted = sorted(set(int(i) for i in keep_class_indices))
        _full_to_keep = {full: i for i, full in enumerate(_keep_sorted)}
    else:
        _full_to_keep = None
    # pred_output is on GPU. We keep it there for stage 2 (compute the
    # per-organ binary lazily, paste-back on GPU) and only move to CPU
    # at the function boundary. The original full-volume float32 binary
    # (~35 GB on CPU for Turkish-sized CTs) is replaced with small
    # per-organ binaries computed inside the loop.
    B, C, D, H, W = img.shape
    win_d, win_h, win_w = args.window_size
    _dev = pred_output.device

    cls_class_list = sorted([c for c in class_list if 'lesion' in c])

    for batch in range(B):
        pred_per_ch = {}
        counter_per_ch = {}
        cls_per_ch = {}
        cls_counter_per_ch = {}
        one_count = torch.ones((win_d, win_h, win_w), dtype=torch.bfloat16, device=_dev)

        for org in organs_with_tumors:
            org_idx_full = class_list.index(org)
            if _full_to_keep is not None:
                if org_idx_full not in _full_to_keep:
                    # Organ channel isn't in the kept subset — can't read its
                    # stage-1 binary to crop, so skip. (Caller should include
                    # all organs_with_tumor in keep_class_indices.)
                    continue
                org_idx = _full_to_keep[org_idx_full]
            else:
                org_idx = org_idx_full
            organ_prob = pred_output[batch, org_idx]
            # float32 because crop_foreground_3d → denoise_mask does
            # `.cpu().numpy().astype(bool)` which rejects bfloat16.
            organ_mask = (organ_prob > 0.5).float()
            if organ_mask.sum() == 0:
                continue
            x = img[batch, 0]
            # tensor_lab is sliced but the cropped value is discarded
            # downstream — pass the tiny per-organ binary instead of the
            # 74-channel global binary so we don't ferry tens of GB.
            out = crop_foreground_3d(tensor_ct=x, tensor_lab=organ_mask, foreground=organ_mask,
                                     crop_size=[win_d, win_h, win_w], rand=False, return_coordinate=True)
            if not isinstance(out, tuple):
                print(f"Failed to crop on organ {org}: {out}")
                continue
            cropped_ct, _, cropped_organ, coord = out
            d_start_idx,d_end_idx, h_start_idx,h_end_idx, w_start_idx,w_end_idx = coord

            lesion_name = _lesion_like_name(org)
            if lesion_name not in class_list:
                print(f"[stage2] no lesion channel for {org} (lesion_name={lesion_name}) — skip")
                continue
            lesion_ch_full = class_list.index(lesion_name)
            if _full_to_keep is not None:
                if lesion_ch_full not in _full_to_keep:
                    continue
                lesion_ch = _full_to_keep[lesion_ch_full]
            else:
                lesion_ch = lesion_ch_full

            with torch.no_grad():
                model_output = net(cropped_ct.unsqueeze(0).unsqueeze(0),age=age,sex=sex,
                                   skip_report_informed_decoder=(not args.use_transformer_decoder))
                pred_cls = None
                if isinstance(model_output, dict):
                    if not use_transformer_decoder:
                        pred = model_output['segmentation']
                    else:
                        pred = model_output['refined_segmentation']
                if isinstance(pred, tuple) or isinstance(pred, list):
                    pred = pred[0]
                if isinstance(pred, tuple) or isinstance(pred, list):
                    pred = pred[0]
                if isinstance(model_output, dict):
                    pred_cls = classification_to_3D(model_output, pred.shape[-3], pred.shape[-2], pred.shape[-1])

                if not args.epai_stage_2:
                    pred = torch.sigmoid(pred)
                else:
                    pred = F.softmax(pred, dim=1)

            if lesion_ch not in pred_per_ch:
                pred_per_ch[lesion_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16, device=_dev)
                counter_per_ch[lesion_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16, device=_dev)
            # `pred` (model output) is full-channel; pred_per_ch is keyed by
            # the keep-translated index — read with lesion_ch_full, write with lesion_ch.
            pred_per_ch[lesion_ch][d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred[0, lesion_ch_full].to(torch.bfloat16).to(_dev)
            counter_per_ch[lesion_ch][d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count

            if pred_cls is not None and lesion_name in cls_class_list:
                cls_ch = cls_class_list.index(lesion_name)
                if cls_ch not in cls_per_ch:
                    cls_per_ch[cls_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16, device=_dev)
                    cls_counter_per_ch[cls_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16, device=_dev)
                cls_per_ch[cls_ch][d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred_cls[0, cls_ch].to(torch.bfloat16).to(_dev)
                cls_counter_per_ch[cls_ch][d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count

        # Blend per-channel on GPU; free as we go.
        for ch in list(pred_per_ch.keys()):
            pred_t = pred_per_ch.pop(ch)
            counter_t = counter_per_ch.pop(ch)
            ch_mask = (counter_t > 0).to(torch.bfloat16)
            counter_safe = counter_t * ch_mask + 1e-6 * (1 - ch_mask)
            avg = pred_t / counter_safe
            pred_output[batch, ch] = (1 - ch_mask) * pred_output[batch, ch] + ch_mask * avg

        if cls_output is not None:
            for ch in list(cls_per_ch.keys()):
                pred_t = cls_per_ch.pop(ch)
                counter_t = cls_counter_per_ch.pop(ch)
                ch_mask = (counter_t > 0).to(torch.bfloat16)
                counter_safe = counter_t * ch_mask + 1e-6 * (1 - ch_mask)
                avg = pred_t / counter_safe
                cls_output[batch, ch] = (1 - ch_mask) * cls_output[batch, ch] + ch_mask * avg

    # Return on GPU; caller thresholds on GPU and only moves the
    # smaller uint8 label_pred to CPU. Avoids a 17.9 GB bf16 PCIe
    # transfer per case.
    return pred_output, cls_output


def crop_foreground_3d(tensor_ct, tensor_lab, foreground, crop_size, margin=1, refine_iterations=3, rand=True, return_coordinate=False):
    """
    Crops a 3D CT & binary label around the label's nonzero region, returning EXACT [d,h,w].
    
    If rand=True, the bounding box is randomly shifted within the volume if possible.
    If rand=False, it is centered if possible.

    1) If label is empty => return "zero mask"
    2) If bounding box is bigger than crop_size => morphological denoise => 
       if still doesn't fit => return "mask does not fit crop size"
    3) If bounding box <= crop_size => compute the valid range of random shifts
       for each dimension. If no valid shift is possible => "mask does not fit crop size"
    4) Otherwise, pick a random shift and return (cropped_ct, cropped_label).

    Args:
        tensor_ct (torch.Tensor): shape [D,H,W] or [1,D,H,W]
        foreground (torch.Tensor): shape [D,H,W] or [1,D,H,W], binary
        
        crop_size (tuple/list): (d,h,w)
        margin (int or tuple): extra margin
        refine_iterations (int): # of erosions/dilations

    Returns:
        (cropped_ct, cropped_label) or
        "zero mask" or
        "mask does not fit crop size"
    """

    ##### 1) Unify shapes #####
    if tensor_ct.ndim == 3:
        D, H, W = tensor_ct.shape
        ct_has_channel = False
        ct_has_batch = False
    elif tensor_ct.ndim == 4 and tensor_ct.shape[0] == 1:
        _, D, H, W = tensor_ct.shape
        ct_has_channel = True
        ct_has_batch = False
    elif tensor_ct.ndim == 5 and tensor_ct.shape[0] == 1 and tensor_ct.shape[1] == 1:
        _, _, D, H, W = tensor_ct.shape
        ct_has_channel = True
        ct_has_batch = True
    else:
        raise ValueError(f"CT must be [D,H,W] or [1,D,H,W] or [1,1,D,H,W], got {tensor_ct.shape}")

    # --- replace the old squeeze block ----------------------------------------
    if foreground.ndim == 4 and foreground.shape[0] == 1:
        # foreground is [1, D, H, W]  → drop the leading 1
        label_3d = foreground[0].clone()
    elif foreground.ndim == 3:
        label_3d = foreground.clone()
    else:
        raise ValueError(
            f"Foreground must be [D,H,W] or [1,D,H,W], got {foreground.shape}"
        )
    
    assert foreground.shape[-3:]==tensor_ct.shape[-3:], f"Foreground shape must match CT shape, got {foreground.shape} and {tensor_ct.shape}"
    
    backup_foreground = label_3d.clone()
        
    # Check empty
    if torch.count_nonzero(label_3d) == 0:
        return "zero mask"

    ##### 2) Get bounding box #####
    coords = torch.nonzero(label_3d, as_tuple=False)
    zmin, zmax = coords[:, 0].min().item(), coords[:, 0].max().item()
    ymin, ymax = coords[:, 1].min().item(), coords[:, 1].max().item()
    xmin, xmax = coords[:, 2].min().item(), coords[:, 2].max().item()

    if isinstance(margin, int):
        margin = (margin, margin, margin)
    mz, my, mx = margin

    # Apply margin---this is the foreground bounding box
    zmin = max(zmin - mz, 0)
    zmax = min(zmax + mz, D - 1)
    ymin = max(ymin - my, 0)
    ymax = min(ymax + my, H - 1)
    xmin = max(xmin - mx, 0)
    xmax = min(xmax + mx, W - 1)
    
    # After applying margin and clamping:
    if xmin > xmax:   xmin, xmax = xmax, xmin
    if ymin > ymax:   ymin, ymax = ymax, ymin
    if zmin > zmax:   zmin, zmax = zmax, zmin

    desired_d, desired_h, desired_w = crop_size
    if desired_d > D or desired_h > H or desired_w > W:
        return "requesting crop larger than the CT!!"

    def bbox_dim(z0, z1, y0, y1, x0, x1):
        return (z1 - z0 + 1), (y1 - y0 + 1), (x1 - x0 + 1)

    bbox_d, bbox_h, bbox_w = bbox_dim(zmin, zmax, ymin, ymax, xmin, xmax)

    # Check if bounding box is bigger
    if bbox_d > desired_d or bbox_h > desired_h or bbox_w > desired_w:
        # Hybrid GPU morphology + scipy CC on bbox-cropped tensor when
        # the mask is on cuda. Same speedup as the teacher path; matches
        # scipy denoise's largest-CC output. Fallback to scipy on CPU.
        if label_3d.is_cuda:
            from inference.inference3d_teacher import _denoise_mask_gpu
            refined = _denoise_mask_gpu(label_3d, iterations=refine_iterations)
        else:
            refined = denoise_mask(label_3d, iterations=refine_iterations)
        label_3d = refined.clone()
        if torch.count_nonzero(refined) == 0:
            return "zero mask"

        # Recompute bounding box
        coords = torch.nonzero(refined, as_tuple=False)
        zmin, zmax = coords[:, 0].min().item(), coords[:, 0].max().item()
        ymin, ymax = coords[:, 1].min().item(), coords[:, 1].max().item()
        xmin, xmax = coords[:, 2].min().item(), coords[:, 2].max().item()

        zmin = max(zmin - mz, 0)
        zmax = min(zmax + mz, D - 1)
        ymin = max(ymin - my, 0)
        ymax = min(ymax + my, H - 1)
        xmin = max(xmin - mx, 0)
        xmax = min(xmax + mx, W - 1)
        
        if xmin > xmax:   xmin, xmax = xmax, xmin
        if ymin > ymax:   ymin, ymax = ymax, ymin
        if zmin > zmax:   zmin, zmax = zmax, zmin

        bbox_d, bbox_h, bbox_w = bbox_dim(zmin, zmax, ymin, ymax, xmin, xmax)
        if bbox_d > desired_d or bbox_h > desired_h or bbox_w > desired_w:
            return "mask does not fit crop size"

    ##### 3) We know bounding box is <= crop_size. Let's find valid shifts. #####

    # We want subvolume [zstart : zstart+desired_d-1] to fully contain [zmin : zmax].
    # => zstart <= zmin
    # => zstart+desired_d-1 >= zmax => zstart >= zmax - (desired_d-1)
    # So zstart in [ zmax-(desired_d-1), zmin ]
    # Also zstart cannot be negative, and zstart+desired_d-1 cannot extend beyound the volume.
    # We'll define a helper:

    def valid_shifts_1D(min_bb, max_bb, vol_size, crop_size):
        """
        Returns a range (low, high) of all valid starting positions 
        such that [start : start+crop_size-1] fully contains [min_bb : max_bb]
        and stays within [0, vol_size-1].
        If there's no valid integer in [low, high], no shift is possible.
        """
        min_start = max_bb - (crop_size - 1)  # bounding box forced at the 'end'
        max_start = min_bb                    # bounding box forced at the 'start'

        # clamp to [0, vol_size - crop_size]
        lower_bound = 0
        upper_bound = vol_size - crop_size

        # intersection
        final_low = max(min_start, lower_bound)
        final_high = min(max_start, upper_bound)
        return int(final_low), int(final_high)

    # z dimension
    z_low, z_high = valid_shifts_1D(zmin, zmax, D, desired_d)
    # y dimension
    y_low, y_high = valid_shifts_1D(ymin, ymax, H, desired_h)
    # x dimension
    x_low, x_high = valid_shifts_1D(xmin, xmax, W, desired_w)

    # If any dimension has final_low > final_high, 
    # there's no integer that can satisfy bounding box constraints.
    if z_low > z_high or y_low > y_high or x_low > x_high:
        return "mask does not fit crop size"

    # Helper to pick shift in one dimension
    # If there's no valid shift (low>high), we 'crop in place' by placing bounding box at zmin
    # (clamped so we stay inside [0, vol_size - crop_size]).
    def pick_shift_1d(low, high, bb_min, vol_size, csize, rand_flag):
        if low > high:
            # No shift range => just place bounding box at bb_min (clamp to valid range)
            return max(0, min(bb_min, vol_size - csize))
        else:
            if rand_flag:
                return random.randint(int(low), int(high))
            else:
                return (low + high) // 2

    ##### 4) Pick the shift (or no shift if none is possible) #####
    z_start = pick_shift_1d(z_low, z_high, zmin, D, desired_d, rand)
    y_start = pick_shift_1d(y_low, y_high, ymin, H, desired_h, rand)
    x_start = pick_shift_1d(x_low, x_high, xmin, W, desired_w, rand)

    z_end = z_start + desired_d
    y_end = y_start + desired_h
    x_end = x_start + desired_w
    
    def dbg(dim, low, high, start, bb_min, bb_max, size):
        print(f"{dim}:  bb=({bb_min},{bb_max})  "
            f"shift_range=[{low},{high}]  chosen={start}  "
            f"crop=({start},{start+size-1})")
    #dbg('z', z_low, z_high, z_start, zmin, zmax, desired_d)
    #dbg('y', y_low, y_high, y_start, ymin, ymax, desired_h)
    #dbg('x', x_low, x_high, x_start, xmin, xmax, desired_w)

    # Now we check if indeed we are inside the volume
    if z_end > D or y_end > H or x_end > W:
        raise ValueError(f"Crop failed. Why? It should not fail here.")

    ##### 5) Final Crop #####
    if ct_has_channel and not ct_has_batch:
        cropped_ct = tensor_ct[:, z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[:, z_start:z_end, y_start:y_end, x_start:x_end]
    elif ct_has_channel and ct_has_batch:
        cropped_ct = tensor_ct[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
    else:
        cropped_ct = tensor_ct[z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[z_start:z_end, y_start:y_end, x_start:x_end]

    if cropped_ct.shape[-3:] != (desired_d, desired_h, desired_w):
        raise ValueError(f"Crop failed, got {cropped_ct.shape[-3:]}. Why? It should not fail here.")
    
    cropped_fg = label_3d[z_start:z_end, y_start:y_end, x_start:x_end]
    if torch.count_nonzero(cropped_fg) == 0 or \
        (torch.count_nonzero(cropped_fg) >= torch.count_nonzero(label_3d)*1.5) or \
        (torch.count_nonzero(cropped_fg) <= torch.count_nonzero(label_3d)*0.5):
        #is the original foreground 0?
        print('Original foreground total:', torch.count_nonzero(label_3d))
        print('Cropped foreground total:',torch.count_nonzero(cropped_fg))
        #check for inplace changes at foreground
        print('Inplace changes in foreground:',(not torch.equal(foreground,backup_foreground)))
        #is the problem in random??
        z_start = pick_shift_1d(z_low, z_high, zmin, D, desired_d, False)
        y_start = pick_shift_1d(y_low, y_high, ymin, H, desired_h, False)
        x_start = pick_shift_1d(x_low, x_high, xmin, W, desired_w, False)

        z_end = z_start + desired_d
        y_end = y_start + desired_h
        x_end = x_start + desired_w
        
        cropped_fg_deter = backup_foreground[z_start:z_end, y_start:y_end, x_start:x_end]
        
        print('Deter foreground total:',torch.count_nonzero(cropped_fg_deter))
        raise ValueError("zero mask after crop")
    
    if return_coordinate:
        coord=[z_start,z_end, y_start,y_end, x_start,x_end]
        return (cropped_ct, cropped_label, cropped_fg, coord)
    else:
        return (cropped_ct, cropped_label, cropped_fg)
    

from scipy.ndimage import binary_erosion, binary_dilation, label

def denoise_mask(mask_3d, iterations=2, connected_component=True):
    """
    Perform `iterations` binary erosions + `iterations` binary dilations,
    then AND with the original mask to remove small/noisy regions.
    Then keep only the largest connected component of the result.
    """
    device = mask_3d.device
    #check if mask is torch tensor
    if isinstance(mask_3d, torch.Tensor):
        np_mask = mask_3d.cpu().numpy().astype(bool)
    else:
        np_mask = mask_3d.astype(bool)

    # 1) Morphological denoise
    eroded  = binary_erosion(np_mask, iterations=iterations)
    dilated = binary_dilation(eroded,  iterations=iterations)
    final   = dilated & np_mask  # shape: (D,H,W), bool

    if connected_component:
        # 2) Label connected components in `final`
        labeled, num_components = label(final)  # labeled: int array with [1..num_components] labels

        if num_components == 0:
            # No foreground at all
            refined_mask = torch.from_numpy(final).to(device)
        elif num_components == 1:
            # Only one component, so it's already the largest
            refined_mask = torch.from_numpy(final).to(device)
        else:
            # More than one => pick largest
            # counts[i] = number of voxels with label i
            counts = np.bincount(labeled.ravel())
            # Index 0 is background, so ignore it by zeroing it out.
            counts[0] = 0  
            largest_label = np.argmax(counts)     # The label with the most voxels
            largest_mask = (labeled == largest_label)
            refined_mask = torch.from_numpy(largest_mask).to(device)
    else:
        # No connected component analysis, just return the mask
        refined_mask = torch.from_numpy(final).to(device)

    return refined_mask