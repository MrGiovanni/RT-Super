import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import split_idx
import pdb
import numpy as np
import random
#from ..training.augmentation import crop_foreground_3d

# tumor_info_builder lives at the MedFormer root; this module is one level
# deeper. Importing lazily inside the function would also work, but a top
# import is cheap and makes the dependency explicit.
from tumor_info_builder import lesion_like_name as _lesion_like_name


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
                             age = None, sex = None):
    if args.inference_2_stages:
        pred_output, cls_output =  inference_2_stages(net, img, args, organs_with_tumors=args.organs_with_tumor, class_list=args.class_list,
                                                      use_transformer_decoder=use_transformer_decoder,age=age,sex=sex)
    else:
        pred_output, cls_output = inference_sliding_window_one_pass(net, img, args, pancreas=pancreas, gaussian=gaussian,
                                                                    use_transformer_decoder=use_transformer_decoder,age=age,sex=sex)
        # inference_sliding_window_one_pass now returns on GPU; move to
        # CPU before returning to predict_abdomenatlas_teacher.prediction().
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

    # `keep_class_indices` (when set) caps the channels allocated in
    # pred_output. On big Turkish CTs the full 74-channel bf16 buffer is
    # 21+ GB on GPU which busts the activation budget; restricting to
    # the ~16 keep classes drops it to ~4.5 GB and lets us stay on GPU.
    _keep_idx_t = None
    if keep_class_indices is not None and len(keep_class_indices) > 0:
        _keep_sorted = sorted(set(int(i) for i in keep_class_indices))
        _keep_idx_t = torch.as_tensor(_keep_sorted, dtype=torch.long, device=img.device if img.is_cuda else 'cpu')
        _n_out = len(_keep_sorted)
    else:
        _n_out = args.classes

    # Accumulators on GPU when the input is on GPU AND the pred buffer
    # fits with activation headroom; else CPU fallback. Margin 28 GB
    # covers model (~5 GB), counter, gauss_w, and the stage-2 activations
    # on big T_oplc CTs (we hit 45 GB peak with 20 GB margin on 700-slice
    # Turkish cases).
    _pred_bytes = B * _n_out * D * H * W * 2
    _free_bytes = torch.cuda.mem_get_info(img.device)[0] if img.is_cuda else 0
    _gpu_ok = img.is_cuda and (_pred_bytes < max(_free_bytes - 28 * (1 << 30), 0))
    _acc_dev = img.device if _gpu_ok else torch.device('cpu')
    print(f'[SW] pred_output device={_acc_dev}  pred_GB={_pred_bytes/(1<<30):.1f}  free_GB={_free_bytes/(1<<30):.1f}  n_out={_n_out}/{args.classes}')
    pred_output = torch.zeros((B, _n_out, D, H, W), dtype=torch.bfloat16, device=_acc_dev)
    cls_output = None
    cls_counter = torch.zeros((B, 1, D, H, W), dtype=torch.bfloat16, device=_acc_dev)

    counter = torch.zeros((B, 1, D, H, W), dtype=torch.bfloat16, device=_acc_dev)
    one_count = torch.ones((B, 1, win_d, win_h, win_w), dtype=torch.bfloat16, device=_acc_dev)
    # Move keep-index tensor to accumulator device if pred_output ended
    # up on CPU.
    if _keep_idx_t is not None and _keep_idx_t.device != _acc_dev:
        _keep_idx_t_acc = _keep_idx_t.to(_acc_dev)
    else:
        _keep_idx_t_acc = _keep_idx_t
    if gaussian:
        gauss_w = gauss_w.to(_acc_dev)

    # --- profiling counters (active only when DEBUG_SW_PROFILE=1) ---
    import os as _os_prof, time as _time_prof
    _prof = (_os_prof.environ.get('DEBUG_SW_PROFILE') == '1')
    _t_fwd = _t_acc = 0.0
    _n_fwd = _n_skip = 0
    SW_BATCH = int(_os_prof.environ.get('SW_BATCH', '1'))
    if _prof:
        torch.cuda.synchronize()
    _t_loop_start = _time_prof.time() if _prof else 0.0

    # Phase 1: collect kept window coordinates (those that pass the
    # pancreas/ROI skip check). This decouples the skip filter from the
    # forward pass so we can batch the latter.
    kept = []
    for i in range(D // half_win_d):
        for j in range(H // half_win_h):
            for k in range(W // half_win_w):
                d0, d1 = split_idx(half_win_d, D, i)
                h0, h1 = split_idx(half_win_h, H, j)
                w0, w1 = split_idx(half_win_w, W, k)
                if pancreas is None or pancreas[:, :, d0:d1, h0:h1, w0:w1].sum() > 0:
                    kept.append((d0, d1, h0, h1, w0, w1))
                else:
                    _n_skip += 1

    # Phase 2: batched forward + per-window accumulate. The model is
    # heavily under-utilised at SW_BATCH=1 (a 128^3 patch through ~100M
    # params is well under what the GPU can do in a single kernel
    # launch). Batching gives near-linear throughput up to GPU
    # saturation; SW_BATCH=4 is a safe default on RTX 6000 Ada with
    # the rest of the GPU state in this pipeline.
    with torch.no_grad():
        for _b_start in range(0, len(kept), SW_BATCH):
            coords_batch = kept[_b_start:_b_start + SW_BATCH]
            input_batch = torch.cat(
                [img[:, :, d0:d1, h0:h1, w0:w1] for (d0, d1, h0, h1, w0, w1) in coords_batch],
                dim=0,
            )  # (Bsw, C, 128, 128, 128)

            if _prof:
                torch.cuda.synchronize(); _t0 = _time_prof.time()
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
            if _prof:
                torch.cuda.synchronize(); _t_fwd += _time_prof.time() - _t0; _n_fwd += len(coords_batch)

            if _prof:
                torch.cuda.synchronize(); _t0 = _time_prof.time()

            # When keep_class_indices is set, restrict pred to those
            # channels BEFORE the cross-device transfer so we move 16
            # channels instead of 74 (~5x less data per window).
            if _keep_idx_t is not None:
                pred = pred.index_select(dim=1, index=_keep_idx_t)
            pred_bf = pred.to(torch.bfloat16).to(_acc_dev)  # (Bsw, n_out, 128, 128, 128)
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
                    w_patch_cls = w_patch.expand(B, _n_out, *w_patch.shape[2:])
                    w_patch_cnt = w_patch.expand(B, 1,      *w_patch.shape[2:])
                    pred_output[:, :, d0:d1, h0:h1, w0:w1] += (pred_bf[b_idx:b_idx+1] * w_patch_cls)
                    counter[:, :, d0:d1, h0:h1, w0:w1] += w_patch_cnt
            if _prof:
                torch.cuda.synchronize(); _t_acc += _time_prof.time() - _t0

    if _prof:
        torch.cuda.synchronize()
        _t_loop = _time_prof.time() - _t_loop_start
        print(f"[SW-PROF] loop={_t_loop:.2f}s  fwd_n={_n_fwd} fwd_total={_t_fwd:.2f}s avg={_t_fwd/max(_n_fwd,1):.3f}s  "
              f"skip_n={_n_skip}  acc={_t_acc:.2f}s  "
              f"residual={_t_loop - _t_fwd - _t_acc:.2f}s  SW_BATCH={SW_BATCH}", flush=True)

    pred_output /= (counter + 1e-6)
    if flag:
        pred_output = pred_output[:, :, :origin_D, :origin_H, :origin_W]

    # Caller decides where to move it — inference_2_stages_teacher keeps
    # it on GPU for stage 2 and only moves to CPU at the function
    # boundary. inference_sliding_window (the non-stage-2 path) moves to
    # CPU at its own return.
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


def inference_2_stages(net, img, args, organs_with_tumors=None, class_list=None, use_transformer_decoder=False,age=None,sex=None):
    #stage 1: coarse, run over all the CT
    pred_output, cls_output = inference_sliding_window_one_pass(net, img, args, pancreas=None,gaussian=True, use_transformer_decoder=use_transformer_decoder,age=age,sex=sex)
    pred_output_stage_1_binary = (pred_output > 0.5).float()
    #stage 2: identify organs with lesion, crop on them, substitute over output. For overlaps in stage 2, do simple average.
    
    #prepare zero tensor for output
    B, C, D, H, W = img.shape
    win_d, win_h, win_w = args.window_size
    
    #run over each organ that may have tumors
    for batch in range(B):
        pred_output_stage_2 = torch.zeros((1, args.classes, D, H, W)).to(torch.bfloat16).cpu()#.to(img.device)
        pred_output_cls_stage_2 = None
        
        counter = torch.zeros((1, 1, D, H, W)).to(torch.bfloat16).cpu()#.to(img.device)
        one_count = torch.ones((1, 1, win_d, win_h, win_w),dtype=torch.bfloat16).cpu()#.to(img.device)
        for org in organs_with_tumors:
            org_idx = class_list.index(org)
            #check if the organ is present
            if pred_output_stage_1_binary[batch, org_idx, :, :, :].sum() > 0:
                #get the mask of the organ
                organ_mask = pred_output_stage_1_binary[batch, org_idx, :, :, :]
                x=img[batch,0]
                out = crop_foreground_3d(tensor_ct=x, tensor_lab=pred_output_stage_1_binary[batch], foreground=organ_mask, 
                                         crop_size=[win_d, win_h, win_w],rand=False,return_coordinate=True)
                if not isinstance(out, tuple):
                    #failed crop on organ
                    print(f"Failed to crop on organ {org}: {out}")
                    continue
                cropped_ct, _, cropped_organ, coord = out
                d_start_idx,d_end_idx, h_start_idx,h_end_idx, w_start_idx,w_end_idx = coord
                #run inference on the cropped ct
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
                        #print('using sigmoid')
                    else:
                        pred = F.softmax(pred, dim=1)
                #add the prediction to the output of stage 2
                pred_output_stage_2[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred.to(torch.bfloat16).cpu()
                if pred_cls is not None:
                    if pred_output_cls_stage_2 is None:
                        pred_output_cls_stage_2 = torch.zeros((1, pred_cls.shape[1], D, H, W),dtype=torch.bfloat16).cpu()
                    pred_output_cls_stage_2[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred_cls.to(torch.bfloat16).cpu()
                counter[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count.to(torch.bfloat16).cpu()
        mask = (counter > 0).float()
        #add epsilon to avoid division by zero
        counter = counter*mask + 1e-6*(1-mask)
        pred_output_stage_2 /= counter
        if pred_output_cls_stage_2 is not None:
            pred_output_cls_stage_2 /= counter

        #use mask to merge pred_output_stage_2 with pred_output
        pred_output[batch] = (1-mask) * pred_output[batch] + mask * pred_output_stage_2
        if pred_output_cls_stage_2 is not None:
            cls_output[batch] = (1-mask) * cls_output[batch] + mask * pred_output_cls_stage_2
        
    return pred_output, cls_output


def _canonical_organ(name):
    """Mirror dataset.canonical_organ at line 5306. Used to map an entry of
    `args.organs_with_tumor` (e.g. 'adrenal_gland_right', 'gall_bladder') to
    the canonical organ name the TumorInfoBuilder expects.
    """
    if isinstance(name, list):
        name = name[0]
    base = name.replace('_lesion', '')
    name = name.replace('_right', '').replace('_left', '')
    lower_base = base.lower()
    if 'pancrea' in lower_base:
        return 'pancreas'
    if 'kidney' in lower_base:
        return 'kidney'
    if 'adrenal' in lower_base:
        return 'adrenal'
    if 'lung' in lower_base:
        return 'lung'
    if 'femur' in lower_base:
        return 'femur'
    if 'gall' in lower_base:
        return 'gall_bladder'
    return base


_SIDED_ORGANS = {
    'kidney_left', 'kidney_right',
    'adrenal_gland_left', 'adrenal_gland_right',
    'lung_left', 'lung_right',
    'femur_left', 'femur_right',
}


def _organ_side(name):
    """Return 'right' / 'left' / None for class-list organ names."""
    if name.endswith('_right'):
        return 'right'
    if name.endswith('_left'):
        return 'left'
    return None


def _summarise_tensor_for_dump(t):
    if not isinstance(t, torch.Tensor):
        return {'type': type(t).__name__, 'value': str(t)[:200]}
    return {
        'shape': list(t.shape),
        'dtype': str(t.dtype),
        'sum': float(t.float().sum().item()),
        'nonzero_count': int((t != 0).sum().item()),
        'min': float(t.min().item()) if t.numel() else 0.0,
        'max': float(t.max().item()) if t.numel() else 0.0,
        'first_voxels_flat': t.float().flatten()[:5].tolist() if t.numel() else [],
    }


def _vector_or_summary_for_dump(t):
    if not isinstance(t, torch.Tensor):
        return _summarise_tensor_for_dump(t)
    if t.numel() <= 100:
        return {'shape': list(t.shape), 'dtype': str(t.dtype), 'values': t.detach().cpu().tolist()}
    return _summarise_tensor_for_dump(t)


def _get_lesion_crop_centers(prob_map_3d, binary_map_3d, n_target):
    """Pick up to `n_target` probability-weighted center-of-mass crop
    centers from the student lesion-channel output map.

    Connected components of `binary_map_3d` are scored by total probability
    inside each CC (using `prob_map_3d`); the top-`n_target` CCs by score
    are kept and the COM of each is returned (each voxel weighted by its
    student probability — treats the prob map as a density).

    Returns a list of (z,y,x) int tuples, length min(n_target, n_components).
    Empty if the binary map is empty or n_target == 0.
    """
    from scipy.ndimage import label as cc_label
    if torch.is_tensor(binary_map_3d):
        binary_np = binary_map_3d.detach().cpu().numpy().astype(bool)
    else:
        binary_np = np.asarray(binary_map_3d).astype(bool)
    if torch.is_tensor(prob_map_3d):
        prob_np = prob_map_3d.detach().float().cpu().numpy()
    else:
        prob_np = np.asarray(prob_map_3d, dtype=np.float32)
    if binary_np.sum() == 0 or n_target == 0:
        return []
    labeled, n_cc = cc_label(binary_np)
    if n_cc == 0:
        return []
    cc_scores = []
    for cc_id in range(1, n_cc + 1):
        mask = (labeled == cc_id)
        cc_scores.append((cc_id, float(prob_np[mask].sum())))
    cc_scores.sort(key=lambda x: -x[1])
    centers = []
    for cc_id, _ in cc_scores[:n_target]:
        mask = (labeled == cc_id)
        coords = np.argwhere(mask)
        w = prob_np[mask]
        ws = float(w.sum())
        if ws <= 0.0:
            c = coords.mean(axis=0)
        else:
            c = (coords * w[:, None]).sum(axis=0) / ws
        centers.append((int(c[0]), int(c[1]), int(c[2])))
    return centers


def _centered_window(center, win, total):
    """Return [lo, hi] of an axis-aligned window of size `win` centered at
    `center`, clipped to [0, total]. Slides the window if the centered box
    overflows the volume (matching crop_around_coordinate_3d's clip behaviour).
    """
    c = int(center)
    lo = c - win // 2
    if lo < 0:
        lo = 0
    if lo + win > total:
        lo = total - win
    return lo, lo + win


def inference_2_stages_teacher(student_net, teacher_net, img, args,
                                tumor_info_builder, bdmap_id,
                                organs_with_tumors, class_list,
                                age, sex,
                                give_size=True,
                                reproduce_sided_organ_bug=False,
                                filter_laterality=False,
                                dump_dir=None,
                                sw_roi_mask=None,
                                keep_class_indices=None):
    """Two-stage inference where stage 2 calls the report-informed teacher
    decoder per organ, with a tumor_info dict built from the per-tumor
    report (replicating the training-time `tumor_info_input`).

    Stage 1 is identical to `inference_2_stages` stage 1: a sliding-window
    pass over the full image with the student head (skip_report_informed_decoder
    True). Stage 2 crops on each `organs_with_tumors` channel of the stage-1
    binary mask, exactly like `inference_2_stages` line 246-252, then calls
    the teacher with `report_info`, `use_report=ones`,
    `skip_report_informed_decoder=False` and pastes ONLY the lesion channel
    for that organ back into the full-image volume (per user direction).

    The benign skip on crop failure (`if not isinstance(out, tuple)`) is the
    only allowed silent skip; non-crop failures propagate.
    """
    # Stage 1 — student over the full CT.
    # `sw_roi_mask` (when non-None) is a binary union of GT organ masks
    # passed as the SW window-skip ROI — windows that don't intersect it
    # have their forward pass skipped (existing behaviour of the
    # `pancreas` parameter in inference_sliding_window_one_pass).
    import os as _os_prof, time as _time_prof
    _prof2 = (_os_prof.environ.get('DEBUG_SW_PROFILE') == '1')
    pred_output, cls_output = inference_sliding_window_one_pass(
        student_net, img, args, pancreas=sw_roi_mask, gaussian=True,
        use_transformer_decoder=False, age=age, sex=sex,
        keep_class_indices=keep_class_indices)
    # Build full_idx → keep_idx mapping. If keep_class_indices was given,
    # pred_output's channel layout is sorted(keep_class_indices) (mirrors
    # the SW's allocation). Use the mapping to translate stage-2's
    # full-class indices to pred_output's channel indices.
    if keep_class_indices is not None and len(keep_class_indices) > 0:
        _keep_sorted = sorted(set(int(i) for i in keep_class_indices))
        _full_to_keep = {fi: ki for ki, fi in enumerate(_keep_sorted)}
    else:
        _full_to_keep = None  # 1:1, no translation needed
    if _prof2:
        torch.cuda.synchronize(); _t_s2_start = _time_prof.time()
        _t_org_total = 0.0; _t_org_inner = {}
    # pred_output is on GPU; we keep it there for stage 2 and only move
    # to CPU at the end of this function. The global pred_output_stage_1_binary
    # used to be (74, D, H, W) float32 ≈ 35 GB on CPU — replaced here
    # with per-organ binary masks computed lazily inside the loop.

    B, C, D, H, W = img.shape
    win_d, win_h, win_w = args.window_size
    _dev = pred_output.device

    for batch in range(B):
        # Per-channel sparse accumulators on the same device as pred_output
        # (GPU). Each (D, H, W) bfloat16 buffer is ~240 MB; with ≤9 unique
        # lesion channels touched per case, total GPU overhead ~4 GB —
        # negligible vs the 5 GB model + 18 GB pred_output already on GPU.
        pred_per_ch = {}      # lesion_ch -> tensor (D, H, W) bfloat16 sum (on _dev)
        counter_per_ch = {}   # lesion_ch -> tensor (D, H, W) bfloat16 counter (on _dev)
        one_count = torch.ones((win_d, win_h, win_w), dtype=torch.bfloat16, device=_dev)

        for org in organs_with_tumors:
            if _prof2: torch.cuda.synchronize(); _t_org0 = _time_prof.time()
            org_idx_full = class_list.index(org)
            # Translate to pred_output channel index. If keep_class_indices
            # was set, pred_output's channels are a subset; the organ must
            # be present in keep (it is, since predict_abdomenatlas_teacher
            # builds keep_classes from args.organs_with_tumor).
            if _full_to_keep is not None:
                if org_idx_full not in _full_to_keep:
                    print(f"[teacher] {org} (full_idx={org_idx_full}) not in keep_classes — skipped")
                    continue
                org_idx = _full_to_keep[org_idx_full]
            else:
                org_idx = org_idx_full
            organ_prob = pred_output[batch, org_idx]
            organ_mask = (organ_prob > 0.5).float()
            if _prof2: torch.cuda.synchronize(); _t_org_inner.setdefault(org,{})['mask']=_time_prof.time()-_t_org0
            if organ_mask.sum() == 0:
                continue

            organ_canonical = _canonical_organ(org)
            lesion_name = _lesion_like_name(org)
            if lesion_name not in class_list:
                print(f"[teacher] no lesion channel for {org} (lesion_name={lesion_name}) — skipped")
                continue
            lesion_ch_full = class_list.index(lesion_name)
            if _full_to_keep is not None:
                if lesion_ch_full not in _full_to_keep:
                    print(f"[teacher] lesion {lesion_name} not in keep_classes — skipped")
                    continue
                lesion_ch = _full_to_keep[lesion_ch_full]
            else:
                lesion_ch = lesion_ch_full
            x = img[batch, 0]
            organ_pred = (organ_mask > 0)
            lesion_prob = pred_output[batch, lesion_ch]
            lesion_pred = (lesion_prob > 0.5)

            # ----- Determine the list of crop windows for this organ -----
            # Each entry is (cropped_ct[3D], coord=[d0,d1,h0,h1,w0,w1]).
            #
            # Policy ("organ + outside lesion-COMs"):
            #   1. Always crop on the stage-1 organ prediction (centered, 128^3).
            #   2. For each lesion reported in the report (N = number of report
            #      rows for this organ/side), compute the probability-weighted
            #      center of mass of one connected component of the student's
            #      lesion-channel binary prediction (top-N by total CC prob).
            #   3. If a lesion COM falls OUTSIDE the organ-crop window, add a
            #      separate 128^3 crop centered on that COM. COMs already
            #      inside the organ crop are skipped (already covered).
            crop_windows = []
            # tensor_lab only needs to share spatial dims with tensor_ct
            # (crop_foreground_3d slices it but stage 2 discards the
            # cropped label). Pass the small per-organ binary instead of
            # the original 74-channel global binary so we don't move 35 GB
            # through crop_foreground_3d 9 times.
            if _prof2: torch.cuda.synchronize(); _t_pre_crop = _time_prof.time()
            out = crop_foreground_3d(
                tensor_ct=x, tensor_lab=organ_mask,
                foreground=organ_mask, crop_size=[win_d, win_h, win_w],
                rand=False, return_coordinate=True)
            if _prof2: torch.cuda.synchronize(); _t_org_inner.setdefault(org,{})['crop']=_time_prof.time()-_t_pre_crop
            organ_coord = None
            if not isinstance(out, tuple):
                print(f"Failed to crop on organ {org}: {out} — no organ-crop window")
            else:
                cropped_ct_o, _, _, organ_coord = out
                organ_coord = list(organ_coord)
                crop_windows.append((cropped_ct_o, organ_coord))

            _side = _organ_side(org) if filter_laterality else None
            try:
                _rows = tumor_info_builder._select_rows(
                    bdmap_id, organ_canonical, side=_side)
            except Exception as e:
                print(f"[teacher] _select_rows failed for {bdmap_id}/{organ_canonical}: {e}")
                _rows = []
            n_target = len(_rows)
            if n_target > 0:
                centers = _get_lesion_crop_centers(
                    lesion_prob, lesion_pred, n_target)
                if organ_coord is not None:
                    oz0, oz1, oh0, oh1, ow0, ow1 = organ_coord
                for (cz, cy, cx) in centers:
                    inside_organ = (
                        organ_coord is not None
                        and oz0 <= cz < oz1
                        and oh0 <= cy < oh1
                        and ow0 <= cx < ow1
                    )
                    if inside_organ:
                        print(f"[teacher] lesion COM ({cz},{cy},{cx}) inside "
                              f"organ crop — covered, no extra window")
                        continue
                    d0, d1 = _centered_window(cz, win_d, D)
                    h0, h1 = _centered_window(cy, win_h, H)
                    w0, w1 = _centered_window(cx, win_w, W)
                    cct = x[d0:d1, h0:h1, w0:w1]
                    crop_windows.append((cct, [d0, d1, h0, h1, w0, w1]))
                    print(f"[teacher] adding lesion-COM crop at ({cz},{cy},{cx}) "
                          f"outside organ crop for {org}")
            if len(crop_windows) == 0:
                print(f"[teacher] no usable windows for {org} — skipping")
                continue

            # Spatial mask given to the teacher: binarize(elementwise-max(
            # stage-1 organ_prob, stage-1 lesion_prob)) at th=0.5. The teacher
            # was trained with binary masks; this is the union of the two
            # stage-1 binary predictions.
            full_res_spatial_mask = (torch.maximum(organ_prob, lesion_prob) > 0.5).float()

            # ----- Per-window teacher pass -----
            # 1 organ-crop window + up to N outside-organ lesion-COM windows.
            for _w_idx, (cropped_ct, coord) in enumerate(crop_windows):
                d0, d1, h0, h1, w0, w1 = coord
                # Bug-reproduction mode: at training, the dataset's
                # estimate_tumor_volume('adrenal_gland_right') (and other
                # sided sub-segments) returns zero rows due to a string
                # quirk — so the model saw the no_tumor branch for sided
                # crops. Mirror that here when requested.
                if reproduce_sided_organ_bug and org in _SIDED_ORGANS:
                    tumor_info = tumor_info_builder.build(
                        bdmap_id=bdmap_id,
                        organ_canonical='__bug_no_tumor__',
                        full_res_lesion_mask_3d=full_res_spatial_mask,
                        crop_coords=list(coord),
                    )
                    print(f"[teacher] bug-reproduce: forcing no-tumor info for {org}")
                else:
                    side = _organ_side(org) if filter_laterality else None
                    tumor_info = tumor_info_builder.build(
                        bdmap_id=bdmap_id,
                        organ_canonical=organ_canonical,
                        full_res_lesion_mask_3d=full_res_spatial_mask,
                        crop_coords=list(coord),
                        side=side,
                    )
                if dump_dir is not None:
                    import os, json
                    os.makedirs(dump_dir, exist_ok=True)
                    rec = {
                        'bdmap_id': bdmap_id,
                        'organ_internal': org,
                        'organ_canonical': organ_canonical,
                        'window_idx': _w_idx,
                        'crop_coord_zyx': [int(c) for c in coord],
                        'spatial_mask_full_res': _summarise_tensor_for_dump(full_res_spatial_mask),
                        'tumor_info_input': {k: _vector_or_summary_for_dump(v) for k, v in tumor_info.items()},
                    }
                    org_safe = org.replace(' ', '_')
                    suffix = f'__w{_w_idx}' if len(crop_windows) > 1 else ''
                    with open(os.path.join(dump_dir, f'{bdmap_id}__{org_safe}{suffix}.json'), 'w') as f:
                        json.dump(rec, f, indent=2, default=str)

                tumor_info_gpu = {
                    k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else v)
                    for k, v in tumor_info.items()
                }

                with torch.no_grad():
                    _t_kwargs = {}
                    if getattr(args, 'time_points', 1) > 1:
                        _t_kwargs = dict(
                            dates=torch.zeros(1, 1, dtype=torch.float32, device='cuda'),
                            patient_ids=torch.zeros(1, dtype=torch.int64, device='cuda'),
                            cropped_organs=torch.zeros(1, dtype=torch.int64, device='cuda'),
                            dist=None,
                            organ_cropped_cannon=[organ_canonical],
                        )

                    import os as _os_d
                    if _os_d.environ.get('DEBUG_TEACHER_INPUT') == '1':
                        print(f'[DEBUG_TEACHER_INPUT] bdmap={bdmap_id} organ={org} canonical={organ_canonical} '
                              f'crop=({d0}:{d1},{h0}:{h1},{w0}:{w1}) window={_w_idx}', flush=True)
                        for _k, _v in tumor_info_gpu.items():
                            if isinstance(_v, torch.Tensor):
                                _s = float(_v.float().sum().item()) if _v.numel() else 0.0
                                _nz = int((_v != 0).sum().item()) if _v.numel() else 0
                                _head = (_v[0].detach().float().cpu().tolist() if _v.numel() <= 60
                                         else _v[0].detach().float().cpu().tolist()[:20] + ['...'])
                                print(f'  {_k:<55} shape={tuple(_v.shape)} sum={_s:.4f} nz={_nz} head={_head}', flush=True)
                            else:
                                print(f'  {_k:<55} type={type(_v).__name__} value={_v}', flush=True)
                        _ct = cropped_ct
                        print(f'  cropped_ct shape={tuple(_ct.shape)} '
                              f'sum={float(_ct.float().sum().item()):.4f} '
                              f'min={float(_ct.float().min().item()):.4f} '
                              f'max={float(_ct.float().max().item()):.4f}', flush=True)
                        for _k, _v in _t_kwargs.items():
                            if isinstance(_v, torch.Tensor):
                                print(f'  [t_kw] {_k:<20} shape={tuple(_v.shape)} value={_v.detach().cpu().tolist()}', flush=True)
                            else:
                                print(f'  [t_kw] {_k:<20} {_v!r}', flush=True)
                        print('  no_mask_training=', (not give_size), 'use_report=ones(1)', 'skip_report_informed_decoder=False', 'annotated_per_voxel=True', flush=True)

                    model_output = teacher_net(
                        cropped_ct.unsqueeze(0).unsqueeze(0).cuda(),
                        report_info=tumor_info_gpu,
                        use_report=torch.ones(1, dtype=torch.bool, device='cuda'),
                        skip_report_informed_decoder=False,
                        annotated_per_voxel=True,
                        no_mask_training=(not give_size),
                        age=age, sex=sex,
                        **_t_kwargs,
                    )
                    if isinstance(model_output, dict):
                        if 'refined_segmentation' not in model_output:
                            raise RuntimeError(
                                f"Teacher output missing 'refined_segmentation'; "
                                f"keys={list(model_output.keys())}")
                        pred = model_output['refined_segmentation']
                    else:
                        pred = model_output
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    if not args.epai_stage_2:
                        pred = torch.sigmoid(pred)
                    else:
                        pred = F.softmax(pred, dim=1)

                # Paste back ONLY the lesion channel for this organ — into the
                # per-channel sparse buffer. With multiple windows per organ
                # (organ-crop plus any outside-organ lesion-COM crops), each
                # window adds to the running sum and the counter tracks the
                # number of windows that touched each voxel; the blend at the
                # end averages across overlapping windows.
                if lesion_ch not in pred_per_ch:
                    pred_per_ch[lesion_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16, device=_dev)
                    counter_per_ch[lesion_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16, device=_dev)
                # `pred` is the teacher's full 74-channel output; index by
                # lesion_ch_full. `pred_per_ch` is keyed by pred_output's
                # channel index (keep-translated lesion_ch) so it matches
                # the final blend's pred_output[batch, ch] write.
                pred_per_ch[lesion_ch][d0:d1, h0:h1, w0:w1] += pred[0, lesion_ch_full].to(torch.bfloat16).to(_dev)
                counter_per_ch[lesion_ch][d0:d1, h0:h1, w0:w1] += one_count
                print(f"[teacher] pasted {org} -> ch {lesion_ch} ({lesion_name}) "
                      f"w={_w_idx} @ ({d0}:{d1},{h0}:{h1},{w0}:{w1})")

            if _prof2:
                torch.cuda.synchronize()
                _t_org_inner.setdefault(org,{})['total']=_time_prof.time()-_t_org0
                _t_org_total += _t_org_inner[org]['total']

        # Blend: average teacher's paste-backs over their counter, then
        # replace stage-1 voxels where counter>0 with the average. Stage-1
        # values on other channels and on untouched voxels are preserved.
        # Operates on GPU; pred_output is moved to CPU after the batch loop.
        for ch in list(pred_per_ch.keys()):
            pred_t = pred_per_ch.pop(ch)
            counter_t = counter_per_ch.pop(ch)
            mask = (counter_t > 0).to(torch.bfloat16)
            counter_safe = counter_t * mask + 1e-6 * (1 - mask)
            avg = pred_t / counter_safe
            pred_output[batch, ch] = (1 - mask) * pred_output[batch, ch] + mask * avg

    if _prof2:
        torch.cuda.synchronize()
        _t_s2 = _time_prof.time() - _t_s2_start
        print(f"[S2-PROF] stage2_total={_t_s2:.2f}s  per_organ_sum={_t_org_total:.2f}s", flush=True)
        for k,v in _t_org_inner.items():
            print(f"  {k:<25} mask={v.get('mask',0):.3f}s  crop={v.get('crop',0):.3f}s  total={v.get('total',0):.3f}s", flush=True)
    # Return on GPU; caller (prediction()) thresholds on GPU and only
    # moves the much-smaller uint8 label_pred to CPU. Avoids a 17.9 GB
    # bf16 PCIe transfer per case.
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
        # Hybrid GPU morphology + scipy CC on the bbox-cropped opened
        # mask. Matches the original full-volume scipy denoise's
        # largest-CC output, at ~5× the speed (avoids running scipy
        # label on 75 M voxels).
        if label_3d.is_cuda:
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


def _denoise_mask_gpu(mask_3d, iterations=2, connected_component=True):
    """Hybrid GPU/CPU denoise.

      - GPU morphological opening (`iterations` erosions then dilations
        via F.max_pool3d). Radius-1 per step, equivalent to scipy's
        binary_erosion/binary_dilation with the default 3^3 cross.
      - Largest-connected-component (when connected_component=True) via
        scipy.ndimage.label run on the BBOX-CROPPED opened mask. The
        full-volume tensor is ~75 M voxels (~1.5 s in scipy); the
        bbox-cropped tensor is typically ≤ 128^3 = 2 M voxels (~50 ms),
        producing an identical largest-CC result.

    Returns a (D, H, W) float32 binary mask on the same GPU device as
    `mask_3d`.
    """
    assert mask_3d.is_cuda
    m = mask_3d.unsqueeze(0).unsqueeze(0).float()  # (1, 1, D, H, W)

    # --- Morphological opening (radius-1 per iter) on GPU. ---
    inv = 1.0 - m
    for _ in range(iterations):
        inv = F.max_pool3d(inv, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - inv
    dilated = eroded
    for _ in range(iterations):
        dilated = F.max_pool3d(dilated, kernel_size=3, stride=1, padding=1)
    # Clamp to original — opening is always ⊂ original.
    opened = (dilated * m).squeeze(0).squeeze(0)  # (D, H, W)

    if not connected_component:
        return opened

    # --- Largest-CC via scipy on the BBOX-CROPPED opened mask. ---
    # 1) GPU bbox of the opened mask (cheap reductions; avoids the
    #    slow torch.nonzero on a noisy 75 M-voxel tensor).
    opened_bool = opened > 0
    z_any = opened_bool.any(dim=2).any(dim=1)  # (D,)
    y_any = opened_bool.any(dim=2).any(dim=0)  # (H,)
    x_any = opened_bool.any(dim=1).any(dim=0)  # (W,)
    if not bool(z_any.any().item()):
        return opened  # already empty
    z_nz = z_any.nonzero(as_tuple=True)[0]
    y_nz = y_any.nonzero(as_tuple=True)[0]
    x_nz = x_any.nonzero(as_tuple=True)[0]
    z0, z1 = int(z_nz[0].item()), int(z_nz[-1].item()) + 1
    y0, y1 = int(y_nz[0].item()), int(y_nz[-1].item()) + 1
    x0, x1 = int(x_nz[0].item()), int(x_nz[-1].item()) + 1

    # 2) Crop, move to CPU, scipy CC, keep-largest.
    cropped_np = opened[z0:z1, y0:y1, x0:x1].cpu().numpy().astype(bool)
    if not cropped_np.any():
        return opened
    labeled, n_cc = label(cropped_np)
    if n_cc > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background
        largest_label = int(sizes.argmax())
        cropped_np = (labeled == largest_label)

    # 3) Paste back into a zero full-volume tensor on GPU.
    out = torch.zeros_like(opened)
    out[z0:z1, y0:y1, x0:x1] = torch.from_numpy(cropped_np.astype(np.float32)).to(out.device)
    return out


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