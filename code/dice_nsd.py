#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Row‑wise Dice + NSD evaluator for pancreatic_lesion masks
(+ optional connected‑component pruning).
"""
import argparse, glob, os, sys, csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from filelock import FileLock
import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import torch
from monai.metrics import compute_surface_dice

# --------------------------- CSV layout ------------------------------ #
CSV_COLS = ["case", "dice", "nsd", "error"]


def write_row(row, csv_path):
    lock = FileLock(csv_path + ".lock")
    with lock:
        hdr = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if hdr:
                w.writeheader()
            w.writerow(row)


# --------------------------- metrics --------------------------------- #
def dice_score(pred, gt):
    tp = (pred & gt).sum().item()
    return 2.0 * tp / (pred.sum().item() + gt.sum().item() + 1.0)


def surf_dice(pred, gt, spacing, tol):
    return compute_surface_dice(pred.unsqueeze(0).unsqueeze(0).float(),
                                gt.unsqueeze(0).unsqueeze(0).float(),
                                include_background=True,
                                spacing=tuple(float(s) for s in spacing),
                                class_thresholds=[tol]).item()


# --------------------------- helpers --------------------------------- #
def load_mask(path):
    img = nib.load(path)
    return img.get_fdata() > 0.5, img.header.get_zooms()[:3]


def cc_filter(mask, min_vox):
    if min_vox <= 0:
        return mask
    lab, n = ndimage.label(mask, np.ones((3, 3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    remove = sizes < min_vox
    remove[0] = False
    mask[remove[lab]] = 0
    return mask


# --------------------------- worker ---------------------------------- #
def eval_case(task):
    cid, pred_p, gt_p, tol, cc_thr, csvp, skip_neg = task
    try:
        pred_np, spacing = load_mask(pred_p)
        gt_np,  _        = load_mask(gt_p)

        # optional CC pruning
        if cc_thr > 0:
            pred_np = cc_filter(pred_np.astype(np.uint8), cc_thr)

        # handle empty‑GT logic
        if not gt_np.any():
            if skip_neg:                # simply ignore this scan
                return None
            else:                       # score it
                empty_pred = not pred_np.any()
                dice = nsd = 1.0 if empty_pred else 0.0
                row = dict(case=cid,
                           dice=f"{dice:.4f}",
                           nsd=f"{nsd:.4f}",
                           error="")
                write_row(row, csvp)
                return row

        # normal (non‑empty) case
        p = torch.from_numpy(pred_np.astype(np.uint8))
        g = torch.from_numpy(gt_np.astype(np.uint8))
        row = dict(case=cid,
                   dice=f"{dice_score(p, g):.4f}",
                   nsd=f"{surf_dice(p, g, spacing, tol):.4f}",
                   error="")
        write_row(row, csvp)
        return row

    except Exception as e:
        write_row(dict(case=cid, dice="nan", nsd="nan", error=str(e)), csvp)
        return None


# --------------------------- CLI / main ------------------------------ #
def split_parts(lst, n, idx):
    base, extra = divmod(len(lst), n)
    start = idx * base + min(idx, extra)
    end   = start + base + (1 if idx < extra else 0)
    return lst[start:] if idx == n - 1 else lst[start:end]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pred_root", required=True)
    p.add_argument("--gt_root",   required=True)
    p.add_argument("--out_csv",   required=True)
    p.add_argument("--tolerance", type=float, default=3.0)
    p.add_argument("--cc_post",   type=int,   default=100,
                   help="min voxels per connected component (0 = disable)")
    p.add_argument("--skip_negatives", action="store_true",
                   help="Ignore scans with empty GT mask")
    p.add_argument("--workers",   type=int, default=cpu_count())
    p.add_argument("--num_parts", type=int, default=1)
    p.add_argument("--part",      type=int, default=0)
    p.add_argument("--continue",  dest="cont", action="store_true",
                   help="Skip IDs already in output CSV")
    return p.parse_args()


def main():
    a = parse_args()

    gt_map = {Path(p).parent.name: p
              for p in glob.glob(os.path.join(a.gt_root,
                                              "BDMAP_*",
                                              "pancreatic_lesion.nii.gz"))}

    pred_map = {Path(p).parents[1].name[:-7]: p
                for p in glob.glob(os.path.join(a.pred_root,
                                                "BDMAP_*.nii.gz",
                                                "predictions",
                                                "pancreatic_lesion.nii.gz"))}

    ids = sorted(set(gt_map) & set(pred_map))
    if not ids:
        sys.exit("No matching IDs")

    if a.cont and os.path.exists(a.out_csv):
        with FileLock(a.out_csv + ".lock"):
            done = {r.split(",")[0] for r in open(a.out_csv)}
        ids = [i for i in ids if i not in done]
    else:
        with FileLock(a.out_csv + ".lock"):
            if os.path.exists(a.out_csv):
                os.remove(a.out_csv)

    if a.num_parts > 1:
        ids = split_parts(ids, a.num_parts, a.part)

    print(f"cases={len(ids)}  part={a.part}/{a.num_parts-1}  "
          f"workers={a.workers}  cc_post={a.cc_post}  "
          f"skip_negatives={a.skip_negatives}")

    tasks = [(cid, pred_map[cid], gt_map[cid],
              a.tolerance, a.cc_post,
              a.out_csv, a.skip_negatives) for cid in ids]

    if a.workers == 1:
        for t in tqdm(tasks, desc="eval"):
            eval_case(t)
    else:
        with Pool(a.workers) as pool:
            for _ in tqdm(pool.imap_unordered(eval_case, tasks),
                          total=len(tasks), desc="eval"):
                pass

    print("Finished – rows appended to", a.out_csv)


if __name__ == "__main__":
    main()