#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dice & NSD for pancreatic_lesion **probability** maps.

GT  : <gt_root>/BDMAP_<ID>/pancreatic_lesion.nii.gz
PR  : <pred_root>/BDMAP_<ID>.nii.gz/predictions_raw/pancreatic_lesion.nii.gz
"""
import argparse, glob, os, sys, csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from filelock import FileLock
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
from monai.metrics import compute_surface_dice

# -------------------------------------------------------------------- #
THR = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
CSV_COLS = (
    ["case"] +
    [f"dice_{t}" for t in THR] +
    [f"nsd_{t}"  for t in THR]
)


def write_csv_row(row, csv_path):
    lock = FileLock(csv_path + ".lock")
    with lock:
        header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if header:
                w.writeheader()
            w.writerow(row)


# --------------------------- metrics --------------------------------- #
def dice(pred, gt):
    tp = (pred & gt).sum()
    den = pred.sum() + gt.sum() + 1.0
    return 2.0 * tp / den


def surf_dice(pred, gt, spacing, tol):
    p = pred.unsqueeze(0).unsqueeze(0).float()
    g = gt.unsqueeze(0).unsqueeze(0).float()
    return compute_surface_dice(p, g,
                                include_background=True,
                                spacing=tuple(float(s) for s in spacing),
                                class_thresholds=[tol]).item()


# --------------------------- IO -------------------------------------- #
def load_prob(path):
    img = nib.load(path)
    return img.get_fdata().astype(np.float32), img.header.get_zooms()[:3]


def load_gt(path):
    img = nib.load(path)
    return (img.get_fdata() > 0.5).astype(np.uint8)


# --------------------------- worker ---------------------------------- #
def eval_case(task):
    cid, prob_p, gt_p, tol, csv_path = task
    prob, spacing = load_prob(prob_p)
    gt   = load_gt(gt_p)
    if not gt.any():
        return None                    # skip empty GT

    row = {"case": cid}
    gt_t = torch.from_numpy(gt).to(torch.uint8)

    for t in THR:
        bin_pred = (prob > t).astype(np.uint8)
        p_t = torch.from_numpy(bin_pred).to(torch.uint8)

        row[f"dice_{t}"] = f"{dice(p_t, gt_t):.4f}"
        row[f"nsd_{t}"]  = f"{surf_dice(p_t, gt_t, spacing, tol):.4f}"

    write_csv_row(row, csv_path)
    return row


# --------------------------- CLI ------------------------------------- #
def split_parts(lst, n, idx):
    base, extra = divmod(len(lst), n)
    start = idx * base + min(idx, extra)
    end   = start + base + (1 if idx < extra else 0)
    return lst[start:] if idx == n - 1 else lst[start:end]


def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pred_root", required=True)
    p.add_argument("--gt_root",   required=True)
    p.add_argument("--out_csv",   required=True)
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument("--workers",   type=int, default=cpu_count())
    p.add_argument("--num_parts", type=int, default=1)
    p.add_argument("--part",      type=int, default=0)
    p.add_argument("--continue",  action="store_true",
                   help="Skip IDs already in CSV")
    return p.parse_args()


def main():
    a = parse()
    gt_files = glob.glob(os.path.join(a.gt_root,
                                      "BDMAP_*",
                                      "pancreatic_lesion.nii.gz"))
    gt_map = {Path(p).parent.name: p for p in gt_files}

    pred_files = glob.glob(os.path.join(a.pred_root,
                                        "BDMAP_*.nii.gz",
                                        "predictions_raw",
                                        "pancreatic_lesion.nii.gz"))
    pred_map = {Path(p).parents[1].name[:-7]: p for p in pred_files}

    ids = sorted(set(gt_map) & set(pred_map))
    if not ids:
        sys.exit("No matching IDs")

    if a.__dict__["continue"] and os.path.exists(a.out_csv):
        with FileLock(a.out_csv + ".lock"):
            done = {r.split(",")[0] for r in open(a.out_csv)}
        ids = [i for i in ids if i not in done]
    else:
        with FileLock(a.out_csv + ".lock"):
            if os.path.exists(a.out_csv):
                os.remove(a.out_csv)

    if a.num_parts > 1:
        ids = split_parts(ids, a.num_parts, a.part)

    print(f"cases={len(ids)}  part={a.part}/{a.num_parts-1}  workers={a.workers}")

    tasks = [(cid, pred_map[cid], gt_map[cid],
              a.tolerance, a.out_csv) for cid in ids]

    if a.workers == 1:
        for t in tqdm(tasks, desc="eval"):
            eval_case(t)
    else:
        with Pool(a.workers) as pool:
            for _ in tqdm(pool.imap_unordered(eval_case, tasks),
                          total=len(tasks), desc="eval"):
                pass

    print("finished")


if __name__ == "__main__":
    main()