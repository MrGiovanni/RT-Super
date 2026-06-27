#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from filelock import FileLock
from tqdm import tqdm

# ---------------- I/O utils ---------------- #
def write_csv_row(row, csv_columns, csv_file_path):
    lock_file = csv_file_path + ".lock"
    with FileLock(lock_file, timeout=60):
        write_header = not os.path.exists(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

def _normalize_id(file_name: str) -> str:
    # Matches your earlier normalization
    return file_name.replace('_0000.', '.').replace('.nii.gz', '')

# ---------------- imaging ---------------- #
def resample_image(image, original_spacing, target_spacing=(1, 1, 1), order=1):
    try:
        image = image.get_fdata()
    except AttributeError:
        pass
    resize_factor = np.array(original_spacing, dtype=float) / np.array(target_spacing, dtype=float)
    return ndimage.zoom(image, resize_factor, order=order), resize_factor

def get_spacing(ct_scan_path):
    img = nib.load(ct_scan_path)
    # Get first 3 zooms even if header has time dimension
    return img.header.get_zooms()[:3]

def detection(tumor_mask, spacing, th=0.5, erode=True):
    # Load & binarize
    arr = nib.load(tumor_mask)
    if spacing is None:
        spacing = arr.header.get_zooms()[:3]
    arr=arr.get_fdata() > th
    # Resample to 1 mm
    arr, _ = resample_image(arr.astype(np.uint8), original_spacing=spacing, target_spacing=(1, 1, 1), order=0)
    arr = arr.astype(bool)
    if erode:
        original = arr.copy()
        arr = ndimage.binary_erosion(arr, structure=np.ones((3, 3, 3), dtype=np.uint8), iterations=1)
        arr = ndimage.binary_dilation(arr, structure=np.ones((3, 3, 3), dtype=np.uint8), iterations=2)
        arr &= original
    return int(arr.sum())  # voxels == mm^3

# ---------------- discovery ---------------- #
SUFFIXES = [
    '_lesion.nii.gz',   # generic lesion
    '_pdac.nii.gz',
    '_pnet.nii.gz',
    '_cyst.nii.gz',
]

MASK_SUBDIR_CANDIDATES = ['segmentations', 'predictions', 'prediction', '']

def _find_mask_dir(outputs_folder: str, case_id: str) -> str | None:
    base = os.path.join(outputs_folder, case_id)
    for sub in MASK_SUBDIR_CANDIDATES:
        p = os.path.join(base, sub) if sub else base
        if os.path.isdir(p):
            return p
    return None

def discover_columns(outputs_folder: str, any_lesion: bool) -> list[str]:
    """
    Scan outputs to discover all organs/suffix columns so we can fix the CSV header
    before processing in parallel.
    """
    columns = {"BDMAP_ID"}
    cases = [f for f in os.listdir(outputs_folder) if os.path.isdir(os.path.join(outputs_folder, f))]
    for case in cases:
        mdir = _find_mask_dir(outputs_folder, case)
        if not mdir:
            continue
        try:
            for f in os.listdir(mdir):
                for s in SUFFIXES:
                    if f.endswith(s):
                        organ = f[:-len(s)]
                        if any_lesion:
                            # In any_lesion mode we still name columns by organ/suffix you discovered
                            pass
                        if s == '_lesion.nii.gz':
                            columns.add(f"{organ} tumor volume predicted")
                        elif s == '_pdac.nii.gz':
                            columns.add(f"{organ} pdac volume predicted")
                        elif s == '_pnet.nii.gz':
                            columns.add(f"{organ} pnet volume predicted")
                        elif s == '_cyst.nii.gz':
                            columns.add(f"{organ} cyst volume predicted")
        except Exception:
            continue
    return sorted(columns)

# ---------------- per-case processing ---------------- #
def _process_single_file(file, outputs_folder, ct_folder, th, any_lesion: bool):
    file_path = os.path.join(outputs_folder, file)
    if not (os.path.isdir(file_path) or file.endswith('.nii.gz')):
        return None

    # Case ID (dir name or file stem)
    case_id = file
    if case_id.endswith('.nii.gz'):
        case_id = case_id.replace('.nii.gz', '')
    row = {"BDMAP_ID": _normalize_id(case_id)}

    # CT spacing
    ct_path = os.path.join(ct_folder, case_id)
    if '.nii.gz' not in ct_path:
        ct_path = ct_path + '.nii.gz'
    #if not os.path.exists(ct_path):
        # Try subfile e.g. <case>/ct.nii.gz
    #    alt = os.path.join(ct_folder, case_id, 'ct.nii.gz')
    #    if os.path.exists(alt):
    #        ct_path = alt
    #if not os.path.exists(ct_path):
        # Can't compute volumes without spacing; leave row with ID only
    #    spacing = None #we will compute it from the prediction
    #    spacing_from_prediction = True
    #else:
    #    spacing = get_spacing(ct_path)
    #    spacing_from_prediction = False
   
    spacing=None # better to get the spacing from the prediction, since sometimes we predict from 1x1x1 npz files

    # Where are masks?
    mdir = _find_mask_dir(outputs_folder, case_id)
    if not mdir:
        return row  # nothing to add

    # Iterate masks
    files_here = []
    try:
        files_here = os.listdir(mdir)
    except Exception:
        return row

    for f in files_here:
        for s in SUFFIXES:
            if not f.endswith(s):
                continue
            organ = f[:-len(s)]
            tumor_mask_path = os.path.join(mdir, 'any_lesion.nii.gz') if any_lesion else os.path.join(mdir, f)
            if not os.path.exists(tumor_mask_path):
                continue
            try:
                volume_mm3 = detection(tumor_mask_path, spacing, th)
            except Exception:
                continue

            if s == '_lesion.nii.gz':
                col_name = f"{organ} tumor volume predicted"
            elif s == '_pdac.nii.gz':
                col_name = f"{organ} pdac volume predicted"
            elif s == '_pnet.nii.gz':
                col_name = f"{organ} pnet volume predicted"
            elif s == '_cyst.nii.gz':
                col_name = f"{organ} cyst volume predicted"
            else:
                continue

            row[col_name] = volume_mm3

            # We found a matching suffix in this filename; no need to check other suffixes for this f
            break

    return row

# ---------------- splitting ---------------- #
def split_into_parts(file_list, n_parts, part_index):
    if n_parts <= 0:
        raise ValueError("n_parts must be a positive integer.")
    if part_index < 0 or part_index >= n_parts:
        raise ValueError("part_index must be between 0 and n_parts-1.")
    total_files = len(file_list)
    base, extra = divmod(total_files, n_parts)
    start = part_index * base + min(part_index, extra)
    end = start + base + (1 if part_index < extra else 0)
    return file_list[start:] if part_index == n_parts - 1 else file_list[start:end]

# ---------------- main driver ---------------- #
def process_outputs(args, outputs_folder, ct_folder, th, workers=10, continue_processing=False, cases=None):
    csv_file_path = os.path.join(
        outputs_folder,
        "tumor_detection_results_global_lesion.csv" if args.any_lesion else "tumor_detection_results.csv"
    )

    # Candidates to process (dirs or .nii.gz files)
    all_files = [
        f for f in os.listdir(outputs_folder)
        if os.path.isdir(os.path.join(outputs_folder, f)) or f.endswith('.nii.gz')
    ]

    # Filter by provided case list (optional)
    if cases is not None:
        cases_df = pd.read_csv(cases)
        ids = set(cases_df['BDMAP_ID'].astype(str).str.replace('.nii.gz', '').str.replace('_0000', '', regex=False))
        all_files = [f for f in all_files if f.replace('.nii.gz','').replace('_0000','') in ids]

    # Resume logic
    if os.path.exists(csv_file_path) and continue_processing:
        existing = pd.read_csv(csv_file_path)
        done = set(existing['BDMAP_ID'].astype(str))
        to_process = [f for f in all_files if _normalize_id(f.replace('.nii.gz','')) not in done]
        print(f"Resuming: skipping {len(done)} already in CSV; processing {len(to_process)} new.")
        need_header = False
        open_mode = "a"
        # Keep previous columns (header) if resuming
        csv_columns = list(existing.columns)
    else:
        to_process = all_files
        need_header = True
        open_mode = "w"
        # Discover all possible columns first
        csv_columns = discover_columns(outputs_folder, any_lesion=args.any_lesion)
        if not csv_columns:
            csv_columns = ["BDMAP_ID"]

        # If starting fresh, remove any old file
        if os.path.exists(csv_file_path) and (args.parts == 1 or args.part == 0):
            os.remove(csv_file_path)

    # Split into parts if needed
    if args.parts > 1:
        to_process = split_into_parts(sorted(to_process), args.parts, args.part)

    if not to_process:
        print("Nothing to process.")
        return

    # Ensure header exists if starting fresh
    if need_header:
        with FileLock(csv_file_path + ".lock", timeout=60):
            with open(csv_file_path, open_mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writeheader()

    # Parallel processing with progress bar
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_single_file, f, outputs_folder, ct_folder, th, args.any_lesion): f
            for f in to_process
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
            file_name = futures[fut]
            try:
                row = fut.result()
                if row is None:
                    continue
                # Guarantee all header columns present; fill missing as empty
                out_row = {k: row.get(k, "") for k in csv_columns}
                write_csv_row(out_row, csv_columns, csv_file_path)
            except Exception as e:
                print(f"[error] {file_name}: {e}")

    print("CSV file saved at:", csv_file_path)

# ---------------- CLI ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect tumors from segmentation masks and write per-organ volumes.")
    parser.add_argument("--outputs_folder", type=str, required=True, help="Path to the outputs folder (segmentations)")
    parser.add_argument("--ct_folder", type=str, required=True, help="Path to the original CT scans folder (NIfTI)")
    parser.add_argument("--th", type=float, default=0.5, help="Threshold for binarizing tumor mask")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--cases", default=None, help="CSV with BDMAP_ID list to include")
    parser.add_argument("--continuing", action='store_true', help="Continue appending to existing CSV")
    parser.add_argument("--parts", type=int, default=1, help="Split dataset into N parts")
    parser.add_argument("--part", type=int, default=0, help="Which part (0-indexed)")
    parser.add_argument("--any_lesion", action='store_true', help="Use any_lesion.nii.gz in each case folder")
    args = parser.parse_args()

    process_outputs(
        args,
        outputs_folder=args.outputs_folder,
        ct_folder=args.ct_folder,
        th=args.th,
        workers=args.workers,
        continue_processing=args.continuing,
        cases=args.cases,
    )