import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation
import os
import yaml
import time
import sys
import pandas as pd
import json
import copy
import shutil
import re
import  importlib
from pathlib import Path
import hashlib
from typing import Any, Optional
#python dataset_abdomenatlas.py --dataset abdomenatlas --model medformer --dimension 3d --batch_size 2 --crop_on_tumor --save_destination /fastwork/psalvador/JHU/data/atlas_300_medformer_augmented_npy_augmented_multich_crop_on_tumor/ --crop_on_tumor --multi_ch_tumor --workers_overwrite 10

from filelock import FileLock, Timeout
# ---------------------------------------------------------



def string_to_tensor(x: Any, *, salt: str, allow_numeric: bool = True) -> torch.Tensor:
    """Deterministic string/obj -> int64. 0 means 'unknown/unpairable'."""
    if x is None:
        return torch.tensor(0, dtype=torch.int64)

    # numeric fast path (only if allowed)
    if allow_numeric:
        if isinstance(x, int) and x != 0:
            return torch.tensor(int(x), dtype=torch.int64)
        if isinstance(x, float):
            if math.isnan(x) or x == 0.0:
                return torch.tensor(0, dtype=torch.int64)
            return torch.tensor(int(x), dtype=torch.int64)

    # string handling
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("nan", "none", "unknown", "random", "unk"):
            return torch.tensor(0, dtype=torch.int64)
        if allow_numeric:
            try:
                v = float(s)
                if v == 0.0 or math.isnan(v):
                    return torch.tensor(0, dtype=torch.int64)
                return torch.tensor(int(v), dtype=torch.int64)
            except ValueError:
                pass
    else:
        s = str(x)

    # deterministic hash -> 53 bits
    u64 = int.from_bytes(hashlib.sha1((salt + "|" + s).encode("utf-8")).digest()[:8], "big", signed=False)
    u53 = u64 & ((1 << 53) - 1)
    if u53 == 0:
        u53 = 1
    return torch.tensor(u53, dtype=torch.int64)

def _load_numpy_array(path: str) -> np.ndarray:
    """
    Loads an ndarray from either .npy or .npz.
    For .npz, returns:
      - data["arr"] if present (recommended if you saved with key 'arr')
      - else data["arr_0"] if present (default np.savez_compressed)
      - else the first array in the archive
    """
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        # prefer a stable key if you used one
        if "arr" in obj.files:
            arr = obj["arr"]
        elif "arr_0" in obj.files:
            arr = obj["arr_0"]
        else:
            arr = obj[obj.files[0]]
        obj.close()
        return arr
    return obj

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    keys = list_of_dicts[0].keys()
    for d in list_of_dicts:
        if d.keys() != keys:
            raise ValueError("All dicts must have the same keys")
    return {k: [d[k] for d in list_of_dicts] for k in keys}

def tumor_to_organ2(tumor_name):
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
    

def dice(x,y):
    
    assert x.shape == y.shape, f"Expected x and y to have the same shape, but got x.shape={x.shape} and y.shape={y.shape}"
    eps = 1e-6
    intersection = (x*y).sum()
    union = x.sum() + y.sum()
    return (2*intersection + eps) / (union + eps)

def get_sample_weight(labels,proportions,class_names,balancer=None,loading_augmented=False):
    weights = []
    tumors = []
    eps = 1e-4
    if balancer is not None:
        tumor_prop = 1-proportions['healthy'] 
        if loading_augmented:
            #read yaml with tumor proportions
            #with open(os.path.join(self.save_destination, 'tumor_proportions.yaml'), 'w') as f:
            with open(os.path.join(balancer.save_destination, 'tumor_proportions.yaml'), 'r') as f:
                proportions = yaml.load(f, Loader=yaml.SafeLoader)
                if proportions is None:
                    raise ValueError('Tumor proportions could not be loaded from yaml file!')
        else:
            #get proportions from balancer
            proportions = balancer.tumor_proportions
        for k,v in proportions.items():
            proportions[k] = v * tumor_prop 
        
    for i, c in enumerate(class_names):
        if c in proportions.keys():
            if labels[i].sum() > 0: #positive sample for class
                weights.append(1.0 / (eps + proportions[c]))
                tumors.append(c) # keep track of which tumors are present in the labels
            else: #negative sample for class
                weights.append(1.0 / (eps + (1-proportions[c])))
        else:
            # If the class is not in proportions, assign a default weight
            weights.append(1.0)
    # Normalize weights to sum to 1*number of classes
    weights = torch.tensor(weights)
    weights = weights / weights.sum() # Normalize to sum to 1
    weights = weights*len(class_names) # Scale by number of classes to keep the relative weights
    
    #print
    #print('Sample tumors:',tumors,' ; sample weights:', weights, flush=True)
    return weights

def get_class_proportions(meta,sample_list,lesion_class_names):
    """
    Get class weights based on the sample list and the meta information.
    This function will return a weight for each class in the sample list.
    :param meta: pandas dataframe with the meta information
    :param sample_list: list of sample names to consider
    :param lesion_classes: list of lesion classes to consider
    :return: list of weights for each class in the sample list
    """
    # Get the meta information for the samples in the sample list
    if isinstance(meta, str):
        meta = pd.read_csv(meta)
    
    # For each sample in sample_list, we add one row to the meta dataframe, accept duplicates!
    tmp = []
    for sample in sample_list:
        # If the sample is not in the meta, add it with zeros for all classes
        tmp.append(meta[meta['BDMAP ID'] == sample]) # get the row for the sample
    meta = pd.concat(tmp, ignore_index=True) # concatenate all the rows for the sample list
    if len(meta) == 0:
        raise ValueError("No samples found in the meta file for the provided sample list.")
    
    #print('Lesion class names:', lesion_class_names, flush=True)
    #organs
    organs_lesion_classes = {n.replace('_lesion','').replace('_','').replace('adrenal','adrenal gland'): n for n in lesion_class_names} # remove lesion suffix to get organs

    # Get the counts for each class
    cols = [f'number of {organ} lesion instances' for organ in list(organs_lesion_classes.keys())] # get the columns for each organ lesion instance
    
    #print cols missing from meta
    for col in cols:
        if col not in meta.columns:
            lesion_cols_meta = sorted([col for col in meta.columns if 'number of' in col.lower() and 'lesion instances' in col.lower()])
            raise ValueError(f"Column '{col}' not found in the meta dataframe. Lesion columns found in meta: {lesion_cols_meta}. Please check the meta file or the lesion class names provided.")
    
    # Get the counts for each class in the sample list
    meta = meta[cols]
    
    #make int
    meta = meta.fillna(0).astype(int) # fill NaN with 0 and convert to int
    
    #binarize
    meta = (meta >= 1).astype(int) # convert to binary, 1 if there is at least one instance, 0 otherwise
    
    #sum
    counts = meta.sum(axis=0)  # Sum across all samples for each class
    
    #create a dict of class proportions
    total = len(meta)  # Total number of samples in the sample list
    assert total == len(sample_list), f"Total number of samples in the meta ({total}) does not match the sample list ({len(sample_list)})"
    proportions = {}
    for i, organ in enumerate(list(organs_lesion_classes.keys())):
        proportions[organs_lesion_classes[organ]] = counts[f'number of {organ} lesion instances'] / total if total > 0 else 0  # Avoid division by zero
    #now calculate how many samples have no lesion
    meta['no_lesion'] = (meta[cols].sum(axis=1) == 0).astype(int)
    no_lesion_count = meta['no_lesion'].sum() # how many samples have no lesions
    # Add the no lesion count to the proportions
    proportions['healthy'] = no_lesion_count / total if total > 0 else 0  # Proportion of healthy samples
    
    #print the proportions for debugging
    #print('Class proportions:', proportions, flush=True)
    #print('Number of samples:', total, flush=True)
    
    return proportions
def clone_tensors_inplace(retur: dict):
    """
    Recursively detach→cpu→contiguous() tensors in retur.
    Also raise if any tumor-shaped output has more than 10 entries.
    """
    import sys
    import torch

    sample_name = str(retur.get("name", "UNKNOWN_SAMPLE"))

    def _raise(key, t):
        msg = (f"[clone_tensors_inplace] Sample: {sample_name} → "
               f"'{key}' has >10 tumors; shape={tuple(t.shape)}")
        print(msg, flush=True, file=sys.stderr)
        raise RuntimeError(msg)

    # clone like your original
    def _clone(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().contiguous()
        if isinstance(x, dict):
            for k, v in x.items():
                x[k] = _clone(v)
            return x
        if isinstance(x, list):
            return [_clone(e) for e in x]
        if isinstance(x, tuple):
            return tuple(_clone(e) for e in x)
        return x

    _clone(retur)  # mutate nested dicts/lists; tensors replaced

    # helper: check len along axis 0 only
    def _check_len0(key, t):
        if isinstance(t, torch.Tensor):
            try:
                if t.shape[0] > 10:
                    _raise(key, t)
            except Exception:
                # if axis-0 doesn't exist, just ignore as requested
                pass

    # fixed tumor-shaped outputs (axis 0 is the tumor dimension)
    for key in ("volumes", "attenuation", "diameters", "sizes_slices"):
        t = retur.get(key, None)
        if t is not None:
            _check_len0(key, t)

    # per-voxel dicts (handle your typo + the correct key)
    for per_key in ("voumes_per_voxel", "volumes_per_voxel", "diameters_per_voxel"):
        dv = retur.get(per_key, None)
        if isinstance(dv, dict):
            for k, v in dv.items():
                _check_len0(f"{per_key}.{k}", v)

    return retur

def balance_classes(class1, class2):
    """
    Balances two lists of strings by repeating the smaller one until its length
    matches the larger one and then shuffling both lists.
    
    Parameters:
        class1 (list of str): The first class.
        class2 (list of str): The second class.
        
    Returns:
        tuple: A tuple (balanced_class1, balanced_class2) with both lists balanced.
    """
    # Determine which list is smaller
    if len(class1) < len(class2):
        # Compute how many times to repeat class1 to match class2's size
        times = len(class2) // len(class1)
        remainder = len(class2) % len(class1)
        balanced_class1 = class1 * times + class1[:remainder]
        balanced_class2 = class2[:]  # Make a copy of class2
    elif len(class2) < len(class1):
        times = len(class1) // len(class2)
        remainder = len(class1) % len(class2)
        balanced_class2 = class2 * times + class2[:remainder]
        balanced_class1 = class1[:]  # Make a copy of class1
    else:
        # If they are already equal in size, just copy them
        balanced_class1, balanced_class2 = class1[:], class2[:]
    
    # Shuffle both lists in place
    random.shuffle(balanced_class1)
    random.shuffle(balanced_class2)
    
    return balanced_class1, balanced_class2
    

def normalize_no_lesion(col: pd.Series) -> pd.Series:
    """Return a boolean Series where True = healthy (no lesion), False = not healthy."""
    if pd.api.types.is_bool_dtype(col):
        return col

    # Numeric path first (handles ints/floats AND numeric-looking strings)
    num = pd.to_numeric(col, errors='coerce')
    has_num = num.notna()
    out = pd.Series(False, index=col.index)  # default False (not healthy)
    out[has_num] = num[has_num].eq(1)

    # Textual path for the rest
    txt = col[~has_num].astype(str).str.strip().str.lower()
    true_tokens  = {'1', '1.0', 'true', 't', 'yes', 'y'}
    false_tokens = {'0', '0.0', 'false', 'f', 'no', 'n', '', 'nan', 'none', 'null'}

    out.loc[~has_num & txt.isin(true_tokens)]  = True
    out.loc[~has_num & txt.isin(false_tokens)] = False
    # anything else remains False

    return out

def coerce_to_bool(x):
    # Treat missing as False
    if x is pd.NA or x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    if isinstance(x, str):
        s = x.strip().lower()
        return s in {"1","true","t","yes","y","on"}
    # numbers/bools
    return bool(x)

def max_diameter_like_estimator(size_cell):
    """
    Parse 'Tumor Size (mm)' using the same strategy as estimate_tumor_volume:
      - If 1 number  -> use [d, d, d]
      - If 2 numbers -> use [d1, d2, (d1+d2)/2]
      - If 3+        -> take the top 3 numbers
    Return the maximum of those three. np.nan if nothing numeric is found.
    """
    # Extract all numeric tokens (handles '10', '10.5', '10 x 12', '10x12x8', etc.)
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', str(size_cell))]
    if not nums:
        return np.nan

    if len(nums) == 1:
        d1 = d2 = d3 = nums[0]
    elif len(nums) == 2:
        d1, d2 = nums[:2]
        d3 = (d1 + d2) / 2.0
    else:
        d1, d2, d3 = sorted(nums, reverse=True)[:3]

    return float(max(d1, d2, d3))


def generate_sanity_one_per_tumor_malignancy(
    img_list,
    tumors_per_type,
    slice_only=False,
    reports=None,
    unknown_sizes_only=False,
    debug_malignancy=False,
    atlas_meta=None,
    get_malignancy_fn=None,
    atlas_malignancy=None,   # <— NEW: malignancy table for per-voxel (atlas) IDs
):
    import random
    import numpy as np
    import pandas as pd
    import re

    # ---------- helpers ----------
    def _id_col(df):
        if df is None:
            return None
        if 'BDMAP_ID' in df.columns: return 'BDMAP_ID'
        if 'BDMAP ID' in df.columns: return 'BDMAP ID'
        raise ValueError("Expected 'BDMAP_ID' or 'BDMAP ID' in provided dataframe.")

    def _canon_org(s: str):
        return str(s).strip().lower().replace('_', ' ')

    def _canon_series(s: pd.Series):
        return s.astype(str).str.strip().str.lower().str.replace('_', ' ', regex=False)

    def _extract_id_from_path(p):
        m = re.search(r'BDMAP_\d{8}', str(p))
        return m.group(0) if m else None

    def _default_get_malignancy(row):
        def _norm(x):
            if x is None: return ""
            if isinstance(x, float) and np.isnan(x): return ""
            return str(x).strip().lower()
        cand_m = [c for c in row.index if 'malignan' in c.lower()]
        cand_b = [c for c in row.index if 'benign'   in c.lower()]
        m = _norm(row[cand_m[0]]) if cand_m else ""
        b = _norm(row[cand_b[0]]) if cand_b else ""
        syn = {"y":"yes","1":"yes","true":"yes","malignant":"yes",
               "n":"no","0":"no","false":"no","benign":"no"}
        m = syn.get(m, m); b = syn.get(b, b)
        if m == "yes" and b != "yes": return 1.0
        if b == "yes" and m != "yes": return 0.0
        return float("nan")

    get_malig = get_malignancy_fn or _default_get_malignancy

    # ---------- optional prefilter for slice_only / unknown_sizes_only ----------
    allowed_ids = None
    if slice_only:
        if reports is None:
            raise ValueError("slice_only=True requires `reports`.")
        rep = reports.copy()
        idcol = _id_col(rep)

        if 'series matches report' in rep.columns:
            sr = rep['series matches report']
            if sr.dtype == bool:
                mask_sr = sr.fillna(False)
            else:
                mask_sr = sr.astype(str).str.strip().str.lower().isin({'1','true','yes','y'})
            rep = rep[mask_sr]

        rep = rep[pd.to_numeric(rep['Image'], errors="coerce").notna()]
        allowed_ids = set(rep[idcol].dropna().astype(str).tolist())

    if unknown_sizes_only:
        if reports is None:
            raise ValueError("unknown_sizes_only=True requires `reports`.")
        idcol = _id_col(reports)
        df = reports.dropna(subset=[idcol, 'Standardized Organ']).copy()

        if 'no lesion' in df.columns:
            try:
                col = df['no lesion']
                if col.dtype == bool:
                    healthy_mask = col.fillna(False)
                else:
                    healthy_mask = col.astype(str).str.strip().str.lower().isin({'1','true','yes','y'})
                df = df[~healthy_mask]
            except Exception:
                pass

        sz = df['Tumor Size (mm)'].astype(str)
        df['is_unknown'] = sz.eq('u') | (~sz.str.contains(r'\d', na=False))

        agg = (df
               .groupby([idcol, 'Standardized Organ'])['is_unknown']
               .agg(all_unknown='all', any_row='size')
               .reset_index())
        only_unknown = agg[(agg['all_unknown']) & (agg['any_row'] > 0)][[idcol, 'Standardized Organ']]

        tumors_per_type = {
            organ: (only_unknown.loc[_canon_series(only_unknown['Standardized Organ']) == _canon_org(organ), idcol]
                    .astype(str).unique().tolist())
            for organ in tumors_per_type.keys()
        }

    # ---------- map ID -> first path (works for both UFO & Atlas) ----------
    id_to_first_path = {}
    for path in img_list:
        bid = _extract_id_from_path(path)
        if bid and bid not in id_to_first_path:
            id_to_first_path[bid] = path

    # Atlas ID set (if atlas_meta is given)
    atlas_id_set = set()
    if atlas_meta is not None:
        am = atlas_meta.copy()
        idcol_am = _id_col(am)
        atlas_id_set = set(am[idcol_am].astype(str).tolist())

    # ---------- candidate pools ----------
    # pools by organ
    mal_pool = {k: set() for k in tumors_per_type.keys()}   # IDs with malignant
    ben_pool = {k: set() for k in tumors_per_type.keys()}   # IDs with benign
    single_pool = {k: set() for k in tumors_per_type.keys()}  # IDs with exactly-one tumor (UFO) or exactly-one lesion (Atlas)

    # UFO: exactly-one tumor row per (ID, Organ) + malignancy from `reports`
    if reports is not None:
        rep = reports.copy()
        idcol = _id_col(rep)
        rep[idcol] = rep[idcol].astype(str)

        if allowed_ids is not None:
            rep = rep[rep[idcol].isin(allowed_ids)]

        grp = rep.groupby([idcol, 'Standardized Organ'])
        # one-row groups
        sdf = grp.filter(lambda g: len(g) == 1).copy()
        sdf['_g_malignancy_'] = [get_malig(row) for _, row in sdf.iterrows()]

        for organ in tumors_per_type.keys():
            org_mask = _canon_series(sdf['Standardized Organ']) == _canon_org(organ)
            sub = sdf[org_mask]
            # UFO "single tumor"
            single_pool[organ].update(sub[idcol].tolist())
            # malignancy
            mal_ids = sub.loc[sub['_g_malignancy_'] == 1.0, idcol].tolist()
            ben_ids = sub.loc[sub['_g_malignancy_'] == 0.0, idcol].tolist()
            mal_pool[organ].update(mal_ids)
            ben_pool[organ].update(ben_ids)

    # Atlas: exactly-one lesion instance (from atlas_meta) AND (optionally) malignancy from `atlas_malignancy`
    atlas_single = {k: set() for k in tumors_per_type.keys()}
    if atlas_meta is not None:
        am = atlas_meta.copy()
        idcol_am = _id_col(am)
        am[idcol_am] = am[idcol_am].astype(str)
        # columns like "Number of pancreatic lesions" or "... instances"
        cols = [c for c in am.columns if ('number of' in c.lower()) or ('instances' in c.lower())]
        for organ in tumors_per_type.keys():
            ocanon = _canon_org(organ)
            cand_cols = [c for c in cols if all(tok in c.lower() for tok in ocanon.split())]
            if not cand_cols:
                cand_cols = [c for c in cols if (ocanon in c.lower()) or ('lesion' in c.lower() and any(t in c.lower() for t in ocanon.split()))]
            if not cand_cols:
                continue
            cc = cand_cols[0]
            try:
                one_ids = am.loc[am[cc] == 1, idcol_am].astype(str).tolist()
                atlas_single[organ].update(one_ids)
                # include in the global "single_pool"
                single_pool[organ].update(one_ids)
            except Exception:
                pass

    # malignancy for Atlas (from a dedicated table if provided; else try reports restricted to atlas IDs)
    if atlas_malignancy is not None:
        amg = atlas_malignancy.copy()
        idcol_mg = _id_col(amg)
        amg[idcol_mg] = amg[idcol_mg].astype(str)
        # groups with exactly-one row per (ID, Organ)
        grp = amg.groupby([idcol_mg, 'Standardized Organ'])
        mg1 = grp.filter(lambda g: len(g) == 1).copy()
        mg1['_g_malignancy_'] = [get_malig(row) for _, row in mg1.iterrows()]
        for organ in tumors_per_type.keys():
            org_mask = _canon_series(mg1['Standardized Organ']) == _canon_org(organ)
            sub = mg1[org_mask]
            # keep only atlas IDs
            sub = sub[sub[idcol_mg].isin(atlas_id_set)]
            mal_pool[organ].update(sub.loc[sub['_g_malignancy_'] == 1.0, idcol_mg].tolist())
            ben_pool[organ].update(sub.loc[sub['_g_malignancy_'] == 0.0, idcol_mg].tolist())
    elif reports is not None and atlas_id_set:
        # fallback: use reports, but restrict to atlas IDs
        rep = reports.copy()
        idcol = _id_col(rep)
        rep[idcol] = rep[idcol].astype(str)
        rep = rep[rep[idcol].isin(atlas_id_set)]
        grp = rep.groupby([idcol, 'Standardized Organ'])
        sdf = grp.filter(lambda g: len(g) == 1).copy()
        sdf['_g_malignancy_'] = [get_malig(row) for _, row in sdf.iterrows()]
        for organ in tumors_per_type.keys():
            org_mask = _canon_series(sdf['Standardized Organ']) == _canon_org(organ)
            sub = sdf[org_mask]
            mal_pool[organ].update(sub.loc[sub['_g_malignancy_'] == 1.0, idcol].tolist())
            ben_pool[organ].update(sub.loc[sub['_g_malignancy_'] == 0.0, idcol].tolist())

    # ---------- selection ----------
    sanity_mapping = {}
    rng = random.Random()

    for organ, id_list in tumors_per_type.items():
        shuffled_ids = [str(x) for x in id_list]
        rng.shuffle(shuffled_ids)

        if allowed_ids is not None:
            shuffled_ids = [bid for bid in shuffled_ids if bid in allowed_ids]

        chosen_id = None

        if debug_malignancy and (mal_pool[organ] or ben_pool[organ]):
            want_malig = rng.random() < 0.5
            pri = mal_pool[organ] if want_malig else ben_pool[organ]
            sec = ben_pool[organ] if want_malig else mal_pool[organ]

            # prefer IDs that also are "single lesion/tumor"
            singles = single_pool.get(organ, set())
            pri_ids = [bid for bid in shuffled_ids if (bid in pri) and (bid in singles)]
            sec_ids = [bid for bid in shuffled_ids if (bid in sec) and (bid in singles)]

            if pri_ids:
                chosen_id = rng.choice(pri_ids)
            elif sec_ids:
                chosen_id = rng.choice(sec_ids)

        # if not set yet, still try to enforce single-lesion
        if chosen_id is None and single_pool.get(organ):
            cands = [bid for bid in shuffled_ids if bid in single_pool[organ]]
            if cands:
                chosen_id = rng.choice(cands)

        # final fallback: first available id that we have a path for
        if chosen_id is None:
            for bid in shuffled_ids:
                if bid in id_to_first_path:
                    chosen_id = bid
                    break

        if chosen_id is not None:
            p = id_to_first_path.get(chosen_id)
            if p is not None:
                sanity_mapping[organ] = p

    return sanity_mapping

def generate_sanity_one_per_tumor(img_list, tumors_per_type, slice_only=False, reports=None, unknown_sizes_only=False):
    """
    Generates a mapping from tumor type to a single image path (from img_list).
    
    For every tumor type (key in tumors_per_type), this function shuffles the associated
    list of BDMAP_IDs for randomness. Then it searches through img_list to find the first
    image whose extracted base ID exactly matches one of these shuffled IDs. It returns a
    dictionary mapping tumor type -> image path.
    
    Parameters:
        img_list (list): List of full image file paths (e.g., self.img_list).
        tumors_per_type (dict): Dictionary with tumor types as keys and lists of BDMAP_ID strings as values.
        
    Returns:
        dict: Mapping of tumor type (key) to the corresponding image path (value) where a match is found.
    """
    sanity_mapping = {}
    if slice_only:
        reports_slice = reports[reports['series matches report']==True]
        dirty = reports_slice[pd.to_numeric(reports_slice['Image'], errors="coerce").isna()]
        reports_slice = reports_slice[~reports_slice['BDMAP_ID'].isin(dirty['BDMAP_ID'].tolist())]
        allowed_ids = set(reports_slice['BDMAP_ID'].dropna().astype(str).tolist())
        
    if unknown_sizes_only:
        if reports is None:
            raise ValueError("unknown_sizes_only=True requires `reports` to be provided.")

        df = reports.dropna(subset=['BDMAP_ID', 'Standardized Organ']).copy()

        # Drop clearly-healthy rows if that column exists
        if 'no lesion' in df.columns:
            try:
                healthy_mask = normalize_no_lesion(df['no lesion'])
                df = df[~healthy_mask]
            except Exception:
                pass

        # An entry is "unknown size" if it's 'u' OR contains no digits
        sz = df['Tumor Size (mm)'].astype(str)
        df['is_unknown'] = sz.eq('u') | (~sz.str.contains(r'\d', na=False))

        # Keep (BDMAP_ID, Organ) groups where ALL sizes are unknown (and there is at least one row)
        agg = (df
            .groupby(['BDMAP_ID', 'Standardized Organ'])['is_unknown']
            .agg(all_unknown='all', any_row='size')
            .reset_index())
        only_unknown = agg[(agg['all_unknown']) & (agg['any_row'] > 0)][['BDMAP_ID', 'Standardized Organ']]

        # Build tumors_per_type with ONLY-unknown groups per organ
        tumors_per_type = {
            organ: (only_unknown.loc[only_unknown['Standardized Organ'] == organ, 'BDMAP_ID']
                    .astype(str).unique().tolist())
            for organ in tumors_per_type.keys()
        }
        
    id_to_first_path = {}
    for path in img_list:
        bid = extract_id(path)  # must return the BDMAP_ID comparable to your lists
        if bid is None:
            continue
        if bid not in id_to_first_path:
            id_to_first_path[bid] = path
    
    # Iterate over each tumor type
    for tumor, id_list in tumors_per_type.items():
        # Shuffle IDs to introduce randomness
        shuffled_ids = id_list.copy()
        shuffled_ids = [str(x) for x in shuffled_ids]
        random.shuffle(shuffled_ids)
        
        if slice_only:
            # Filter img_list to only include slice images
            shuffled_ids_filtered = [bid for bid in shuffled_ids if bid in allowed_ids]
            if len(shuffled_ids_filtered)>0:
                shuffled_ids = shuffled_ids_filtered
        
        # choose the first BID that has a mapped image path
        for bid in shuffled_ids:
            path = id_to_first_path.get(bid)
            if path is not None:
                sanity_mapping[tumor] = path
                break  # one per tumor type
    
    return sanity_mapping


def get_lesion_channels(out, classes, assertion = False, return_class_names = False):
    assert len(out.shape)==5, f"Expected out to have shape (batch_size, channels, D, H, W), but got {out.shape}"
    
    #merge lesion channels if they are in the same organ. Outputs will have only lesion channels, removes organ channels.
    assert out.shape[1] == len(classes), f"Expected out.shape[1] to be equal to len(classes), but got out.shape={out.shape}, len(classes)={len(classes)}"
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
        kys = [x.replace('pancreas_lesion','pancreatic_lesion') for x in kys]
        return lesion_out, kys
    else:
        return lesion_out
    

def sanity_assert_no_lesion_mask(label, classes, sample_name):
    #ensure all per-voxel labels for lesions are zero
    label = label.unsqueeze(0)
    lesion_channels,class_names = get_lesion_channels(label, classes, return_class_names=True)
    
    if lesion_channels.sum() != 0:
        for i,c in enumerate(class_names,0):
            if lesion_channels[:,i].sum() > 0:
                raise ValueError(f'When using the argument no_mask, we expect lesion labels to be zero, but their sum was: {lesion_channels.sum()}, class with lesion: {c}, sum: {lesion_channels[:,i].sum()}, sample name: {sample_name}')

def get_group_upsample_coeficients(group_sizes, target_proportions):
    """
    This will be used to get the factors at which we will upsample each report quality group (and per-voxel samples)
    when using balance_supervision_report_quality. Target proportions are the percentage of the final augmented dataset that
    each group will take. We look for the minimal augmentation that enforces this proportion.
    """
    assert len(group_sizes) == len(target_proportions), "group_sizes and target_proportions must have the same length"
    assert group_sizes.min() > 0, "All group sizes must be greater than zero"
    
    smallest_factor_idx = torch.argmin(target_proportions/group_sizes)
    #smallest upsampling coef should be 1. Thus:
    target_dataset_size = group_sizes[smallest_factor_idx]/target_proportions[smallest_factor_idx]
    augmentation_coef = target_dataset_size*target_proportions/group_sizes
    
    #upsampled group sizes
    upsampled_group_sizes = (augmentation_coef*group_sizes).round().long()
    
    return augmentation_coef, upsampled_group_sizes

def supersample_group(group,target_size):
    assert len(group) > 0, "Group must have at least one element to be supersampled"
    if len(group) == target_size:
        return group
    assert len(group) < target_size, "Group must be smaller than target_size to be supersampled"
    n = len(group)
    q, r = divmod(target_size, n)
    out = group * q                    # fast replicate
    if r:
        out += random.sample(group, r) # fill remainder
    assert len(out) == target_size
    return out

def _count_atlas_ufo(img_list, atlas_root, ufo_root):
    """
    Count how many deduped IDs belong to Atlas vs UFO roots.
    """
    img_list = list(set(img_list))  # dedupe
    atlas_count, ufo_count = 0, 0

    for p in img_list:
        if str(p).startswith(str(atlas_root)):
            atlas_count += 1
        elif str(p).startswith(str(ufo_root)):
            ufo_count += 1

    #raise ValueError(f"Number of Atlas files in dataset: {atlas_count}")
    print(f"Number of Atlas files in dataset: {atlas_count}", flush=True, file=sys.stderr)
    print(f"Number of UFO files in dataset: {ufo_count}", flush=True, file=sys.stderr)
    print(f"Total unique files: {len(img_list)}", flush=True, file=sys.stderr)
    
    
def to_jsonable(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x

def define_malignancy(row, malignancy_column='pathology_and_radiology_malignant',
                      benign_column='radiology_benign_ICD_pathology_ok',
                      relaxed_malignancy_col=None):
    if row['is_healthy'] == True:
        return 'healthy'
    malig = str(row[malignancy_column]).strip().lower()
    benign = str(row[benign_column]).strip().lower()
    if malig == 'yes':
        return 'malignant'
    elif benign == 'yes':
        return 'benign'
    elif malig == 'no':
        return 'benign'
    else:
        if relaxed_malignancy_col is not None:#not pathology confirmed
            malig_relaxed = str(row[relaxed_malignancy_col]).strip().lower()
            if malig_relaxed == 'yes':
                return 'malignant'
            elif malig_relaxed == 'no':
                return 'benign'
        return 'u'
    
    


def clean_ufo(reports,annotated_tumors,limit_healthy=True,
              slice_only=False,patch_size=128,
              use_all_data=False,
              benign_maligant_only=False,
              malignancy_column='pathology_and_radiology_malignant',
              benign_column='radiology_benign_ICD_pathology_ok',
              relaxed_malignancy_col=None,#change if you want a relaxed definition of malignancy to be acceptable (e.g., radiology report based)
              ):
    """
    This function gets a list of reports and removes cases of no interest:
    - We get the healthy patients
    - We get, for each tumor we have annotated organs, all reports that have known tumor size and number, unless you set use_all_data
    - We remove, for organs that have rignr and left (adrenal glands, kidneys), the reports that have unknown sub-segment (not right or left)
    Then, we print the number of useful cases per tumor
    """
    if benign_maligant_only:
        if malignancy_column is None or benign_column is None:
            raise ValueError("benign_maligant_only=True requires malignancy_column and benign_column to be set")
    
    # Preserve "multiple" flag BEFORE the size standardization below collapses
    # 'multiple' to 'u'. Idempotent across repeated clean_ufo calls on the
    # same dataframe: only set if the column doesn't exist yet.
    if '_was_multiple' not in reports.columns:
        reports['_was_multiple'] = reports['Tumor Size (mm)'].astype(str).str.lower() == 'multiple'

    #standardize: make tumor size=='u' if it does not contain a number or 'Unknow Tumor Size' is different from 'no'
    reports['unknown size'] =    reports.apply(lambda row: 'yes' if (not bool(re.search(r'\d', str(row['Tumor Size (mm)']))) or row['Unknow Tumor Size'] != 'no') else 'no', axis=1)
    reports['Tumor Size (mm)'] = reports.apply(lambda row: 'u'   if (not bool(re.search(r'\d', str(row['Tumor Size (mm)'])))) else row['Tumor Size (mm)'], axis=1)



    #drop LLM hallucinations
    hallucination=reports[(reports['Tumor Size (mm)'].astype(str).str.contains(r'^0\.0\s*x', regex=True, na=False)) | (reports['Tumor Size (mm)']=='0.0') | (reports['Tumor Size (mm)']=='0')]
    reports = reports[~reports['BDMAP_ID'].isin(hallucination['BDMAP_ID'].tolist())]

    #drop tumors that do not fit the crop size
    max_diam = reports['Tumor Size (mm)'].apply(max_diameter_like_estimator)
    too_big_mask = max_diam > patch_size
    if too_big_mask.any():
        big_ids = reports.loc[too_big_mask, 'BDMAP_ID'].dropna().astype(str).unique().tolist()
        print(f'Dropping {len(big_ids)} BDMAP_IDs with tumor diameter > {patch_size} mm', flush=True, file=sys.stderr)
        reports = reports[~reports['BDMAP_ID'].astype(str).isin(big_ids)]
    else:
        print(f'No BDMAP_ID with tumor diameter > {patch_size} mm found', flush=True, file=sys.stderr)
    # -------------------------------------------------------------------------
    
    if slice_only:
        # keep healthy regardless of slices
        healthy_mask = normalize_no_lesion(reports['no lesion'])
        healthy_df = reports[healthy_mask].copy()

        # consider only rows where the series matches the report
        m = (reports['series matches report'] == True)
        # numeric slice indices among those rows
        img_num = pd.to_numeric(reports.loc[m, 'Image'], errors='coerce').notna()

        # patients with at least one known (numeric) slice
        good_ids = (
            reports.loc[m & img_num, 'BDMAP_ID']
            .dropna().astype(str).unique().tolist()
        )

        # keep ALL rows for those patients + all healthy rows
        reports_slice = reports[reports['BDMAP_ID'].astype(str).isin(good_ids)].copy()
        reports = pd.concat([reports_slice, healthy_df], ignore_index=True)

        print('We are using only reports with slices, number of reports:',
            len(reports), flush=True, file=sys.stderr)
    
    if (malignancy_column is not None) and (benign_column is not None):
        healthy_mask = normalize_no_lesion(reports['no lesion'])
        reports['is_healthy'] = healthy_mask
        reports['malignant_benign'] = reports.apply(lambda row: define_malignancy(row, malignancy_column, benign_column,relaxed_malignancy_col), axis=1)
    if benign_maligant_only:
        # keep healthy regardless of benign/malignant
        benign_vals = reports[benign_column].astype(str).str.strip().str.lower()
        if (benign_vals == 'no').any():
            bad_ids = reports.loc[benign_vals == 'no', 'BDMAP_ID'].astype(str).unique()[:20]
            raise ValueError(f"Found benign=='no' in {benign_column}. Example IDs: {bad_ids}")
        
        #get cases that are yes or no in the column malignancy_column or benign_column
        malign_mask = reports[malignancy_column].astype(str).str.lower().isin(['yes','no'])
        benign_mask = reports[benign_column].astype(str).str.lower().isin(['yes'])
        
        
        #filter reports
        good_ids = reports[malign_mask | benign_mask | healthy_mask]
        good_ids = good_ids[good_ids['Standardized Organ'].isin(annotated_tumors)]
        good_ids = good_ids['BDMAP_ID'].dropna().astype(str).unique().tolist()
        reports = reports[reports['BDMAP_ID'].astype(str).isin(good_ids)].copy()
        

        
        
    
    interest = {}
    
    maxi=0
    for organ in annotated_tumors:
        interest[organ] = reports[reports['Standardized Organ'] == organ]
        interest[organ] = interest[organ][interest[organ]['Tumor Size (mm)'].astype(str).str.contains(r'\d', na=False)]#only cases with number
        interest[organ] = interest[organ][interest[organ]['Unknow Tumor Size'] == 'no']
        if organ in ['kidney','adrenal_gland','lung','breast','femur']:
            loc_series = interest[organ]['Standardized Location'].astype(str).str.lower()
            interest[organ] = interest[organ][
                loc_series.str.contains('right', na=False) | loc_series.str.contains('left', na=False)
            ]
        print('Number of useful cases for %s: %s'%(organ, interest[organ]['BDMAP_ID'].nunique()))
        if interest[organ]['BDMAP_ID'].nunique()>maxi:
            maxi=interest[organ]['BDMAP_ID'].nunique()

    # Healthy
    healthy_mask = normalize_no_lesion(reports['no lesion'])
    healthy_df = reports[healthy_mask].copy()

    if limit_healthy and maxi > 0:
        ids_healthy = healthy_df['BDMAP_ID'].dropna().unique().tolist()
        k = min(maxi, len(ids_healthy))
        ids_healthy = random.sample(ids_healthy, k) if k > 0 else []
        healthy_df = healthy_df[healthy_df['BDMAP_ID'].isin(ids_healthy)]

    print('Number of healthy cases:', healthy_df['BDMAP_ID'].nunique())
    interest['healthy'] = healthy_df

    print('Number of healthy cases:', interest['healthy']['BDMAP_ID'].nunique())
    #concat
    tumors_per_type = {}
    for k,v in interest.items():
        tumors_per_type[k]=v['BDMAP_ID'].unique().tolist()
    interest = pd.concat(interest.values())
    interest = interest.drop_duplicates()
    print('Total number of useful cases:', interest['BDMAP_ID'].nunique())
    ids_of_interest = interest['BDMAP_ID'].unique().tolist()
    
    #ALWAYS, we need ALL TUMORS for the patients with ids in interest, not only those with size!
    if not use_all_data:
        # Compute, per (BDMAP_ID, Organ), whether ANY tumor has unknown size
        any_unk_by_pair_all = (
            reports
            .assign(any_unk = reports['unknown size'].astype(str).eq('yes'))
            .groupby(['BDMAP_ID', 'Standardized Organ'], as_index=False)['any_unk']
            .any()
        )
        
        reports = reports[reports['BDMAP_ID'].isin(ids_of_interest)]
        # Build allowed (BDMAP_ID, Organ) pairs from interest (these have KNOWN sizes)
        allowed_pairs = (
            interest[['BDMAP_ID', 'Standardized Organ']]
            .dropna(subset=['Standardized Organ'])
            .drop_duplicates()
            .assign(allow_interest=True)
        )
        
        
        # Join both onto reports
        #if reports has a column 'allow_interest', drop it before the merge
        if 'allow_interest' in reports.columns:
            reports = reports.drop(columns=['allow_interest'])
        if 'any_unk' in reports.columns:
            reports = reports.drop(columns=['any_unk'])
        if 'prohibit crop' in reports.columns:
            reports = reports.drop(columns=['prohibit crop'])
        reports = reports.merge(allowed_pairs, on=['BDMAP_ID', 'Standardized Organ'], how='left')
        reports['allow_interest'] = reports['allow_interest'].fillna(False)
        reports = reports.merge(any_unk_by_pair_all, on=['BDMAP_ID', 'Standardized Organ'], how='left')
        reports['any_unk'] = reports['any_unk'].fillna(False)
        # allow only if (patient, organ) is in interest AND there are NO unknown sizes for that pair
        reports['prohibit crop'] = ~(reports['allow_interest'] & (~reports['any_unk']))
        bad = reports.loc[(~reports['prohibit crop']) & reports['any_unk']]
        assert bad.empty, "Logic error: allowed crops still have unknown-size tumors"
        assert reports.loc[(~reports['prohibit crop']) & (reports['any_unk'])].empty
    else:
        reports['prohibit crop'] = False

    #always prohibit crop on organs with unknown laterality of tumors
    #ignore tumors (not patients) outside the organs of interest
    annotated_tumors = set(annotated_tumors)
    for i,row in reports.iterrows():
        if row['Standardized Organ'] in ['kidney','adrenal_gland','lung','breast','femur']:
            loc = str(row.get('Standardized Location', '')).lower()
            if not ('right' in loc or 'left' in loc):
                reports.at[i,'prohibit crop'] = True
        if row['Standardized Organ'] not in annotated_tumors:
            reports.at[i,'prohibit crop'] = True
            
    #prohibit crop on tumors w/o malignancy info if benign_maligant_only
    if benign_maligant_only:
        unknown_malignancy_mask = reports['malignant_benign'] == 'u'
        reports.loc[unknown_malignancy_mask, 'prohibit crop'] = True

        bad = reports[(reports['malignant_benign'] == 'u') & (~reports['prohibit crop'])]
        assert bad.empty, f"Unknown malignancy rows still crop-allowed: {bad[['BDMAP_ID','Standardized Organ']].head(10)}"
        
        
    print(f"Size of the full dataset: {reports['BDMAP_ID'].nunique()} BDMAP_IDs, {len(reports)} tumors, number of healthy BDMAP_IDs: {reports.loc[normalize_no_lesion(reports['no lesion']), 'BDMAP_ID'].nunique()}", flush=True, file=sys.stderr)
    
    if not use_all_data:
        #assertion: check if all organs with allowed crop have known size
        allowed_crop = reports[~reports['prohibit crop']]
        #check if size is numeric
        bad = allowed_crop[~allowed_crop['Tumor Size (mm)'].astype(str).str.contains(r'\d', na=False)]
        if len(bad)>0:
            raise ValueError(f"Logic error: some allowed crops have unknown-size tumors: {bad[['BDMAP_ID','Standardized Organ','Tumor Size (mm)']]}")
    
    ids_of_interest = (
        reports['BDMAP_ID'].dropna().astype(str).unique().tolist()
    )
    return reports, ids_of_interest,tumors_per_type

def extract_id(image_path):
    """
    Extract the base ID from an image file path by removing the directory,
    extension, and a trailing '_gt' if present.
    """
    file_name = os.path.basename(image_path)
    base = os.path.splitext(file_name)[0]  # removes .npy or .npz
    if base.endswith('_gt'):
        base = base[:-3]  # remove the '_gt'
    return base

def tumor_to_organ(tumor_name):
    base = tumor_name.replace('_lesion', '')
    lb = base.lower()
    if lb == 'pancreatic':
        return 'pancreas'
    elif lb in ('adrenal_gland', 'adrenal'):
        return 'adrenal gland'
    else:
        return base

class AbdomenAtlasDataset(Dataset):
    def __init__(self, args, mode='train', k_fold=10, k=0, seed=0, all_train=False,
                crop_on_tumor=True,
                 save_destination=None,  
                 load_augmented=False,
                 gigantic_length=False,
                 samples_per_epoch=3000,
                 save_augmented=False,
                 tumor_classes= ['adrenal gland', 'bladder', 'colon', 'duodenum',
                 'esophagus', 'gallbladder','prostate','spleen','stomach','uterus'],#we will only crop on these tumors
                 balance_supervision=True,
                 debug=False,
                 UFO_only=False,
                 Atlas_only=False,
                 load_slices=True,
                 load_per_patient=None,
                 maximum_loaded_per_patient=None,
                 ):    
        
        if load_per_patient is None:
            load_per_patient = args.load_per_patient
        if maximum_loaded_per_patient is None:
            maximum_loaded_per_patient = args.maximum_loaded_per_patient
            
        if args.no_mask:
            UFO_only = True
        if hasattr(args,'load_malignancy') and args.load_malignancy:
            self.load_malignancy = args.load_malignancy
            self.malignant_col = args.malignant_col
            self.benign_col=args.benign_col
        else:
            self.load_malignancy = False
        self.UFO_only=UFO_only
        self.current_epoch = 0
        self.db_count = 0
        self.mode = mode
        self.args = args
        self.load_augmented = load_augmented   
        self.save_counter = 0 
        self.save_destination = save_destination
        self.gigantic_length=gigantic_length
        self.save_augmented = save_augmented
        self.samples_per_epoch = samples_per_epoch
        self.reports = pd.read_csv(args.reports)
        self.maximum_loaded_per_patient = maximum_loaded_per_patient
        if 'series matches report' in self.reports.columns:
            self.reports['series matches report'] = self.reports['series matches report'].apply(coerce_to_bool) #nans become False
        self.load_slices= load_slices
        self.tumor_classes = tumor_classes
        #read the ORIGINAL spacing list for the whole dataset
        self.original_spacing_list = pd.read_csv(args.original_spacing_list)
        if 'BDMAP ID' in self.reports.columns:  # no ()
            self.reports = self.reports.rename(columns={'BDMAP ID': 'BDMAP_ID'})
            
        self.reports_original = self.reports.copy()
        healthy_mask = normalize_no_lesion(self.reports['no lesion'])
        self.reports_original['is_healthy'] = healthy_mask
        if self.load_malignancy:
            self.reports_original['malignant_benign'] = self.reports_original.apply(lambda row: define_malignancy(row, malignancy_column=self.malignant_col, benign_column=self.benign_col,
                                                                                                                  relaxed_malignancy_col=(args.relaxed_malignancy_col if (self.load_malignancy and hasattr(args,'relaxed_malignancy_col')) else None)), axis=1)

        self.main_pid  = os.getpid()
        self.data_root = args.data_root
        print('Reports loaded from:', args.reports, flush=True, file=sys.stderr)
        print('Number of reports:', len(self.reports), flush=True, file=sys.stderr)
        self.zero_masks={}
        assert mode in ['train', 'test']
        self.counter=0
        
        if debug:
            self.sanity_path_debug=args.sanity_path_debug

        atlas_name_list = list(set([f[:len('BDMAP_00000000')] for f in os.listdir(self.data_root) if 'BDMAP' in f and '_gt' in f]))
        atlas_name_list2 = list(set([f[:len('BDMAP_00000000')] for f in os.listdir(self.data_root) if 'BDMAP' in f and '_gt' not in f]))
        atlas_name_list = list(set(atlas_name_list) & set(atlas_name_list2))
        if self.UFO_only:
            atlas_name_list = []
        
           #print('Number of Atlas Images:', len(atlas_name_list), flush=True, file=sys.stderr)
        self.UFO_root = args.UFO_root
        img_name_list_UFO = list(set([f[:len('BDMAP_00000000')] for f in os.listdir(self.UFO_root) if 'BDMAP' in f and '_gt' in f]))
        img_name_list_UFO2 = list(set([f[:len('BDMAP_00000000')] for f in os.listdir(self.UFO_root) if 'BDMAP' in f and '_gt' not in f]))
        img_name_list_UFO = list(set(img_name_list_UFO) & set(img_name_list_UFO2))
           #print('UFO root:', args.UFO_root, flush=True, file=sys.stderr)
           #print('Number of UFO Images:', len(img_name_list_UFO), flush=True, file=sys.stderr)

        #from reports, get only those in the dataset
        img_name_list_UFO_set =  set(img_name_list_UFO)
        ids = [case.replace('_0000.nii.gz','').replace('.nii.gz','') for case in img_name_list_UFO_set]

        if args.ucsf_ids is not None:
            cases = pd.read_csv(args.ucsf_ids)
            if 'BDMAP_ID' not in cases.columns:
                cases.rename(columns={'BDMAP ID': 'BDMAP_ID'}, inplace=True)
            cases = cases['BDMAP_ID'].tolist()
            ids = [case for case in ids if case in cases]
            #filter out img_name_list_UFO
            cases_set = set(cases)
            img_name_list_UFO = [case for case in img_name_list_UFO if case in cases_set]
            
            
        print()
        print('NUMBER OF SELECTED UFO IDs:', len(ids), flush=True, file=sys.stderr)
        print('NUMBER OF SELECTED UFO IMAGES:', len(img_name_list_UFO), flush=True, file=sys.stderr)
        print()

        #concatenate the two lists 
        img_name_list = atlas_name_list + img_name_list_UFO
        random.Random(seed).shuffle(img_name_list)
        
        if args.exclude_ids:
            exclude = pd.read_csv(args.exclude_ids)
            exclude = set(exclude['BDMAP ID'].tolist())
            print(f'Number of cases prior to exclusion: {len(img_name_list)}', flush=True, file=sys.stderr)
            img_name_list = [case for case in img_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') not in exclude]
            print(f'Number of cases after exclusion: {len(img_name_list)}', flush=True, file=sys.stderr)
            atlas_name_list = [case for case in atlas_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') not in exclude]
            img_name_list_UFO = [case for case in img_name_list_UFO if case.replace('_0000.nii.gz','').replace('.nii.gz','') not in exclude]
            ids = [case for case in ids if case not in exclude]
            
        
            
        if hasattr(args, 'test_ids_exclude') and args.test_ids_exclude is not None:
            exclude = pd.read_csv(args.test_ids_exclude)
            exclude = set(exclude['BDMAP ID'].tolist())
            print(f'Number of cases prior to exclusion (test ids): {len(img_name_list)}', flush=True, file=sys.stderr)
            img_name_list = [case for case in img_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') not in exclude]
            print(f'Number of cases after exclusion (test ids): {len(img_name_list)}', flush=True, file=sys.stderr)
            atlas_name_list = [case for case in atlas_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') not in exclude]
            img_name_list_UFO = [case for case in img_name_list_UFO if case.replace('_0000.nii.gz','').replace('.nii.gz','') not in exclude]
            ids = [case for case in ids if case not in exclude]
            
        self.prohibited_mask_ids = []
        if hasattr(args, 'atlas_ids') and args.atlas_ids is not None:
            include = pd.read_csv(args.atlas_ids)
            include = set(include['BDMAP ID'].tolist())
            print(f'Size of include list (atlas ids): {len(include)}', flush=True, file=sys.stderr)
            print(f'Number of cases prior to filtering (include train ids): {len(img_name_list)}', flush=True, file=sys.stderr)
            atlas_name_list_new = [case for case in atlas_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') in include]
            self.prohibited_mask_ids = list(set(atlas_name_list) - set(atlas_name_list_new)) #we cannot allow these masks to be loaded, but these cases may be loaded with reports, if they are in UFO dataset
            self.prohibited_mask_ids = set([case.replace('_0000.nii.gz','').replace('.nii.gz','') for case in self.prohibited_mask_ids])
            atlas_name_list = atlas_name_list_new
            img_name_list = atlas_name_list + img_name_list_UFO
            random.Random(seed).shuffle(img_name_list)
            print(f'Number of cases after filtering (include train ids): {len(img_name_list)}', flush=True, file=sys.stderr)
            
            
        if args.load_clip:
            #get only images for which we have embeddings
            embed_list = os.listdir(args.clip_source)
            img_name_list = [case for case in img_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') in embed_list]
            img_name_list_UFO = [case for case in img_name_list_UFO if case.replace('_0000.nii.gz','').replace('.nii.gz','') in embed_list]
            atlas_name_list = [case for case in atlas_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') in embed_list]
            if len(atlas_name_list) == 0:
                UFO_only = True
                print('WARNING: No Atlas images with embeddings found, using only UFO dataset', flush=True, file=sys.stderr)
            
            print('NUMBER OF SELECTED IMAGES AFTER EMBEDDING FILTER:', len(img_name_list), flush=True, file=sys.stderr)
            print('NUMBER OF SELECTED UFO IMAGES AFTER EMBEDDING FILTER:', len(img_name_list_UFO), flush=True, file=sys.stderr)

        self.tumor_annotated_seg = {}


        if not all_train:
            length = len(img_name_list)
            test_name_list = img_name_list[:min(200, length//10)]
            train_name_list = list(set(img_name_list) - set(test_name_list))
        else:
            train_name_list = img_name_list
            test_name_list = None
        
        if mode == 'train':
            img_name_list = train_name_list
        else:
            img_name_list = test_name_list
            
        #update ids according to img_name_list
        img_name_list_UFO_set = set(img_name_list_UFO)
        ids = [case.replace('_0000.nii.gz','').replace('.nii.gz','') for case in img_name_list if case in img_name_list_UFO_set]

        self.reports = self.reports[self.reports['BDMAP_ID'].isin(ids)]
        if hasattr(args,'benign_maligant_only'):
            benign_maligant_only = args.benign_maligant_only
        else:
            benign_maligant_only = False
        self.reports, ids, tumors_per_type = clean_ufo(self.reports,tumor_classes,slice_only=args.train_on_slices_only,
                                                       use_all_data=args.use_all_data, 
                                                       limit_healthy = (not(args.use_all_data) or benign_maligant_only),
                                                       benign_maligant_only=benign_maligant_only,
                                                       malignancy_column=self.malignant_col if self.load_malignancy else None,
                                                       benign_column=self.benign_col if self.load_malignancy else None,
                                                       relaxed_malignancy_col=args.relaxed_malignancy_col if (self.load_malignancy and hasattr(args,'relaxed_malignancy_col')) else None)
        
        
        if benign_maligant_only:
            #we also need to clean atlas
            #filter atlas_name_list to only cases in self.reports plus cases between bdmap ids 14484-14637 and 22412-22490
            with_reports = [case for case in atlas_name_list if case in self.reports['BDMAP_ID'].tolist()]
            esophagus_cases = [case for case in atlas_name_list if 'BDMAP' in case and re.search(r'BDMAP_(\d+)', case) and 14484 <= int(re.search(r'BDMAP_(\d+)', case).group(1)) <= 14637]
            uterus_cases = [case for case in atlas_name_list if 'BDMAP' in case and re.search(r'BDMAP_(\d+)', case) and 22412 <= int(re.search(r'BDMAP_(\d+)', case).group(1)) <= 22490]
            atlas_name_list = list(set(with_reports + esophagus_cases + uterus_cases))
            
        if mode=='train' and hasattr(args,'upsample_malig_benign') and args.upsample_malig_benign:    
            _, bdmap_ids_mb, _ = clean_ufo(self.reports,tumor_classes,slice_only=args.train_on_slices_only,
                                                       use_all_data=args.use_all_data, 
                                                       limit_healthy = True,
                                                       benign_maligant_only=True,
                                                       malignancy_column=self.malignant_col,
                                                       benign_column=self.benign_col,
                                                       relaxed_malignancy_col=args.relaxed_malignancy_col if (self.load_malignancy and hasattr(args,'relaxed_malignancy_col')) else None)
            esophagus_cases = [case for case in atlas_name_list if 'BDMAP' in case and re.search(r'BDMAP_(\d+)', case) and 14484 <= int(re.search(r'BDMAP_(\d+)', case).group(1)) <= 14637]
            uterus_cases = [case for case in atlas_name_list if 'BDMAP' in case and re.search(r'BDMAP_(\d+)', case) and 22412 <= int(re.search(r'BDMAP_(\d+)', case).group(1)) <= 22490]
            bdmap_ids_mb = list(set(bdmap_ids_mb + esophagus_cases + uterus_cases))
            
            
        
        #use ids to filter img_name_list and img_name_list_UFO
        ids_set = set(ids)
        img_name_list_UFO = [case for case in img_name_list \
            if case.replace('_0000.nii.gz','').replace('.nii.gz','') in ids_set]
        atlas_name_list_set = set(atlas_name_list)
        img_name_list_UFO_set = set(img_name_list_UFO)
        img_name_list = [case for case in img_name_list \
            if ((case in img_name_list_UFO_set) or (case in atlas_name_list_set))]
        img_name_list_UFO = list(set(img_name_list_UFO))
        img_name_list = list(set(img_name_list))
        assert len(img_name_list_UFO) == len(ids), f'Number of UFO images {len(img_name_list_UFO)} does not match number of ids {len(ids)}'
        assert len(img_name_list) >= len(ids), f'Number of images {len(img_name_list)} is less than number of ids {len(ids)}'
        
            
        #print(img_name_list)
        print('Start loading %s data'%self.mode)
        if args.balance_pos_neg and mode == 'train':
            ufo_meta = pd.read_csv(args.UFO_meta)
            ufo_healthy = ufo_meta[ufo_meta['no lesion']==1]['BDMAP ID'].tolist()
            ufo_disease = ufo_meta[ufo_meta['no lesion']==0]['BDMAP ID'].tolist()
            #get only the cases in self.img_name_list_ufo
            img_name_list_set = set(img_name_list)
            ufo_healthy = [i for i in ufo_healthy if i in img_name_list_set]
            ufo_disease = [i for i in ufo_disease if i in img_name_list_set]
            assert len(ufo_healthy) > 0
            assert len(ufo_disease) > 0
            print('ufo healthy cases:', len(ufo_healthy))
            print('ufo disease cases:', len(ufo_disease))
            ufo_healthy, ufo_disease = balance_classes(ufo_healthy, ufo_disease)
            print('After balancing ufo, healthy cases:', len(ufo_healthy))
            print('After balancing ufo, disease cases:', len(ufo_disease))
            
            
            atlas_meta = pd.read_csv(args.atlas_meta)
            cols = [col for col in atlas_meta.columns if 'number of' in col.lower() or 'instances' in col.lower()]
            # Filter the rows where all selected columns are 0
            atlas_healthy = atlas_meta[(atlas_meta[cols] == 0).all(axis=1)]
            atlas_diasease = atlas_meta[(atlas_meta[cols] > 0).any(axis=1)]
            #get only the cases in img_name_list
            img_name_list_set = set(img_name_list)
            atlas_healthy = [i for i in atlas_healthy['BDMAP ID'].tolist() if i in img_name_list_set]
            atlas_diasease = [i for i in atlas_diasease['BDMAP ID'].tolist() if i in img_name_list_set]
            assert len(atlas_healthy) > 0, 'No healthy cases found in atlas metadata!'
            assert len(atlas_diasease) > 0, 'No disease cases found in atlas metadata!'
            print('Atlas healthy cases:', len(atlas_healthy))
            atlas_healthy, atlas_diasease = balance_classes(atlas_healthy, atlas_diasease)
            print('After balancing Atlas, healthy cases:', len(atlas_healthy))
            print('After balancing Atlas, disease cases:', len(atlas_diasease))
            
            # Combine ufo and Atlas
            if UFO_only:
                img_name_list = ufo_healthy + ufo_disease
                balance_supervision = False
            elif Atlas_only:
                img_name_list =  atlas_healthy + atlas_diasease
                balance_supervision = False
            else:
                img_name_list = ufo_healthy + ufo_disease + atlas_healthy + atlas_diasease
            print('After balancing ufo and Atlas, total image name list length:', len(img_name_list))
            
        if args.balance_supervision_report_quality and (Atlas_only):
            raise ValueError('balance_supervision_report_quality cannot be used with UFO_only or Atlas_only, not implemented')
            
        if args.balance_supervision_report_quality and mode == 'train' and (not Atlas_only):
            if not args.use_all_data:
                raise ValueError('balance_supervision_report_quality requires use_all_data to be True')
            #check if use_sample_weigths in args and is True
            if hasattr(args, 'use_sample_weights') and args.use_sample_weights:
                raise ValueError('balance_supervision_report_quality should not be used with use_sample_weights')
                
            #here, we will supersample the dataset to give more power to reports of better quality
            #high quality: has tumor size and slice
            #mid quality: has tumor size but no slice
            #low quality: no tumor size
            #fourth category: annotated per voxel
            reports_high, ids_high, _ = clean_ufo(self.reports,tumor_classes,slice_only=True, use_all_data=False, limit_healthy = (not args.use_all_data), benign_maligant_only=benign_maligant_only)
            reports_mid,  ids_mid, _ = clean_ufo(self.reports,tumor_classes,slice_only=False, use_all_data=False, limit_healthy = (not args.use_all_data), benign_maligant_only=benign_maligant_only)
            reports_low,  ids_low, _ = clean_ufo(self.reports,tumor_classes,slice_only=False, use_all_data=True, limit_healthy = (not args.use_all_data), benign_maligant_only=benign_maligant_only)
            #remove overlaps
            reports_low = reports_low[~reports_low['BDMAP_ID'].isin(reports_high['BDMAP_ID'].to_list())]
            reports_low = reports_low[~reports_low['BDMAP_ID'].isin(reports_mid['BDMAP_ID'].to_list())]
            ids_low = list(set(ids_low) - set(ids_high) - set(ids_mid))
            reports_mid = reports_mid[~reports_mid['BDMAP_ID'].isin(reports_high['BDMAP_ID'].to_list())]
            ids_mid = list(set(ids_mid) - set(ids_high))
            
            #remove any id not in reports
            original_ids = set(self.reports['BDMAP_ID'].tolist())
            ids_high = list(set(ids_high) & original_ids)
            ids_mid  = list(set(ids_mid)  & original_ids)
            ids_low  = list(set(ids_low)  & original_ids)
            
            #get the ids from img_name_list_UFO
            ufo_present = set(img_name_list_UFO)
            atlas_ids   = list(set(atlas_name_list))
            ids_high = sorted(set(ids_high) & ufo_present)  
            ids_mid  = sorted(set(ids_mid)  & ufo_present)   
            ids_low  = sorted(set(ids_low)  & ufo_present)  
            
            #raise ValueError('Number of per-voxel annotated cases:', len(set(atlas_name_list)))
            print('Number of per-voxel annotated cases:', len(set(atlas_name_list)),flush=True)
            print('Number of high quality tumors (with slices and tumor size):', len(reports_high), 'CTs:', len(ids_high),flush=True)
            print('Number of mid quality tumors (with tumor size):', len(reports_mid), 'CTs:', len(ids_mid),flush=True)
            print('Number of low quality tumors (no tumor size):', len(reports_low), 'CTs:', len(ids_low),flush=True)
            
            if not UFO_only:
                target_props = torch.tensor([0.3, 0.4, 0.2, 0.1], dtype=torch.float)
                group_sizes = torch.tensor([len(atlas_ids),len(ids_high),len(ids_mid),len(ids_low)], dtype=torch.float)
                augmentation_coef, upsampled_group_sizes = get_group_upsample_coeficients(
                    group_sizes, target_props
                )

                super_atlas = supersample_group(atlas_ids, upsampled_group_sizes[0].item())
                super_high  = supersample_group(ids_high,  upsampled_group_sizes[1].item())
                super_mid   = supersample_group(ids_mid,   upsampled_group_sizes[2].item())
                super_low   = supersample_group(ids_low,   upsampled_group_sizes[3].item())
                
                img_name_list_UFO = super_high + super_mid + super_low
                atlas_name_list = super_atlas
                img_name_list = atlas_name_list + img_name_list_UFO # combine again after balancing
            else:
                target_props = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float)
                group_sizes = torch.tensor([len(ids_high),len(ids_mid),len(ids_low)], dtype=torch.float)
                augmentation_coef, upsampled_group_sizes = get_group_upsample_coeficients(
                    group_sizes, target_props
                )

                super_high  = supersample_group(ids_high,  upsampled_group_sizes[0].item())
                super_mid   = supersample_group(ids_mid,   upsampled_group_sizes[1].item())
                super_low   = supersample_group(ids_low,   upsampled_group_sizes[2].item())
                
                img_name_list_UFO = super_high + super_mid + super_low
                img_name_list = img_name_list_UFO # combine again after balancing
                
            
            
            print(f'Size of augmented dataset: {len(img_name_list)} = Atlas {len(atlas_name_list)} + UFO {len(img_name_list_UFO)}')
            if hasattr(args,'upsample_malig_benign') and args.upsample_malig_benign:
                raise ValueError('upsample_malig_benign cannot be used with balance_supervision_report_quality')
        else:
            
            if mode=='train' and hasattr(args,'upsample_malig_benign') and args.upsample_malig_benign:
                #idea: upsample malignant and benign cases to have equal representation as all other cases
                #if UFO_only or Atlas_only:
                #    raise ValueError('upsample_malig_benign cannot be used with UFO_only or Atlas_only, not implemented')
                if benign_maligant_only:
                    raise ValueError('upsample_malig_benign cannot be used with benign_maligant_only; initial dataset creation must not be filtered by malignancy')
                #cases with benign/malignant: bdmap_ids_mb
                atlas_name_list_set = set(atlas_name_list)
                atlas_name_list = [x for x in img_name_list if x in atlas_name_list_set] # filter to only those in the atlas
                img_name_list_UFO_set = set(img_name_list_UFO)
                img_name_list_UFO = [x for x in img_name_list if x in img_name_list_UFO_set] # filter to only those in the UFO dataset
                atlas_mb = [case for case in atlas_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') in bdmap_ids_mb]
                ufo_mb = [case for case in img_name_list_UFO if case.replace('_0000.nii.gz','').replace('.nii.gz','') in bdmap_ids_mb]
                #upsample both atlas_mb and ufo_mb to match the size of atlas_name_list and img_name_list_UFO respectively
                if len(atlas_mb)==0 and len(ufo_mb)==0:
                    raise ValueError('No malignant/benign cases found for upsampling!')
                if len(atlas_mb)>0:
                    assert len(atlas_mb) <= len(atlas_name_list), f'Number of malignant/benign atlas cases {len(atlas_mb)} exceeds total atlas cases {len(atlas_name_list)}. How??'
                    times_atlas = len(atlas_name_list) // len(atlas_mb)
                    remainder_atlas = len(atlas_name_list) % len(atlas_mb)
                    atlas_mb = atlas_mb * times_atlas + random.sample(atlas_mb, k=remainder_atlas)
                    random.shuffle(atlas_mb)
                if len(ufo_mb)>0:
                    assert len(ufo_mb) <= len(img_name_list_UFO), f'Number of malignant/benign ufo cases {len(ufo_mb)} exceeds total ufo cases {len(img_name_list_UFO)}. How??'
                    times_ufo = len(img_name_list_UFO) // len(ufo_mb)
                    remainder_ufo = len(img_name_list_UFO) % len(ufo_mb)
                    ufo_mb = ufo_mb * times_ufo + random.sample(ufo_mb, k=remainder_ufo)
                    random.shuffle(ufo_mb)
                print(f'Before upsampling malignant/benign cases, Atlas size: {len(atlas_name_list)}, UFO size: {len(img_name_list_UFO)}, dataset size: {len(img_name_list)}', flush=True)
                img_name_list_UFO = img_name_list_UFO + ufo_mb
                atlas_name_list = atlas_name_list + atlas_mb
                if UFO_only:
                    img_name_list = img_name_list_UFO
                elif Atlas_only:
                    img_name_list = atlas_name_list
                else:
                    img_name_list = atlas_name_list + img_name_list_UFO
                print(f'After upsampling malignant/benign cases, Atlas size: {len(atlas_name_list)}, UFO size: {len(img_name_list_UFO)}, dataset size: {len(img_name_list)}', flush=True)
                
            if mode == 'train' and balance_supervision and (not UFO_only) and (not Atlas_only):
                if len(atlas_name_list) == 0 or len(img_name_list_UFO) == 0:
                    raise ValueError('balance_supervision requires both Atlas and UFO datasets to be used, but one is empty')
                #get the atlas and ufo items following class balancing
                print(f'Balancing supervision between Atlas and UFO datasets for {mode} mode')
                atlas_name_list_set = set(atlas_name_list)
                atlas_name_list = [x for x in img_name_list if x in atlas_name_list_set] # filter to only those in the atlas
                img_name_list_UFO_set = set(img_name_list_UFO)
                img_name_list_UFO = [x for x in img_name_list if x in img_name_list_UFO_set] # filter to only those in the UFO dataset
                if len(atlas_name_list)>len(img_name_list_UFO):
                    times = len(atlas_name_list) // len(img_name_list_UFO)
                    rem = len(atlas_name_list) % len(img_name_list_UFO)
                    img_name_list_UFO = img_name_list_UFO * times + random.sample(img_name_list_UFO, k=rem)
                    random.shuffle(img_name_list_UFO)
                elif len(img_name_list_UFO)>len(atlas_name_list):
                    #randomly select some from atlas
                    times = len(img_name_list_UFO) // len(atlas_name_list)
                    rem = len(img_name_list_UFO) % len(atlas_name_list)
                    atlas_name_list = atlas_name_list * times + random.sample(atlas_name_list, k=rem)
                    random.shuffle(atlas_name_list)
                img_name_list = atlas_name_list + img_name_list_UFO # combine again after balancing
            elif UFO_only:
                img_name_list = img_name_list_UFO
            elif Atlas_only:
                img_name_list = atlas_name_list
            else:
                img_name_list = atlas_name_list + img_name_list_UFO

        random.shuffle(img_name_list) #shuffle
        
        
        _CANON = re.compile(r"^(BDMAP_[A-Za-z0-9]{8})\.(npy|npz)$", re.IGNORECASE)          # only pure ids, accepts BDMAP_ BDMAP_A and BDMAP_V
        def _build_pair_index(root):
            files = set(os.listdir(root))
            idx = {}

            # prefer .npz if both exist (you said they won’t, but this keeps it robust)
            for ext in (".npz", ".npy"):
                for f in files:
                    m = _CANON.match(f)
                    if not m or not f.endswith(ext):
                        continue
                    name = m.group(1)
                    lab  = f.replace(ext, f"_gt{ext}")              # exact match for this f
                    if lab in files:
                        idx[name] = (os.path.join(root, f), os.path.join(root, lab))
            return idx

        # --- build indices once (outside the hot loop) ---
        atlas_idx = _build_pair_index(args.data_root)
        ufo_idx   = _build_pair_index(args.UFO_root)

        # --- fast, single-pass materialization ---
        self.img_list = []
        self.lab_list = []
        self.spacing_list = []
        self.UFO_paths = []
        self.Atlas_paths = []

        for name in img_name_list:
            atlas_img=False
            UFO_img=False
            assert 'BDMAP_' in name, f'We expect IDs to begin with BDMAP_, followed by 8 chars. We got: {name}'
            sample_bdmap_id = name[name.find('BDMAP_'):name.find('BDMAP_')+len('BDMAP_00001111')]
            this_mask_is_prohibited = sample_bdmap_id in self.prohibited_mask_ids
            if UFO_only or this_mask_is_prohibited:
                pair = ufo_idx.get(name)
                UFO_img=True
                if pair is None:
                    raise ValueError(f"Image {name} not found in UFO indices")
            elif Atlas_only:
                pair = atlas_idx.get(name)
                atlas_img=True
                if pair is None:
                    raise ValueError(f"Image {name} not found in atlas indices")
            else:
                pair = atlas_idx.get(name)
                atlas_img=True
                if pair is None:
                    pair = ufo_idx.get(name) #preference to atlas
                    atlas_img=False
                    UFO_img=True
                if pair is None:
                    raise ValueError(f"Image {name} not found in atlas or UFO indices")
                
            img_path, lab_path = pair
            if atlas_img:
                self.Atlas_paths.append(img_path)
                self.tumor_annotated_seg[img_path] = True
                assert not UFO_only
            elif UFO_img:
                self.UFO_paths.append(img_path)
                self.tumor_annotated_seg[img_path] = False
                assert not Atlas_only
            else:
                raise ValueError('This should never happen')

            self.img_list.append(img_path)
            self.lab_list.append(lab_path)

        # fixed spacing per item; create once without per-iteration list creation
        self.spacing_list = [(1.0, 1.0, 1.0)] * len(self.img_list)  # itk reverse doesn't matter for 1,1,1

        self.crop_on_tumor = crop_on_tumor
        
        with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
            classes = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
            classes = sorted(classes)
            print('Classes list loaded from %s'%f, flush=True, file=sys.stderr)
            #print('Classes:', classes, flush=True, file=sys.stderr)
            print('Number of Classes:', len(classes), flush=True, file=sys.stderr)

        with open(os.path.join(args.UFO_root, 'list', 'label_names.yaml'), 'r') as f:
            classes_UFO = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
            classes_UFO = sorted(classes_UFO)
            
          
        for c in classes_UFO:
            if 'lesion' in c.lower() or ' tumor' in c.lower() or ' mass' in c.lower() or 'cyst' in c.lower() or 'pdac' in c.lower() or 'pnet' in c.lower():
                raise ValueError('UFO classes should not contain tumor or lesion classe. our assumption is that the UFO data does not have lesions annotated per voxel. Classes found:', classes_UFO)


        self.classes = classes
        self.classes_UFO = classes_UFO
        self.num_classes = len(classes)
        self.num_classes_UFO = len(classes_UFO)

        print('Classes:')
        for i, c in enumerate(classes):
            print(i, c)
        print('Classes UFO:')
        for i, c in enumerate(classes_UFO):
            print(i, c)

        lesion_classes = []
        lesion_class_names = []
        for i, c in enumerate(classes):
            if 'lesion' in c.lower() and any(tumor.replace(' ','_').replace('_gland','').replace('pancreas','pancreatic') in c.lower() for tumor in tumor_classes):
                lesion_classes.append(i)
                lesion_class_names.append(c)
        self.lesion_classes = lesion_classes
        assert len(lesion_classes) == len(tumor_classes), f'Number of lesion classes {len(lesion_classes)} does not match number of tumor classes {len(tumor_classes)}'
        print('Lesion classes:', lesion_classes)
        print('Lesion class names:', lesion_class_names)
        self.lesion_class_names=lesion_class_names

        self.saved_count = 0  # Reset the saved count on instantiation
        #print('Load done, length of dataset:', len(self.img_list))

        #check if all ids are in the reports
        # Convert IDs from the DataFrame to a set for faster lookup
        report_ids = set(self.reports['BDMAP_ID'].values)
        # Find IDs not present in the DataFrame
        report_ids_set = set(report_ids)
        missing_ids = [id for id in ids if id not in report_ids_set]
        # Raise an error if there are missing IDs
        if missing_ids:
            raise ValueError(f"IDs not in reports: {missing_ids}. Length of reports: {len(self.reports)}, number of missing ids: {len(missing_ids)}")

        if args.balanced_cropper:
            self.cropper = augmentation.choose_organ_class_match_tumor(class_names=self.classes, lesion_classes=lesion_class_names,
                                                                       scale=self.args.scale, rotate=self.args.rotate, translate=self.args.translate)
            self.cropper_UFO = augmentation.choose_organ_class_match_tumor(class_names=self.classes_UFO, 
                                                                           lesion_classes=lesion_class_names,
                                                                           reports = True,
                                                                       scale=self.args.scale, rotate=self.args.rotate, translate=self.args.translate)
        else:
            self.cropper = None
            self.cropper_UFO = None
        self.balancing_crops = args.balanced_cropper
        
        if args.class_weights:
            meta = args.atlas_meta
            meta_ufo = args.UFO_meta
            meta = pd.read_csv(meta)
            meta_ufo = pd.read_csv(meta_ufo)
            meta = pd.concat([meta, meta_ufo], ignore_index=True) # combine JHH and UFO meta
            self.class_proportions = get_class_proportions(
                meta=meta, 
                sample_list=img_name_list,
                lesion_class_names=lesion_class_names
            )
        else:
            self.class_proportions = None
        
        
        
        #get one ct scan per tumor type for debugging
        
        self.debug_img_list = generate_sanity_one_per_tumor(self.img_list, tumors_per_type)
        self.debug_img_list_UFO = generate_sanity_one_per_tumor(self.UFO_paths, tumors_per_type, 
                                                                slice_only = (self.load_slices and (not args.use_all_data)), reports=self.reports,
                                                                unknown_sizes_only = args.use_all_data)
        self.debug_img_list_Atlas = generate_sanity_one_per_tumor(self.Atlas_paths, tumors_per_type)
        #self.debug_img_list_UFO = generate_sanity_one_per_tumor(self.UFO_paths, tumors_per_type,
        #                                                        slice_only=False,
        #                                                        reports=self.reports,
        #                                                        unknown_sizes_only=False,
        #                                                        debug_malignancy=True,                 # << enable 50/50 mal/ben
        #                                                        get_malignancy_fn=self.get_malignancy  # << reuse your logic
        #                                                        )
        #self.debug_img_list_Atlas = generate_sanity_one_per_tumor(self.Atlas_paths, tumors_per_type,
        #                                                            debug_malignancy=True,     # enforces "single tumor" via atlas_meta
        #                                                            atlas_meta=self.reports_original
        #                                                        )
        self.crop_registry = os.path.join(self.save_destination,'list','crop_registry.db')
        self.LOCKFILE = str(Path(self.crop_registry)) + ".lock"
        
        
        
        
        if args.model_genesis_pretrain:
            # 1) Find MedFormer/ (it’s four levels up from this file):
            medformer_root = Path(__file__).resolve().parents[3]
            if not medformer_root.joinpath("baselines").is_dir():
                raise ImportError(f"Cannot find baselines/ under {medformer_root}")
            # 2) Ensure Python will search there
            if str(medformer_root) not in sys.path:
                sys.path.insert(0, str(medformer_root))
            # 3) Import the utils module and bind your method
            mg = importlib.import_module("baselines.model_genesis.utils")
            self.generate_pair = mg.generate_one_pair
        else:
            self.generate_pair = None
            


        self.load_per_patient=load_per_patient
            
        # --- Subsegment availability flags (so we can fall back cleanly) ---
        self._has_liver_subsegs_UFO   = all(f"liver_segment_{i}" in self.classes_UFO for i in range(1, 9))
        self._has_pancreas_subsegs_UFO = all(f"pancreas_{p}" in self.classes_UFO for p in ("head", "body", "tail"))
            
        #if debug and self.mode == 'train' and self.save_augmented:
        #    #load_augmented_bck = self.load_augmented
            #self.load_augmented = False
        #    self.sanity_path = self.sanity_path_debug
        #    self.non_tumor_crop_chance = 0
        #    self.non_tumor_crop_chance_unk_size = 0
        #    self.debug()
        #    #self.load_augmented = load_augmented_bck
        self.sanity_path = "./DatasetSanityMultiTumorPerPatient"
        self.non_tumor_crop_chance = 0.2
        self.non_tumor_crop_chance_unk_size = 0.5
        self.debugging=False

        # ----- Ablation: mask report metadata after sample selection is done -----
        # Done HERE (after all clean_ufo / supersampling / id-list construction)
        # so sample selection / weighting see the real data, but every read_report()
        # in __getitem__ returns the masked values. Both model-input fields
        # (tumor_info_input) and loss-bound fields (volumes, diameters, attenuation)
        # then carry the trained "unknown" sentinels.
        # No-op when the flags are off.
        if getattr(args, 'remove_size', False):
            print('[ABLATION] --remove_size: setting Tumor Size (mm) to "u" for all rows', flush=True, file=sys.stderr)
            self.reports['Tumor Size (mm)'] = 'u'
            self.reports['unknown size'] = 'yes'
        if getattr(args, 'remove_attenuation', False):
            print('[ABLATION] --remove_attenuation: setting Standardized Attenuation to "" for all rows', flush=True, file=sys.stderr)
            # estimate_tumor_volume treats any value that is not 'high' or 'low' as 999 (unknown);
            # get_input_info_report then encodes 999 as the "unknown" attenuation index.
            self.reports['Standardized Attenuation'] = ''

        print(f'The size of the {self.mode} dataset is: {len(self.img_list)}', flush=True, file=sys.stderr)

        _count_atlas_ufo(self.img_list, args.data_root, args.UFO_root)
        
        
        
    def debug(self):
        self.debugging=True
        if os.path.exists(self.sanity_path):
            #remove
            os.system('rm -r %s'%self.sanity_path)
            
        if "gallbladder" in self.debug_img_list_Atlas.keys():
            gb_val = self.debug_img_list_Atlas.pop("gallbladder")
            self.debug_img_list_Atlas = {"gallbladder": gb_val, **self.debug_img_list_Atlas}
            
            gb_val = self.debug_img_list_UFO.pop("gallbladder")
            self.debug_img_list_UFO = {"gallbladder": gb_val, **self.debug_img_list_UFO}

            
        print('Debugging satrted')
        for tumor, img_name in self.debug_img_list_Atlas.items():
            print()
            print()
            print('Debugging Atlas image:', img_name, 'Tumor type:', tumor)
            dta = self.__getitem__(idx=None, name=img_name,debug_dict=True)
            BDMAP_ID = img_name[img_name.find('BDMAP_'):img_name.find('BDMAP_')+len('BDMAP_00001111')]
            #add a txt file in the sanity path with the tumor type
            with open(os.path.join(self.sanity_path, tumor+'_'+BDMAP_ID+'_per_voxel_anno.json'), 'w') as f:
                json.dump(to_jsonable(dta), f)
        for tumor, img_name in self.debug_img_list_UFO.items():
            print()
            print()
            print('Debugging UFO image:', img_name, 'Tumor type:', tumor)
            dta = self.__getitem__(idx=None, name=img_name,debug_dict=True)
            BDMAP_ID = img_name[img_name.find('BDMAP_'):img_name.find('BDMAP_')+len('BDMAP_00001111')]
            with open(os.path.join(self.sanity_path, tumor+'_'+BDMAP_ID+'_report_anno.json'), 'w') as f:
                json.dump(to_jsonable(dta), f)
        self.debugging=False
        print('Debugging finished')
        
    def _fallback_to_organ_label(self, lbl: str, classes: list):
        """If a sub-segment label is missing, return the organ label present in `classes`."""
        if lbl.startswith("liver_segment_") and "liver" in classes:
            return "liver"
        if lbl.startswith("pancreas_") and "pancreas" in classes:
            return "pancreas"
        return None

    def read_report(self, idx, original=False):
        if not original:
            reports = self.reports
        else:
            reports = self.reports_original
        id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
        if id not in reports['BDMAP_ID'].values:
            #print('ID is: ',id)
            raise ValueError('ID is not in the reports:', id, 'Length of reports:', len(reports))
            return None #no tumor
        else:
            tumors=reports[reports['BDMAP_ID']==id]
            #tumors=tumors.to_dict(orient='records')
            return tumors
        
    
    def get_patient_id(self, idx):
        try:
            report = self.read_report(idx, original=True)
        except:
            return "unknown"

        patient_id = report["Patient ID"].iloc[0]

        if pd.isna(patient_id):
            return "unknown"

        if isinstance(patient_id, np.generic):
            patient_id = patient_id.item()  # np.str_ / np.int64 -> str / int

        return str(patient_id)
        
    def get_date(self,idx):
        try:
            report = self.read_report(idx, original=True)
        except:
            return torch.tensor([0]).float()
        dates = pd.to_datetime(report["Exam Completed Date"], errors="coerce")
        date = dates.iloc[0]
        if pd.isna(date):
            return torch.tensor([0]).float()
        else:
            #date should be the number of years since 1971, the invention of CT scan, in years (float)
            #we can use any reference really, we just care about the delta between dates for 2 ct scans
            date = (date - pd.Timestamp('1971-01-01')).days / 365.25
            date = torch.tensor([date]).float()
            return date
        
    def get_age_and_sex(self, idx):
        try:
            report = self.read_report(idx, original=True)
        except:
            return torch.tensor([0]).float(), torch.tensor([0]).float()
        age = report['Patient Age'].iloc[0]
        sex = report['Patient Sex'].iloc[0]
        
        # --- age ---
        # handle NaN/None early
        if pd.isna(age):
            age_val = 0.0
        else:
            # numeric already
            if isinstance(age, (int, float, np.integer, np.floating)):
                age_val = float(age)
            else:
                # string: extract first number like "65", "65.0", "065", "65Y"
                s = str(age).strip()
                m = re.search(r"(\d+(\.\d+)?)", s)
                age_val = float(m.group(1)) if m else 0.0

        # clamp to a sane range (optional but recommended)
        if not np.isfinite(age_val) or age_val < 0 or age_val > 120:
            age_val = 0.0

        age = torch.tensor([age_val / 100.0], dtype=torch.float32)

        if not isinstance(sex, str):
            sex = 'unknown'
        if sex.lower() == 'male':
            sex = torch.tensor([1]).float()
        elif sex.lower() == 'female':
            sex = torch.tensor([0.5]).float()
        elif sex.lower() == 'unknown':
            sex = torch.tensor([0]).float()
        else:
            raise ValueError(f'Unexpected value for patient sex, {sex}, it should be Male, Female or unknown/NaN')
        return age, sex
        
    def get_contrast_phase(self, idx):
        try:
            report = self.read_report(idx, original=True)
        except:
            return 'unknown'
        if 'contrast' not in report.columns:
            return 'unknown'
        contrast = report['contrast'].iloc[0]
        #if nan, return unk
        if not isinstance(contrast, str):
            return 'unknown'
        return contrast
            

    def __len__(self):
        if self.mode == 'train':
            if self.gigantic_length:
                return len(self.img_list) * 100000
            else:
                return len(self.img_list)
        else:
            return len(self.img_list)
        
    def sample_ids(self,bdmap_ids_with_dates,bdmap_ids_without_dates,number_to_sample,preferential_id=None):
        if len(bdmap_ids_with_dates) >= number_to_sample:
            to_sample = bdmap_ids_with_dates[:]
        else:
            to_sample = bdmap_ids_with_dates[:]
            #add bdmap_ids_without_dates until maximum_loaded_per_patient or until we run out of bdmap_ids_without_dates
            needed = number_to_sample - len(bdmap_ids_with_dates)
            if len(bdmap_ids_without_dates) > 0:
                if len(bdmap_ids_without_dates)>= needed:
                    to_sample += random.sample(bdmap_ids_without_dates, k=needed)
                else:
                    to_sample += bdmap_ids_without_dates
        
        if preferential_id is not None:
            sampled_ids = [[id_sel,date] for id_sel, date in bdmap_ids_with_dates if id_sel == preferential_id]
            sampled_ids += [[id_sel,date] for id_sel, date in bdmap_ids_without_dates if id_sel == preferential_id]
            assert len(sampled_ids) == 1, f"preferential_id must be in the sampled ids and only once. our sampled_ids: {sampled_ids}"
            #remove preferential_id from to_sample to avoid sampling it twice
            to_sample = [[id_sel,date] for id_sel, date in to_sample if id_sel != preferential_id]
            number_to_sample -= 1
        else:
            sampled_ids = []
            
        if number_to_sample == 0:
            assert len(sampled_ids) > 0, "If number_to_sample is 0, we should have already added the preferential_id to sampled_ids, but we have none. sampled_ids: {}".format(sampled_ids)
            return sampled_ids
        
        if len(to_sample) <= number_to_sample:
            sampled_ids += to_sample
        else:
            sampled_ids += random.sample(to_sample, k=number_to_sample)
        return sampled_ids
            
    def replicate_return(self,returned):
        #torch collate will complain when we return different number of BDMAP IDs for different samples,
        #so we must pad by replicating the first BDMAP ID until we reach maximum_loaded_per_patient
        assert isinstance(returned,list), f'we want a list of the returns of getitem_real, got {type(returned)}'
        if len(returned) == 0:
            raise ValueError('We should have at least one BDMAP ID to return, but we got an empty list.')
        if len(returned) > self.maximum_loaded_per_patient:
            #get only the first maximum_loaded_per_patient
            returned = returned[:self.maximum_loaded_per_patient]
        while len(returned) < self.maximum_loaded_per_patient:
            returned.append(returned[0])  
        returned = list_of_dicts_to_dict_of_lists(returned)
        return returned
    
    def __getitem__(self, idx, name=None, BDMAP_ID=None, debug_dict=False,try_2=False,do_not_retry=False):
        if (not self.load_per_patient) or (name is not None):
            return self.getitem_real(idx=idx, name=name, BDMAP_ID=BDMAP_ID, debug_dict=debug_dict,try_2=try_2)
        else:
            if BDMAP_ID is not None:
                if (not self.UFO_only) and (BDMAP_ID not in self.prohibited_mask_ids):
                    name = os.path.join(self.data_root, BDMAP_ID+'.npy')
                    if not os.path.exists(name):
                        name = os.path.join(self.data_root, BDMAP_ID+'.npz')
                else:
                    name = os.path.join(self.UFO_root, BDMAP_ID+'.npz')
                if not os.path.exists(name):
                    name = os.path.join(self.UFO_root, BDMAP_ID+'.npy')
                if not os.path.exists(name):
                    name = os.path.join(self.UFO_root, BDMAP_ID+'.npz')
                if not os.path.exists(name):
                    raise ValueError('Image %s not found in npy nor npz'%name)
                idx = self.img_list.index(name)
            
            print(f'Loading per patient for idx {idx}', flush=True)         
            assert 'BDMAP_' in self.img_list[idx], f'We expect IDs to begin with BDMAP_, followed by 8 chars. We got: {self.img_list[idx]}'
            original_bdmap_id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
            #check if idx is in reports
            if original_bdmap_id in self.reports['BDMAP_ID'].values:
                report = self.read_report(idx, original=False)#idx represents a BDMAP_ID---we keep the resampling we did (e.g., augment cases pathology confirmed)
            else:
                if original_bdmap_id in self.reports_original['BDMAP_ID'].values:
                    #this id must be annotated per voxel, or we raise
                    if not self.tumor_annotated_seg[self.img_list[idx]]:
                        raise ValueError(f'BDMAP_ID {original_bdmap_id} exists in original reports but not in processed reports, and it is not annotated per voxel')
                #no report for this patient, we just load it
                print(f'No report found for BDMAP_ID {original_bdmap_id}, loading non-longitudinal', flush=True)
                try:
                    data = self.getitem_real(idx=None, BDMAP_ID=original_bdmap_id, try_2=True,do_not_change_patient=True)
                except:
                    if (not try_2) and (not do_not_retry):
                        idx = random.randint(0, len(self.img_list)-1)
                        return self.__getitem__(idx=idx, name=name, BDMAP_ID=BDMAP_ID, debug_dict=debug_dict, try_2=True, do_not_retry=True)
                    else:
                        raise ValueError('Failed to load non-longitudinal data for BDMAP_ID {} during longitudinal loading.'.format(original_bdmap_id))
                        
                data['dates'] = torch.tensor([0.0]).float()
                data['Patient ID'] = 'unknown_'+str(random.randint(0,1000000))
                data['organ_cropped'] = data['subsegment_cropped_in'] if data['subsegment_cropped_in'] is not None else 'random'
                return self.replicate_return([data])
                
            patient_id = report['Patient ID'].iloc[0]
            patient_reports = self.reports.loc[self.reports['Patient ID'] == patient_id].copy()
            patient_id = self.get_patient_id(idx)
            
            if len(patient_reports) == 0 or patient_id == 'unknown':
                print(f'No more reports found for BDMAP_ID {original_bdmap_id}, loading non-longitudinal', flush=True)
                try:
                    data = self.getitem_real(idx=None, BDMAP_ID=original_bdmap_id, try_2=True,do_not_change_patient=True)
                except:
                    if (not try_2) and (not do_not_retry):
                        idx = random.randint(0, len(self.img_list)-1)
                        return self.__getitem__(idx=idx, name=name, BDMAP_ID=BDMAP_ID, debug_dict=debug_dict, try_2=True, do_not_retry=True)
                    else:
                        raise ValueError('Failed to load non-longitudinal data for BDMAP_ID {} during longitudinal loading.'.format(original_bdmap_id))
                data['dates'] = torch.tensor([0.0]).float()
                data['Patient ID'] = 'unknown_'+str(random.randint(0,1000000))
                data['organ_cropped'] = data['subsegment_cropped_in'] if data['subsegment_cropped_in'] is not None else 'random'
                return self.replicate_return([data])
            
            patient_reports['Exam Completed Date'] = pd.to_datetime(
                patient_reports['Exam Completed Date'], errors='coerce'
            )

            # If multiple rows per BDMAP_ID exist, pick one date per BDMAP_ID (earliest non-NaT)
            date_per_bdmap = (
                patient_reports
                .sort_values('Exam Completed Date')
                .groupby('BDMAP_ID', as_index=False)['Exam Completed Date']
                .min()
            )

            # Define bdmap_ids order (e.g., chronological by date; NaT at the end)
            date_per_bdmap = date_per_bdmap.sort_values('Exam Completed Date', na_position='last')
            bdmap_ids = date_per_bdmap['BDMAP_ID'].tolist()
            
            print(f'Patient {patient_id} has BDMAP IDs: {bdmap_ids} with corresponding dates: {date_per_bdmap["Exam Completed Date"].tolist()}', flush=True)

            dates = date_per_bdmap['Exam Completed Date'].tolist()
            baseline = next((d for d in dates if pd.notna(d)), pd.NaT)  # earliest valid date, or NaT if none

            years = []
            for d in dates:
                if pd.isna(d) or pd.isna(baseline):
                    years.append(-5.0)
                else:
                    delta_days = (d - baseline).total_seconds() / 86400.0
                    years.append(float(delta_days / 365.25))
                    
            years = [y for y in years]  # scale down to keep in a reasonable range for the model; adjust as needed

            #dates: a list with a data for each bdmap_id, in the same order as bdmap_ids. 
            # The first date should be 0, and the others should be the number of years (with decimals) 
            # between the exam completed date of the first scan and the exam completed date of the scan corresponding to the bdmap_id
            # if the date is missing, we will use -0.5, which means unknown.
            
            print(f'Patient {patient_id} has BDMAP IDs: {bdmap_ids} with corresponding years from baseline: {years}', flush=True)
            load_augmented = self.load_augmented
            if not os.path.exists(os.path.join(self.save_destination, original_bdmap_id + ".json")):
                load_augmented = False
            if load_augmented:
                # get the organ cropped for the original_bdmap_id
                with open(os.path.join(self.save_destination, original_bdmap_id + ".json"), "r") as f:
                    dta=json.load(f)
                selected_organ_cropped = dta['subsegment_cropped_in'] if 'subsegment_cropped_in' in dta.keys() and dta['subsegment_cropped_in'] is not None else 'random'
                if selected_organ_cropped=='random':
                    #if random, we load only one crop of the patient.
                    organs_cropped={'random':[original_bdmap_id]}
                else:
                    #where other ids cropped on?
                    organs_cropped = {}
                    for bdmap_id in bdmap_ids:
                        if not os.path.exists(os.path.join(self.save_destination, bdmap_id + ".json")):
                            continue
                        with open(os.path.join(self.save_destination, bdmap_id + ".json"), "r") as f:
                            dta=json.load(f)
                        organ_cropped = dta['subsegment_cropped_in'] if 'subsegment_cropped_in' in dta.keys() and dta['subsegment_cropped_in'] is not None else 'random' #tumor in crop is actually the tumor sub-segment, such as 'pancreas_head' or 'adrenal_left'
                        if organ_cropped in organs_cropped.keys():
                            organs_cropped[organ_cropped].append(bdmap_id)
                        else:
                            organs_cropped[organ_cropped] = [bdmap_id]
                            
                bdmap_ids_with_dates = [[bdmap_id,d] for bdmap_id, d in zip(bdmap_ids, years) if (d>=0 and (bdmap_id in organs_cropped[selected_organ_cropped]))]
                bdmap_ids_without_dates = [[bdmap_id,d] for bdmap_id, d in zip(bdmap_ids, years) if (d<0 and (bdmap_id in organs_cropped[selected_organ_cropped]))]
                sampled_ids = self.sample_ids(bdmap_ids_with_dates,bdmap_ids_without_dates,self.maximum_loaded_per_patient,preferential_id = original_bdmap_id)
                
                assert any(bid == original_bdmap_id for bid, _ in sampled_ids), "original_bdmap_id must be in sampled_ids"
                
                loaded = []
                for bdmap_id,date in sampled_ids:
                    try:
                        data = self.getitem_real(idx=None, BDMAP_ID=bdmap_id, try_2=True,do_not_change_patient=True)
                    except:
                        if (not try_2) and (not do_not_retry):
                            #randomly select a patient from 0 to self.__len__()-1
                            new_bdmap_idx = random.randint(0, self.__len__()-1)
                            return self.__getitem__(idx=new_bdmap_idx, try_2=True, do_not_retry=True)
                        else:
                            raise ValueError(f'Failed to load BDMAP ID {bdmap_id} during longitudinal loading.')
                    data['dates'] = torch.tensor([date]).float()
                    data['Patient ID'] = patient_id
                    data['organ_cropped'] = data['subsegment_cropped_in'] if data['subsegment_cropped_in'] is not None else 'random'
                    loaded.append(data)
                
            else: #not load_augmented
                bdmap_ids_with_dates = [[bdmap_id,d] for bdmap_id, d in zip(bdmap_ids, years) if (d>=0)]
                bdmap_ids_without_dates = [[bdmap_id,d] for bdmap_id, d in zip(bdmap_ids, years) if (d<0)]
                bdmap_ids_all = bdmap_ids_with_dates + bdmap_ids_without_dates
                
                try:
                    loaded_first = self.getitem_real(idx=None, BDMAP_ID=original_bdmap_id, try_2=True,do_not_change_patient=True)
                except:
                    if (not try_2) and (not do_not_retry):
                        idx = random.randint(0, len(self.img_list)-1)
                        return self.__getitem__(idx=idx, name=name, BDMAP_ID=BDMAP_ID, debug_dict=debug_dict, try_2=True, do_not_retry=True)
                    else:
                        raise ValueError(f'Failed to load original BDMAP ID {original_bdmap_id}')
                    
                subseg_cropped_init = loaded_first['subsegment_cropped_in'] if loaded_first['subsegment_cropped_in'] is not None else 'random'
                organ_cropped_init = loaded_first['slices_cropped_dict']['selected_organ_non_canonical']
                organ_cropped_init = organ_cropped_init if organ_cropped_init is not None else 'random'
                
                if not self.save_augmented:
                    if organ_cropped_init=='random':
                        sampled_ids = [[bdmap_id,d] for bdmap_id, d in zip(bdmap_ids, years) if (bdmap_id == original_bdmap_id)]
                    else:
                        #randomly select self.maximum_loaded_per_patient BDMAP IDs, give preference to cases with dates
                        sampled_ids = self.sample_ids(bdmap_ids_with_dates,bdmap_ids_without_dates,self.maximum_loaded_per_patient,
                                                    preferential_id = original_bdmap_id)
                else:
                    sampled_ids = bdmap_ids_all
                    
                assert any(bid == original_bdmap_id for bid, _ in sampled_ids), "original_bdmap_id must be in sampled_ids"
                    
                

                loaded = []
                #now load the remaining bdmap_ids, giving preference to cropping on the same organ
                for bdmap_id, date in sampled_ids:
                    print(f'Patient {patient_id}; BDMAP ID {bdmap_id}; date {date}; Crop organ {organ_cropped_init}; subsegment {subseg_cropped_init}', flush=True)
                    if bdmap_id == original_bdmap_id:
                        data = loaded_first
                    else:
                        try:
                            data = self.getitem_real(idx=None, BDMAP_ID=bdmap_id, try_2=True,do_not_change_patient=True,
                                                        preferential_subsegment_crop=subseg_cropped_init,
                                                        preferential_organ_crop=organ_cropped_init)
                        except:
                            if (not try_2) and (not do_not_retry):
                                idx = random.randint(0, len(self.img_list)-1)
                                return self.__getitem__(idx=idx, name=name, BDMAP_ID=BDMAP_ID, debug_dict=debug_dict, try_2=True, do_not_retry=True)
                            else:
                                raise ValueError(f'Failed to load original BDMAP ID {bdmap_id}')
                    subseg_cropped = data['subsegment_cropped_in'] if data['subsegment_cropped_in'] is not None else 'random'
                    print(f'Patient {patient_id}; BDMAP ID {bdmap_id}; date {date}; true subsegment {subseg_cropped}', flush=True)
                    if bdmap_id == original_bdmap_id:
                        data['dates'] = torch.tensor([date]).float()
                        data['Patient ID'] = patient_id
                        data['organ_cropped'] = subseg_cropped
                        loaded.append(data)
                    elif subseg_cropped != 'random' and subseg_cropped == subseg_cropped_init:
                        data['dates'] = torch.tensor([date]).float()
                        data['Patient ID'] = patient_id
                        data['organ_cropped'] = subseg_cropped
                        loaded.append(data)
                    
            return self.replicate_return(loaded)

    def getitem_real(self, idx, name=None, BDMAP_ID=None, debug_dict=False,try_2=False,
                        do_not_change_patient=False,preferential_subsegment_crop=None,preferential_organ_crop=None):
        if name is not None:
            idx = self.img_list.index(name)#for debugging, let's you request a sample by name
        if BDMAP_ID is not None:
            if (not self.UFO_only) and (BDMAP_ID not in self.prohibited_mask_ids):
                name = os.path.join(self.data_root, BDMAP_ID+'.npy')
                if not os.path.exists(name):
                    name = os.path.join(self.data_root, BDMAP_ID+'.npz')
            else:
                name = os.path.join(self.UFO_root, BDMAP_ID+'.npz')
            if not os.path.exists(name):
                name = os.path.join(self.UFO_root, BDMAP_ID+'.npy')
            if not os.path.exists(name):
                name = os.path.join(self.UFO_root, BDMAP_ID+'.npz')
            if not os.path.exists(name):
                raise ValueError('Image %s not found in npy nor npz'%name)
            idx = self.img_list.index(name)
        
        #print('Loading:', self.img_list[idx], self.lab_list[idx])
        self.current_sample = self.img_list[idx]
        sample_id=self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
        mask_is_prohibited = sample_id in self.prohibited_mask_ids
        if self.load_augmented and (not mask_is_prohibited):#we cannot load augmented cases from prohibited masks, as it may have been saved with mask in the augment_forever code
            #return self.load_augmented_data(idx)
            try:
                return self.load_augmented_data(idx)#loads and returns data already augmented and pre-saved
            except:
                if not do_not_change_patient:
                    idx = np.random.randint(len(self.img_list))
                    try:
                        return self.load_augmented_data(idx)
                    except:
                        print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx])
                else:
                    print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx])

        try:
            np_img = np.load(self.img_list[idx], mmap_mode='r', allow_pickle=False)
            if '.npz' in self.img_list[idx]:
                np_img = np_img['arr_0']
        except:
            print('Error loading:', self.img_list[idx])
            try:
                np_img = np.load(self.img_list[idx])
                if '.npz' in self.img_list[idx]:
                    np_img = np_img['arr_0']
            except:
                if (not try_2) and (not do_not_change_patient):
                    print(f'CAREFUL!!! ERROR LOADING {self.img_list[idx]}', flush=True, file=sys.stderr)
                    #get a random index and load
                    rnd_idx = random.randint(0, self.__len__()-1)
                    return self.getitem_real(rnd_idx, try_2=True)
                raise ValueError('Error loading:', self.img_list[idx])
        try:
            np_lab = np.load(self.lab_list[idx], mmap_mode='r', allow_pickle=False)
            if '.npz' in self.lab_list[idx]:
                np_lab = np_lab['arr_0']
        except:
            print('Error loading:', self.lab_list[idx])
            try:
                np_lab = np.load(self.lab_list[idx])
                if '.npz' in self.lab_list[idx]:
                    np_lab = np_lab['arr_0']
            except:
                if (not try_2) and (not do_not_change_patient):
                    print(f'CAREFUL!!! ERROR LOADING {self.img_list[idx]}', flush=True, file=sys.stderr)
                    #get a random index and load
                    rnd_idx = random.randint(0, self.__len__()-1)
                    return self.getitem_real(rnd_idx, try_2=True)
                raise ValueError('Error loading:', self.lab_list[idx])
            
        bdmap_id=self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
        original_spacing = self.original_spacing_list[self.original_spacing_list['BDMAP_ID']==bdmap_id]['z_mm'].iloc[0]
        original_spacing = float(original_spacing)
        
            
        #get the slices
        if self.load_slices and not self.tumor_annotated_seg[self.img_list[idx]]:
            try:
                #here we extract the slices from report. Actually, we begin by just checking which organs have slices.
                #we will prioritize them in cropping
                segments,tumor_dict=self.get_tumor_segment_labels(idx)
                slices_dict= self.get_slices_from_report(idx,segments)
                
                if slices_dict is not None and slices_dict['slices_found']:
                    np_slices = np.load(self.lab_list[idx].replace('_gt','_slice'), mmap_mode='r', allow_pickle=False)
                    if '.npz' in self.lab_list[idx]:
                        np_slices = np_slices['arr_0']
                else:
                    slices_dict = None
                    np_slices = None
            except:
                if not try_2:
                    print(f'CAREFUL!!! ERROR LOADING {self.img_list[idx]}', flush=True, file=sys.stderr)
                    if idx==self.__len__()-1:
                        print('Error loading slices for last sample:', self.lab_list[idx], flush=True, file=sys.stderr)
                        return self.getitem_real(0, try_2=True)
                    return self.getitem_real(idx+1, try_2=True)
                raise ValueError('Error loading:', self.lab_list[idx])
        else:
            slices_dict = None
            np_slices = None
            

        if self.img_list[idx] in self.UFO_paths:
            classes = self.classes_UFO
        else:
            classes = self.classes
            if mask_is_prohibited:
                raise ValueError('Prohibited mask trying to be loaded, this should not happen!')

        if np_lab.shape[0] != len(classes):
            # 4. Unpack the bits along the same axis.
            try:
                np_lab = np.unpackbits(np_lab, axis=0)
            except:
                raise ValueError('Unpack bits failed for sample:', self.img_list[idx])
            assert np_lab.shape[0] < len(classes) +10
            np_lab = np_lab[:len(classes)]
            ##print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)


        if self.mode == 'train':
            d, h, w = self.args.training_size
            #np_img, np_lab = augmentation.np_crop_3d(np_img, np_lab, [d+40, h+40, w+40], mode='random')

            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)
            assert tensor_img[0,0].shape == tensor_lab[0,0].shape, f'Image spatial shape {tensor_img[0,0].shape} does not match label shape {tensor_lab[0,0].shape}, sample is {self.img_list[idx]}'
            #make tensor_lab int16
            tensor_lab = tensor_lab.to(torch.int16) 
            if self.load_slices and np_slices is not None:
                tensor_slices = torch.from_numpy(np_slices).unsqueeze(0).unsqueeze(0)
                assert tensor_slices.shape==tensor_img.shape, f'Slices shape {tensor_slices.shape} does not match image shape {tensor_img.shape}, sample is {self.img_list[idx]}'
                assert len(tensor_slices.shape) == 5, f'Slices shape {tensor_slices.shape} is not 5D, sample is {self.img_list[idx]}'
            else:
                tensor_slices = None
            ##print('Time to load data:', time.time() - start, flush=True, file=sys.stderr)
            aug_start = time.time()

            del np_img, np_lab
            if self.load_slices and tensor_slices is not None:
                del np_slices
                tensor_lab = torch.cat((tensor_lab, tensor_slices), dim=1)  # concatenate slices to labels in channel dim
            tensor_img, tensor_lab = tensor_img.contiguous(), tensor_lab.contiguous()
            #pad with zeros if the image is smaller than the training patch size + a little margin
            tensor_img, tensor_lab = augmentation.pad_volume_pair(tensor_img, tensor_lab, d+40, h+40, w+40)
            
            if self.load_slices and tensor_slices is not None:
                #now split
                tensor_slices = tensor_lab[:, -1, :, :, :]  # last channel is slices
                tensor_lab = tensor_lab[:, :-1, :, :, :]  # all but last channel are labels
            else:
                tensor_slices = None
            
            #print('Starting crop:', flush=True, file=sys.stderr)
            tensor_img, tensor_lab, tumor_dict, selected_tumor, selected_organ_non_canonical, reason, tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel, slices_cropped_dict, subseg_cropped_in, organ_cropped_in = self.crop(tensor_img, tensor_lab, idx, d, h, w,
                                                                                                                                                                                                  slices_mask=tensor_slices, slices_dict=slices_dict, 
                                                                                                                                                                                                  preferential_subsegment_crop=preferential_subsegment_crop,
                                                                                                                                                                                                  preferential_organ_crop=preferential_organ_crop)
            #print('Cropped.', flush=True, file=sys.stderr)'
            if self.debugging:print('Selected organ:', selected_organ_non_canonical, flush=True, file=sys.stderr)
            selected_organ = canonical_organ(selected_organ_non_canonical)
            if self.debugging:print('Selected organ canonical:', selected_organ, flush=True, file=sys.stderr)
            
            if slices_cropped_dict is not None and slices_cropped_dict['slices_used']:
                print(f'!>!>!>!>!>!> Successfully cropped slices for {selected_organ} in {self.img_list[idx]} <!<!<!<!<!<!<!', flush=True, file=sys.stderr)
            
            if not self.save_augmented:
                #this augmentation is online.
                if np.random.random() < 0.3:
                    tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
                if np.random.random() < 0.3:
                    tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
                if np.random.random() < 0.3:
                    std = np.random.random() * 0.2 
                    tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
        
        else:
            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)#.float()
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)#.to(torch.uint8)
            #assert type is int8
            assert tensor_lab.dtype == torch.int16
            assert tensor_img.dtype == torch.float32
            del np_img, np_lab

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape[1:] == tensor_lab.shape[1:]
        
        #if the item is from the UFO dataset, we convert the labels to the atlas format--negative classes are set to 0, unknown classes are SET TO NAN.
        if self.img_list[idx] in self.UFO_paths:
            #convert to atlas format
            tensor_lab, unk_channels, unk_channels_tensor = self.assign_labels(tensor_lab,idx)#send to atlas label space
            tumor_volumes_in_crop,tumor_diameters,tumor_attenuation, tumor_slices, tumor_malignancy=self.estimate_tumor_volume(idx,tumor_segment_crop=selected_tumor)
            #print('Tumor volume estimated from report:', tumor_volumes_in_crop, flush=True, file=sys.stderr)
            chosen_segment_mask=self.get_chosen_segment_mask(tensor_lab, selected_tumor)
            if torch.tensor(tumor_volumes_in_crop).float().sum() == 0 and chosen_segment_mask.sum()>0:
                raise ValueError(f'{self.img_list[idx]}tumor_volumes_report should not be all zeros if chosen_segment_mask is not all zeros')
            # Mirrors the loss-side assertion at losses_foundation.py:4108-4110
            # AND the same check we added in load_augmented_data (~line 3982).
            # When the augmented loader's retries exhaust, getitem_real falls
            # through to this non-augmented path (line 2224+), so without this
            # check the bad sample reaches the loss and crashes the run.
            # Raising here triggers the outer __getitem__ retry (line 2009+).
            if unk_channels_tensor.sum() == 0 and chosen_segment_mask.sum() > 0:
                raise ValueError(f'{self.img_list[idx]} unk_channels should not be all zeros if chosen_segment_mask is not all zeros')
            tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.measure_tumors_mask_zeros(tensor_lab)#just zeros
            malignancy_per_voxel = torch.tensor([np.nan]*len(self.classes)).float().to(tensor_img.device)
        else:
            unk_channels_tensor = torch.zeros(tensor_lab.shape).type_as(tensor_lab)
            unk_channels = {}
            tumor_volumes_in_crop = [0,0,0,0,0,0,0,0,0,0]
            tumor_diameters = torch.zeros((10,3)).float()
            #tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel : we got it in crop, before rotation
            tumor_attenuation = [0,0,0,0,0,0,0,0,0,0]
            tumor_attenuation = torch.tensor(tumor_attenuation).float().type_as(tensor_img)
            chosen_segment_mask = torch.zeros(tensor_lab.shape).type_as(tensor_lab)#it is important to define this as 0--or it will cause loss problems!
            tumor_slices = torch.tensor([[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan]]).float().type_as(tensor_img)
            tumor_malignancy = tumor_slices.clone()
            malignancy_per_voxel = self.get_malignancy_annotated_per_voxel(idx,tensor_lab)
            
        dta={'tumor_in_crop':selected_tumor,
             'unknown_per_voxel':unk_channels,
             'subsegment_cropped_in': subseg_cropped_in,
             'organ_cropped_in': organ_cropped_in,}
        

        if self.save_augmented:
            slices_mask = slices_cropped_dict['slices_mask'] if slices_cropped_dict is not None else None
            binary_slices_mask = slices_cropped_dict['binary_slices_mask'] if slices_cropped_dict is not None else None
            max_slice = slices_cropped_dict['max_slice'] if (slices_cropped_dict is not None) else None  
            tumor_df_json = slices_cropped_dict['tumor_df'].replace({np.nan: None}).to_json(orient="records") if slices_cropped_dict is not None else None
            slices_used = slices_cropped_dict['slices_used'] if slices_cropped_dict is not None else False
            saved_dict = self.save(tensor_img, tensor_lab, idx, tumor_dict, dta, unk_channels_tensor=unk_channels_tensor, 
                      tumor_volumes_in_crop=tumor_volumes_in_crop,chosen_segment_mask=chosen_segment_mask,tumor_diameters=tumor_diameters,
                      selected_organ=selected_organ, tumor_attenuation=tumor_attenuation,reason=reason,
                      tumor_volumes_in_crop_per_voxel=tumor_volumes_in_crop_per_voxel,
                      tumor_diameters_per_voxel=tumor_diameters_per_voxel,
                      slices_mask=slices_mask,
                      binary_slices_mask=binary_slices_mask,
                      tumor_df = tumor_df_json,
                      max_slice=max_slice,
                      selected_organ_non_canonical=selected_organ_non_canonical,
                      slices_used = slices_used,
                      original_spacing = original_spacing,
                      tumor_slices = tumor_slices,
                      tumor_malignancy = tumor_malignancy,
                      malignancy_per_voxel = malignancy_per_voxel)
        ##print('Time to augment data:', time.time() - aug_start, flush=True, file=sys.stderr)
        
        if self.args.load_clip:
            embedding = self.load_clip(idx, selected_organ)
        
        if self.mode == 'train':
            if self.class_proportions: 
                sample_weights = get_sample_weight(tensor_lab,self.class_proportions,self.classes, balancer=self.cropper if self.balancing_crops else None) 
            else:
                sample_weights = torch.ones_like(tensor_lab)
            ##print('Shapes:', tensor_img.shape, tensor_lab.shape)
            
            
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float().clone(), torch.from_numpy(tensor_lab).float().clone()
                return {"image":           tensor_img.clone().float(),
                        "label":           tensor_lab.clone().float(),
                        "unk_channels":    unk_channels_tensor.clone().float(),
                        "volumes":         torch.tensor(tumor_volumes_in_crop).float().clone(),
                        "mask":            chosen_segment_mask.float().clone(),
                        "diameters":       tumor_diameters.float().clone(),}
            
            if slices_cropped_dict is None:
                #we cannot return none, it causes errors in the batch
                #check if annotated per voxel:
                tumor_df='[]'
                slices_cropped_dict={'slices_mask': torch.zeros_like(tensor_img).float(), 
                                     'tumor_df':tumor_df, #this will be needed in the ball loss, as we want to assign each tumor size to one slice only
                                     'binary_slices_mask': torch.ones_like(tensor_img).float(), #we can use this in the volume loss, we first dilate the organ, then we multiply by this mask
                                     'max_slice': 0,
                                     'slices_used': False,
                                     "selected_organ_non_canonical": selected_organ_non_canonical,
                                     "z_spacing": original_spacing,
                                     }
            else:
                slices_cropped_dict['slices_mask']= slices_cropped_dict['slices_mask'].float()
                slices_cropped_dict["selected_organ_non_canonical"] = selected_organ_non_canonical
                
            if not isinstance(slices_cropped_dict['tumor_df'], str):
                slices_cropped_dict['tumor_df'] =  slices_cropped_dict['tumor_df'].replace({np.nan: None}).to_json(orient="records") #for collation
            else:
                if slices_cropped_dict['tumor_df']!= '[]':
                    raise ValueError(f'Expected tumor_df to be a dataframe or a [], but got {type(slices_cropped_dict["tumor_df"])}. Why?')
            
            self.SanityAssertOutput(tensor_lab, unk_channels_tensor,torch.tensor(tumor_volumes_in_crop).float(),chosen_segment_mask.float(),
                                        binary_slices_mask=slices_cropped_dict['binary_slices_mask'],slices_mask=slices_cropped_dict['slices_mask'])
            
            if self.img_list[idx] in self.UFO_paths:   
                if self.debugging:print(f'sizes_slices is: {tumor_slices}')
            if slices_cropped_dict['slices_used']:
                slices_cropped_dict['binary_slices_mask'] = slices_cropped_dict['binary_slices_mask'].float()
            else:
                slices_cropped_dict['binary_slices_mask'] = torch.ones_like(tensor_img).float()
            
            if not (slices_cropped_dict and slices_cropped_dict['slices_used'] and slices_cropped_dict['max_slice'] > 0):
                # sizes_slices is (10, 2): [size_mm, slice_idx]; clear the slice column
                tumor_slices[:, 1] = float('nan')
                
            contrast = self.get_contrast_phase(idx)
            age,sex = self.get_age_and_sex(idx)
            date = self.get_date(idx)
            patient_id = self.get_patient_id(idx)
            #print(f'Contrast phase for {self.img_list[idx]}: {contrast}', flush=True, file=sys.stderr)
            subsegment_cropped_in = subseg_cropped_in if subseg_cropped_in is not None else 'random'
            
            subsegment_cropped_in_no_hash = subsegment_cropped_in
            if not self.load_per_patient:
                #hash patient ids and subsegment_cropped_in to help all_gather downstream
                patient_id = string_to_tensor(patient_id, salt="pid_v1", allow_numeric=True)
                subsegment_cropped_in = string_to_tensor(subsegment_cropped_in, salt="organ_v1", allow_numeric=False)
                
            retur = {"image":          tensor_img.clone().float(),
                    "label":           tensor_lab.clone().to(torch.int16),
                    "unk_channels":    unk_channels_tensor.clone().to(torch.int16),
                    "volumes":         torch.tensor(tumor_volumes_in_crop).float().clone(),
                    "mask":            chosen_segment_mask.float().clone(),
                    "diameters":       tumor_diameters.clone().float(),
                    "weights":         sample_weights.float(),
                    "attenuation":     tumor_attenuation.float(),
                    "voumes_per_voxel": {k: torch.tensor(v, dtype=torch.float32) for k, v in tumor_volumes_in_crop_per_voxel.items()},
                    "diameters_per_voxel": {k: torch.tensor(v, dtype=torch.float32) for k, v in tumor_diameters_per_voxel.items()},
                    "name": self.img_list[idx],
                    "slices_cropped_dict" : slices_cropped_dict,
                    "sizes_slices": tumor_slices.clone().float(),
                    "sizes_malignancy": tumor_malignancy.clone().float(),
                    "malignancy_per_voxel": malignancy_per_voxel,
                    "contrast": contrast,
                    "annotated_per_voxel": self.tumor_annotated_seg[self.img_list[idx]],
                    "age": age, 
                    "sex": sex,
                    'tumor_in_crop': dta['tumor_in_crop'] if dta['tumor_in_crop'] is not None else 'random',#we cannot return None
                    'date': date,
                    'patient_id': patient_id,
                    'subsegment_cropped_in': subsegment_cropped_in,
                    'organ_cropped_in': organ_cropped_in if organ_cropped_in is not None else 'random',
                    'organ_cropped_cannon': canonical_organ(subsegment_cropped_in_no_hash),
                    }
            
            retur['tumor_info_input'] = self.get_input_info_report(retur, idx)
            
            if self.UFO_only:
                sanity_assert_no_lesion_mask(retur['label'], self.classes, self.img_list[idx])
                
            #print malignancy_per_voxel
            print()
            print()
            print(f'Sample is {self.img_list[idx]}, malignancy_per_voxel is {malignancy_per_voxel}', flush=True, file=sys.stderr)
            print()
            print()
            print()
            print()
            print(f'Sample is {self.img_list[idx]}, sizes_malignancy is {tumor_malignancy}', flush=True, file=sys.stderr)
            print()
            print()
            
            assert not torch.isnan(tensor_img.clone().float()).any(), f'Input is nan for {self.img_list[idx]}'
            
            #assertion: if not use_all_data, then we cannot have negative tumor volumes
            if not self.args.use_all_data:
                volumes = retur['volumes']
                if not (volumes.sum(dim=-1)>=0).all():
                    raise ValueError(f'Negative tumor volumes found in {self.img_list[idx]} but use_all_data is False. Volumes are {volumes}, sample is: {self.img_list[idx]}')
            
            clone_tensors_inplace(retur)

            if self.args.load_clip:
                retur["clip_embedding"] = embedding
            if debug_dict:
                return saved_dict
            else:
                return retur
        else:
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float().clone(), torch.from_numpy(tensor_lab).float().clone()
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])
        
        
        
    def load_clip(self, idx, selected_organ):
        """
        Loads the CLIP embedding for the given organ from the specified sample.
        """
        if not self.balancing_crops:
            raise ValueError('Balancing crops is not enabled, but load_clip is set to True.')
        selected_organ = canonical_organ(selected_organ)
        file = selected_organ.replace('adrenal', 'adrenal_glands').replace('random','full_report').replace('gall_bladder','gallbladder')
        source = self.args.clip_source
        id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
        pth = os.path.join(source, id, file+'.npy')
        if not os.path.exists(pth):
            pth = os.path.join(source, id, 'full_report.npy')
        if not os.path.exists(pth):
            raise ValueError(f'Missing clip embedding at {pth}')
        clip = np.load(pth)
        clip = torch.from_numpy(clip).float()
        return clip
        
        
    def clean_subseg_list(self, tumor_segments):
        #split tumor segments that have /
        tmp=[]
        for segment in tumor_segments:
            if pd.isna(segment) or segment == 'u':
                continue
            else:
                sublist=segment.split(' / ')
                if sublist not in tmp:
                    tmp.append(sublist)
        tumor_segments = tmp
        tumor_segments_flat = list(set([item for sublist in tmp for item in sublist]))
        return tumor_segments, tumor_segments_flat
    
    def get_tumor_segment_labels(self, idx):
        """
        This function reads the LLM output for a given report, and its most importat outputs are subseg_with_only_known_sizes and organs_with_only_known_sizes_n_segments.
        These outputs represent organ/organ subsegments that contain tumors but do not contain tumors with unknown size.  
        """
        tumors=self.read_report(idx)
        if tumors is None:
            #no tumor, just do random crop
            retur = {'tumor_segments':[],
                    'tumor_segments_flat':[],
                    'tumor_organs':[],
                    'organs_with_unk_tumor_segment':[],
                    'organs_with_unk_tumor_size':[],
                    'organs_with_only_known_sizes_n_segments':[],
                    'subseg_with_only_known_sizes':[],
                    'subseg_with_unk_tumor_size':[],
                    'subsegs_in_organs_with_unk':[],
                    'prohibit_crop':[],
                    }
            #print('No tumor found for:', self.img_list[idx], flush=True, file=sys.stderr)
            return retur,tumors
        else:
            #tumor is present
            #drops 'no lesion' rows and organs not in our classes
            tumors = tumors[tumors['Standardized Organ'].isin(self.tumor_classes)]
            
            tumor_segments = tumors['Standardized Location'].tolist()
            tumor_sizes = tumors['Tumor Size (mm)'].tolist()
            tumor_organs = tumors['Standardized Organ'].tolist()
            prohibit_crop = tumors[tumors['prohibit crop']==True]['Standardized Organ'].tolist()
            
            #add organ names to segments
            if self.debugging:print('Tumor segments before adding organ names:', tumor_segments, flush=True, file=sys.stderr)
            if self.debugging:print('Tumor organs before adding organ names:', tumor_organs, flush=True, file=sys.stderr)
            tmp=[]
            for i,s in enumerate(tumor_segments,0):
                if pd.isna(s) or s == 'u':
                    tmp.append(s)
                else:
                    if ' / ' not in s:
                        tmp.append(tumor_organs[i]+'_'+s)
                    else:
                        #add the organ name to each sub-segment. Their names are separated by ' / '
                        t=s.split(' / ')
                        t = [tumor_organs[i]+'_'+seg for seg in t]
                        #join back into one string
                        t = ' / '.join(t)
                        tmp.append(t)
            tumor_segments = tmp
            if self.debugging:print('Tumor segments after adding organ names:', tumor_segments, flush=True, file=sys.stderr)


            #check which organs have tumors with unknown size or segment
            organs_with_unk_tumor_segment = []
            organs_with_unk_tumor_size = []
            #and which subsegments have unknown size
            subseg_with_unk_tumor_size = []
            for i in list(range(len(tumor_organs))):
                if pd.isna(tumor_sizes[i]) or tumor_sizes[i] == 'u' or tumor_sizes[i] == 'multiple' or not bool(re.search(r'\d', str(tumor_sizes[i]))):
                    organs_with_unk_tumor_size.append(tumor_organs[i])
                    subseg_with_unk_tumor_size.append(tumor_segments[i])
                if pd.isna(tumor_segments[i]) or tumor_segments[i] == 'u' or tumor_sizes[i] == 'multiple' or not bool(re.search(r'\d', str(tumor_sizes[i]))):
                    organs_with_unk_tumor_segment.append(tumor_organs[i])

            #check which segments are in an organ with some unknown tumor size or segment
            subsegs_in_organs_with_unk = []
            for i in list(range(len(tumor_organs))):
                #check if the organ is not in the list of organs with unknown tumor segment
                if tumor_organs[i] in organs_with_unk_tumor_segment or tumor_organs[i] in organs_with_unk_tumor_size:
                    subsegs_in_organs_with_unk.append(tumor_segments[i])

            

            tumor_segments, tumor_segments_flat = self.clean_subseg_list(tumor_segments)
            subseg_with_unk_tumor_size, subseg_with_unk_tumor_size_flat = self.clean_subseg_list(subseg_with_unk_tumor_size)
            subsegs_in_organs_with_unk, subsegs_in_organs_with_unk_flat = self.clean_subseg_list(subsegs_in_organs_with_unk)

            tumor_organs = list(set(organ for organ in tumor_organs if not pd.isna(organ) and organ != 'u'))
            organs_with_unk_tumor_segment = list(set(organ for organ in organs_with_unk_tumor_segment if not pd.isna(organ) and organ != 'u'))
            organs_with_unk_tumor_size = list(set(organ for organ in organs_with_unk_tumor_size if not pd.isna(organ) and organ != 'u'))

            #subsegments with only known sizes
            subseg_with_only_known_sizes = list(set(tumor_segments_flat) - set(subseg_with_unk_tumor_size_flat) - set(subsegs_in_organs_with_unk_flat))
            #organs with only known sizes and locations of tumors
            organs_with_only_known_sizes_n_segments = list(set(tumor_organs) - set(organs_with_unk_tumor_segment) - set(organs_with_unk_tumor_size))
            organs_with_only_known_sizes = list(set(tumor_organs) - set(organs_with_unk_tumor_size))

            #for subseg_with_only_known_sizes, you must check tumor_segments, and consider segments that come in pairs
            #check if some sub-segment is in more than one item in the list, if so, merge the items
            tmp=[]
            for segment in subseg_with_only_known_sizes:
                #get all items that contain the segment in the list tumor_segments
                items = [item for item in tumor_segments if segment in item]
                #flatten
                items = list(set([item for sublist in items for item in sublist]))
                #items represent a list of sub-segments that share tumors with segment
                #check if any of them is in the list of prohibted segments
                if any(item in subseg_with_unk_tumor_size_flat for item in items) or \
                   any(item in subsegs_in_organs_with_unk_flat for item in items):
                    continue
                else:
                    tmp.append(items)
            subseg_with_only_known_sizes=tmp
                

            #create a big dict with the variables here
            retur = {'tumor_segments':tumor_segments,
                    'tumor_segments_flat':tumor_segments_flat,
                    'tumor_organs':tumor_organs,
                    'organs_with_unk_tumor_segment':organs_with_unk_tumor_segment,
                    'organs_with_unk_tumor_size':organs_with_unk_tumor_size,
                    'organs_with_only_known_sizes_n_segments':organs_with_only_known_sizes_n_segments,
                    'organs_with_only_known_sizes':organs_with_only_known_sizes,
                    'subseg_with_only_known_sizes':subseg_with_only_known_sizes,
                    'subseg_with_unk_tumor_size':subseg_with_unk_tumor_size,
                    'subsegs_in_organs_with_unk':subsegs_in_organs_with_unk,
                    'prohibit_crop':prohibit_crop,
                    }
            #print()
            #print('Retur:', retur, flush=True, file=sys.stderr)
            #raise ValueError('You must change the handling of this function output everywhere it is used')
            #print()
            #print('Tumor Dict:', tumors[['Standardized Location','Tumor Size (mm)','Standardized Organ','no lesion']])
            #print()
            #print('subseg_with_only_known_sizes:', retur['subseg_with_only_known_sizes'], flush=True, file=sys.stderr)
            #print()
            #print('organs_with_only_known_sizes_n_segments:', retur['organs_with_only_known_sizes_n_segments'], flush=True, file=sys.stderr)
            #print()
            #print('organs_with_only_known_sizes:',retur['organs_with_only_known_sizes'], flush=True, file=sys.stderr)
            #print()
            #check if tumor_organs is not nan
            
            #if isinstance(retur['tumor_organs'],str):#if nan it is a normal case
            #    #print('XXXXXXXX Tumor Found for:', self.img_list[idx],f'tumor is: {tumors[["Standardized Location","Tumor Size (mm)","Standardized Organ"]]}', flush=True, file=sys.stderr)
            return retur,tumors
        

    
    def get_slices_from_report(self,idx,segments):
        """
        Here, we get from the report organs_with_all_sizes_and_slices_known, organs_with_all_sizes_and_some_slices_known,
        and we also return the dataframe with tumors. This dataframe can be used to determine slices and respective tumor sizes.
        We return None in case there is no tumor, if the series does not match the report, or if there is no slice.
        """
        tumor_df = self.read_report(idx)
        if tumor_df is None:
            return None
        if (not coerce_to_bool(tumor_df['series matches report'].iloc[0])) or coerce_to_bool(tumor_df['impossible slice'].iloc[0]):
            return None

        base_orgs = list(set(segments.get('organs_with_only_known_sizes', [])))
        organs_all = []
        organs_some = []

        for org in base_orgs:
            slc = tumor_df.loc[tumor_df['Standardized Organ'] == org, 'Image']
            # Convert to numeric; non-numeric becomes NaN
            s_num = pd.to_numeric(slc, errors='coerce')
            if len(s_num) == 0:
                continue
            if s_num.notna().all():
                organs_all.append(org)
            elif s_num.notna().any():
                organs_some.append(org)

        if not organs_all and not organs_some:
            return None

        return {
            'organs_with_all_sizes_and_slices_known': organs_all,
            'organs_with_all_sizes_and_some_slices_known': organs_some,
            'slices_found': ((len(organs_all)+len(organs_some))>0)
        }
            
            
    def get_random_tumor_seg_mask(self, tensor_lab, tumor_segment, exclude=None,classes=None):
        #print('get_random_tumor_seg_mask - Selected tumor segment:', tumor_segment, flush=True, file=sys.stderr)
        #get the mask for a given segment/organ or segment list
        
        
        if not isinstance(tumor_segment, list):
            tumor_segment = [tumor_segment]

        if len(tumor_segment)==1 and tumor_segment[0] == 'pancreas':
            if 'pancreas' not in self.classes_UFO:
                #pancreas is a special case, we have pancreas labels but they are not in the atlas format
                #we assign all pancreas labels to 1
                tumor_segment = ['pancreas head','pancreas body','pancreas tail']
            else:
                tumor_segment = ['pancreas','pancreas head','pancreas body','pancreas tail']
        if len(tumor_segment)==1 and tumor_segment[0] == 'liver':
            if 'liver' not in self.classes_UFO:
                #liver is a special case, we have liver labels but they are not in the atlas format
                #we assign all liver labels to 1
                tumor_segment = ['liver segment 1','liver segment 2','liver segment 3','liver segment 4',
                                'liver segment 5','liver segment 6','liver segment 7','liver segment 8']
            else:
                tumor_segment = ['liver','liver segment 1','liver segment 2','liver segment 3','liver segment 4',
                                'liver segment 5','liver segment 6','liver segment 7','liver segment 8']
        
        #get the labels of the tumor segment
        segment_labels=[seg.replace(' ','_').replace('gallbladder','gall_bladder').replace('adrenal gland','adrenal_gland').replace('uterus','prostate') for seg in tumor_segment]
        for i,seg in enumerate(segment_labels,0):
            if seg=='tail':
                segment_labels[i]='pancreas_tail'
            elif seg=='body':
                segment_labels[i]='pancreas_body'
            elif seg=='head':
                segment_labels[i]='pancreas_head'
            elif seg.startswith('segment'):
                segment_labels[i]='liver_'+seg
                
        #remove subsegments that are not in the classes_UFO
        if not self._has_liver_subsegs_UFO:
            tmp=[]
            for seg in segment_labels:
                if ('liver' in seg) and ('segment' in seg):
                    tmp.append('liver')
                else:
                    tmp.append(seg)
            segment_labels=list(set(tmp))
        if not self._has_pancreas_subsegs_UFO:
            tmp=[]
            for seg in segment_labels:
                if ('pancreas' in seg) and (('head' in seg) or ('body' in seg) or ('tail' in seg)):
                    tmp.append('pancreas')
                else:
                    tmp.append(seg)
            segment_labels=list(set(tmp))

        #print('Segment labels are:', segment_labels, flush=True, file=sys.stderr)
        if classes is None:
            raise ValueError('Classes is mandatory')
        for label in segment_labels:
            if label not in classes:
                raise ValueError('Label %s not in classes'%label)

        if len(tensor_lab.shape) == 4:
            tensor_lab = tensor_lab.unsqueeze(0)
        assert len(tensor_lab.shape) == 5, f'Label tensor must have 5 dimensions, but got {len(tensor_lab.shape)} dimensions and shape {tensor_lab.shape}'

        tumor_segment_labels = []
        for i,clss in enumerate(classes,0):
            if clss in segment_labels:
                tumor_segment_labels.append(i)

        #print('Label indices of tumor segment are:', tumor_segment_labels, flush=True, file=sys.stderr)
        
        #get the tumor segment mask
        tumor_segment_mask=[]
        #print('The shape of tensor_lab is:', tensor_lab.shape, flush=True, file=sys.stderr)
        for i in range(tensor_lab.shape[1]):
            if i in tumor_segment_labels:
                tumor_segment_mask.append(tensor_lab[:,i])
        if len(tumor_segment_mask) == 0:
            print('No tumor segment mask found for segment:', tumor_segment, 'in sample:', self.current_sample, flush=True, file=sys.stderr)
            print(f'tumor_segment_labels are: {tumor_segment_labels}, classes are: {classes}', flush=True, file=sys.stderr)
            raise ValueError('No tumor segment mask found for segment %s in sample %s'%(tumor_segment,self.current_sample))
        tumor_segment_mask=torch.stack(tumor_segment_mask,axis=0)
        tumor_segment_mask=tumor_segment_mask.sum(0)
        #binarize
        tumor_segment_mask[tumor_segment_mask>0]=1
        #assert tumor_segment_mask.sum().item()!=0.0, f'problem in case {self.current_sample}, tumor segment mask is empty, crop is in {tumor_segment}'
        return tumor_segment_mask

    def get_chosen_segment_mask(self, tensor_lab, tumor_segment):
        if tumor_segment == 'random' or ('random' in tumor_segment):
            return torch.zeros_like(tensor_lab).type_as(tensor_lab)
        
        if self.debugging:print('Chosen segment (getting its mask):', tumor_segment, flush=True, file=sys.stderr)
        segment_mask = self.get_random_tumor_seg_mask(tensor_lab, tumor_segment,classes=self.classes).squeeze(0)
        #apply it to the lesion classes
        segment_mask_lesion_ch = []
        assert segment_mask.sum().item()!=0.0, f'problem in case {self.current_sample}, segment_mask is empty, crop is in {tumor_segment}'
        
        tumor_segment = [tumor_segment] if not isinstance(tumor_segment, list) else tumor_segment
        
        lesion_like_names = []
        for s in tumor_segment:
            lesion_like_name = s.replace(' ','_').replace('_right','').replace('_left','').replace('_gland','').replace('gall_bladder','gallbladder')
            if ('liver' in lesion_like_name) or ('segment' in lesion_like_name):
                lesion_like_name = 'liver'
            elif ('pancrea' in lesion_like_name) or ('head' in lesion_like_name) or ('body' in lesion_like_name) or ('tail' in lesion_like_name):
                lesion_like_name = 'pancreatic'
            lesion_like_name+='_lesion'
            lesion_like_names.append(lesion_like_name)
            if self.debugging:print('Lesion like name:', lesion_like_name, flush=True, file=sys.stderr)
            
        added=False
        if self.debugging:print('Segment is (2):', tumor_segment, flush=True, file=sys.stderr)
        for c in self.classes:
            #print('Classes are:', self.classes, flush=True, file=sys.stderr)
            if 'lesion' in c and c in lesion_like_names:
                segment_mask_lesion_ch.append(segment_mask)
                added=True
                #print('Segment added to class:', c, flush=True, file=sys.stderr)
            else:
                segment_mask_lesion_ch.append(torch.zeros_like(tensor_lab[0]).type_as(tensor_lab))
                #print('zero mask added for:', c, flush=True, file=sys.stderr)
        segment_mask_lesion_ch = torch.stack(segment_mask_lesion_ch,axis=0)
        assert segment_mask_lesion_ch.sum().item()!=0.0, f'problem in case {self.current_sample}, chosen segment mask is empty, crop is in {tumor_segment}'
        #assert only one channel is zero:
        for i in range(segment_mask_lesion_ch.shape[0]):
            # For sample i, lo has shape (num_lesion_channels, ...spatial dimensions...)
            lo = segment_mask_lesion_ch[i]
            # Sum over all dimensions except the channel, regardless of the number of spatial dims.
            lo_sum = lo.sum(dim=(-1,-2,-3))
            # Create a boolean mask for channels with any nonzero value.
            active_mask = lo_sum > 0
            active_count = active_mask.sum().item()
            if active_count > 1:  # If more than one lesion channel is active
                # Prepare the names of the lesion channels that are active.
                active_names = [self.classes[j] for j in range(len(self.classes)) if active_mask[j]]
                raise ValueError(
                    f"Error: For sample index {i}, more than one lesion channel has active elements. "
                    f"Active lesion channels: {active_names}"
                    f"lo.sum(dim=(-1,-2,-3)): {lo.sum(dim=(-1,-2,-3))}"
                )
        assert (segment_mask_lesion_ch.sum((-1,-2,-3))>0).float().sum()<=1, 'Only one channel should be non-zero in the chosen segment mask!'
        if not added:
            raise ValueError('No segment added to lesion class:', c, flush=True, file=sys.stderr)
        return segment_mask_lesion_ch
           
    def random_crop(self, tensor_img, tensor_lab, d, h, w):
        tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d+40, h+40, w+40], mode='random')
        if self.args.aug_device == 'gpu':
            tensor_img = tensor_img.cuda(self.args.proc_idx).float()
            tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
        if np.random.random() < 0.4:
            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
        else:
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
        return tensor_img, tensor_lab
    
    def random_crop_on_tumor(self, tensor_img, tensor_lab, d, h, w, tumor_case=None,
                             tumor_prob=None, foreground_prob=None, background_prob=None,
                             ufo=False,return_sizes = False):
        if ufo:
            clss = self.classes_UFO
        else:
            clss = self.classes
            
        
        
        if self.balancing_crops:
            if ufo:
                cropper = self.cropper_UFO
                lesion_classes = [] # for ufo, our labels here do not contain lesion classes, it only has the ufo classes (organs)
            else:
                cropper = self.cropper
                lesion_classes = self.lesion_classes
        if ufo:
            lesion_classes = [] # for ufo, our labels here do not contain lesion classes, it only has the ufo classes (organs)
        else:
            lesion_classes = self.lesion_classes
        if (np.random.random() < 0.4) and not (ufo and self.balancing_crops):
            #we do not rotate here for ufo, we rotate inside cropper
            #crop large, then rotate and crop small
            assert len(tensor_lab.shape) == 5
            if tumor_case is None:
                tumor_case = tensor_lab[:,lesion_classes].sum()>0
            if self.debugging and tumor_case and (tumor_prob is None) and (foreground_prob is None) and (background_prob is None):
                tumor_prob=1
                foreground_prob=0
                background_prob=0
            if self.balancing_crops:
                tensor_img, tensor_lab, crop_organ = cropper(tensor_img, tensor_lab, d+40, h+40, w+40,tumor_case,
                                                      tumor_prob=tumor_prob, foreground_prob=foreground_prob, 
                                                      background_prob=background_prob,return_crop_organ=True,
                                                      report_anno=ufo)
            else:
                tensor_img, tensor_lab, crop_organ = augmentation.random_crop_on_tumor(tensor_img, tensor_lab, lesion_classes, 
                                                                           d+40, h+40, w+40,tumor_case,
                                                                           tumor_prob=tumor_prob, 
                                                                           foreground_prob=foreground_prob, 
                                                                           background_prob=background_prob,
                                                                           return_crop_organ=True,
                                                                           class_names=clss)
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
            if return_sizes:
                if not ufo:
                    #measure tumors before rotation
                    tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.measure_tumors_mask(tensor_lab.squeeze(0))
                else:
                    tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.measure_tumors_mask_zeros(tensor_lab.squeeze(0))
            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
            ##print('Shape of tensor after rotate tumor crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

        else:
            #just crop on tumor
            assert len(tensor_lab.shape) == 5
            if tumor_case is None:
                tumor_case = tensor_lab[:,lesion_classes].sum()>0
            if self.debugging and tumor_case and (tumor_prob is None) and (foreground_prob is None) and (background_prob is None):
                tumor_prob=1
                foreground_prob=0
                background_prob=0
            if self.balancing_crops:
                tensor_img, tensor_lab, crop_organ = cropper(tensor_img, tensor_lab, d, h, w,tumor_case,
                                                      tumor_prob=tumor_prob, foreground_prob=foreground_prob,
                                                      background_prob=background_prob, return_crop_organ=True,
                                                      report_anno=ufo)
            else:
                tensor_img, tensor_lab, crop_organ = augmentation.random_crop_on_tumor(tensor_img, tensor_lab, 
                                                                           lesion_classes, d, h, w,tumor_case,
                                                                           tumor_prob=tumor_prob, 
                                                                           foreground_prob=foreground_prob,
                                                                           background_prob=background_prob,
                                                                           return_crop_organ=True,
                                                                           class_names=clss)
                
            if return_sizes:
                if not ufo:
                    #measure tumors before rotation
                    tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.measure_tumors_mask(tensor_lab.squeeze(0))
                else:
                    tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.measure_tumors_mask_zeros(tensor_lab.squeeze(0))
        ##print('Crop on tumor successful for:', self.img_list[idx], flush=True, file=sys.stderr)
        crop_organ = tumor_to_organ(crop_organ)#when we crop on tumor per-voxel, this is returned as a tumor class, like duodenum_lesion
        #find the actual subsegment we cropped on
        crop_subseg = self.find_cropped_subsegment(crop_organ, tensor_lab, tumor_per_voxel=((not ufo) and tumor_case), classes=clss)

        if not return_sizes:
            return tensor_img, tensor_lab, crop_organ, crop_subseg
        else:
            return tensor_img, tensor_lab, crop_organ, crop_subseg, tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel
   
    def max_diameter_mm(self,size):
        if 'x' not in size:
            #single diameter provided. Use ball.
            diameter=float(size)
        else:
            #2 or 3 diameterts, use ellipsoid
            sizes=size.split(' x ')
            sizes=[float(s) for s in sizes]
            diameter=max(sizes)
            
        return int(round(diameter))
    
    def binarize_membership(self, x: torch.Tensor, values, *, out_dtype=None):
        xi   = x.round().to(torch.int32)                   # enforce ints
        vals = torch.as_tensor(values, device=xi.device, dtype=xi.dtype)
        mask = torch.isin(xi, vals)
        return mask.to(out_dtype or x.dtype)

        
    def cut_tumor_segment_with_slices(self, tumor_segment_mask, slices_mask, chosen_organ, metadata_df, idx, radius_tolerance=2):
        """
        We use the tumor slices for the chosen organ, 
        with z-axis mirroring (we are unsure if the radiologist counted slices from top to bottom or bottom to top),
        to "cut" the segment mask. Also consider tumor sizes when cutting. 
        To crop the slice mask itself, we use a trick: we just concatenate it to the labels, and separate afterwards.
        """
        #raise ValueError('In save, you must save the value of the maximum slice, so you can do mirroring in the ball loss when loading augmented data.')
        #raise ValueError('Precision in the ball conv: the binary_slices_mask must have slices with diameter (use radius in construction), tumor_segment_mask can be larger for cropping')
        #raise ValueError('The binary erosion or connected componnent can break masks with slice. Maybe just return binary_slices_mask here, and apply it inside augmentation, AFTER these denoising operations. Check if you will need to apply it at many places.')
        #Get the slices for the chosen organ
        #slices = metadata_df[metadata_df['Standardized Organ'] == chosen_organ]
        slices = df_filter_by_canonical_organ(metadata_df, chosen_organ)
        mtch =  slices['series matches report'].to_list()
        if any([(isinstance(x,bool) and x==False) for x in mtch]):
            print(f'Some tumors have wrong series!', flush=True, file=sys.stderr)
            return None
        
        imp = slices['impossible slice'].to_list()
        if any([(isinstance(x,bool) and x==True) for x in imp]):
            print(f'Some tumors have impossible slice!', flush=True, file=sys.stderr)
            return None
        
        #drop rows with non-numeric values in 'Image' column, keep as a df
        # Convert to numeric, invalid parsing -> NaN
        slices["Image"] = pd.to_numeric(slices["Image"], errors="coerce")
        # Drop rows where Image is NaN (non-numeric or missing)
        len_before = len(slices)
        slices = slices.dropna(subset=["Image"])
        len_after = len(slices)
        
        if len(slices) == 0:
            print(f'No slices found for organ {chosen_organ} in metadata_df', flush=True, file=sys.stderr)
            return None
        
        if len_before != len_after:
            print(f'The tumor organ, {chosen_organ}, has tumors with unknown slices, we are cropping on the whole organ. Tumor dict: {metadata_df}', flush=True, file=sys.stderr)
            return None

        #mirror slices
        max_slice = int(slices_mask.max().item())
        slices['Mirror Image'] = max_slice - slices['Image']
        #check if any negative values in 'Mirror Image'
        if (slices['Mirror Image'] < 0).any():
            #raise ValueError(f'Negative slice values found in {slices["Mirror Image"]}, this should not happen. Max slice is {max_slice}, and the slices are {slices["Image"]}')
            return None

        #now we need to create a range of allowed slices: each slice in the df, we consider the slice +/- the tumor diameter
        slices["Max Diameter (mm)"] = slices["Tumor Size (mm)"].apply(self.max_diameter_mm).astype("Int64")
        bdmap_id=self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
        original_spacing = self.original_spacing_list[self.original_spacing_list['BDMAP_ID']==bdmap_id]['z_mm'].iloc[0]
        original_spacing = float(original_spacing)
            
        allowed_slices = []
        for _, row in slices.iterrows():
            slice_value = row['Mirror Image']
            diameter = row['Max Diameter (mm)']
            radius = max(round(diameter / 2),1)*radius_tolerance  # at least 1 slice radius
            radius = int(round(radius / original_spacing))  # number of the ORIGINAL slices this radius (in mm) corresponds to
            #create a range of allowed slices
            allowed_slices.extend(range(int(slice_value - radius), int(slice_value + radius + 1)))
            if self.debugging:print(f'Slices allowed mirror: from {int(slice_value - radius)} to {int(slice_value + radius + 1)}, slice is {slice_value}, size is {diameter}, max slice is {max_slice}', flush=True, file=sys.stderr)
            
            slice_value = row['Image']
            allowed_slices.extend(range(int(slice_value - radius), int(slice_value + radius + 1)))
            if self.debugging:print(f'Slices allowed: from {int(slice_value - radius)} to {int(slice_value + radius + 1)}, slice is {slice_value}, size is {diameter}', flush=True, file=sys.stderr)
            
            
        allowed_slices= list(set(allowed_slices))  # remove duplicates
        allowed_slices = [s for s in allowed_slices if (s >= 0 and s <= max_slice)]
        
        #use the allowed slices to cut the tumor segment mask, pick up only voxels whose VALUES (not position) are in allowed_slices
        binary_slices_mask = self.binarize_membership(slices_mask, allowed_slices)
        
        if binary_slices_mask.sum() == 0:
            raise ValueError(f'No slices found for organ {chosen_organ} in metadata_df, after applying tumor sizes. Allowed slices: {allowed_slices}, max slice: {max_slice}', flush=True, file=sys.stderr)
        
        #now we can cut the tumor segment mask with the slices mask
        tumor_segment_mask = tumor_segment_mask * binary_slices_mask
        
        if tumor_segment_mask.sum() == 0:
            print(f'No tumor segment mask found for organ {chosen_organ} after cutting with slices mask. Allowed slices: {allowed_slices}, max slice: {max_slice}', flush=True, file=sys.stderr)
            return None
        
        return tumor_segment_mask, binary_slices_mask
        
        
        
            
        

        
    
    def random_crop_on_segment(self, tensor_img, tensor_lab, tumor_segment_mask, d, h, w,
                               slices_mask=None, chosen_organ=None, metadata_df=None, slices_used=False,
                               idx=None):
        """
        This key function will crop around the segment with tumor. 
        Here, we can also use slices. If so, we use the tumor slices for the chosen organ, 
        with z-axis mirroring (we are unsure if the radiologist counted slices from top to bottom or bottom to top),
        to "cut" the segment mask. Also consider tumor sizes when cutting. 
        To crop the slice mask itself, we use a trick: we just concatenate it to the labels, and separate afterwards.
        """
        
        
        #make a copy of the tumor segment mask, so we do not modify the original one
        tumor_segment_mask_backup = tumor_segment_mask.clone()
        
        foreground_cut_with_slices = False
        max_slice = 0
        if slices_mask is not None and slices_used:
            if len(tensor_lab.shape) == 5:
                concat_dim = 1
            elif len(tensor_lab.shape) == 4:
                concat_dim = 0
                
            max_slice = int(slices_mask.max().item())
        
            #cut the tumor segment mask with slices
            tmp = self.cut_tumor_segment_with_slices(tumor_segment_mask.clone(), slices_mask.clone(), 
                                                     chosen_organ, metadata_df, idx=idx)
            if tmp is not None:
                foreground_cut_with_slices = True
                tumor_segment_mask, binary_slices_mask = tmp
                if tumor_segment_mask.sum() == 0:
                    raise ValueError(f'Tumor segment mask is empty after cutting with slices for organ {chosen_organ} in image {self.img_list[idx]}, this should not happen, we should not use slices if the mask becomes empty, and cut_tumor_segment_with_slices should have returned None')
            else:
                binary_slices_mask = torch.ones_like(tumor_segment_mask, dtype=torch.bool)
                slices_used=False
                tumor_segment_mask = tumor_segment_mask_backup
                if self.debugging:print(f'Slices not used for image {self.img_list[idx]} for organ {chosen_organ}', flush=True, file=sys.stderr)
            
                
            #add to tensor_lab the tumor_segment_mask_backup and slices_mask
            while len(tumor_segment_mask_backup.shape) < len(tensor_lab.shape):
                tumor_segment_mask_backup = tumor_segment_mask_backup.unsqueeze(0)
            while len(slices_mask.shape) < len(tensor_lab.shape):
                slices_mask = slices_mask.unsqueeze(0)
            while len(binary_slices_mask.shape) < len(tensor_lab.shape):
                binary_slices_mask = binary_slices_mask.unsqueeze(0)
            
            tensor_lab = torch.cat((tensor_lab.float(), 
                                    tumor_segment_mask_backup.float(), 
                                    slices_mask.float(),
                                    binary_slices_mask.float()), dim=concat_dim)
        else:
            slices_mask = None
            
                
        
        if np.random.random() < 0.4:
            #crop large, then rotate and crop small
            assert len(tensor_lab.shape) == 5
            #crop large on segment
            uncut_organ_mask = tumor_segment_mask_backup if foreground_cut_with_slices else None
            binary_slices = binary_slices_mask if foreground_cut_with_slices else None
            out = augmentation.crop_foreground_3d(tensor_ct=tensor_img, tensor_lab=tensor_lab, foreground=tumor_segment_mask, crop_size=[d+40, h+40, w+40],
                                                  rand=False, uncut_organ_mask=uncut_organ_mask, binary_slices_mask=binary_slices)
            if isinstance(out, tuple):
                tensor_img, tensor_lab, tumor_segment_mask = out
            else:
                return out
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
            tensor_img, tensor_lab, tumor_segment_mask = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, 
                                                                                                       self.args.scale, self.args.rotate, 
                                                                                                       self.args.translate, foreground=tumor_segment_mask)
            # get uncut_organ_mask and binary_slices_mask from tensor_lab
            if foreground_cut_with_slices:
                if concat_dim == 1:
                    uncut_organ_mask = tensor_lab[:, -3]
                    binary_slices = tensor_lab[:, -1]
                elif concat_dim == 0:
                    uncut_organ_mask = tensor_lab[-3]
                    binary_slices = tensor_lab[-1]
            else:
                uncut_organ_mask = None
                binary_slices = None
            out =  augmentation.crop_foreground_3d(tensor_ct=tensor_img, tensor_lab=tensor_lab, foreground=tumor_segment_mask, crop_size=[d, h, w],
                                                  rand=True, uncut_organ_mask=uncut_organ_mask, binary_slices_mask=binary_slices)
            
        else:
            uncut_organ_mask = tumor_segment_mask_backup if foreground_cut_with_slices else None
            binary_slices = binary_slices_mask if foreground_cut_with_slices else None
            out = augmentation.crop_foreground_3d(tensor_ct=tensor_img, tensor_lab=tensor_lab, foreground=tumor_segment_mask, crop_size=[d, h, w],
                                                  rand=True, uncut_organ_mask=uncut_organ_mask, binary_slices_mask=binary_slices)
        if isinstance(out, tuple):
            tensor_img, tensor_lab, tumor_segment_mask = out
            if tumor_segment_mask.sum()==0:
                raise ValueError('Tumor mask is empty after cropping, this should be impossible.')
            
            if slices_mask is not None:
                #get the uncut_segment_mask and the slices_mask for the tensor_lab
                if concat_dim == 1:
                    binary_slices_mask = tensor_lab[:, -1]
                    slices_mask = tensor_lab[:, -2]
                    tumor_segment_mask_uncut = tensor_lab[:, -3]
                    tensor_lab = tensor_lab[:, :-3]
                elif concat_dim == 0:
                    binary_slices_mask = tensor_lab[-1]
                    slices_mask = tensor_lab[-2]
                    tumor_segment_mask_uncut = tensor_lab[-3]
                    tensor_lab = tensor_lab[:-3]
                    
                bdmap_id=self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
                original_spacing = self.original_spacing_list[self.original_spacing_list['BDMAP_ID']==bdmap_id]['z_mm'].iloc[0]
                original_spacing = float(original_spacing)
                slices_cropped_dict={'slices_mask': slices_mask, 
                                     #'tumor_segment_mask_cut': tumor_segment_mask_cut, #ideally, we do not want to use this directly in our loss, because we want to dilate the organ mask to create resistance to mask errors, but we do not want to dilate the slice ranges
                                     'tumor_df':metadata_df[metadata_df['Standardized Organ'] == chosen_organ], #this will be needed in the ball loss, as we want to assign each tumor size to one slice only
                                     'binary_slices_mask': binary_slices_mask, #we can use this in the volume loss, we first dilate the organ, then we multiply by this mask
                                     'max_slice': max_slice,
                                     'slices_used': slices_used,
                                     'z_spacing': original_spacing #this will be needed in the ball loss, where you will need to cut the organ mask with the slices again. 
                                     }
                
            else:
                slices_cropped_dict=None
                    
                
            
            return (tensor_img, tensor_lab, slices_cropped_dict)
        else:
            return out
            
    def organ_to_subsegment(self,tumor_segment, segments):
        flatten = segments['tumor_segments_flat']
        bilateral = ['kidney','adrenal','lung','femur','breast']
        if 'pancrea' in tumor_segment:
            tumor_segment=[f for f in flatten if 'pancrea' in f]
            if len(tumor_segment)==0:
                tumor_segment=['pancreas']
            #if len(sub_segs)>0:
            #    tumor_segment=random.choice(sub_segs) DON'T, if many subsegs have tumor, we must crop on all of them!
            return tumor_segment
        elif 'liver' in tumor_segment:
            tumor_segment=[f for f in flatten if 'segment' in f]
            if len(tumor_segment)==0:
                tumor_segment=['liver']
            #if len(sub_segs)>0:
            #    tumor_segment=random.choice(sub_segs)
            return tumor_segment
        elif any(b in tumor_segment for b in bilateral):
            sub_segs = []
            for sublist in segments['subseg_with_only_known_sizes']:
                for f in sublist:
                    if not isinstance(f, str):
                        raise ValueError('Subsegment is not a string:', f)
                    if tumor_segment in f:
                        sub_segs.append(f)
            if len(sub_segs)>0:
                tumor_segment=random.choice(sub_segs) #here it is correct, I cannot crop on both kidneys, I must choose one
                return tumor_segment
            else: 
                #print('No subsegment found for:', tumor_segment,'; available subsegments:', segments['subseg_with_only_known_sizes'])
                return None #we cannot crop on kidney, we need to know if it is left or right. None signals a problem and asks for other organ.
        else:
            return tumor_segment #no subsegment, return organ
        
    class segment_options:
        """
        Idea: I want a set of priority. If some organs have only tumors with known slices, I will only choose the target
        organ in these organs. 
        """
        def __init__(self, segments, slices=None, use_all_data=False): 
            #start running priority:
            if slices is not None:
                self.priority = []
                self.priority.append(list(set(slices['organs_with_all_sizes_and_slices_known'])))#firt piority
                self.priority.append(list(set(slices['organs_with_all_sizes_and_some_slices_known'])))#second priority
                self.priority.append(list(set(segments['organs_with_only_known_sizes'])))#third piority
                
                #last priority is organs with unknown sizes
                if use_all_data:
                    self.priority.append(list(set(segments['organs_with_unk_tumor_size'])))
            else:
                #no slice, no priority levels (for now, see TODO)
                self.priority = [list(set(segments['organs_with_only_known_sizes']))]
                
                #last priority is organs with unknown sizes
                if use_all_data:
                    self.priority.append(list(set(segments['organs_with_unk_tumor_size'])))
                
            #remove organs in prohibit_crop
            prohibit_crop = segments['prohibit_crop']
            for i in range(len(self.priority)):
                self.priority[i] = [seg for seg in self.priority[i] if seg not in prohibit_crop]
            self.use_all_data = use_all_data
                
        def get_options(self):
            """
            send the highest non-empty priority list
            """
            for p in self.priority:
                if len(p)>0:
                    return list(p)
            return []
        
        def remove_option(self,to_remove):
            """
            When we could not crop on a given organ, this function allows us to remove it from the priority list.
            """
            for i in range(len(self.priority)):
                self.priority[i] = [seg for seg in self.priority[i] if seg not in [to_remove]]
                
        def __len__(self):
            """
            Returns the number of organs in all priority levels
            """
            flatten=[]
            for p in self.priority:
                flatten.extend(p)
            flatten=list(set(flatten))
            return len(flatten)
        
        def only_unknown_size(self):
            """
            Check if we are in the last tier of the priority list
            """
            if not self.use_all_data:
                return False
            
            non_empty_tiers = [p for p in self.priority if len(p)>0]
            if len(non_empty_tiers) == 1 or len(non_empty_tiers) == 0:
                return True
            else:
                return False
            
    
                
    def find_cropped_subsegment(self, crop_organ, tensor_lab, tumor_per_voxel, classes):
        #given crop_organ and tensor_lab, we need to find what organ sub-segments we cropped on
        #first, does this organ even have subsegments?
        organs_with_subsegs = {
            'liver': [c for c in classes if 'segment' in c],
            'pancreas': [c for c in classes if 'head' in c or 'body' in c or 'tail' in c],
            'kidney': ['kidney_left', 'kidney_right'],
            'adrenal_gland': ['adrenal_gland_left', 'adrenal_gland_right'],
            'femur': ['femur_left', 'femur_right'],
            'lung': ['lung_left', 'lung_right'],
        }
        
        if crop_organ not in organs_with_subsegs:
            return crop_organ
        else:
            sub_segs = organs_with_subsegs[crop_organ]
            if len(sub_segs) == 1:
                return sub_segs[0]
            if tumor_per_voxel:
                #now we need to get the labels for each sub-segment, and check which one intersects with the tumor the most
                tumor_class = augmentation.organ_to_tumor(crop_organ)
                tumor_mask = tensor_lab[classes.index(tumor_class)]
                max_dsc = -1
                for sub_seg in sub_segs:
                    tumor_segment_mask=self.get_random_tumor_seg_mask(tensor_lab, sub_seg,classes=classes) #this should only be used by samples annotated by voxel
                    dsc = dice(tumor_mask, tumor_segment_mask)
                    if dsc > max_dsc:
                        max_dsc = dsc
                        chosen_subseg = sub_seg
            else:
                #no tumor mask, just find the subsegment with largest volume in the crop
                max_vol= -1
                for sub_seg in sub_segs:
                    tumor_segment_mask=self.get_random_tumor_seg_mask(tensor_lab, sub_seg,classes=classes)
                    vol = tumor_segment_mask.sum().item()
                    if vol > max_vol:
                        max_vol = vol
                        chosen_subseg = sub_seg
            return chosen_subseg
        
    def _norm_organ_name(self, s: str) -> str:
        s = str(s).strip().lower()
        s = s.replace(" ", "_").replace("-", "_")
        s = re.sub(r"_+", "_", s).strip("_")
        s = s.replace("gallbladder", "gall_bladder")
        s = s.replace("adrenal_gland", "adrenal")  # covers both "adrenal gland" and already-underscored
        return s

    def organ_is_in(self, organ1, organ_list):
        o1 = self._norm_organ_name(organ1)
        norm_list = {self._norm_organ_name(o) for o in organ_list}
        return o1 in norm_list
        
    def crop(self, tensor_img, tensor_lab, idx, d, h, w, slices_mask=None, slices_dict=None,
             preferential_subsegment_crop=None, preferential_organ_crop=None):
        #preferential_subsegment_crop is in the same format as tumor_segment
        if (preferential_organ_crop is not None) and (preferential_organ_crop != 'random'): #if random, we ignore the preferential organ crop
            skip = False
            bilateral = ['kidney','adrenal','lung','femur','breast']
            if any([b in preferential_organ_crop for b in bilateral]):
                if (preferential_subsegment_crop == 'random') and not (('left' in preferential_organ_crop) or ('right' in preferential_organ_crop)):
                    skip = True 
                if (preferential_subsegment_crop != 'random') and not (('left' in preferential_subsegment_crop) or ('right' in preferential_subsegment_crop)):
                    skip = True
            if not skip: #we cannot determine left or right in this case, skip
                if preferential_subsegment_crop == 'random': #some branches output subsegment as random, bug organ is not random so we can infer subsegment
                    preferential_subsegment_crop = preferential_organ_crop
                return_9 = None
                if self.tumor_annotated_seg[self.img_list[idx]]:
                    tumor_segment_mask = self.get_random_tumor_seg_mask(tensor_lab, preferential_subsegment_crop, classes=self.classes)
                    tumor_dict = None
                else:
                    tumor_segment_mask = self.get_random_tumor_seg_mask(tensor_lab, preferential_subsegment_crop, classes=self.classes_UFO)
                    segments,tumor_dict=self.get_tumor_segment_labels(idx)
                    tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = None, None
                    
                if tumor_segment_mask.sum().item()>0.0:
                    #ignore the preferential crop if the chosen segment is 0 (not present)
                    
                    out = self.random_crop_on_segment(tensor_img, tensor_lab, tumor_segment_mask, d, h, w,
                                                        slices_mask=slices_mask, chosen_organ=preferential_organ_crop,
                                                        metadata_df=tumor_dict, slices_used = (slices_dict is not None),
                                                        idx=idx)
                    
                    if isinstance(out, tuple):
                        #we got a successful crop. 
                        tensor_img, tensor_lab, slices_cropped_dict = out
                        if tensor_lab.sum().item()==0.0:
                            raise ValueError('Crop produced a zero mask for:', preferential_subsegment_crop, flush=True, file=sys.stderr)
                        print('>>>>>>>>> Per-patient cropped around tumor segment:', preferential_subsegment_crop, flush=True, file=sys.stderr)
                        assert preferential_organ_crop is not None, 'If you use preferential subsegment crop, you must provide a preferential organ crop, to know which organ slices to use for cropping with slices.'
                        crop_subseg = preferential_subsegment_crop
                        crop_organ = preferential_organ_crop
                        
                        if self.tumor_annotated_seg[self.img_list[idx]]:
                            tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.measure_tumors_mask(tensor_lab.squeeze(0))
                        
                        if not self.tumor_annotated_seg[self.img_list[idx]]:
                            #in case there is no tumor in this subsegment, we must return 'random' in the 4th return of this function
                            #check the report to see if there is a tumor in this subsegment
                            print(f'preferential_subsegment_crop is {preferential_subsegment_crop}, tumor segments in report are {segments["tumor_segments_flat"]}; segments organs: {segments["tumor_organs"]}', flush=True, file=sys.stderr)
                            if self.organ_is_in(preferential_subsegment_crop,segments['tumor_segments_flat']+segments['tumor_organs']):
                                subseg_tumor_return=preferential_subsegment_crop
                                message = 'tumor crop success'
                                return_9 = slices_cropped_dict
                            else:
                                subseg_tumor_return='random'
                                message='no tumor to crop on'
                        else:
                            #here I do not care what the 4th return will be, because this will not be used for downstream processing
                            subseg_tumor_return = None
                            message = 'annotated per voxel'
                            
                        print(f'Tumor segment: {subseg_tumor_return}', flush=True, file=sys.stderr)
                        
                        return tensor_img, tensor_lab, tumor_dict, subseg_tumor_return, preferential_organ_crop, message, tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel, return_9, crop_subseg, crop_organ
                    else:
                        print(f'Crop on preferential organ/subsegment {preferential_organ_crop}/{preferential_subsegment_crop} failed in image {self.img_list[idx]}, reason is {out}, trying normal cropping.', flush=True, file=sys.stderr)
                else:
                    print(f'Preferred organ/subsegment crop {preferential_organ_crop}/{preferential_subsegment_crop} empty (zero) in image {self.img_list[idx]}, proceeding with normal cropping.', flush=True, file=sys.stderr)
                # if you are here, it means that the crop on the preferential organ did not work (not reaching the return above). Thus, we proceed with the usual cropping strategy below, which will probably crop on another organ.
        
        if self.tumor_annotated_seg[self.img_list[idx]]:
            #for data with per-voxel tumor annotations
            tensor_img, tensor_lab, crop_organ, crop_subseg, tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, return_sizes=True)
            if self.debugging:print('This is an image with per-voxel annotations:'+self.img_list[idx], flush=True, file=sys.stderr)
            
            return tensor_img, tensor_lab, None, None, crop_organ, 'annotated per voxel', tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel, None, crop_subseg, crop_organ
        
        else:
            if not self.crop_on_tumor:
                raise ValueError('You should set crop on tumor for UFO data, as the tumor crops here are necessary for the report loss. And it makes little sense to not crop on tumor for Atlas and crop on tumor for UFO.')
            if self.debugging:print('This is an image with tumor annotations from reports:'+self.img_list[idx], flush=True, file=sys.stderr)
            #data without per-voxel tumor annotations, just reports mentioning tumors
            segments,tumor_dict=self.get_tumor_segment_labels(idx)
            
            #segment_options=segments['organs_with_only_known_sizes']
            segment_options= self.segment_options(segments=segments, slices=slices_dict, use_all_data=self.args.use_all_data)
            
            #print('Segment options:', segment_options, flush=True, file=sys.stderr)
            if len(segment_options)==0:
                #no tumor, crop like a per-voxel annotated case.
                tensor_img, tensor_lab, crop_organ, crop_subseg = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, 
                                                                               tumor_case=False, ufo=True)
                
                #print('No segment options, random crop', flush=True, file=sys.stderr)
                return tensor_img, tensor_lab, tumor_dict, 'random', crop_organ, 'no tumor to crop on', None, None, None, crop_subseg, crop_organ
            
            if segment_options.only_unknown_size():
                non_tumor_crop_chnc = self.non_tumor_crop_chance_unk_size
                #we set a higher chance of cropping on background for these cases. For cases of unk size, the volume loss has a big
                #margin of tolerance (0.5-10 cm). So we are constantly pushing the model to crop at least a small tumor on the organ
                #where we cropped at. This is not ideal, as the model may learn to always predict a small tumor. By cropping on the
                #background with a higher chance, we use the negative crops to push back and compensate for this tendency.
            else:
                non_tumor_crop_chnc = self.non_tumor_crop_chance
            
            #crop around the tumor organ/subsegment 
            # 80% chance of cropping on tumor:
            if np.random.random() < non_tumor_crop_chnc:
                #foreground crop or background crop
                tensor_img, tensor_lab, crop_organ, crop_subseg = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, tumor_case=False,
                                                                   tumor_prob=0, foreground_prob=0.5, background_prob=0.5,
                                                                   ufo=True)
                #print('Random crop by chance', flush=True, file=sys.stderr)
                return tensor_img, tensor_lab, tumor_dict, 'random', crop_organ, 'random crop by chance', None, None, None, crop_subseg, crop_organ
                
            else:
                out = None
                
                while len(segment_options)>=1 and (not isinstance(out, tuple) or tumor_segment_mask.sum().item()==0.0):
                    out = None
                    #tumor crop
                    #randomly pick an organ
                    if not self.args.balanced_cropper:
                        tumor_organ = segment_options.get_options()
                        #random choice now
                        tumor_organ = random.choice(tumor_organ)
                    else:
                        #get class indices
                        tumor_organ = self.cropper_UFO.choose_tumor_class(segment_options.get_options(), update_EMA=False)
                        
                    #print(f'Chosen organ to crop on: {tumor_organ}', flush=True, file=sys.stderr)
                        
                    #try to get its segment
                    if self.debugging:print(f'Tumor organ chosen: {tumor_organ}', flush=True, file=sys.stderr)
                    tumor_segment = self.organ_to_subsegment(tumor_organ, segments)
                    if self.debugging:print('Chosen segment to crop on:', tumor_segment, flush=True, file=sys.stderr)
                    if self.debugging:print('Possibilities were:', segments['subseg_with_only_known_sizes'], flush=True, file=sys.stderr)
                    
                    if tumor_segment is None:
                        #this indicates an impossible crop, like crop on kidney, but we do not know if it is left or right
                        #then, we remove the organ from the segment_options and continue
                        segment_options.remove_option(tumor_organ)
                        if isinstance(tumor_organ, list):
                            reason = 'no subsegment for organs '+' / '.join(tumor_organ)
                        else:
                            reason = 'no subsegment for organ '+tumor_organ
                        continue
                    #print('Chosen segment a:', tumor_segment, flush=True, file=sys.stderr)
                    
                    #get the mask for the tumor segment
                    tumor_segment_mask=self.get_random_tumor_seg_mask(tensor_lab, tumor_segment,classes=self.classes_UFO)
                    if tumor_segment_mask.sum().item()==0.0:
                        #if we cannot segment the organ segment containing the tumor, we try another organ
                        #print('Zero mask for:', tumor_segment, flush=True, file=sys.stderr)
                        self.zero_masks[self.current_sample]=tumor_segment
                        #save as yaml
                        with open('zero_masks.yaml', 'w') as f:
                            yaml.dump(self.zero_masks, f)
                        segment_options.remove_option(tumor_organ)
                        if isinstance(tumor_segment, list):
                            reason = 'tumor segment mask is empty for '+' / '.join(tumor_segment)
                        else:
                            reason = 'tumor segment mask is empty for '+tumor_segment
                        continue #try next segment
                    
                    out = self.random_crop_on_segment(tensor_img, tensor_lab, tumor_segment_mask, d, h, w,
                                                      slices_mask=slices_mask, chosen_organ=tumor_organ,
                                                      metadata_df=tumor_dict, slices_used = (slices_dict is not None),
                                                      idx=idx)
                    
                    #suceeded_crop = False
                    if isinstance(out, tuple):
                        #we got a successful crop. 
                        tensor_img, tensor_lab, slices_cropped_dict = out
                        if tensor_lab.sum().item()==0.0:
                            raise ValueError('Crop produced a zero mask for:', tumor_segment, flush=True, file=sys.stderr)
                        if self.args.balanced_cropper:
                            if isinstance(tumor_segment, list):
                                #random choice
                                tumor_segment_update = random.choice(tumor_segment)
                            else:
                                tumor_segment_update = tumor_segment
                            self.cropper_UFO.update_crop_proportions_EMA(self.cropper_UFO.tumor_proportions, tumor_segment_update)
                        if self.debugging:print('>>>>>>>>> Cropped around tumor segment:', tumor_segment, flush=True, file=sys.stderr)
                        crop_organ = tumor_organ
                        crop_subseg = tumor_segment
                        return tensor_img, tensor_lab, tumor_dict, tumor_segment, tumor_organ, 'tumor crop success', None, None, slices_cropped_dict, crop_subseg, crop_organ
                    else:
                        reason = out
                        #this is a failed crop, we try another segment
                        if self.debugging:print('Failed crop for:', tumor_segment,'out is:',out, flush=True, file=sys.stderr)
                        if True:#debug
                            import csv
                            # Append the failure details to the CSV.
                            with open('failed_crops_multi_tumor.csv', 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                # Write a row with image ID, tumor_segment, and out. Convert 'out' to string if needed.
                                writer.writerow([self.img_list[idx], tumor_segment, str(out)])
                        segment_options.remove_option(tumor_organ)
                        continue

                if len(segment_options)==0:
                    #if we cannot crop in any tumor, we fall back to random crop
                    tensor_img, tensor_lab, crop_organ, crop_subseg = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, tumor_case=False,
                                                                tumor_prob=0, foreground_prob=0.5, background_prob=0.5,
                                                                   ufo=True)
                    if self.debugging:print(f'----------- Tumor crop failed for {self.img_list[idx]}, reason: {reason}', flush=True, file=sys.stderr)
                    return tensor_img, tensor_lab, tumor_dict, 'random', crop_organ, reason, None, None, None, crop_subseg, crop_organ
                
                raise ValueError('You should not be here, this is a bug')


    def save(self, tensor_img, tensor_lab, idx, tumor_dict=None, dta=None, unk_channels_tensor=None,
            tumor_volumes_in_crop=None, chosen_segment_mask=None,tumor_diameters=None,
            selected_organ=None, tumor_attenuation=None, reason=None,
            tumor_volumes_in_crop_per_voxel=None, tumor_diameters_per_voxel=None,
            slices_mask=None,binary_slices_mask=None,
            selected_organ_non_canonical=None,
            max_slice=None, tumor_df=None, slices_used = False,
            original_spacing=None,
            tumor_slices = None,
            tumor_malignancy = None, malignancy_per_voxel = None):
        """
        Saves the augmented image/label pair to disk if a destination was specified.
        Uses numpy .npz format and keeps the original naming scheme.
        """
        os.makedirs(self.save_destination, exist_ok=True)

        # Keep the same filenames as the original
        base_img_name = os.path.basename(self.img_list[idx])   # e.g. "xxx.npz"
        base_lab_name = os.path.basename(self.lab_list[idx])   # e.g. "xxx_gt.npz"

        img_filename = os.path.join(self.save_destination, base_img_name)
        lab_filename = os.path.join(self.save_destination, base_lab_name)

        np_img = tensor_img.cpu().numpy()
        np_lab = tensor_lab.cpu().numpy().astype(np.bool_)  

        #print('Number of labels:',np_lab.shape[0], flush=True, file=sys.stderr)
        np_lab = np.packbits(np_lab, axis=0) #from bool to uint8 - reduce the channels dimension by 8. Each voxel is saved a a byte anyway. This reduce the size of the file by 8.
        ##print('Shape of label after packing:', np_lab.shape)
        
        img_filename = img_filename.replace('.npy','.npz')
        lab_filename = lab_filename.replace('.npy','.npz')
        
        #remove original npy file if they exist:
        if os.path.exists(img_filename.replace('.npz','.npy')):
            os.remove(img_filename.replace('.npz','.npy'))
        if os.path.exists(lab_filename.replace('.npz','.npy')):
            os.remove(lab_filename.replace('.npz','.npy'))
        if os.path.exists(lab_filename.replace('.npz','_slices.npy')):
            os.remove(lab_filename.replace('.npz','_slices.npy'))
        if os.path.exists(lab_filename.replace('.npz','_binary_slices.npy')):
            os.remove(lab_filename.replace('.npz','_binary_slices.npy'))
        if os.path.exists(lab_filename.replace('.npz','_unk.npy')):
            os.remove(lab_filename.replace('.npz','_unk.npy'))
        if os.path.exists(lab_filename.replace('.npz','_chosen_tumor_segment.npy')):
            os.remove(lab_filename.replace('.npz','_chosen_tumor_segment.npy'))

        # Save as .npz
        np.savez_compressed(img_filename, np_img)
        np.savez_compressed(lab_filename, np_lab)
        #print('Saved:',img_filename, flush=True, file=sys.stderr)
        #print('Saved:',lab_filename, flush=True, file=sys.stderr)
        
        if slices_mask is not None:
            #slices_mask is a tensor, we need to convert it to numpy
            slices_mask = slices_mask.cpu().numpy()
            #to uint16
            slices_mask = np.clip(slices_mask, 0, 65535).astype(np.uint16)
            np.savez_compressed(lab_filename.replace('.npz','_slices.npz'), slices_mask)
            
        if binary_slices_mask is not None:
            #binary_slices_mask is a tensor, we need to convert it to numpy
            binary_slices_mask = binary_slices_mask.cpu().numpy().astype(np.bool_)
            np.savez_compressed(lab_filename.replace('.npz','_binary_slices.npz'), binary_slices_mask)


        if unk_channels_tensor is not None:
            unk_ch = unk_channels_tensor.cpu().numpy().astype(np.bool_)
            unk_channels_tensor = np.packbits(unk_ch, axis=0)
            np.savez_compressed(lab_filename.replace('.npz','_unk.npz'), unk_channels_tensor)

        if chosen_segment_mask is not None:
            chosen_segment_mask = chosen_segment_mask.cpu().numpy().astype(np.bool_)
            chosen_segment_mask = np.packbits(chosen_segment_mask, axis=0)
            np.savez_compressed(lab_filename.replace('.npz','_chosen_tumor_segment.npz'), chosen_segment_mask)

        if tumor_dict is not None:
            tumor_dict.to_csv(os.path.join(self.save_destination, img_filename.replace('.npz','.csv')), index=False)
            
        
        #annotated per voxel?
        per_vox = self.tumor_annotated_seg[self.img_list[idx]]
            
            
        if dta is not None:
            if tumor_diameters is not None:
                tumor_diameters = tumor_diameters.cpu().numpy().tolist()
                dta['tumor diameters'] = tumor_diameters
                
            if tumor_volumes_in_crop is not None:
                dta['tumor volumes in crop'] = tumor_volumes_in_crop
                    
            if tumor_volumes_in_crop_per_voxel is not None:
                dta['tumor volumes in crop per voxel'] = tumor_volumes_in_crop_per_voxel
                
            if tumor_diameters_per_voxel is not None:
                dta['tumor diameters in crop per voxel'] = tumor_diameters_per_voxel
            
            if tumor_attenuation is not None:
                tumor_attenuation = tumor_attenuation.cpu().numpy().tolist()
                dta['tumor attenuation'] = tumor_attenuation
                
            if tumor_slices is not None:
                tumor_slices = tumor_slices.cpu().numpy().tolist()
                dta['sizes_slices'] = tumor_slices
            
            if reason is not None:
                #add reason to the dict
                dta['reason of crop'] = reason
                
            if selected_organ_non_canonical is not None:
                dta['selected organ non canonical'] = selected_organ_non_canonical
                
            if max_slice is not None:
                dta['max_slice'] = int(max_slice)
            
            if tumor_df is not None:
                if isinstance(tumor_df, pd.DataFrame):
                    dta['tumor_df'] = tumor_df.replace({np.nan: None}).to_json(orient="records")
                elif isinstance(tumor_df, str):
                    dta['tumor_df'] = tumor_df  # assume it’s already a JSON string
                else:
                    raise ValueError(f"Unexpected type for tumor_df. Expected DataFrame or JSON string.")
                
            dta['slices_used'] = slices_used
            
            if original_spacing is not None:
                dta['original_spacing'] = original_spacing
                
            if tumor_malignancy is not None:
                dta['tumor malignancy'] = tumor_malignancy.cpu().numpy().tolist()
                
            if malignancy_per_voxel is not None:
                dta['tumor malignancy per voxel'] = malignancy_per_voxel.cpu().numpy().tolist()
                
            with open(os.path.join(self.save_destination, img_filename.replace('.npz','.json')), "w") as f:
                json.dump(dta, f)

        if self.cropper is not None:
            #update the sql registry with selected_organ
            id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
            self.safe_upsert(id, selected_organ)

        self.save_counter += 1
        
        return dta
        
    
    def safe_upsert(self, key: str, value: str, retries: int = 3) -> None:
        """
        Atomically “upsert” a single key/value pair into the YAML store.

        The record lives in  <self.yaml_dir>/<KEY>_crop.yaml
            └── content:  - <value>

        Concurrent writers are serialised with a .lock file next to the YAML.
        """
        if self.debugging:print('Attempting to upsert:', key, value, flush=True, file=sys.stderr)
        yaml_dir   = Path(self.save_destination)          # make sure you set this attr!

        yml_path   = yaml_dir / f"{key}_crop.yaml"
        lock_path  = yml_path.with_suffix(".lock")
        lock       = FileLock(lock_path, timeout=30)

        for attempt in range(retries):
            try:
                with lock:                        # one writer at a time
                    # 1. write to a temp file first
                    tmp = yml_path.with_suffix(".tmp")
                    with tmp.open("w") as f:
                        yaml.safe_dump([value], f, default_flow_style=False)

                    # 2. atomic replace → readers never see a partial file
                    os.replace(tmp, yml_path)
                if self.debugging:print(f"[yaml‑upsert] {key} → {value}", flush=True, file=sys.stderr)
                return                            # ☑ success
            except Timeout:
                # another writer is busy, wait a little then retry
                time.sleep(0.2 * (attempt + 1))

        # all retries exhausted
        raise RuntimeError(f"Could not write {key=} after {retries} attempts")
    
  
                    
    def load_augmented_data(self, idx):
        # We'll assume the user has already run the dataset once to save the augmented data.
        if self.save_destination is None:
            raise ValueError("load_augmented=True but save_destination=None. Cannot load augmented data.")
        
        #print('Loading augmented data for:', self.img_list[idx], flush=True, file=sys.stderr)

        start = time.time()

        # Derive the filenames from the original naming scheme
        base_img_name = os.path.basename(self.img_list[idx])    # e.g. "xxx.npy"
        base_lab_name = os.path.basename(self.lab_list[idx])    # e.g. "xxx_gt.npy"
        
        # Replace npz by npy
        base_img_name = base_img_name.replace('.npz', '.npy')
        base_lab_name = base_lab_name.replace('.npz', '.npy')

        aug_img_path = os.path.join(self.save_destination, base_img_name)
        aug_lab_path = os.path.join(self.save_destination, base_lab_name)
        
        #check if npy file exists, otherwise, try npz
        if not os.path.exists(aug_img_path):
            aug_img_path = aug_img_path.replace('.npy', '.npz')
            aug_lab_path = aug_lab_path.replace('.npy', '.npz')
            base_img_name = base_img_name.replace('.npy', '.npz')
            base_lab_name = base_lab_name.replace('.npy', '.npz')
            
        
            
        
        # Load the augmented data
        np_img = _load_numpy_array(aug_img_path)  # shape as saved
        tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).float()
        ##print('Time to load augmented image:', time.time() - start, flush=True, file=sys.stderr)
        start = time.time()
        ##print shapes
        ##print('Shape:', np_img.shape, np_lab.shape)

        # Convert to torch
        # The code expects image to be float32 and label int8 (for checking).
        np_lab = _load_numpy_array(aug_lab_path)  # uint8

        # 4. Unpack the bits along the same axis.
        if np_lab.shape[0] != len(self.classes):
            #print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
            start_unpack = time.time()
            # 4. Unpack the bits along the same axis.
            np_lab = np.unpackbits(np_lab, axis=0)
            assert np_lab.shape[0] < self.num_classes +10
            assert np_lab.shape[0] >= self.num_classes
            np_lab = np_lab[:self.num_classes]
            #print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)
            ##print('Time to unpack:', time.time() - start_unpack, flush=True, file=sys.stderr)

        tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)

        ##print('Time to load augmented label:', time.time() - start, flush=True, file=sys.stderr)
        aug_start = time.time()

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        
        if self.mode == 'train':
            #this augmentation is online.
            if np.random.random() < 0.3:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
            if np.random.random() < 0.3:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            if np.random.random() < 0.3:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
            if np.random.random() < 0.3:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
            if np.random.random() < 0.3:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
            if np.random.random() < 0.3:
                std = np.random.random() * 0.2 
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
            ##print('Applied augmentation online!')
        
        ##print('Augmentation deactivated!')

        # You can still call save_sanity_check if desired
        #self.save_sanity_check(tensor_img, tensor_lab, idx)

        ##print('Time augmenting data:', time.time() - aug_start, flush=True, file=sys.stderr)

        tensor_img = tensor_img.squeeze(0)

        ##print('Shapes:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

        if self.mode == 'train':
            if self.img_list[idx] not in self.UFO_paths:
                #annotated per-voxel, no unknnown voxel
                unk_channels_list=torch.zeros(tensor_lab.shape).type_as(tensor_lab)
                tumor_volumes_in_crop=[0,0,0,0,0,0,0,0,0,0]
                tumor_diameters=torch.zeros((10,3)).float()
                #try:
                with open(os.path.join(self.save_destination, base_img_name.replace('.npy','.json').replace('.npz','.json')), "r") as f:
                    dta=json.load(f)
                tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = dta['tumor volumes in crop per voxel'], dta['tumor diameters in crop per voxel']
                #except:
                #    print('!!!!!!! CAREFUL: MEASURING TUMORS IN DATA LOADING, ISNTEAD OF BEFORE ROTATION !!!!!!!!!!!!!!!!!')
                #    tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.measure_tumors_mask(tensor_lab)
                tumor_attenuation=torch.zeros((10)).float()
                chosen_segment_mask=torch.zeros_like(tensor_lab).type_as(tensor_lab)
                unk_channels_tensor = torch.zeros_like(tensor_lab).type_as(tensor_lab)
                tumor_slices = torch.tensor([[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan]]).float().type_as(tensor_img)
                tumor_malignancy = tumor_slices.clone()
                malignancy_per_voxel = self.get_malignancy_annotated_per_voxel(idx,tensor_lab)
            else:
                #try loading unk_channels_list if saved
                unk_pth=aug_lab_path.replace('_gt.npy','_gt_unk.npy').replace('_gt.npz','_gt_unk.npz')
                if os.path.exists(unk_pth):
                    unk_channels_tensor = _load_numpy_array(unk_pth)
                    if unk_channels_tensor.shape[0] != len(self.classes):
                        unk_channels_tensor = np.unpackbits(unk_channels_tensor, axis=0)
                        unk_channels_tensor = unk_channels_tensor[:len(self.classes)]
                    unk_channels_list = torch.from_numpy(unk_channels_tensor.astype(np.float32))
                    #print(f'----------------UNK WAS LOADED FROM {unk_pth}', flush=True, file=sys.stderr)
                else:
                    unk_channels_list=self.define_unknown_voxels(tensor_lab,idx)
                    #print('----------------UNK WAS CREATED', flush=True, file=sys.stderr)
                #load the json file
                with open(os.path.join(self.save_destination, base_img_name.replace('.npy','.json').replace('.npz','.json')), "r") as f:
                    dta=json.load(f)
                tumor_volumes_in_crop,tumor_diameters,tumor_attenuation,tumor_slices,tumor_malignancy=self.estimate_tumor_volume(idx,tumor_segment_crop=dta['tumor_in_crop'])
                if os.path.exists(aug_lab_path.replace('.npy','_chosen_tumor_segment.npy').replace('.npz','_chosen_tumor_segment.npz')):
                    chosen_segment_mask = _load_numpy_array(aug_lab_path.replace('.npy','_chosen_tumor_segment.npy').replace('.npz','_chosen_tumor_segment.npz'))
                    if chosen_segment_mask.shape[0] != len(self.classes):
                        chosen_segment_mask = np.unpackbits(chosen_segment_mask, axis=0)
                        chosen_segment_mask = chosen_segment_mask[:self.num_classes]
                    chosen_segment_mask = torch.from_numpy(chosen_segment_mask.astype(np.float32))
                else:
                    chosen_segment_mask=self.get_chosen_segment_mask(tensor_lab, dta['tumor_in_crop'])
                tumor_volumes_in_crop_per_voxel, tumor_diameters_per_voxel = self.measure_tumors_mask_zeros(tensor_lab)
                if torch.tensor(tumor_volumes_in_crop).float().sum() == 0 and chosen_segment_mask.sum()>0:
                    raise ValueError(f'{self.img_list[idx]}tumor_volumes_report should not be all zeros if chosen_segment_mask is not all zeros')
                # Mirrors the loss-side assertion at losses_foundation.py:4108-4110.
                # If we hit it here we raise so the existing augmented-load retry
                # in getitem_real (lines ~2212-2222) picks a different sample,
                # rather than crashing the run from inside the loss.
                if unk_channels_list.sum() == 0 and chosen_segment_mask.sum() > 0:
                    raise ValueError(f'{self.img_list[idx]} unk_channels should not be all zeros if chosen_segment_mask is not all zeros')
                malignancy_per_voxel = torch.tensor([np.nan]*len(self.classes)).float().to(tensor_img.device)
            #print('LOADED AUGMENTED DATA', tensor_lab.shape, 'From:', os.path.join(self.save_destination, base_lab_name), flush=True, file=sys.stderr)
            
            if self.class_proportions: 
                sample_weights = get_sample_weight(tensor_lab,self.class_proportions,self.classes, balancer=self.cropper if self.balancing_crops else None) 
            else:
                sample_weights = torch.ones_like(tensor_lab)
            
            
            
            if self.load_slices and dta['tumor_in_crop'] != 'random':
                #get slices_mask and binary_slices_mask
                slices_mask = None
                binary_slices_mask = None
                if os.path.exists(aug_lab_path.replace('.npy','_slices.npy').replace('.npz','_slices.npz')):
                    slices_mask = _load_numpy_array(aug_lab_path.replace('.npy','_slices.npy').replace('.npz','_slices.npz'))
                    slices_mask = torch.from_numpy(slices_mask.astype(np.float32))
                if os.path.exists(aug_lab_path.replace('.npy','_binary_slices.npy').replace('.npz','_binary_slices.npz')):
                    binary_slices_mask = _load_numpy_array(aug_lab_path.replace('.npy','_binary_slices.npy').replace('.npz','_binary_slices.npz'))
                    binary_slices_mask = torch.from_numpy(binary_slices_mask.astype(np.float32))
                if (slices_mask is None) or (binary_slices_mask is None) or (dta['slices_used'] == False) or (dta['tumor_df'] is None) or (dta['tumor_df']=='[]') or (dta['max_slice'] is None) or (dta['original_spacing'] is None):
                    slices_cropped_dict = None
                else:
                    slices_cropped_dict = {'slices_mask': slices_mask.float(),
                                        'binary_slices_mask': binary_slices_mask.float(),
                                        'tumor_df': dta['tumor_df'],
                                        'max_slice': dta['max_slice'],
                                        'slices_used': dta['slices_used'],
                                        'z_spacing': dta['original_spacing'],
                                        'selected_organ_non_canonical': dta['selected organ non canonical']}
            else:
                slices_cropped_dict = None
                
            if slices_cropped_dict is not None and (slices_cropped_dict['binary_slices_mask'] is not None) and (slices_cropped_dict['slices_mask'] is not None) and slices_cropped_dict['slices_used']:
                self.SanityAssertOutput(tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float(),
                                        binary_slices_mask=slices_cropped_dict['binary_slices_mask'],slices_mask=slices_cropped_dict['slices_mask'],)
            
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float().clone(), torch.from_numpy(tensor_lab).float().clone()
                return {"image":           tensor_img.clone().float(),
                        "label":           tensor_lab.clone().float(),
                        "unk_channels":    unk_channels_list.clone().float(),
                        "volumes":         torch.tensor(tumor_volumes_in_crop).float().clone(),
                        "mask":            chosen_segment_mask.float().clone(),
                        "diameters":       tumor_diameters.float().clone(),}
                
            if slices_cropped_dict is None:
                tumor_df = '[]'
                slices_cropped_dict={'slices_mask': torch.zeros_like(tensor_img).float(), 
                                     #'tumor_segment_mask_cut': tumor_segment_mask_cut, #ideally, we do not want to use this directly in our loss, because we want to dilate the organ mask to create resistance to mask errors, but we do not want to dilate the slice ranges
                                     'tumor_df':tumor_df, #this will be needed in the ball loss, as we want to assign each tumor size to one slice only
                                     'binary_slices_mask': torch.ones_like(tensor_img).float(), #we can use this in the volume loss, we first dilate the organ, then we multiply by this mask
                                     'max_slice': 0,
                                     'slices_used': False,
                                     "selected_organ_non_canonical": dta['selected organ non canonical'],
                                     "z_spacing": dta['original_spacing']
                                     }
            else:
                slices_cropped_dict['slices_mask']= slices_cropped_dict['slices_mask'].float()
                slices_cropped_dict['binary_slices_mask'] = slices_cropped_dict['binary_slices_mask'].float()
                
                
            if slices_cropped_dict['slices_used']:
                slices_cropped_dict['binary_slices_mask'] = slices_cropped_dict['binary_slices_mask'].float()
            else:
                slices_cropped_dict['binary_slices_mask'] = torch.ones_like(tensor_img).float()
                
            if not (slices_cropped_dict and slices_cropped_dict['slices_used'] and slices_cropped_dict['max_slice'] > 0):
                # sizes_slices is (10, 2): [size_mm, slice_idx]; clear the slice column
                tumor_slices[:, 1] = float('nan')
                
            contrast = self.get_contrast_phase(idx)
            age,sex = self.get_age_and_sex(idx)
            date = self.get_date(idx)
            patient_id = self.get_patient_id(idx)
            #print(f'Contrast phase for {self.img_list[idx]}: {contrast}', flush=True, file=sys.stderr)
            subsegment_cropped_in = dta['subsegment_cropped_in'] if 'subsegment_cropped_in' in dta else dta['tumor_in_crop']
            if subsegment_cropped_in is None:
                subsegment_cropped_in = 'random'
            organ_cropped_in = dta['organ_cropped_in'] if 'organ_cropped_in' in dta else dta['tumor_in_crop']
            if organ_cropped_in is None:
                organ_cropped_in = 'random'
                
            if not self.load_per_patient:
                #hash patient ids and subsegment_cropped_in to help all_gather downstream
                subsegment_cropped_in_no_hash = subsegment_cropped_in
                patient_id = string_to_tensor(patient_id, salt="pid_v1", allow_numeric=True)
                subsegment_cropped_in = string_to_tensor(subsegment_cropped_in, salt="organ_v1", allow_numeric=False)
            
            retur = {"image":          tensor_img.float(),
                    "label":           tensor_lab.to(torch.int16),
                    "unk_channels":    unk_channels_list.to(torch.int16),
                    "volumes":         torch.tensor(tumor_volumes_in_crop).float(),
                    "mask":            chosen_segment_mask.float(),
                    "diameters":       tumor_diameters.float(),
                    "weights":         sample_weights.float(),
                    "attenuation":     tumor_attenuation.float(),
                    "voumes_per_voxel": {k: torch.tensor(v, dtype=torch.float32) for k, v in tumor_volumes_in_crop_per_voxel.items()},
                    "diameters_per_voxel": {k: torch.tensor(v, dtype=torch.float32) for k, v in tumor_diameters_per_voxel.items()},
                    "name":             self.img_list[idx],
                    "slices_cropped_dict": slices_cropped_dict,
                    "sizes_slices":  tumor_slices.clone().float() if tumor_slices is not None else None,
                    "sizes_malignancy":  tumor_malignancy.clone().float() if tumor_malignancy is not None else None,
                    "malignancy_per_voxel": malignancy_per_voxel,
                    "contrast": contrast,
                    "annotated_per_voxel": self.tumor_annotated_seg[self.img_list[idx]],
                    "age": age,
                    "sex": sex,
                    'tumor_in_crop': dta['tumor_in_crop'] if dta['tumor_in_crop'] is not None else 'random',
                    'date': date,
                    'patient_id': patient_id,
                    'subsegment_cropped_in': subsegment_cropped_in,
                    'organ_cropped_in': organ_cropped_in,
                    'organ_cropped_cannon': canonical_organ(subsegment_cropped_in_no_hash),
                    }
            
            retur['tumor_info_input'] = self.get_input_info_report(retur, idx)
            
            
            if self.UFO_only:
                sanity_assert_no_lesion_mask(retur['label'], self.classes, self.img_list[idx])
            
            clone_tensors_inplace(retur)
            assert not torch.isnan(tensor_img.float()).any(), f'Input is nan for {self.img_list[idx]}'
                
            if self.args.load_clip:
                if self.cropper is None:
                    raise ValueError('We are only using CLIP for the custom cropper mode.')
                #get the organ from the crop registry
                id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
                yml_path = Path(self.save_destination) / f"{id}_crop.yaml"
                with yml_path.open("r") as f:
                    data = yaml.safe_load(f) or []
                selected_organ = data[0]
                embedding = self.load_clip(idx, selected_organ)
                retur["clip_embedding"] = embedding
             
            #assertion: if not use_all_data, then we cannot have negative tumor volumes
            if not self.args.use_all_data:
                volumes = retur['volumes']
                if not (volumes.sum(dim=-1)>=0).all():
                    raise ValueError(f'Negative tumor volumes found in {self.img_list[idx]} but use_all_data is False. Volumes are {volumes}, sample is: {self.img_list[idx]}')
            
            return retur
        else:
            #print('LOADED AUGMENTED DATA', tensor_lab.shape, 'From:', os.path.join(self.save_destination, base_lab_name), flush=True, file=sys.stderr)
            raise ValueError('Loading cropped data in testing. Are you sure? No sliding window?')
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])
        
        
    def get_input_info_report(self, retur, idx):
        """
        For the model input, tumor information should be sent in a dictionary with the following keys:
        tumor_organ: mask for the organ where the tumor is located
        tumor_slices: mask showing the slices where the tumor is located
        tumor_attenuation: attenuation of each tumor
        tumor_malignancy: malignancy of each tumor
        """
    
            
        lesion_labels = ['unknown','no tumor'] + [c for c in self.classes if 'lesion' in c]
        attenuation_labels = ['unknown','no tumor', 'hypo', 'hyper']
        malignancy_labels = ['unknown','no tumor', 'benign', 'malignant']
        denom_labels = max(len(lesion_labels) - 1, 1)
        denom_att = max(len(attenuation_labels) - 1, 1)
        denom_malig = max(len(malignancy_labels) - 1, 1)
        
        unknown_tumor_count = False
        
        tumor_info = {}
        
        #ufo or atlas case?
        annotated_per_voxel = self.tumor_annotated_seg[self.img_list[idx]]
        
        #get tumor organ mask
        if not annotated_per_voxel:
            #check if there is any tumor
            #get tumor count: number of non-zero in sizes_malignancy first dim
            # Count both known-size tumors (volume > 0) and unknown-size tumors
            # (volume == -9999999 sentinel). Padding entries are exactly 0 and
            # are excluded. This avoids the no-tumor cascade for cases where
            # every reported tumor in the crop has unknown size.
            volumes = retur['volumes']
            tumor_count = (volumes!=0).sum().item()#I had a bug here, volumes > 0, which ignored tumors of unknown size
            if tumor_count ==0:
                #no tumor
                tumor_class = 'no tumor'
                tumor_class_idx = lesion_labels.index(tumor_class)/(denom_labels)
                #broadcast to 10 elements (10 tumors)
                tumor_attenuation = 'no tumor'
                tumor_attenuation = attenuation_labels.index(tumor_attenuation)/(denom_att)
                tumor_attenuation = [tumor_attenuation]*10
                tumor_malignancy = 'no tumor'
                tumor_malignancy = malignancy_labels.index(tumor_malignancy)/(denom_malig)
                tumor_malignancy = [tumor_malignancy]*10
                tumor_location_mask = torch.zeros_like(retur['image'].squeeze(0))
                allowed_tumor_slices = torch.ones_like(retur['image'].squeeze(0))
                tumor_diameters = torch.tensor([0]*30).type_as(retur['image'])
                tumor_volumes = torch.tensor([0]*10).type_as(retur['image'])
            else:
                mask_subsegs = retur["mask"]
                #get tumor organ class name
                tumor_ch = torch.argmax(mask_subsegs.sum(dim=(1,2,3)), dim=0).item()
                tumor_class = self.classes[tumor_ch]
                organ_class = tumor_to_organ(tumor_class)
                tumor_class_idx = lesion_labels.index(tumor_class)/(denom_labels)
                tumor_location_mask = mask_subsegs[tumor_ch]
                assert mask_subsegs.sum().item() == tumor_location_mask.sum().item(), f'Only one channel should be active in the tumor subseg maks (we do one tumor at a time)'
                allowed_tumor_slices = retur['slices_cropped_dict']['binary_slices_mask']
                #check if the tumor count is actually unknown
                report = self.read_report(idx)
                tumors_in_organ = report[report['Standardized Organ']==organ_class]
                if len(tumors_in_organ)==0:
                    raise ValueError(f'No tumors found in organ {organ_class} in report for {self.img_list[idx]}')
                # _was_multiple is set in clean_ufo before size standardization.
                # Old check looked for 'multiple' in the post-standardization size
                # column, which is dead code (clean_ufo collapses 'multiple' -> 'u').
                if '_was_multiple' in tumors_in_organ.columns and tumors_in_organ['_was_multiple'].any():
                    unknown_tumor_count = True
                else:
                    unknown_tumor_count = False
                sizes_malignancy = retur['sizes_malignancy']
                #attenuation: 1 = hyper, -1 = hypo, 999 = unknownm, 0 = padding (ignore)
                #size and volume: -9999999 = unknown size or unknown number of tumors, 0 = padding (ignore)
                #malignancy: 1 = malignant, 0 = benign, NaN = unknown, padding = NaN
                if sizes_malignancy is None:
                    tumor_malignancy = []
                    for i in range(tumor_count):
                        tumor_malignancy.append(malignancy_labels.index('unknown')/(denom_malig))
                    #now complete until 10 tumors
                    while len(tumor_malignancy) <10:
                        tumor_malignancy.append(malignancy_labels.index('no tumor')/(denom_malig))
                if sizes_malignancy is not None:
                    if sizes_malignancy[:,0].sum() == 0:
                        raise ValueError(f'No tumor found in sizes_malignancy for {self.img_list[idx]}')
                    #check if all nan
                    #where size is zero, make 
                    tumor_malignancy = []
                    for i in range(sizes_malignancy.shape[0]):
                        s = sizes_malignancy[i][0].item()
                        if s == 0: #padding
                            tumor_malignancy.append(malignancy_labels.index('no tumor')/(denom_malig))
                            continue               
                        m = sizes_malignancy[i][1].item()
                        if np.isnan(m):
                            tumor_malignancy.append(malignancy_labels.index('unknown')/(denom_malig))
                        elif m == 1:
                            tumor_malignancy.append(malignancy_labels.index('malignant')/(denom_malig))
                        elif m == 0:
                            tumor_malignancy.append(malignancy_labels.index('benign')/(denom_malig))
                        else:
                            raise ValueError(f'Unexpected malignancy value {m} in sizes_malignancy for {self.img_list[idx]}')
                attenuation = retur['attenuation']
                if attenuation is None:
                    tumor_attenuation = []
                    for i in range(tumor_count):
                        tumor_attenuation.append(attenuation_labels.index('unknown')/(denom_att))
                    #now complete until 10 tumors
                    while len(tumor_attenuation) <10:
                        tumor_attenuation.append(attenuation_labels.index('no tumor')/(denom_att))
                if attenuation is not None:
                    if torch.equal(attenuation, torch.zeros_like(attenuation)):
                        raise ValueError(f'No tumor found in attenuation for {self.img_list[idx]}')
                    tumor_attenuation = []
                    for i in range(attenuation.shape[0]):
                        a = attenuation[i].item()
                        if a == 0:#padding
                            tumor_attenuation.append(attenuation_labels.index('no tumor')/(denom_att))
                        elif a == 999:
                            tumor_attenuation.append(attenuation_labels.index('unknown')/(denom_att))
                        elif a == 1:
                            tumor_attenuation.append(attenuation_labels.index('hyper')/(denom_att))
                        elif a == -1:
                            tumor_attenuation.append(attenuation_labels.index('hypo')/(denom_att))
                        else:
                            raise ValueError(f'Unexpected attenuation value {a} in attenuation for {self.img_list[idx]}')
                tumor_diameters = retur["diameters"] #shape (10, 3), we need (30)
                tumor_diameters = torch.cat([tumor_diameters[i] for i in range(tumor_diameters.shape[0])], dim=0)
                tumor_volumes = retur["volumes"]
                #cap at -0.5, so -9999 (unknown) becomes -0.5, a reasonable number for the MLP to process
                tumor_volumes = torch.clamp(tumor_volumes, min=-0.5)
                tumor_diameters = torch.clamp(tumor_diameters, min=-0.5)
        else:#annotated per voxel
            labels = retur["label"]
            #get lesion channels
            lesion_channels,class_names = get_lesion_channels(labels.unsqueeze(0), self.classes, return_class_names=True)
            lesion_channels = lesion_channels.squeeze(0)
            if lesion_channels.sum() == 0:
                #no tumor
                tumor_class = 'no tumor'
                tumor_class_idx = lesion_labels.index(tumor_class)/(denom_labels)
                #broadcast to 10 elements (10 tumors)
                tumor_attenuation = 'no tumor'
                tumor_attenuation = attenuation_labels.index(tumor_attenuation)/(denom_att)
                tumor_attenuation = [tumor_attenuation]*10
                tumor_malignancy = 'no tumor'
                tumor_malignancy = malignancy_labels.index(tumor_malignancy)/(denom_malig)
                tumor_malignancy = [tumor_malignancy]*10
                tumor_location_mask = torch.zeros_like(retur['image'].squeeze(0))
                allowed_tumor_slices = torch.ones_like(retur['image'].squeeze(0))
                tumor_count = 0
                tumor_volumes = torch.tensor([0]*10).type_as(retur['image'])
                tumor_diameters = torch.tensor([0]*30).type_as(retur['image'])
            else:
                #get largest tumor by volume
                tumor_vols = lesion_channels.sum(dim=(1,2,3))
                largest_tumor_idx = torch.argmax(tumor_vols).item()
                tumor_class = class_names[largest_tumor_idx]
                tumor_class_idx = lesion_labels.index(tumor_class)/(denom_labels)
                tumor_segmentation_mask = labels[self.classes.index(tumor_class)]
                organ_class_for_mask = tumor_to_organ2(tumor_class)
                if organ_class_for_mask == 'uterus':
                    organ_class_for_mask = 'prostate'
                unknown_organ_mask = False
                if isinstance(organ_class_for_mask, list):
                    if any(o not in self.classes for o in organ_class_for_mask):
                        raise ValueError(f'Organ {organ_class_for_mask} not found in self.classes: {self.classes}.')
                    else:
                        masks = [labels[self.classes.index(o)] for o in organ_class_for_mask]
                        # element‑wise max across sides
                        from functools import reduce
                        tumor_location_mask = reduce(torch.maximum, masks)
                else:
                    if organ_class_for_mask not in self.classes:
                        #raise ValueError(f'Organ {organ_class_for_mask} not found in index mapping: {self.classes}.')
                        #we can have some mask for an organ like bones that is not in the organ classes
                        tumor_location_mask = tumor_segmentation_mask
                        unknown_organ_mask = True
                    else:
                        tumor_location_mask = labels[self.classes.index(organ_class_for_mask)]
                #attenuation from masks
                allowed_tumor_slices = torch.ones_like(retur['image'].squeeze(0))
                tumor_attenuation,tumor_count = attenutation_tensor_from_mask(tumor_location_mask, tumor_segmentation_mask, retur['image'].squeeze(0),
                                                                              skip_attenuation = unknown_organ_mask)
                tumor_attenuation = [attenuation_labels.index(a)/(denom_att) for a in tumor_attenuation]
                #malignancy
                malignancy_per_voxel = retur['malignancy_per_voxel']
                if malignancy_per_voxel is None:
                    tumor_malignancy = []
                    for i in range(tumor_count):
                        tumor_malignancy.append(malignancy_labels.index('unknown')/(denom_malig))
                    #now complete until 10 tumors
                    while len(tumor_malignancy) <10:
                        tumor_malignancy.append(malignancy_labels.index('no tumor')/(denom_malig))
                else:
                    tumor_malignancy = []
                    for i in list(range(tumor_count)):
                        if len(tumor_malignancy)>=10:
                            break
                        m = malignancy_per_voxel[i].item()
                        #nan is unk
                        if np.isnan(m):
                            tumor_malignancy.append(malignancy_labels.index('unknown')/(denom_malig))
                        elif m == 1:
                            tumor_malignancy.append(malignancy_labels.index('malignant')/(denom_malig))
                        elif m == 0:
                            tumor_malignancy.append(malignancy_labels.index('benign')/(denom_malig))
                        else:
                            raise ValueError(f'Unexpected malignancy value {m} in malignancy_per_voxel for {self.img_list[idx]}')
                    #complete until 10 tumors
                    while len(tumor_malignancy) <10:
                        tumor_malignancy.append(malignancy_labels.index('no tumor')/(denom_malig))
                tumor_diameters = retur["diameters_per_voxel"][tumor_class] #shape (10, 3), we need (30)
                tumor_diameters = torch.cat([tumor_diameters[i] for i in range(tumor_diameters.shape[0])], dim=0)
                tumor_volumes = retur["voumes_per_voxel"][tumor_class]
                
        
        tumor_info['tumor_location_mask'] = tumor_location_mask.squeeze(0).float() # 3d mask
        tumor_info['tumor_allowed_slices'] = allowed_tumor_slices.squeeze(0).float() # 3d mask
        tumor_info['tumor_organ_name'] = torch.tensor([tumor_class_idx]).float() # one scalar, normalized
        tumor_info['tumor_count'] = torch.tensor([tumor_count/10]) # one scalar, normalized. We add 1 so unknown becomes 0
        tumor_info['tumor_attenuation'] = torch.tensor(tumor_attenuation).float() # list of 10 elements, normalized, one element per tumor
        tumor_info['tumor_malignancy'] = torch.tensor(tumor_malignancy).float() #list of 10 elements, normalized, one element per tumor
        tumor_info['known_tumor_count'] = torch.tensor([0]).float() if unknown_tumor_count else torch.tensor([1]).float()
        tumor_info['tumor_diameters'] = tumor_diameters.float() #list of 30 elements, 3 per tumor
        tumor_info['tumor_volumes'] = tumor_volumes.float() #list of 10 elements, one element per tumor
        
        
        #concatenate all lists as a single tensor
        tumor_info['tumor_info_vector_organ_count_attenuation_malignancy'] = torch.concat(
            [
            tumor_info['tumor_organ_name'],
            tumor_info['tumor_count'],
            tumor_info['known_tumor_count'],
            tumor_info['tumor_attenuation'],
            tumor_info['tumor_malignancy'],
            tumor_info['tumor_diameters'],
            tumor_info['tumor_volumes'],
            ],dim=0
        ).float()
        assert len(tumor_info['tumor_info_vector_organ_count_attenuation_malignancy'].shape) == 1
        
        if tumor_info['tumor_info_vector_organ_count_attenuation_malignancy'].shape[0]!=63:
            print(f'Shape of tumor_info_vector_organ_count_attenuation_malignancy is {tumor_info["tumor_info_vector_organ_count_attenuation_malignancy"].shape[0]}, expected 23', flush=True, file=sys.stderr)
            print(f'Caase with problem: {self.img_list[idx]}', flush=True, file=sys.stderr)
            print(f'Shape and dtype for each tumor info element for {self.img_list[idx]}:',flush=True, file=sys.stderr)
            print(f"  tumor_location_mask: {tumor_info['tumor_location_mask'].shape}, {tumor_info['tumor_location_mask'].dtype}",flush=True, file=sys.stderr)
            print(f"  tumor_allowed_slices: {tumor_info['tumor_allowed_slices'].shape}, {tumor_info['tumor_allowed_slices'].dtype}",flush=True, file=sys.stderr)
            print(f"  tumor_organ_name: {tumor_info['tumor_organ_name'].shape}, {tumor_info['tumor_organ_name'].dtype}",flush=True, file=sys.stderr)
            print(f"  tumor_count: {tumor_info['tumor_count'].shape}, {tumor_info['tumor_count'].dtype}",flush=True, file=sys.stderr)
            print(f"  tumor_attenuation: {tumor_info['tumor_attenuation'].shape}, {tumor_info['tumor_attenuation'].dtype}",flush=True, file=sys.stderr)
            print(f"  tumor_malignancy: {tumor_info['tumor_malignancy'].shape}, {tumor_info['tumor_malignancy'].dtype}",flush=True, file=sys.stderr)
            print(f"  known_tumor_count: {tumor_info['known_tumor_count'].shape}, {tumor_info['known_tumor_count'].dtype}",flush=True, file=sys.stderr)
            print(f"  tumor_info_vector_organ_count_attenuation_malignancy: {tumor_info['tumor_info_vector_organ_count_attenuation_malignancy'].shape}, {tumor_info['tumor_info_vector_organ_count_attenuation_malignancy'].dtype}",flush=True, file=sys.stderr)
            print(f"  tumor_diameters: {tumor_info['tumor_diameters'].shape}, {tumor_info['tumor_diameters'].dtype}",flush=True, file=sys.stderr)
            print(f"  tumor_volumes: {tumor_info['tumor_volumes'].shape}, {tumor_info['tumor_volumes'].dtype}",flush=True, file=sys.stderr)
            
            print(f'Values of each element:',flush=True, file=sys.stderr)
            print(f"  tumor_organ_name: {tumor_info['tumor_organ_name']}",flush=True, file=sys.stderr)
            print(f"  tumor_count: {tumor_info['tumor_count']}",flush=True, file=sys.stderr)
            print(f"  tumor_attenuation: {tumor_info['tumor_attenuation']}",flush=True, file=sys.stderr)
            print(f"  tumor_malignancy: {tumor_info['tumor_malignancy']}",flush=True, file=sys.stderr)
            print(f"  known_tumor_count: {tumor_info['known_tumor_count']}",flush=True, file=sys.stderr)
            print(f"  tumor_info_vector_organ_count_attenuation_malignancy: {tumor_info['tumor_info_vector_organ_count_attenuation_malignancy']}",flush=True, file=sys.stderr)
            print(f"  tumor_diameters: {tumor_info['tumor_diameters']}",flush=True, file=sys.stderr)
            print(f"  tumor_volumes: {tumor_info['tumor_volumes']}",flush=True, file=sys.stderr)
            
            raise ValueError('tumor_info_vector_organ_count_attenuation_malignancy has incorrect shape')
            
        return tumor_info

    def save_sanity_check(self, img, lab, idx):
        """Save the image and labels to NIfTI format for sanity checking."""
        if self.saved_count < 10:
            save_dir = './SanityCheck'
            os.makedirs(save_dir, exist_ok=True)

            img_folder = os.path.join(save_dir, f'img{self.saved_count + 1}')
            os.makedirs(img_folder, exist_ok=True)

            # Save the image
            img_nifti = sitk.GetImageFromArray(img.squeeze().cpu().numpy())
            ##print shape
            ##print('Shape:', img.squeeze().cpu().numpy().shape)
            img_nifti.SetSpacing(self.spacing_list[idx])
            sitk.WriteImage(img_nifti, os.path.join(img_folder, 'CT.nii.gz'))

            # Save the labels
            for i, cls in enumerate(self.classes):
                label_array = (lab[i].squeeze().cpu().numpy()).astype(np.int8)
                if label_array.max() > 0:  # Save only if the label exists
                    label_nifti = sitk.GetImageFromArray(label_array)
                    label_nifti.SetSpacing(self.spacing_list[idx])
                    sitk.WriteImage(label_nifti, os.path.join(img_folder, f'{cls}.nii.gz'))

            self.saved_count += 1
    
    def assign_labels(self, tensor_lab, idx):
        """
        UFO data is not annotated per-voxel for some classes, making classes and classes_UFO missmatch.
        This function adds zero channels for missing classes, converting a label from UFO format to atlas format.
        It also creates a unk_channels dict explining which class is UNKNOWN.
        Some classes are missing but we know they are truly zero, so we do not add them to unk_channels.
        If the missing class is not a lesion (e.g., an organ we do not have pseudo-annotations for) we assign it to unk_channels.
        If it is a lesion, we check tumor_dict. tumor_dict, extracted from the report, explains which organ/segments present tumors. We check if these organs/segments are present in tensor_lab (cropped).
        Tumor labels with a corresponding tumor segment in the crop -> assign to unk_channels (we do not know where the tumor is).
        Tumor labels withour corresponding tumor segment in the crop -> assign label 0 (negative for tumor in the crop).
        """
        clss_to_idx = {clss: i for i, clss in enumerate(self.classes)}
        clss_UFO_to_idx = {clss: i for i, clss in enumerate(self.classes_UFO)}
        all_data,tumor_dict=self.get_tumor_segment_labels(idx)
        tumor_segments=all_data['tumor_segments']
        if self.debugging:print(f'Tumor segments 1: {tumor_segments}', flush=True, file=sys.stderr)
        
        
        #if the tumor sub-segment is not specified, get all subsegments for the tumor organ
        #print('Tumor segments:', tumor_segments, flush=True, file=sys.stderr)
        #print('Tumor organs:', all_data['tumor_organs'], flush=True, file=sys.stderr)
        for tumor_organ in all_data['tumor_organs']:
            if isinstance(tumor_organ,str) and tumor_organ=='liver':
                if not any('segment' in item for item in all_data['tumor_segments_flat']):
                    if 'liver' not in tumor_segments:
                        tumor_segments.append('liver')
            elif isinstance(tumor_organ,str) and tumor_organ=='pancreas':
                if not any('head' in item for item in all_data['tumor_segments_flat']) and not any('body' in item for item in all_data['tumor_segments_flat']) and not any('tail' in item for item in all_data['tumor_segments_flat']):
                    if 'pancreas' not in tumor_segments:
                        tumor_segments.append('pancreas')         
            elif isinstance(tumor_organ,str) and tumor_organ in ['kidney','adrenal gland','breast','lung','femur']:
                flag = False
                for seg in tumor_segments:
                    if isinstance(seg,list):
                        for s in seg:
                            if 'right' in s or 'left' in s:
                                flag = True
                    else:
                        if 'right' in seg or 'left' in seg:
                            flag = True
                if not flag:
                    tumor_segments.append(tumor_organ+'_right')
                    tumor_segments.append(tumor_organ+'_left')
                    #raise ValueError('We do not know if the tumor is in the left or right kidney/adrenal gland/breast/lung. We cannot crop on this organ.')
            else:
                tumor_segments.append(tumor_organ)#if the organ has no segment, we add the organ itself.
                
        if self.debugging:print(f'Tumor segments 2: {tumor_segments}', flush=True, file=sys.stderr)

        #flatten the list of lists
        tmp=[]
        for item in tumor_segments:
            if isinstance(item, list):
                for subitem in item:
                    tmp.append(subitem)
            else:
                if item == 'pancreas':
                    tmp.extend(['pancreas','pancreas head','pancreas body','pancreas tail'])
                elif item == 'liver':
                    tmp.extend(['liver','liver segment 1','liver segment 2','liver segment 3','liver segment 4',
                                    'liver segment 5','liver segment 6','liver segment 7','liver segment 8'])
                else:
                    tmp.append(item)
        tumor_segments=tmp

        tumor_segments=list(set(tumor_segments))
        
        
        
        #convert to standard label names:
        tumor_segments=[seg.replace(' ','_').replace('gallbladder','gall_bladder').replace('adrenal gland','adrenal_gland') for seg in tumor_segments]
        
        
        #remove subsegments that are not in the classes_UFO
        if not self._has_liver_subsegs_UFO:
            tmp=[]
            for seg in tumor_segments:
                if ('liver' in seg) and ('segment' in seg):
                    tmp.append('liver')
                else:
                    tmp.append(seg)
            tumor_segments=list(set(tmp))
        if not self._has_pancreas_subsegs_UFO:
            tmp=[]
            for seg in tumor_segments:
                if ('pancreas' in seg) and (('head' in seg) or ('body' in seg) or ('tail' in seg)):
                    tmp.append('pancreas')
                else:
                    tmp.append(seg)
            tumor_segments=list(set(tmp))
        
        #assert these are in classes:
        for seg in tumor_segments:
            if seg!='uterus' and seg not in self.classes_UFO:
                raise ValueError('Segment not in classes:',seg)
            
        if self.debugging:print(f'Tumor segments 3: {tumor_segments}', flush=True, file=sys.stderr)
        
        #tumor_segments represents all organ/subsegments with tumors in the whole ct
        #which lesion classes to add unk? check which of the tumor_segments are in the crop.
        zeros = torch.zeros((tensor_lab.shape[-3],tensor_lab.shape[-2],tensor_lab.shape[-1])).type_as(tensor_lab)
        unk_segments={}
        #this variable will create a mask of the segments in the crop that have tumors in report annotation
        
        unk_lesions=[]
        for seg in tumor_segments:
            seg_idx=clss_UFO_to_idx[seg.replace('uterus','prostate')]
            #print('Segment being added to unk:',seg,'Its UFO index is:',seg_idx, flush=True, file=sys.stderr)
            if tensor_lab[seg_idx].max()>0:#organ sub-segment with tumor inside the crop
                #there is a tumor segment in the crop
                #what is the organ of the tumor segment?
                if 'uterus' in seg:
                    organ = 'uterus'
                else:
                    organ = None
                    for org in self.classes_UFO:
                        if org in seg:
                            organ = org
                    if organ is None:
                        raise ValueError('Organ not in organs:',seg)
                
                if 'liver' in seg:
                    if 'liver' not in unk_segments:
                        unk_segments['liver']=zeros.clone()
                    unk_segments['liver'][tensor_lab[seg_idx]>0]=1
                elif 'pancreas' in seg:
                    if 'pancreas' not in unk_segments:
                        unk_segments['pancreas']=zeros.clone()
                    unk_segments['pancreas'][tensor_lab[seg_idx]>0]=1
                else:
                    x = zeros.clone()
                    x[tensor_lab[seg_idx]>0] = 1
                    unk_segments[organ] = x
                
                if '_segment' in seg:
                    organ=seg[:seg.rfind('_segment')]
                else:
                    organ=seg
                organ=organ.replace('_head','').replace('_body','').replace('_tail','')
                unk_lesions.append(organ)
            else:
                #raise ValueError(f'Segment not in crop: {seg}')
                print(f'Segment not in crop:',seg, flush=True, file=sys.stderr)
            
                
        unk_lesions=list(set(unk_lesions))
        if self.debugging:print('unk lesions:', unk_lesions, flush=True, file=sys.stderr)
        if self.debugging:print('unk segments keys:', unk_segments.keys(), flush=True, file=sys.stderr)

        unk_channels={}
        unk_channels_list=[]
        label=[]
        #print('Shape of tensor_lab before assigning labels:', tensor_lab.shape, flush=True, file=sys.stderr)
        assert len(tensor_lab.shape) == 4
        for j,clss in enumerate(self.classes,0):
            #print('Class:',clss,flush=True, file=sys.stderr)
            if clss in self.classes_UFO:
                label.append(tensor_lab[clss_UFO_to_idx[clss]])
                unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
            else:
                is_lesion  = any(s in clss.lower() for s in ('lesion', 'cyst', 'pdac', 'pnet','tumor'))
                if not is_lesion:
                    if not self._has_liver_subsegs_UFO and 'liver_' in clss:
                        clss = 'liver'
                    if not self._has_pancreas_subsegs_UFO and 'pancreas_' in clss:
                        clss = 'pancreas'
                    if clss=='liver':
                        unk_channels[clss]=j
                        #join all liver segments
                        l=torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0])
                        for i in [1,2,3,4,5,6,7,8]:
                            l=torch.logical_or(l,tensor_lab[clss_UFO_to_idx['liver_segment_%i'%i]])
                        if 'liver' in self.classes_UFO:
                            #also or the whole liver
                            l=torch.logical_or(l,tensor_lab[clss_UFO_to_idx['liver']])
                        label.append(l)
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))#this channel is knwon, assign zero to unk_channels_list
                    elif clss=='pancreas':
                        unk_channels[clss]=j
                        #join all pancreas segments
                        l=torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0])
                        for i in ['head','body','tail']:
                            l=torch.logical_or(l,tensor_lab[clss_UFO_to_idx['pancreas_%s'%i]])
                        if 'pancreas' in self.classes_UFO:
                            #also or the whole pancreas
                            l=torch.logical_or(l,tensor_lab[clss_UFO_to_idx['pancreas']])
                        label.append(l)
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))#this channel is knwon, assign zero to unk_channels_list
                    else:
                        label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                        unk_channels[clss]=j
                        unk_channels_list.append(torch.ones(tensor_lab[0].shape).type_as(tensor_lab[0]))#no pixel is known for this channel, assign 1 to unk_channels_list

                else:
                    #lesion class
                    #print('Lesion class:',clss,flush=True, file=sys.stderr)
                    #check if there is a tumorous segment for this lesion in the crop
                    tumor_present=False
                    for organ in unk_lesions:
                        if 'bladder' in organ:
                            #as the word bladder is inside the word gallbladder, we must be careful here
                            if (('gallbladder' in organ) or ('gall_bladder' in organ)) and clss=='gallbladder_lesion':
                                    label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                    unk_channels[clss]=j
                                    unk_channels_list.append(unk_segments[organ])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                    tumor_present=True
                                    break
                            else:
                                if organ in clss and clss == 'bladder_lesion':
                                    label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                    unk_channels[clss]=j
                                    unk_channels_list.append(unk_segments[organ])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                    tumor_present=True
                                    break
                        
                        else:
                            if 'adrenal' in organ and clss=='adrenal_lesion':
                                label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                unk_channels[clss]=j
                                unk_channels_list.append(unk_segments[organ])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                tumor_present=True
                                break
                            elif ('uterus' in organ) and clss=='uterus_lesion':
                                label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                unk_channels[clss]=j
                                unk_channels_list.append(unk_segments['uterus'])#remember the uterus class was annotated as prostate
                                tumor_present=True
                                break
                            elif ('kidney' in organ) and clss=='kidney_lesion':
                                label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                unk_channels[clss]=j
                                unk_channels_list.append(unk_segments[organ])
                                if self.debugging:print(f'Added unk for lesion {clss}, using {organ}', flush=True, file=sys.stderr)
                                tumor_present=True
                                break
                            elif organ.replace('pancreas','pancreatic') in clss:
                                label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                unk_channels[clss]=j
                                if 'liver' in clss:
                                    unk_channels_list.append(unk_segments['liver'])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                elif 'pancreatic' in clss:
                                    unk_channels_list.append(unk_segments['pancreas'])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                else:
                                    unk_channels_list.append(unk_segments[organ])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                tumor_present=True
                                if self.debugging:print(f'Added unk for lesion {clss}', flush=True, file=sys.stderr)
                                break
                        
                            
                    #if not:
                    #assign label 0
                    if not tumor_present:
                        #negative for the tumor
                        label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
        
         
    
    
        label=torch.stack(label,dim=0)
        unk_channels_list=torch.stack(unk_channels_list,0)
        
        
        if len(unk_lesions)>0:
            assert unk_channels_list.sum()>0, 'unk_channels_list should have some non-zero voxels if there are tumors in the crop'
        assert len(label.shape) == 4
        return label,unk_channels,unk_channels_list.type_as(label)
    
    def define_unknown_voxels(self, label, idx):
        """
        Defines the unknown voxels in the image. Unlike assign_labels, this function assumes your labels (tensor_lab) are already in the final format, with the correct number of channels, in the order of self.classes.
        unk_channels is a dictionary with the classes that are unknown.
        """

        #we must first re-create the tensor_lab (input of the assign_labels function), from label, the output of the assign_labels function.
        clss_to_idx = {clss: i for i, clss in enumerate(self.classes)}
        clss_UFO_to_idx = {clss: i for i, clss in enumerate(self.classes_UFO)}
        tensor_lab = []
        for j,clss in enumerate(self.classes_UFO,0):
            ##print('j:',j,flush=True, file=sys.stderr)
            ##print('clss:',clss,flush=True, file=sys.stderr)
            if clss=='background':
                #add zeros as placeholder
                tensor_lab.append(torch.zeros(label[0].shape).type_as(label[0]))
                bkg=j
            else:
                tensor_lab.append(label[clss_to_idx[clss]])
        tensor_lab=torch.stack(tensor_lab,dim=0)
        #add to background the opposite of all other classes
        tensor_lab[bkg]=(tensor_lab.sum(dim=0)==0).type_as(tensor_lab[0])
        

        #now we can use the assign_labels function
        #to define the unknown voxels
    
        #convert to atlas format
        label_out,unk_channels,unk_channels_list=self.assign_labels(tensor_lab,idx)
        #sanity check: see if label_out matches label
        assert (torch.equal(label_out,label))
        
        return unk_channels_list
    
    
    def get_malignancy_annotated_per_voxel(self, idx, labels):
        """
        Returns a vector aligned with self.classes (non-lesions = NaN).
        Values: 1 (malignant), 0 (benign), NaN (unknown or empty channel).
        """
        if not self.load_malignancy:
            return torch.tensor([float('nan')]*len(self.classes)).float().to(labels.device)

        

        # --- robust BDMAP numeric id ---
        base_id = extract_id(self.img_list[idx])
        if 'BDMAP' not in base_id:
            raise ValueError(f'Expected BDMAP ID in {self.img_list[idx]}')
        m = re.search(r'(\d+)$', base_id)
        bdmap_num = int(m.group(1))
        
        # which channels are actually present
        present = (labels.sum(dim=(-1, -2, -3)) > 0)

        malignants = []

        # --- fallback windows only label PRESENT lesion channels ---
        if 14484 <= bdmap_num <= 14637:
            #esophagus lesion dataset
            for i, c in enumerate(self.classes):
                if ('lesion' in c) and ('esophagus' in c) and present[i].item():
                    malignants.append(1.0)
                else:
                    malignants.append(float('nan'))
        elif 22412 <= bdmap_num <= 22490:
            #uterus lesion dataset
            for i, c in enumerate(self.classes):
                if ('lesion' in c) and ('uterus' in c) and present[i].item():
                    malignants.append(1.0)
                else:
                    malignants.append(float('nan'))
        else:
            # --- metadata path ---
            try:
                tumor_df = self.read_report(idx,original=True)
            except:
                #cases outside of metadata
                return torch.tensor([float('nan')]*len(self.classes)).float().to(labels.device)

            organs_with_tumor = set()
            for i, c in enumerate(self.classes):
                if 'lesion' in c and present[i].item():
                    organs_with_tumor.add(tumor_to_organ(c).lower())

            # build sets (lowercased)
            mal_set = set(map(str.lower,
                            tumor_df[tumor_df['malignant_benign'] == 'malignant']['Standardized Organ'].tolist()))
            mal_extra = set(map(str.lower,
                                tumor_df[tumor_df['pathology_and_radiology_malignant'] == 'yes']['Standardized Organ'].tolist()))
            if 'duodenum' in mal_extra:
                mal_set.add('duodenum')

            ben_set = set(map(str.lower,
                            tumor_df[tumor_df['malignant_benign'] == 'benign']['Standardized Organ'].tolist()))
            ben_extra = set(map(str.lower,
                                tumor_df[tumor_df['pathology_and_radiology_malignant'] == 'no']['Standardized Organ'].tolist()))
            if 'duodenum' in ben_extra:
                ben_set.add('duodenum')

            for i, c in enumerate(self.classes):
                if 'lesion' not in c:
                    malignants.append(float('nan'))
                    continue
                organ = tumor_to_organ(c).lower()
                if organ not in organs_with_tumor:
                    malignants.append(float('nan'))
                    continue
                if (organ in mal_set) and (organ in ben_set):
                    malignants.append(float('nan'))
                elif organ in mal_set:
                    malignants.append(1.0)
                elif organ in ben_set:
                    malignants.append(0.0)
                else:
                    malignants.append(float('nan'))

        # keep float dtype (for NaNs) and same device as labels
        malignants = torch.tensor(malignants, dtype=torch.float32, device=labels.device)
        assert malignants.shape[0] == labels.shape[0]
        return malignants
        
    
    def resolve_malignancy_conflicts_by_diameter(self,t, *, ignore_zero=True):
        """
        If we have 2 tumors of the same size, one is malignant and the other is not,
        we cannot know which one is which. So we treat both as unknown malignancy.
        t: (N, 2) tensor/array of [diameter, malignancy], malignancy ∈ {0.0, 1.0, NaN}.
        - Group by EXACT diameter (no rounding).
        - A group conflicts if it contains >= 2 distinct labels among {0, 1, NaN}
        (so 0 vs 1, 0 vs NaN, and 1 vs NaN all conflict).
        - For conflicting groups, set malignancy=NaN for all rows in that group.
        - Optionally ignore diameter == 0 rows (padding) via `ignore_zero`.
        """
        is_torch = isinstance(t, torch.Tensor)
        arr = t.detach().cpu().numpy() if is_torch else np.asarray(t, dtype=float)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected shape (N,2), got {arr.shape}")

        diam = arr[:, 0].astype(float, copy=True)
        mal  = arr[:, 1].astype(float, copy=True)

        valid = ~np.isnan(diam)
        if ignore_zero:
            valid &= (diam != 0.0)

        uniques = np.unique(diam[valid])
        for d in uniques:
            idx = np.where((diam == d) & valid)[0]
            if idx.size >= 2:
                vals = mal[idx]
                classes = set()
                if np.any(vals == 0.0): classes.add(0.0)
                if np.any(vals == 1.0): classes.add(1.0)
                if np.any(np.isnan(vals)): classes.add('nan')
                if len(classes) >= 2:      # conflict: two or more of {0, 1, NaN}
                    mal[idx] = np.nan

        out = np.stack([diam, mal], axis=1)
        return torch.from_numpy(out).type_as(t) if is_torch else out
    
    def get_malignancy_old(self,row):
        if (row['Standardized Organ']=='duodenum') and (('ICD' in self.malignant_col) or ('ICD' in self.benign_col)):
            #no ICD for duodenum,
            malignant_col = 'pathology_and_radiology_malignant'
            benign_col = None
        else:
            malignant_col = self.malignant_col
            benign_col = self.benign_col
        
        def _norm(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            return str(x).strip().lower()
        if (malignant_col not in row.index):
            raise ValueError(f'Columns {malignant_col} and {benign_col} must be present in tumor dataframe.')
        m = _norm(row.get(malignant_col, ""))
        if benign_col is None:
            if m == "yes":
                b = "no"
            elif m == "no":
                b = "yes"
            else:
                b = ""
        else:
            b = _norm(row.get(benign_col,    ""))
        if m == "yes" and b != "yes":
            return 1.0
        if b == "yes" and m != "yes":
            return 0.0
        return float("nan")
    
    def get_malignancy(self,row):
        malignant_benign = row['malignant_benign']
        if malignant_benign == 'malignant':
            return 1.0
        elif malignant_benign == 'benign':
            return 0.0
        else:
            return float('nan')
        

    def estimate_tumor_volume(self, idx, tumor_segment_crop):
        """
        Estimates tumor volume from reports. For the segment in the crop.
        Always returns a list of 10 items, padding with 0.
        
        tumors_in_crop_slices: tensor with the paired max diameter and slice for all tumors in the tumor_segment_crop
        """
        _,tumor_dict=self.get_tumor_segment_labels(idx)
        #print('Tumor dict:', tumor_dict)
        #print all column names in tumor_dict
        #print(tumor_dict.columns)
        #print('Sizes:',tumor_dict['Tumor Size (mm)'])
        #print('Cropped on tumor segment:', tumor_segment_crop)
        
        
        if tumor_segment_crop is None or tumor_segment_crop=='random':
            return [0,0,0,0,0,0,0,0,0,0], torch.zeros((10,3)).float(), torch.zeros((10)).float(), torch.tensor([[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan],[0,np.nan]]).float(), torch.tensor([[0, np.nan]] * 10).float() #CT not cropped around a tumor segment
        
        if isinstance(tumor_segment_crop, list):
            pass
        elif isinstance(tumor_segment_crop, str):
            tumor_segment_crop=[tumor_segment_crop]
        else:
            raise ValueError('tumor_segment_crop must be a list or a string.')
        
        #what if tumor_segment_crop has something like head/body?
        
        #is our tumor_segment_crop organ or segment:
        
        if 'segment' in "".join(tumor_segment_crop) or 'head' in "".join(tumor_segment_crop) or 'body' in "".join(tumor_segment_crop) or 'tail' in "".join(tumor_segment_crop) or 'left' in "".join(tumor_segment_crop) or 'right' in "".join(tumor_segment_crop):
            tpe='segment'
            col='Standardized Location'
        else:
            tpe='organ'
            col='Standardized Organ'
        
        tumors_in_crop=[]
        tumor_in_crop_attenuation=[]
        tumors_in_crop_slices=[]
        tumors_in_crop_malignancy=[]
        for row in tumor_dict.iterrows():
            location=row[1][col]
            #print('Location:',location)
            if not isinstance(location, str) or location.lower()=='u':
                continue
            if '/' in location:
                location=location.split(' / ')
            if not isinstance(location, list):
                location=[location]
            if tpe=='segment':
                org_tmp = row[1]['Standardized Organ']
                location = [org_tmp+'_'+loc for loc in location] #e.g. 'liver_segment_1'
            in_crop=True
            for loc in location:
                #print(f'loc is {loc}, tumor_segment_crop is {tumor_segment_crop}', flush=True, file=sys.stderr)
                if loc not in tumor_segment_crop:
                    in_crop=False
                    break
            if in_crop:
                sze = row[1]['Tumor Size (mm)']
                tumors_in_crop.append(sze)
                tumor_in_crop_attenuation.append(row[1]['Standardized Attenuation'])
                if self.load_slices:
                    if coerce_to_bool(row[1]['series matches report']) and (not coerce_to_bool(row[1]['impossible slice'])) and sze != 'u':
                        slc = pd.to_numeric(row[1]['Image'], errors='coerce')
                        slc = np.nan if pd.isna(slc) else float(slc)
                        tumors_in_crop_slices.append(slc)
                    else:
                        tumors_in_crop_slices.append(np.nan)
                else:
                    tumors_in_crop_slices.append(np.nan) #unknown slice
                if self.load_malignancy:
                    tumors_in_crop_malignancy.append(self.get_malignancy(row[1]))
                else:
                    tumors_in_crop_malignancy.append(np.nan)
                    
        #print('Tumors in crop:', tumors_in_crop)#list of strings with sizes
        assert len(tumors_in_crop_slices) == len(tumors_in_crop), f'Tumor slices ({len(tumors_in_crop_slices)}) and tumor sizes ({len(tumors_in_crop)}) must have the same length.'

        #print('Tumor dict:',tumor_dict[['Standardized Organ','Standardized Location','Tumor Size (mm)']])
                
        #estimate volumes for each tumor size
        volumes=[]
        diameters=[]
        for i,size in enumerate(tumors_in_crop):
            if size == 'u':
                #unknown size, -9999999 will represent unknown
                volumes.append(-9999999)
                diameters.append([-9999999,-9999999,-9999999])
                continue
            elif 'x' not in size:
                #single diameter provided. Use ball.
                diameter=float(size)
                volume=(4/3) * math.pi * ((diameter/2) ** 3)#sphere. volume in mm3 (voxels)
                volumes.append(volume)
                diameters.append([diameter,diameter,diameter])
            else:
                #2 or 3 diameterts, use ellipsoid
                sizes=size.split(' x ')
                sizes=[float(s) for s in sizes]
                if self.debugging:print(f'Sizes is: {sizes}', flush=True, file=sys.stderr)
                if len(sizes)==2:
                    #assume 3rd axis is the average of the other two
                    sizes.append(sum(sizes)/2)
                elif len(sizes) > 3:                
                    # more than 3 numbers, take top 3
                    sizes = sorted(sizes, reverse=True)[:3]
                #ellipsoid volume
                volume=(4/3) * math.pi * ((sizes[0]/2) * (sizes[1]/2) * (sizes[2]/2))
                volumes.append(volume)
                diameters.append(sizes)

        #print('Estimated volumes:',volumes)
        
        att=[]
        for attenuation in tumor_in_crop_attenuation:
            if attenuation=='high':
                att.append(1.0)
            elif attenuation=='low':
                att.append(-1.0)
            else:
                att.append(999)
                
        #print('Estimated attenuations:',att)
                
        #pair the sizes and slices:
        assert len(diameters) == len(tumors_in_crop_slices), f'Diameters ({len(diameters)}) and tumor slices ({len(tumors_in_crop_slices)}) must have the same length.'
        assert len(tumors_in_crop_malignancy) == len(diameters), "Malignancy list must align with diameters/slices."
        for i in range(len(diameters)):
            diameter = max(diameters[i])#maximum tumor diameter
            slc = tumors_in_crop_slices[i]
            tumors_in_crop_slices[i] = [diameter,slc]
            tumors_in_crop_malignancy[i] = [diameter,tumors_in_crop_malignancy[i]]
        #print('Tumors in crop slices:', tumors_in_crop_slices)

        #pdding to 10 tumors
        for i in range(len(volumes),10):
            volumes.append(0)
            diameters.append([0,0,0])
            att.append(0)
            tumors_in_crop_slices.append([0,np.nan])
            tumors_in_crop_malignancy.append([0,np.nan])
        #attenuation: 1 = hyper, -1 = hypo, 999 = unknownm, 0 = padding (ignore)
        #size and volume: -9999999 = unknown size or unknown number of tumors, 0 = padding (ignore)
        
        
        #print('Tumors in crop slices 2:', tumors_in_crop_slices)
        
        #cap length to 10 tumors
        if len(volumes)>10 or len(diameters)>10 or len(att)>10 or len(tumors_in_crop_slices)>10 or len(tumors_in_crop_malignancy)>10:
            volumes=volumes[:10]
            diameters=diameters[:10]
            att=att[:10]
            tumors_in_crop_slices=tumors_in_crop_slices[:10]
            tumors_in_crop_malignancy=tumors_in_crop_malignancy[:10]
            
        tumors_in_crop_malignancy=self.resolve_malignancy_conflicts_by_diameter(tumors_in_crop_malignancy)
        
        if self.debugging:print(f'Created diameters: {diameters}')
        if self.debugging:print(f'Created tumors_in_crop_slices: {tumors_in_crop_slices}')
        if self.debugging:print(f'Created tumors_in_crop_malignancy: {tumors_in_crop_malignancy}')
            
        return volumes,torch.tensor(diameters).float(), torch.tensor(att).float(), torch.tensor(tumors_in_crop_slices).float(), torch.tensor(tumors_in_crop_malignancy).float()
    
    def SanityAssertOutput(self, tensor_lab, unk_channels_tensor,tumor_volumes_in_crop,chosen_segment_mask,
                           binary_slices_mask=None,slices_mask=None):
                                #tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float()
        classes=sorted(self.classes)
        #assert shapes
        assert len(tensor_lab.shape)==4 , 'tensor_lab must have 4 dimensions'
        assert tensor_lab.shape[0]==len(classes), 'Number of classes in tensor_lab (%i) does not match number of classes (%i)'%(tensor_lab.shape,len(classes))
        assert unk_channels_tensor.shape[0]==len(classes), f'Number of classes in unk_channels_tensor ({unk_channels_tensor.shape}) does not match number of classes ({len(classes)})'
        assert chosen_segment_mask.shape[0]==len(classes), f'Number of classes in chosen_segment_mask ({chosen_segment_mask.shape}) does not match number of classes ({len(classes)})'
        assert (tensor_lab.shape==unk_channels_tensor.shape) and (tensor_lab.shape==chosen_segment_mask.shape), f'tensor_lab, unk_channels_tensor and chosen_segment_mask must have the same shape. tensor_lab: %s, unk_channels_tensor: %s, tumor_volumes_in_crop: %s'%(tensor_lab.shape,unk_channels_tensor.shape,tumor_volumes_in_crop.shape)

        
        #save examples
        sample=self.current_sample
        sample=sample[sample.rfind('BDMAP_'):sample.rfind('.')]
        if self.counter<10:
            debug_save_labels(tensor_lab,sample+'_y',self.classes,out_dir=self.sanity_path)
            debug_save_labels(chosen_segment_mask,sample+'_chosen_segment_mask',self.classes,out_dir=self.sanity_path)
            if binary_slices_mask is not None:
                debug_save_labels(chosen_segment_mask*binary_slices_mask,sample+'_binary_slice_mask',self.classes,out_dir=self.sanity_path)
                debug_save_labels(chosen_segment_mask*slices_mask,sample+'_slice_mask',self.classes,out_dir=self.sanity_path)
            debug_save_labels(unk_channels_tensor,sample+'_unk_voxels',self.classes,out_dir=self.sanity_path)
            self.counter+=1

        #assert that unk_channels_tensor and chosen_segment_mask are 0 for all non lesion classes
        missing_classes=set(classes)-set(self.classes_UFO)-{'liver','pancreas'}
        missing_classes=list(missing_classes)
        #print('Missing classes:', missing_classes,flush=True, file=sys.stderr)
        unk_cls=[]
        known_cls=[]
        for i,clss in enumerate(classes):
            if 'lesion' in clss.lower() or clss in missing_classes: 
                unk_cls.append(i)
            else:
                known_cls.append(i)
        if not unk_channels_tensor[known_cls].sum().item()==0:
            for i,clss in enumerate(classes,0):
                if i in unk_cls:
                    continue
                else:
                    if unk_channels_tensor[i].sum().item()!=0:
                        if self.debugging:print('Class with unk voxels:',clss,'Sample is:',sample)
                        
        assert unk_channels_tensor[known_cls].sum().item()==0
        assert chosen_segment_mask[known_cls].sum().item()==0

        #print('Assertions passed!',flush=True, file=sys.stderr)

    def measure_tumors_mask(self, tensor_lab):
        """
        measure tumor diameters from per-voxel label
        """
        assert len(tensor_lab.shape) == 4, f"tensor_lab must have 4 dimensions (C, H, W, D), we got {tensor_lab.shape}"
        tumor_volumes = {}
        tumor_diameters ={}
        for i, clss in enumerate(self.classes):
            if 'lesion' in clss.lower():
                l = tensor_lab[i]
                if l.sum() == 0:
                    #no lesion, append 10 zeros
                    tumor_volumes[clss] = torch.tensor([0,0,0,0,0,0,0,0,0,0]).float().tolist()
                    tumor_diameters[clss] = torch.zeros((10, 3)).float().tolist()
                    continue
                #calculate diameters
                dct = analyze_nth_largest_connected_component(l)
                v = []
                d = []
                for tumor_id in sorted(dct.keys()):
                    vol = dct[tumor_id]['volume']
                    diam = [dct[tumor_id]['longest_diameter'],
                            dct[tumor_id]['perpendicular_diameter'],
                            0.5*(dct[tumor_id]['longest_diameter'] + dct[tumor_id]['perpendicular_diameter'])]  # average of the two
                    d.append(diam)
                    v.append(vol)
                #zero padding to 10 tumors
                while len(v) < 10:
                    v.append(0)
                    d.append([0, 0, 0])
                if (len(v) > 10) or (len(d) > 10):
                    pairs = sorted(zip(v, d), key=lambda x: x[0], reverse=True)#sort by descending volume
                    v, d = map(list, zip(*pairs))
                    # cap to top-10
                    v = v[:10]
                    d = d[:10]
                tumor_volumes[clss] = torch.tensor(v).float().tolist()
                tumor_diameters[clss] = torch.tensor(d).float().tolist()
        return tumor_volumes, tumor_diameters
    
    
    def measure_tumors_mask_zeros(self, tensor_lab):
        """
        just to allow dataset output to be the same for per-voxel annotated samples and annotated with reports
        """
        assert len(tensor_lab.shape) == 4, "tensor_lab must have 4 dimensions (C, H, W, D)"
        tumor_volumes = {}
        tumor_diameters ={}
        for i, clss in enumerate(self.classes):
            if 'lesion' in clss.lower():
                tumor_volumes[clss] = torch.tensor([0,0,0,0,0,0,0,0,0,0]).float().tolist()
                tumor_diameters[clss] = torch.zeros((10, 3)).float().tolist()
        #tumor_volumes = {self.img_list[idx]:tumor_volumes}
        #tumor_diameters = {self.img_list[idx]:tumor_diameters}
        return tumor_volumes, tumor_diameters
        

        

def npy_to_nii(npy_path, nii_path, spacing=(1.0, 1.0, 1.0),labels=None):
    """
    Reads a .npy file, converts it to a SimpleITK image, 
    sets spacing, and saves as .nii.gz.

    :param npy_path:    Path to the input .npy file.
    :param nii_path:    Path to the output .nii.gz file.
    :param spacing:     Tuple or list specifying the (z, y, x) spacing. 
                        Default is (1.0, 1.0, 1.0).
    """
    # Load the NumPy array
    array = np.load(npy_path)
    #print('Shape of array:', array.shape)
    #squeeze
    array = array.squeeze()
    #print('Shape after squeeze:',array.shape)

    if labels is not None:
        #load yaml labels
        with open(labels, 'r') as f:
            labels = yaml.load(f, Loader=yaml.SafeLoader)
        #print('Yaml loaded')
        #sort
        labels = sorted(labels)
       #print('Labels:',labels)
       #print('Shape of array:',array.shape)
        if len(array.shape) == 4:
            #label
            if array.shape[0] < len(labels):
                #unpack
                array = np.unpackbits(array, axis=0)
                array = array[:len(labels)]
            os.makedirs(nii_path.replace('.nii.gz',''), exist_ok=True)
            for label in labels:
                #save each label
                sitk_image = sitk.GetImageFromArray(array[labels.index(label)])
                sitk_image.SetSpacing(spacing)
                sitk.WriteImage(sitk_image, os.path.join(nii_path.replace('.nii.gz',''),label+'.nii.gz'))
               #print('Saved:', os.path.join(nii_path.replace('.nii.gz',''),label+'.nii.gz'))
    else:
       #print('No labels provided, saving as a single volume')
        # Convert NumPy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(array)

        # Optionally set image spacing (if known)
        sitk_image.SetSpacing(spacing)

        # Write to .nii.gz
        sitk.WriteImage(sitk_image, nii_path)


def debug_save_labels(labels: torch.Tensor,
                      name='',
                      label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      out_dir: str = "./DatasetSanityMultiTumor",
                      batch_idx: int = 0):
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
    import nibabel as nib
    # 1. Create output folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Load and sort label names
    if not isinstance(label_names, list):
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)  # e.g. ["liver", "kidney", "pancreas", ...]
        
    label_names_sorted = sorted(label_names)  # sort alphabetically
    
    # 3. Basic shape check
    if len(labels.shape)==4:
        labels = labels.unsqueeze(0)
    assert len(labels.shape) == 5
    B, C, H, W, D = labels.shape
    assert batch_idx < B, f"batch_idx={batch_idx} is out of range for B={B}."
    assert C == len(label_names_sorted), (
        f"Number of channels (C={C}) does not match the number of label names "
        f"(={len(label_names_sorted)})."
    )
    
    # 4. Extract just the batch element we want
    #    This will have shape (C, H, W, D).
    label_slice = labels[batch_idx]
    
    # 5. Loop over channels, save each one as a nii.gz
    for c in range(C):
        # Move channel c to CPU numpy for saving
        channel_data = label_slice[c].detach().cpu().numpy()
        
        # Build a simple identity affine; if you have real metadata, replace it
        affine = np.eye(4, dtype=np.float32)
        
        # Convert to float32 (or int16, float64, etc.)
        channel_data = channel_data.astype(np.float32)
        
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(channel_data, affine)
        
        # Derive a filename from the label name
        channel_label_name = label_names_sorted[c]
        try:
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        except:
            #remove the folder if it exists and create it again
            try:
                shutil.rmtree(os.path.join(out_dir, name))
                os.makedirs(os.path.join(out_dir, name), exist_ok=True)
            except:
                pass
        
        try:
            out_path = os.path.join(out_dir, f"{name}/{channel_label_name}.nii.gz")
            # Save
            nib.save(nifti_img, out_path)
        except:
            pass #we may have race conditions that raises here.
        
    #print(f"Saved to {out_path}")

def canonical_organ(tumor_name):
        """
        Convert a tumor class name to an organ class name:
        - Remove '_lesion'
        - Substitutes some known patterns (like 'pancreatic' -> 'pancreas')
        """
        if isinstance(tumor_name, list):
            tumor_name = tumor_name[0] #right and left
            assert ('right' in tumor_name) or ('left' in tumor_name), 'Tumor name must be a list with right and left'
        base = tumor_name.replace('_lesion', '')
        tumor_name = tumor_name.replace('_right', '').replace('_left', '')
        lower_base = base.lower()
        if 'pancrea' in lower_base:
            return 'pancreas'
        elif 'kidney' in lower_base:
            return 'kidney'
        elif 'adrenal' in lower_base:
            return 'adrenal'
        elif 'lung' in lower_base:
            return 'lung'
        elif 'femur' in lower_base:
            return 'femur'
        elif 'gall' in lower_base:
            return 'gall_bladder'#added 28 apr 2025
        else:
            return base
        
def _organ_side(name):
    """
    Extract 'right' / 'left' / None from an organ key like 'kidney_right' or 'adrenal_gland_left'.
    Lists carrying BOTH sides (e.g. ['kidney_right','kidney_left']) collapse to None
    (no side restriction). Inputs without side info return None.
    """
    if isinstance(name, list):
        sides = {_organ_side(item) for item in name}
        sides.discard(None)
        if len(sides) == 1:
            return next(iter(sides))
        return None
    if not isinstance(name, str):
        return None
    s = name.lower()
    has_r = 'right' in s
    has_l = 'left' in s
    if has_r and not has_l:
        return 'right'
    if has_l and not has_r:
        return 'left'
    return None


def df_filter_by_canonical_organ(metadata_df: pd.DataFrame, chosen_organ) -> pd.DataFrame:
    """
    Returns a filtered view/copy of metadata_df where Standardized Organ matches chosen_organ
    after applying canonical_organ() to both sides.

    When chosen_organ encodes a side (e.g. 'kidney_right'), metadata_df is first restricted
    to rows whose 'Standardized Location' carries the matching side word — so a request for
    one paired organ's side never picks up the contralateral side's tumor rows.

    Does NOT modify metadata_df.
    """
    if metadata_df is None or len(metadata_df) == 0:
        return metadata_df

    # side pre-filter: only the paired organs (kidney, adrenal_gland, lung, femur) carry
    # 'right'/'left' in chosen_organ; their Standardized Location uses bare 'left'/'right'/'u'.
    chosen_side = _organ_side(chosen_organ)
    if chosen_side is not None and 'Standardized Location' in metadata_df.columns:
        loc = metadata_df['Standardized Location'].astype(str).str.lower()
        other = 'left' if chosen_side == 'right' else 'right'
        metadata_df = metadata_df.loc[loc.str.contains(chosen_side, na=False) & ~loc.str.contains(other, na=False)]

    # canonicalize the query
    chosen_can = canonical_organ(chosen_organ)

    # canonicalize the column WITHOUT changing the original df
    # (use .astype(str) only on non-null entries; keep nulls as nulls)
    col = metadata_df["Standardized Organ"]
    col_can = col.where(col.notna(), None).map(lambda x: canonical_organ(x) if x is not None else None)

    # boolean mask
    mask = (col_can == chosen_can)

    # return filtered df (copy optional; I’d keep it as a view unless you plan to edit it)
    return metadata_df.loc[mask]
        
#measure tumors
import scipy
import scipy.ndimage as ndimage
from skimage.transform import rotate
from scipy.spatial.distance import pdist, squareform
from skimage import measure

def measure_vertical_span(binary_image):
    y_coords, x_coords = np.where(binary_image == 1)
    vertical_span = np.max(y_coords) - np.min(y_coords)
    return vertical_span

def analyze_nth_largest_connected_component(tensor_3d, ns=None,th=None,erode=0,
                                            ct=None):
    #erode: any tumor that disapears afer the binary erosion is ignored.
    #print('Received segments:',segments)
    assert len(tensor_3d.shape)==3, 'Input tensor must be 3D (C, H, W)'
    array_3d = tensor_3d.cpu().numpy()
    
    from math import atan2, degrees
    
    # Define the connectivity (3x3x3 for 26-connectivity)
    structure = np.ones((3, 3, 3), dtype=int)
    
    # Label the connected components
    labeled_array, num_features = ndimage.label(array_3d, structure=structure)
    
    if num_features == 0:
        return None
    
    # Calculate the size of each connected component
    sizes = ndimage.sum(array_3d, labeled_array, range(1, num_features + 1))
    sorted_indices = np.argsort(sizes)[::-1]
    
    outputs = {}
    if ns is None:
        ns=list(range(1,len(sorted_indices)+1))
    included=0
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

        if ct is not None:
            msk = np.where(nth_largest_component_mask > 0.5, 1, 0)
            msk = scipy.ndimage.binary_erosion(msk,
                                                structure=np.ones((1,1,1)),
                                                iterations=1)
            segmented = ct*msk
            mean_hu = segmented.sum()/msk.sum()
            std_hu = segmented[msk != 0].std()
            #print(ct.shape,segmented[msk != 0].shape,msk.sum())
        else:
            mean_hu = None
            std_hu = None
            
        if th is not None:
            if volume<th:
                break

        # Find the indices of the n-th largest connected component
        component_indices = np.where(nth_largest_component_mask)
        
        # Iterate through the z-axis to find the longest diameter
        max_longest_diameter = 0
        longest_points = None
        for z in range(array_3d.shape[2]):
            slice_mask = nth_largest_component_mask[:, :, z]
            if np.any(slice_mask):
                diam, point1, point2=measure_diameter(slice_mask)
                if diam > max_longest_diameter:
                    max_longest_diameter = diam
                    longest_points = (point1, point2)
                    longest_diameter_slice=z

        longest_diameter = max_longest_diameter

        if longest_points is not None:
            # Compute perpendicular distances to the line defined by the longest points
            z=longest_diameter_slice
            slice_mask = nth_largest_component_mask[:, :, z]
            # Calculate the angle to rotate the image so the line between point1 and point2 is parallel to the x-axis
            point1, point2 = longest_points
            angle = degrees(atan2(point2[0] - point1[0], point2[1] - point1[1]))
            
            # Rotate the image
            #rotated_image = rotate_image(slice_mask, angle)
            rotated_image = rotate(slice_mask, angle, resize=True, order=0,
                                   preserve_range=True, mode='constant', cval=0)
            assert np.array_equal(rotated_image, rotated_image.astype(bool))
            
            #print_slice(rotated_image,spacing=resize_factor[1])

            # Measure the vertical span
            perpendicular_diameter = measure_vertical_span(rotated_image)
            perpendicular_point=None
            
        else:
            perpendicular_diameter = 0
            perpendicular_point = None

        # Measure the size along the x, y, and z axes (canonical meaning)
        x = component_indices[0].max() - component_indices[0].min() + 1
        center_x = int(component_indices[0].min() + x/2)
        
        y = component_indices[1].max() - component_indices[1].min() + 1
        center_y = int(component_indices[1].min() + y/2)
        center_y = array_3d.shape[1] - center_y

        included+=1
        outputs[included] = {
            "center_x": center_x,
            "center_y": center_y,
            "slice": longest_diameter_slice,
            "volume": volume,
            "longest_diameter": max(1,math.ceil(longest_diameter)),
            "perpendicular_diameter": max(1,math.ceil(perpendicular_diameter)),
            "mean_hu": mean_hu,
            "std_hu":std_hu,
            }

    return outputs


def measure_diameter(binary_image):
    """
    Measures the diameter of an arbitrary shape in a binary image and returns the diameter
    along with the two extreme points that define this diameter.

    Parameters:
    binary_image (numpy.ndarray): 2D binary array where the shape is represented by 1s.

    Returns:
    tuple: The diameter of the shape and the coordinates of the two extreme points.
    """
    # Find contours of the shape
    contours = measure.find_contours(binary_image, 0.5)

    if not contours:
        raise ValueError("No contours found in the binary image")

    # Assuming there's only one shape, take the first contour
    contour = contours[0]

    # Compute pairwise distances between all contour points
    distances = pdist(contour)
    distance_matrix = squareform(distances)

    # Find the indices of the maximum distance
    max_distance_idx = np.unravel_index(np.argmax(distance_matrix, axis=None), distance_matrix.shape)

    # Get the coordinates of the extreme points
    point1 = contour[max_distance_idx[0]]
    point2 = contour[max_distance_idx[1]]

    # Find the maximum distance
    max_distance = distance_matrix[max_distance_idx]

    return max_distance, point1, point2

from scipy.ndimage import label  as cc_label

def attenutation_tensor_from_mask(
    organ_mask: torch.Tensor,
    tumor_mask: torch.Tensor,
    ct: torch.Tensor,
    c: float = 0.2,
    thresh: float = 0.5,
    max_tumours: int = 10,
    skip_attenuation: bool = False,
):
    """
    Compare HU of up to `max_tumours` largest connected lesions against their organ.

    Returns
    -------
    hyper   - tumour mean  > organ_mean + organ_std * c
    hypo   - tumour mean  < organ_mean - organ_std * c
    iso - tumor mean within organ_mean ± organ_std * c
    unknown - no organ voxels
    """

    # ----- 1. sanity checks -------------------------------------------------
    assert organ_mask.shape == tumor_mask.shape == ct.shape, f"shape mismatch, shapes: {organ_mask.shape}, {tumor_mask.shape}, {ct.shape}"
    organ_np = organ_mask.cpu().numpy() > thresh
    tumour_np = tumor_mask.cpu().numpy() > thresh
    ct_np     = ct.cpu().numpy()

    # ----- 2. label connected components in the tumour mask ----------------
    lbl, n_cc = cc_label(tumour_np, structure=np.ones((3,3,3)))  # 26‑conn.
    if n_cc == 0:                                             # no tumour voxels
        #return ['no tumor']*10, n_cc                                            # choice: "undefined"
        raise ValueError('No tumor voxels found in the tumor mask.')
    # voxel count per label (label 0 is background)
    sizes = np.bincount(lbl.ravel())[1:]                       # drop background
    # indices of the `max_tumours` largest components
    top_idxs = np.argsort(sizes)[::-1][:max_tumours] + 1       # +1 to skip bg
    
    attenuation_label = []
    if not skip_attenuation:
        # ----- 3. tumour mean HU values ----------------------------------------
        tumour_means = []
        for lab in top_idxs:
            vox = ct_np[lbl == lab]
            if vox.size:                                           # shouldn’t be 0
                tumour_means.append(vox.mean())

        # ----- 4. organ statistics (excluding *all* tumours) --------------------
        organ_only = organ_np & (~tumour_np)
        organ_hu   = ct_np[organ_only]
        if organ_hu.size == 0:
            for i in list(range(len(tumour_means))):
                attenuation_label.append('unknown')
            while len(attenuation_label) < max_tumours:
                attenuation_label.append('no tumor')
            return attenuation_label, n_cc
        organ_mean = organ_hu.mean()
        organ_std  = organ_hu.std()
    else:
        for i in list(range(len(top_idxs))):
            attenuation_label.append('unknown')
        while len(attenuation_label) < max_tumours:
            attenuation_label.append('no tumor')
        return attenuation_label, n_cc

    # ----- 5. decision logic ----------------------------------------------
    for i in list(range(len(tumour_means))): #question: is this going from largest to smallest tumor in volume?
        if skip_attenuation:
            attenuation_label.append('unknown')
            continue
        if abs(tumour_means[i] - organ_mean) < (organ_std * c):
            attenuation_label.append('unknown')
        elif tumour_means[i] > organ_mean:
            attenuation_label.append('hyper')
        elif tumour_means[i] < organ_mean:
            attenuation_label.append('hypo')
        else:
            raise ValueError(f'Tumor should be hyper, hypo or iso. We got tumour mean {tumour_means[i]}, organ mean {organ_mean} and organ std {organ_std}.')
    while len(attenuation_label) < max_tumours: #pad
        attenuation_label.append('no tumor')
    
    return attenuation_label, n_cc