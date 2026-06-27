import os
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import BatchSampler

import pandas as pd




class SamplerPerPatientLongitudinal(BatchSampler):
    """
    Anchor by ONE dataset index per batch.
    Expand to BDMAP IDs from the same patient, BUT ONLY those cropped on the
    same organ/subsegment as the anchor (verified in real-time via JSON files).

    Rules:
      - If anchor JSON missing -> load just anchor (pad by getting random ids)
      - If anchor tumor_in_crop is None or 'random' -> load just anchor (pad)
      - For other BDMAPs: if JSON missing -> ignore; if tumor_in_crop None/'random' -> ignore
      - Otherwise include only if crop_key == anchor crop_key
    """

    def __init__(
        self,
        dataset,
        batch_size_total: int,
        samples_per_epoch: int,
        *,
        bdmap_len: int = len("BDMAP_00001111"),
        bdmap_token: str = "BDMAP_",
        reports_df_name: str = "reports",
        patient_id_col: str = "Patient ID",
        bdmap_col: str = "BDMAP_ID",   # set this to "BDMAP_ID" if that's what your reports use
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        crop_json_root_attr: str = "save_destination",  # dataset.save_destination
        crop_key = "subsegment_cropped_in",
        crop_key_fallback = 'tumor_in_crop',  # if provided, will try this key if crop_key missing in JSON
        debug_every_n: int = 0,        # 0 disables; e.g. 200 prints every 200 batches
        debug_max_excluded: int = 1000,  # cap lists in printout
        metadata=None,  # path to the per-patient me
    ):
        assert batch_size_total % world_size == 0, "batch_size_total must be divisible by world_size"
        super().__init__(sampler=None, batch_size=batch_size_total, drop_last=False)

        self.dataset = dataset
        self.dataset_size = len(dataset.img_list)
        self.batch_size_total = batch_size_total
        self.samples_per_epoch = samples_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.meta = pd.read_csv(metadata) if metadata is not None else None
        
        self.bdmap_meta_col = "BDMAP ID"
        self.date_meta_col = "Exam Completed Date"
        if self.meta is not None:
            assert self.bdmap_meta_col in self.meta.columns, \
                f"Metadata missing '{self.bdmap_meta_col}'. Available: {list(self.meta.columns)}"
            assert self.date_meta_col in self.meta.columns, \
                f"Metadata missing '{self.date_meta_col}'. Available: {list(self.meta.columns)}"

            self.meta[self.date_meta_col] = pd.to_datetime(self.meta[self.date_meta_col], errors="coerce")

            # Build a deterministic bdmap -> date mapping ONCE (fast + avoids duplicates ambiguity)
            # If duplicates exist, keep the first after stable sort.
            meta2 = self.meta[[self.bdmap_meta_col, self.date_meta_col]].copy()
            meta2[self.bdmap_meta_col] = meta2[self.bdmap_meta_col].astype(str)
            meta2 = meta2.sort_values([self.bdmap_meta_col, self.date_meta_col], kind="mergesort")
            meta2 = meta2.drop_duplicates(subset=[self.bdmap_meta_col], keep="first")

            self.bdmap_to_date = dict(zip(meta2[self.bdmap_meta_col], meta2[self.date_meta_col]))
        else:
            self.bdmap_to_date = {}
            
        self.bdmap_len = bdmap_len
        self.bdmap_token = bdmap_token

        # where JSONs live
        self.crop_json_root = getattr(dataset, crop_json_root_attr)
        if self.crop_json_root is None:
            raise ValueError(f"dataset.{crop_json_root_attr} is None, cannot read crop JSONs.")
        self.crop_key = crop_key
        self.crop_key_fallback = crop_key_fallback

        # --- idx -> BDMAP ---
        self.idx_to_bdmap: List[str] = []
        for p in dataset.img_list:
            j = p.find(bdmap_token)
            if j < 0:
                raise ValueError(f"Could not find {bdmap_token} in path: {p}")
            self.idx_to_bdmap.append(p[j : j + bdmap_len])

        self.allowed_bdmap = set(self.idx_to_bdmap)

        # --- bdmap -> list of dataset indices (handles duplicates in img_list) ---
        self.bdmap_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, bd in enumerate(self.idx_to_bdmap):
            self.bdmap_to_indices[bd].append(idx)

        # --- reports are REQUIRED ---
        rep = getattr(dataset, reports_df_name, None)
        if rep is None:
            raise ValueError(f"dataset.{reports_df_name} is missing; cannot build patient mapping.")

        if bdmap_col not in rep.columns:
            raise ValueError(f"reports missing required column '{bdmap_col}'. Available: {list(rep.columns)}")
        if patient_id_col not in rep.columns:
            raise ValueError(f"reports missing required column '{patient_id_col}'. Available: {list(rep.columns)}")
        else:
            rep = rep.copy()

        # only use rows relevant to THIS dataset split
        rep = rep.loc[rep[bdmap_col].isin(self.allowed_bdmap)].copy()

        # ---- fill logic you requested ----
        if "Encrypted Accession Number" in rep.columns:
            rep[patient_id_col] = rep[patient_id_col].fillna(rep["Encrypted Accession Number"])
        rep[patient_id_col] = rep[patient_id_col].fillna(rep[bdmap_col])

        # 1 row per BDMAP deterministically
        rep = rep.dropna(subset=[bdmap_col]).drop_duplicates(subset=[bdmap_col], keep="first")

        # --- bdmap -> pid, default pid=bdmap for missing in reports ---
        self.bdmap_to_pid = {bd: bd for bd in self.allowed_bdmap}
        for bd, pid in zip(rep[bdmap_col].tolist(), rep[patient_id_col].tolist()):
            if bd in self.bdmap_to_pid and pid is not None:
                self.bdmap_to_pid[bd] = str(pid)

        # --- pid -> UNIQUE bdmaps ---
        self.pid_to_bdmaps: Dict[str, List[str]] = defaultdict(list)
        for bd in self.allowed_bdmap:
            self.pid_to_bdmaps[self.bdmap_to_pid[bd]].append(bd)
        for pid in list(self.pid_to_bdmaps.keys()):
            self.pid_to_bdmaps[pid] = sorted(set(self.pid_to_bdmaps[pid]))

        # --- chunk/cycle bookkeeping ---
        self.cycle_length = math.ceil(self.dataset_size / self.samples_per_epoch)
        self.epoch = 0
        self.cycle = -1
        self.shuffled_anchors = list(range(self.dataset_size))

        # optional: tiny cache to avoid re-reading JSON many times in one epoch
        # (safe because JSONs are per-crop saved; if they change during training you said "verify real time"
        # but caching per-epoch still reflects real-time enough; set to False if you truly want zero caching.)
        self._cache_enabled = False
        self._organ_cache: Dict[str, Optional[str]] = {}
        
        
        self.debug_every_n = int(debug_every_n)
        self.debug_max_excluded = int(debug_max_excluded)
        self._dbg_batch_i = 0
        
    def _debug_print_choice(
        self,
        *,
        json_path: str,
        pid: str,
        anchor_idx: int,
        anchor_bd: str,
        anchor_organ: Optional[str],
        patient_bds: List[str],
        chosen_bds: List[str],
        included_same_organ: List[str],
        excluded_reasons: Dict[str, List[str]],
        bd_to_organ: Dict[str, Optional[str]],
        batch_indices: List[int],
        padded_indices: List[int],
        local_batch: List[int],
    ):
        #print the json path
        if self.rank != 0:
            return
        if self.debug_every_n <= 0:
            return
        self._dbg_batch_i += 1
        if (self._dbg_batch_i % self.debug_every_n) != 0:
            return

        def _cap(lst: List[str]) -> List[str]:
            if len(lst) <= self.debug_max_excluded:
                return lst
            return lst[: self.debug_max_excluded] + [f"... (+{len(lst)-self.debug_max_excluded} more)"]

        # Pretty print organs for a few bdmaps
        def _fmt_bd(bd: str) -> str:
            org = bd_to_organ.get(bd, None)
            return f"{bd}:{org}"

        print("\n" + "=" * 120, flush=True)
        print(f"[SamplerPerPatientLongitudinal DEBUG] pid={pid}", flush=True)
        print(f"  JSON path for anchor: {json_path}", flush=True)
        print(f"  anchor_idx={anchor_idx} anchor_bd={anchor_bd} anchor_organ={anchor_organ}", flush=True)
        print(f"  patient_bds (n={len(patient_bds)}): {_cap(patient_bds)}", flush=True)

        print(f"  included_same_organ (n={len(included_same_organ)}): {_cap([_fmt_bd(b) for b in included_same_organ])}", flush=True)

        # Exclusions by reason
        if excluded_reasons:
            print("  excluded:", flush=True)
            for reason, bds in excluded_reasons.items():
                print(f"    - {reason} (n={len(bds)}): {_cap([_fmt_bd(b) for b in bds])}", flush=True)
        else:
            print("  excluded: <none>", flush=True)

        print(f"  chosen_bds (n={len(chosen_bds)}): {_cap([_fmt_bd(b) for b in chosen_bds])}", flush=True)
        print(f"  batch_indices (n={len(batch_indices)}): {batch_indices}", flush=True)
        if padded_indices:
            print(f"  padded_indices (n={len(padded_indices)}): {padded_indices}", flush=True)
        print(f"  local_batch rank={self.rank}/{self.world_size}: {local_batch}", flush=True)
        print("=" * 120 + "\n", flush=True)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        # clear cache each epoch to keep it "real-time enough"
        if self._cache_enabled:
            self._organ_cache.clear()

    def _maybe_new_cycle(self):
        new_cycle = self.epoch // self.cycle_length
        if new_cycle != self.cycle:
            self.cycle = new_cycle
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.cycle)
                self.shuffled_anchors = torch.randperm(self.dataset_size, generator=g).tolist()
            else:
                self.shuffled_anchors = list(range(self.dataset_size))
                
    def _torch_choice(self, g, items, k, anchor, prefer_diff_date_prob=0.8):
        if k <= 0:
            return []
        
        if self.meta is None:
            #raise ValueError("Debugging: you did not pass the metadata with dates.")
            perm = torch.randperm(len(items), generator=g).tolist()
            return [items[i] for i in perm[:k]]

        date_anchor = self.bdmap_to_date.get(anchor, pd.NaT)
        if pd.isna(date_anchor):
            perm = torch.randperm(len(items), generator=g).tolist()
            return [items[i] for i in perm[:k]]

        same_date = []
        other_date = []
        for bd in items:
            d = self.bdmap_to_date.get(bd, pd.NaT)
            if pd.isna(d):
                # treat unknown as same bucket (conservative)
                same_date.append(bd)
            elif d == date_anchor:
                same_date.append(bd)
            else:
                other_date.append(bd)

        select_diff_date = (torch.rand((), generator=g).item() < prefer_diff_date_prob)
        sample_first = other_date if select_diff_date else same_date
        sample_second = same_date if select_diff_date else other_date

        # Shuffle deterministically within buckets
        out = []

        if len(sample_first) > 0:
            perm_first = torch.randperm(len(sample_first), generator=g).tolist()
            out.extend([sample_first[i] for i in perm_first])
        if len(sample_second) > 0 and len(out) < k:
            perm_second = torch.randperm(len(sample_second), generator=g).tolist()
            out.extend([sample_second[i] for i in perm_second])

        return out[:k]

    def _read_crop_organ(self, bdmap_id: str) -> Optional[str]:
        """
        Returns:
          - None if JSON missing OR tumor_in_crop missing/None/'random'
          - else the string tumor_in_crop
        """
        
        if self._cache_enabled and bdmap_id in self._organ_cache:
            return self._organ_cache[bdmap_id]

        path = os.path.join(self.crop_json_root, f"{bdmap_id}.json")
        if not os.path.exists(path):
            organ = None
        else:
            try:
                with open(path, "r") as f:
                    dta = json.load(f)
            except (json.JSONDecodeError, OSError):
                # JSON is being rewritten concurrently; skip without caching.
                return None


            #We had some augmented samples saved with this bug pattern: "tumor_in_crop": "random", "unknown_per_voxel": {"bladder_lesion": 5}, "subsegment_cropped_in": "bladder", "organ_cropped_in": "bladder"...
            #so if tumor_in_crop is random, but organ_cropped_in is inside unknown_per_voxel keys, we shuld reject the sample and not load it.
            tumor_in_crop = dta.get("tumor_in_crop", None)
            unknown_per_voxel = dta.get("unknown_per_voxel", None)
            organ_cropped_in = dta.get(self.crop_key, None)
            if organ_cropped_in is None and self.crop_key_fallback is not None:
                organ_cropped_in = dta.get(self.crop_key_fallback, None)
            organ_cropped_in = organ_cropped_in.replace("subsegment_", "").replace('_right','').replace('_left','') if organ_cropped_in is not None else None
            if organ_cropped_in is not None and tumor_in_crop is not None and unknown_per_voxel is not None:
                unknown_per_voxel_keys = unknown_per_voxel.keys()
                cropped_on_tumor = any([organ_cropped_in in k for k in unknown_per_voxel_keys])
                if cropped_on_tumor and tumor_in_crop == "random":
                    return 'do not select'
            
            organ = dta.get(self.crop_key, None)
            if organ is None and self.crop_key_fallback is not None:
                organ = dta.get(self.crop_key_fallback, None)
            if organ is not None:
                organ = str(organ)
                if organ.lower() == "random":
                    organ = None
        if self._cache_enabled:
            self._organ_cache[bdmap_id] = organ
        return organ

    def __iter__(self):
        self._maybe_new_cycle()

        within_cycle = self.epoch % self.cycle_length
        start = within_cycle * self.samples_per_epoch
        end = min(start + self.samples_per_epoch, self.dataset_size)
        anchors = self.shuffled_anchors[start:end]

        # pad anchors to exactly samples_per_epoch
        shortfall = self.samples_per_epoch - len(anchors)
        if shortfall > 0:
            pool = self.shuffled_anchors[:start] + self.shuffled_anchors[end:]
            if len(pool) == 0:
                pool = self.shuffled_anchors
            gpad = torch.Generator()
            gpad.manual_seed(self.seed + 1337 * (self.cycle + 1) + within_cycle)
            pad_idx = torch.randint(0, len(pool), size=(shortfall,), generator=gpad).tolist()
            anchors = anchors + [pool[i] for i in pad_idx]

        # per-epoch RNG shared across ranks
        g = torch.Generator()
        g.manual_seed(self.seed + 10_000 * (self.cycle + 1) + within_cycle)

        for anchor_idx in anchors:
            anchor_bd = self.idx_to_bdmap[anchor_idx]
            pid = self.bdmap_to_pid.get(anchor_bd, anchor_bd)
            patient_bds = self.pid_to_bdmaps.get(pid, [anchor_bd])

            if anchor_bd not in patient_bds:
                raise ValueError(f"Anchor BDMAP {anchor_bd} not found in patient {pid}'s list. How?")

            # --- real-time crop-organ gating (anchor decides) ---
            anchor_organ = self._read_crop_organ(anchor_bd)
            if isinstance(anchor_organ,str) and anchor_organ == 'do not select':
                continue

            excluded_reasons: Dict[str, List[str]] = defaultdict(list)
            bd_to_organ: Dict[str, Optional[str]] = {}

            if anchor_organ is None:
                chosen_bds = [anchor_bd]
                # optionally explain why anchor had no organ
                # (we can't distinguish missing json vs missing key with _read_crop_organ as written)
                # so we just report it as None.
                bd_to_organ[anchor_bd] = None
                included_same_organ = [anchor_bd]
            else:
                included_same_organ = []
                for bd in patient_bds:
                    organ = self._read_crop_organ(bd)
                    if isinstance(organ,str) and organ == 'do not select':
                        organ = None
                    bd_to_organ[bd] = organ

                    if organ is None:
                        excluded_reasons["missing_json_or_cropkey_or_random"].append(bd)
                        continue
                    if organ != anchor_organ:
                        excluded_reasons[f"organ_mismatch_vs_anchor({anchor_organ})"].append(bd)
                        continue

                    included_same_organ.append(bd)
                    
                

                # must include anchor; if it somehow vanished, fall back to anchor-only
                if anchor_bd not in included_same_organ:
                    chosen_bds = [anchor_bd]
                    bd_to_organ.setdefault(anchor_bd, anchor_organ)
                else:
                    if len(included_same_organ) >= self.batch_size_total:
                        others = [b for b in included_same_organ if b != anchor_bd]
                        chosen_bds = [anchor_bd] + self._torch_choice(g, others, self.batch_size_total - 1, anchor_bd)
                    else:
                        others = [b for b in included_same_organ if b != anchor_bd]
                        if len(others) > 1:
                            perm = torch.randperm(len(others), generator=g).tolist()
                            others = [others[i] for i in perm]
                        chosen_bds = [anchor_bd] + others

            # --- map chosen BDMAPs -> dataset indices ---
            batch_indices: List[int] = []
            for bd in chosen_bds:
                if bd == anchor_bd:
                    batch_indices.append(anchor_idx)
                else:
                    candidates = self.bdmap_to_indices[bd]
                    r = torch.randint(0, len(candidates), size=(1,), generator=g).item()
                    batch_indices.append(candidates[r])

            #padded_indices: List[int] = []
            #while len(batch_indices) < self.batch_size_total:
            #    r = torch.randint(0, self.dataset_size, size=(1,), generator=g).item()
            #    if r in batch_indices:
            #        continue
            #    batch_indices.append(r)
            #    padded_indices.append(r)
                
            #this version below avoids loading buggy samples. After the augmented data finished re-writing (deleting the bugs), go back to using the commented loop above
            padded_indices: List[int] = []
            tries = 0
            max_tries = 1000
            while len(batch_indices) < self.batch_size_total:
                tries += 1
                if tries > max_tries:
                    raise ValueError(f"Too many tries ({tries}) to find a valid random sample for padding. Check if your dataset has enough valid samples or if there is a bug causing most samples to be invalid.")

                r = torch.randint(0, self.dataset_size, size=(1,), generator=g).item()
                if r in batch_indices:
                    continue

                bd = self.idx_to_bdmap[r]
                organ_r = self._read_crop_organ(bd)
                if organ_r == "do not select":
                    continue  # <-- prevents the dataset bug from being padded in

                batch_indices.append(r)
                padded_indices.append(r)

            # --- DDP slicing ---
            local_batch = batch_indices[self.rank :: self.world_size]

            # --- DEBUG PRINT (optional) ---
            self._debug_print_choice(
                json_path=os.path.join(self.crop_json_root, f"{anchor_bd}.json"),
                pid=pid,
                anchor_idx=anchor_idx,
                anchor_bd=anchor_bd,
                anchor_organ=anchor_organ,
                patient_bds=patient_bds,
                chosen_bds=chosen_bds,
                included_same_organ=included_same_organ,
                excluded_reasons=excluded_reasons,
                bd_to_organ=bd_to_organ,
                batch_indices=batch_indices,
                padded_indices=padded_indices,
                local_batch=local_batch,
            )

            yield local_batch


    def __len__(self):
        return self.samples_per_epoch