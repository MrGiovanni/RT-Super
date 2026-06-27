"""Stand-alone replica of dataset.get_input_info_report() for inference time.

The training-time dataset
(`training.dataset.dim3.dataset_abdomenatlas_UFO_multi_tumor.AbdomenAtlasDataset.get_input_info_report`,
defined at line 4138 of that module) builds a 10-key dict with all the
report-derived information the report-informed teacher decoder consumes.
At inference we cannot instantiate that dataset (the preprocessed NPZ files
may not be present and we don't have GT masks), so we reimplement the
*report* side here and accept the spatial mask + crop coordinates from
stage-1 inference.

The output dict is byte-equivalent to the dataset's output for non-spatial
fields and is `full_res_lesion_mask_3d[crop_coords]` (cropped to window) for
the two 3-D fields. The only branch we cover is the UFO branch (no
per-voxel tumor annotation): at inference we never have voxel-level GT
lesion masks, so the dataset's `tumor_annotated_seg[name]==False` path is
the one we replicate.
"""

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

# We import a few helpers from the dataset module so we behave identically.
# The dataset module is heavyweight; importing it is acceptable here because
# the inference path runs once per process.
from training.dataset.dim3.dataset_abdomenatlas_UFO_multi_tumor import (
    clean_ufo,
    coerce_to_bool,
)


# Constant orderings — exactly as in the dataset's get_input_info_report.
_LESION_LABEL_PADDING = ['unknown', 'no tumor']
_ATTENUATION_LABELS = ['unknown', 'no tumor', 'hypo', 'hyper']
_MALIGNANCY_LABELS = ['unknown', 'no tumor', 'benign', 'malignant']

# Default tumor_classes (organs the dataset crops on). Mirrors the constructor
# default of AbdomenAtlasDataset.
_DEFAULT_TUMOR_CLASSES = [
    'adrenal gland', 'bladder', 'colon', 'duodenum',
    'esophagus', 'gallbladder', 'prostate', 'spleen',
    'stomach', 'uterus',
]

# Tumor-classes order used by the dataset's training script (with kidney/lung
# added beyond the default; this is what the trained checkpoint saw). The
# extra entries don't change clean_ufo's per-organ filtering for samples
# whose organ isn't one of these — they only affect which entries appear in
# the `interest` dict and the prohibit_crop computation.
_TRAINING_TUMOR_CLASSES = list(_DEFAULT_TUMOR_CLASSES)


def lesion_like_name(segment_name: str) -> str:
    """Mirror lines 2887-2895 of the dataset module: collapse organ subsegment
    or sided organ names to their canonical lesion channel name.

    Examples:
        'liver' / 'liver_segment_3'   -> 'liver_lesion'
        'pancreas' / 'pancreas_head'  -> 'pancreatic_lesion'
        'adrenal_gland_left'          -> 'adrenal_lesion'
        'gall_bladder'                -> 'gallbladder_lesion'
        'spleen'                      -> 'spleen_lesion'
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


def _standardized_organ_for(organ_canonical: str) -> str:
    """Convert a canonical organ name (the dataset's `organ_cropped_cannon`)
    to the value used in the per-tumor report's `Standardized Organ` column.

    The actual values present in the report (verified empirically):
      'liver', 'pancreatic', 'kidney', 'colon', 'esophagus', 'uterus',
      'spleen', 'adrenal gland', 'bladder', 'gallbladder' (no underscore!),
      'breast', 'stomach', 'lung', 'bone', 'prostate', 'duodenum'.

    Note 'gallbladder' is one word here, while the dataset's class list and
    `canonical_organ` use 'gall_bladder' (with underscore). Both forms are
    handled below.
    """
    base = organ_canonical.replace('_lesion', '')
    lb = base.lower()
    if lb in ('pancreatic', 'pancreas'):
        return 'pancreas'
    if lb in ('adrenal_gland', 'adrenal'):
        return 'adrenal gland'
    if lb in ('gallbladder', 'gall_bladder'):
        return 'gallbladder'
    return base


def _estimate_size_volume(size_str: Union[str, float]) -> Tuple[float, List[float]]:
    """Parse a 'Tumor Size (mm)' cell into (volume_mm3, [d1, d2, d3]).
    Mirrors dataset.estimate_tumor_volume lines 4990-5016 exactly.
    """
    s = str(size_str)
    if s == 'u':
        return -9999999.0, [-9999999.0, -9999999.0, -9999999.0]
    if 'x' not in s:
        d = float(s)
        v = (4 / 3) * math.pi * ((d / 2) ** 3)
        return v, [d, d, d]
    sizes = [float(p) for p in s.split(' x ')]
    if len(sizes) == 2:
        sizes.append(sum(sizes) / 2)
    elif len(sizes) > 3:
        sizes = sorted(sizes, reverse=True)[:3]
    v = (4 / 3) * math.pi * ((sizes[0] / 2) * (sizes[1] / 2) * (sizes[2] / 2))
    return v, sizes


def _resolve_malignancy_conflicts_by_diameter(t: torch.Tensor, ignore_zero: bool = True) -> torch.Tensor:
    """Replica of dataset.resolve_malignancy_conflicts_by_diameter (4826-4863).

    Two tumors with the same diameter but different malignancy labels (or one
    NaN) collapse to NaN — we cannot tell them apart.
    """
    arr = t.detach().cpu().numpy().astype(float, copy=True)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N,2), got {arr.shape}")
    diam = arr[:, 0]
    mal = arr[:, 1]
    valid = ~np.isnan(diam)
    if ignore_zero:
        valid &= (diam != 0.0)
    uniques = np.unique(diam[valid])
    for d in uniques:
        idx = np.where((diam == d) & valid)[0]
        if idx.size >= 2:
            vals = mal[idx]
            classes = set()
            if np.any(vals == 0.0):
                classes.add(0.0)
            if np.any(vals == 1.0):
                classes.add(1.0)
            if np.any(np.isnan(vals)):
                classes.add('nan')
            if len(classes) >= 2:
                mal[idx] = np.nan
    out = np.stack([diam, mal], axis=1)
    return torch.from_numpy(out).type_as(t)


def _malignancy_from_row(row) -> float:
    """Mirror dataset.get_malignancy (line 4896): read 'malignant_benign'."""
    mb = row.get('malignant_benign', None)
    if mb == 'malignant':
        return 1.0
    if mb == 'benign':
        return 0.0
    return float('nan')


def _attenuation_label(att_str) -> float:
    """Mirror lines 5020-5027 of estimate_tumor_volume: map 'Standardized
    Attenuation' string to {1.0 (high), -1.0 (low), 999 (unknown)}."""
    if att_str == 'high':
        return 1.0
    if att_str == 'low':
        return -1.0
    return 999.0


class TumorInfoBuilder:
    """Builds the same `tumor_info_input` dict as the dataset, given a
    full-resolution lesion mask and crop coordinates.

    Parameters
    ----------
    reports_csv_path : str
        Path to the per-tumor report CSV (same as the predict script's
        --reports flag).
    label_names_yaml_path : str
        Path to the YAML file listing the dataset's classes (same YAML the
        dataset reads via `args.data_root/list/label_names.yaml` and the
        predict script reads via --class_list). The dataset's `self.classes`
        is `sorted(yaml.load(...))`.
    tumor_classes : list[str], optional
        Organs to consider as crop-eligible. Mirrors the dataset constructor
        default. Used only for clean_ufo's filter side-effects.
    patch_size : int
        Controls clean_ufo's "drop the entire BDMAP_ID if any tumor diameter
        > patch_size" filter. At training time this matches args.training_size[0]
        (=128) so cases that wouldn't fit a 128^3 crop are excluded. At
        INFERENCE this filter is undesirable — it drops perfectly good cases
        whenever the patient happens to have one large tumor in an unrelated
        organ (e.g. a 135mm abdominal-wall tumor knocking out the gallbladder
        prediction). The default here is therefore set to a huge value so the
        filter is effectively disabled. Pass patch_size=128 explicitly to
        reproduce the dataset behavior byte-for-byte.
    """

    def __init__(self,
                 reports_csv_path: str,
                 label_names_yaml_path: str,
                 tumor_classes: Optional[Sequence[str]] = None,
                 patch_size: int = 10_000_000,
                 use_all_data: bool = True,
                 limit_healthy: bool = False,
                 malignancy_column: str = 'pathology_and_radiology_malignant',
                 benign_column: str = 'radiology_benign_ICD_pathology_ok',
                 relaxed_malignancy_col: Optional[str] = None,
                 load_slices: bool = False,
                 load_malignancy: bool = True,
                 reproduce_volumes_gt_zero_bug: bool = False):
        # bug-repro: training's get_input_info_report used (volumes > 0) for
        # tumor_count, which dropped unknown-size tumors (volume < 0). Active
        # training code was later fixed to (volumes != 0). Set this True to
        # match a checkpoint that was trained against the buggy version.
        self.reproduce_volumes_gt_zero_bug = bool(reproduce_volumes_gt_zero_bug)
        with open(label_names_yaml_path, 'r') as f:
            classes_raw = yaml.load(f, Loader=yaml.SafeLoader)
        self.classes = sorted(classes_raw)
        self.lesion_labels = list(_LESION_LABEL_PADDING) + [c for c in self.classes if 'lesion' in c]
        self.attenuation_labels = list(_ATTENUATION_LABELS)
        self.malignancy_labels = list(_MALIGNANCY_LABELS)
        self.denom_labels = max(len(self.lesion_labels) - 1, 1)
        self.denom_att = max(len(self.attenuation_labels) - 1, 1)
        self.denom_malig = max(len(self.malignancy_labels) - 1, 1)

        self.tumor_classes = list(tumor_classes) if tumor_classes is not None else list(_DEFAULT_TUMOR_CLASSES)
        self.load_slices = bool(load_slices)
        self.load_malignancy = bool(load_malignancy)

        reports = pd.read_csv(reports_csv_path)
        if 'BDMAP ID' in reports.columns:
            reports = reports.rename(columns={'BDMAP ID': 'BDMAP_ID'})
        if 'series matches report' in reports.columns:
            reports['series matches report'] = reports['series matches report'].apply(coerce_to_bool)

        # Replicate the dataset's malignancy logic with fallback handling:
        # - Primary: `malignancy_column` (default: 'pathology_and_radiology_malignant')
        # - Strict-benign:  `benign_column`     (default: 'radiology_benign_ICD_pathology_ok')
        # - Relaxed fallback (when primary value is non-'yes'/'no'):
        #     `relaxed_malignancy_col` — training launchers set this to 'malignancy'.
        #
        # Extra robustness needed for datasets that don't carry the
        # pathology columns at all (e.g. the Turkish per-tumor CSV which
        # only has the 'malignancy' column):
        #   1. If primary column is missing, promote `relaxed_malignancy_col`
        #      to primary (if it exists).
        #   2. If both primary and benign are missing, pass None / None to
        #      clean_ufo — `define_malignancy` is then skipped and every
        #      tumor ends up with 'unknown' malignancy downstream.
        mal_col = malignancy_column if (malignancy_column and malignancy_column in reports.columns) else None
        ben_col = benign_column if (benign_column and benign_column in reports.columns) else None
        relaxed_col = relaxed_malignancy_col if (relaxed_malignancy_col and relaxed_malignancy_col in reports.columns) else None

        if mal_col is None and relaxed_col is not None:
            # Promote relaxed to primary. (The strict/relaxed distinction
            # only matters when both exist — if pathology cols are missing
            # entirely, the relaxed column IS the only signal.)
            print(f'[TumorInfoBuilder] WARN: malignancy_column={malignancy_column!r} not in CSV; '
                  f'promoting relaxed_malignancy_col={relaxed_malignancy_col!r} to primary.',
                  flush=True)
            mal_col = relaxed_col
            relaxed_col = None
            # define_malignancy also needs a benign_column — fake one by
            # treating absence of benign as no info. Use the same column
            # name; define_malignancy reads row[benign_col] then checks
            # 'yes' — if it's never 'yes' we just don't flag it benign.
            if ben_col is None:
                ben_col = mal_col

        if (malignancy_column is not None) and (mal_col is None) and (relaxed_col is None):
            print(f'[TumorInfoBuilder] WARN: neither malignancy_column={malignancy_column!r} nor '
                  f'relaxed_malignancy_col={relaxed_malignancy_col!r} found in CSV; '
                  f'treating all tumors as unknown malignancy.', flush=True)

        reports, _ids, _types = clean_ufo(
            reports, self.tumor_classes,
            limit_healthy=limit_healthy,
            slice_only=False,
            patch_size=patch_size,
            use_all_data=use_all_data,
            benign_maligant_only=False,
            malignancy_column=mal_col,
            benign_column=ben_col,
            relaxed_malignancy_col=relaxed_col,
        )
        self.reports = reports

    # -------------- helpers --------------

    @staticmethod
    def lesion_like_name(segment_name: str) -> str:
        return lesion_like_name(segment_name)

    @staticmethod
    def standardized_organ_for(organ_canonical: str) -> str:
        return _standardized_organ_for(organ_canonical)

    def _select_rows(self, bdmap_id: str, organ_canonical: str,
                     side: Optional[str] = None) -> pd.DataFrame:
        std_organ = _standardized_organ_for(organ_canonical)
        # Mirror dataset's get_tumor_segment_labels (line 2634): rows with
        # Standardized Organ outside tumor_classes are dropped from
        # `tumor_dict`. estimate_tumor_volume then sees zero rows for those
        # organs and returns padding-only arrays. Reproduce that here so we
        # stay byte-equivalent for queries on out-of-class organs.
        if std_organ not in self.tumor_classes:
            return self.reports.iloc[0:0]
        mask = (self.reports['BDMAP_ID'] == bdmap_id) & (self.reports['Standardized Organ'] == std_organ)
        rows = self.reports[mask]
        if side in ('right', 'left'):
            loc = rows['Standardized Location'].astype(str).str.lower()
            rows = rows[loc.str.contains(side, na=False)]
        return rows

    def _per_tumor_arrays(self, rows: pd.DataFrame, organ_canonical: str) -> Dict[str, torch.Tensor]:
        """Build per-tumor volume/diameter/attenuation/slices/malignancy
        arrays from report rows. Mirrors estimate_tumor_volume (4906-5070).

        Pads to 10. Returns the same five tensors the dataset produces:
          volumes (10,), diameters (10,3), attenuation (10,),
          tumors_in_crop_slices (10,2), tumors_in_crop_malignancy (10,2)

        For organ-level queries (no 'segment'/'head'/'body'/'tail'/'left'/
        'right' in the organ name), the dataset uses Standardized Organ as
        the location field — so rows that have NaN/'u' in Standardized
        Location are still kept. We mirror that: apply the 'u' filter only
        when the query is segment-like.
        """
        volumes: List[float] = []
        diameters: List[List[float]] = []
        att: List[float] = []
        slices: List[float] = []
        malignancies: List[float] = []

        # Mirror estimate_tumor_volume line 4935: a "segment" query is one
        # whose name contains a sub-segment / sided keyword.
        seg_kw = ('segment', 'head', 'body', 'tail', 'left', 'right')
        is_segment_query = any(k in organ_canonical for k in seg_kw)

        for _, row in rows.iterrows():
            if is_segment_query:
                # The dataset reads Standardized Location for segments and
                # filters out NaN/'u' there. We don't currently exercise
                # segment queries from inference (organ-level only), but
                # keep the parity for completeness.
                location = row.get('Standardized Location', None)
                if not isinstance(location, str) or location.lower() == 'u':
                    continue
            sze = row['Tumor Size (mm)']
            v, d = _estimate_size_volume(sze)
            volumes.append(v)
            diameters.append(d)
            att.append(_attenuation_label(row.get('Standardized Attenuation', None)))
            if self.load_slices:
                if (coerce_to_bool(row.get('series matches report', False))
                        and (not coerce_to_bool(row.get('impossible slice', False)))
                        and sze != 'u'):
                    slc = pd.to_numeric(row.get('Image', np.nan), errors='coerce')
                    slices.append(np.nan if pd.isna(slc) else float(slc))
                else:
                    slices.append(np.nan)
            else:
                slices.append(np.nan)
            if self.load_malignancy:
                malignancies.append(_malignancy_from_row(row))
            else:
                malignancies.append(np.nan)

        # Pair sizes/slices and sizes/malignancy by max-diameter, like the
        # dataset (lines 5031-5038).
        slc_list: List[List[float]] = []
        mal_list: List[List[float]] = []
        for i in range(len(diameters)):
            max_d = max(diameters[i])
            slc_list.append([max_d, slices[i]])
            mal_list.append([max_d, malignancies[i]])

        # Pad to 10 (line 5042-5047).
        while len(volumes) < 10:
            volumes.append(0.0)
            diameters.append([0.0, 0.0, 0.0])
            att.append(0.0)
            slc_list.append([0.0, float('nan')])
            mal_list.append([0.0, float('nan')])
        # Cap to 10 (line 5055-5060).
        volumes = volumes[:10]
        diameters = diameters[:10]
        att = att[:10]
        slc_list = slc_list[:10]
        mal_list = mal_list[:10]

        # Resolve malignancy conflicts by diameter (line 5062).
        mal_tensor = _resolve_malignancy_conflicts_by_diameter(
            torch.tensor(mal_list, dtype=torch.float32), ignore_zero=True)

        return {
            'volumes': torch.tensor(volumes, dtype=torch.float32),
            'diameters': torch.tensor(diameters, dtype=torch.float32),
            'attenuation': torch.tensor(att, dtype=torch.float32),
            'sizes_slices': torch.tensor(slc_list, dtype=torch.float32),
            'sizes_malignancy': mal_tensor,
        }

    # -------------- main entry --------------

    def build(self,
              bdmap_id: str,
              organ_canonical: str,
              tumor_location_mask_full_res: Optional[torch.Tensor] = None,
              crop_coords: Optional[Sequence[int]] = None,
              side: Optional[str] = None,
              full_res_lesion_mask_3d: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Build the tumor_info dict.

        Parameters
        ----------
        bdmap_id : str
            BDMAP_xxxxxxxx identifier.
        organ_canonical : str
            Canonical organ name (matches dataset's `organ_cropped_cannon`,
            e.g. 'liver', 'pancreas', 'adrenal_gland', 'gall_bladder').
        tumor_location_mask_full_res : torch.Tensor
            Shape (D, H, W). The full-resolution mask that, once cropped
            to `crop_coords`, becomes the model's `tumor_location_mask`
            input. Despite the historical kwarg name `full_res_lesion_mask_3d`,
            in the current pipeline this is typically the organ-region
            mask (training under `--organ_mask_on_lesion` and T_oplc
            inference both put an organ-region mask here).
        crop_coords : sequence of 6 ints
            [z0, z1, y0, y1, x0, x1] (end-exclusive, matching
            crop_foreground_3d's return shape at
            inference/inference3d.py:538). The crop is applied as
            `mask[z0:z1, y0:y1, x0:x1]`.
        full_res_lesion_mask_3d : torch.Tensor, optional
            Deprecated alias for `tumor_location_mask_full_res`. Accepted
            so existing call sites keep working; pass exactly one of the
            two.

        Returns
        -------
        dict with the 10 keys produced by dataset.get_input_info_report
        (lines 4355-4377).
        """
        # Backward-compatible alias resolution. Accept either kwarg name;
        # require exactly one to be supplied.
        if tumor_location_mask_full_res is None and full_res_lesion_mask_3d is None:
            raise TypeError(
                "build() requires `tumor_location_mask_full_res` "
                "(or its deprecated alias `full_res_lesion_mask_3d`)")
        if tumor_location_mask_full_res is not None and full_res_lesion_mask_3d is not None:
            raise TypeError(
                "build() got both `tumor_location_mask_full_res` and the "
                "deprecated alias `full_res_lesion_mask_3d`; pass only one")
        if tumor_location_mask_full_res is None:
            tumor_location_mask_full_res = full_res_lesion_mask_3d
        if crop_coords is None:
            raise TypeError("build() requires `crop_coords`")

        if tumor_location_mask_full_res.ndim == 4 and tumor_location_mask_full_res.shape[0] == 1:
            tumor_location_mask_full_res = tumor_location_mask_full_res[0]
        if tumor_location_mask_full_res.ndim != 3:
            raise ValueError(
                f"tumor_location_mask_full_res must be 3-D (D,H,W); got {tuple(tumor_location_mask_full_res.shape)}")
        z0, z1, y0, y1, x0, x1 = [int(v) for v in crop_coords]

        # Crop the location mask. At smoke-test this matches
        # `retur['mask'][tumor_ch]` voxel-by-voxel (= chosen_segment_mask of
        # the lesion channel built from cropped tensor_lab).
        cropped_loc_mask = tumor_location_mask_full_res[z0:z1, y0:y1, x0:x1].float().clone()
        cropped_shape = cropped_loc_mask.shape

        # Pull report rows for this (BDMAP_ID, organ). Optionally filter by
        # laterality so a per-side crop only sees that side's tumors.
        rows = self._select_rows(bdmap_id, organ_canonical, side=side)
        per_tumor = self._per_tumor_arrays(rows, organ_canonical)
        volumes = per_tumor['volumes']            # (10,)
        diameters = per_tumor['diameters']        # (10, 3)
        attenuation = per_tumor['attenuation']    # (10,)
        sizes_malignancy = per_tumor['sizes_malignancy']  # (10, 2)

        # Tumor count: number of nonzero entries in volumes (line 4170-4171).
        # bug-repro: if reproduce_volumes_gt_zero_bug, drop unknown-size tumors
        # (volume < 0) to match the buggy training-time (volumes > 0) check.
        if self.reproduce_volumes_gt_zero_bug:
            tumor_count = int((volumes > 0).sum().item())
        else:
            tumor_count = int((volumes != 0).sum().item())

        # Determine known_tumor_count from the original-rows _was_multiple
        # column (matches lines 4205-4208 — `tumors_in_organ` may include
        # rows we filtered out at _per_tumor_arrays time, since the dataset
        # checks `_was_multiple` on the unfiltered rows). The dataset's
        # `report` here is `self.read_report(idx)` filtered to just this
        # BDMAP_ID and Standardized Organ — that includes location='u' rows.
        unknown_tumor_count = bool(
            ('_was_multiple' in rows.columns) and rows['_was_multiple'].any()
        )

        if tumor_count == 0:
            # No-tumor branch (lines 4172-4186).
            tumor_class_idx = self.lesion_labels.index('no tumor') / self.denom_labels
            tumor_attenuation = [self.attenuation_labels.index('no tumor') / self.denom_att] * 10
            tumor_malignancy = [self.malignancy_labels.index('no tumor') / self.denom_malig] * 10
            tumor_location_mask = torch.zeros(cropped_shape, dtype=torch.float32)
            allowed_tumor_slices = torch.ones(cropped_shape, dtype=torch.float32)
            tumor_diameters_30 = torch.zeros(30, dtype=torch.float32)
            tumor_volumes_10 = torch.zeros(10, dtype=torch.float32)
        else:
            # Tumor branch (lines 4188-4269).
            tumor_class = lesion_like_name(organ_canonical)
            if tumor_class not in self.lesion_labels:
                raise ValueError(
                    f"Computed lesion class '{tumor_class}' for organ '{organ_canonical}' "
                    f"not in lesion_labels {self.lesion_labels}")
            tumor_class_idx = self.lesion_labels.index(tumor_class) / self.denom_labels
            tumor_location_mask = cropped_loc_mask.float()
            # At inference we don't know slice locations, default to all-ones
            # (matches the dataset's slice_used==False fallback at
            # getitem_real:2489 and the no-tumor / atlas-per-voxel branches).
            allowed_tumor_slices = torch.ones(cropped_shape, dtype=torch.float32)

            # Malignancy from sizes_malignancy (lines 4220-4239).
            tumor_malignancy = []
            for i in range(sizes_malignancy.shape[0]):
                sz = sizes_malignancy[i, 0].item()
                if sz == 0:
                    tumor_malignancy.append(self.malignancy_labels.index('no tumor') / self.denom_malig)
                    continue
                m = sizes_malignancy[i, 1].item()
                if math.isnan(m):
                    tumor_malignancy.append(self.malignancy_labels.index('unknown') / self.denom_malig)
                elif m == 1:
                    tumor_malignancy.append(self.malignancy_labels.index('malignant') / self.denom_malig)
                elif m == 0:
                    tumor_malignancy.append(self.malignancy_labels.index('benign') / self.denom_malig)
                else:
                    raise ValueError(f"Unexpected malignancy value {m} for {bdmap_id}")

            # Attenuation (lines 4248-4263).
            tumor_attenuation = []
            for i in range(attenuation.shape[0]):
                a = attenuation[i].item()
                if a == 0:
                    tumor_attenuation.append(self.attenuation_labels.index('no tumor') / self.denom_att)
                elif a == 999:
                    tumor_attenuation.append(self.attenuation_labels.index('unknown') / self.denom_att)
                elif a == 1:
                    tumor_attenuation.append(self.attenuation_labels.index('hyper') / self.denom_att)
                elif a == -1:
                    tumor_attenuation.append(self.attenuation_labels.index('hypo') / self.denom_att)
                else:
                    raise ValueError(f"Unexpected attenuation value {a} for {bdmap_id}")

            # Diameters (10,3) -> (30,), volumes already (10,), then clamp.
            tumor_diameters_30 = torch.cat([diameters[i] for i in range(diameters.shape[0])], dim=0)
            tumor_volumes_10 = volumes.clone()
            tumor_volumes_10 = torch.clamp(tumor_volumes_10, min=-0.5)
            tumor_diameters_30 = torch.clamp(tumor_diameters_30, min=-0.5)

        # Assemble the dict (lines 4355-4377).
        tumor_info: Dict[str, torch.Tensor] = {
            'tumor_location_mask': tumor_location_mask.squeeze(0).float(),
            'tumor_allowed_slices': allowed_tumor_slices.squeeze(0).float(),
            'tumor_organ_name': torch.tensor([tumor_class_idx]).float(),
            'tumor_count': torch.tensor([tumor_count / 10]).float(),
            'tumor_attenuation': torch.tensor(tumor_attenuation).float(),
            'tumor_malignancy': torch.tensor(tumor_malignancy).float(),
            'known_tumor_count': torch.tensor([0.0]).float() if unknown_tumor_count else torch.tensor([1.0]).float(),
            'tumor_diameters': tumor_diameters_30.float(),
            'tumor_volumes': tumor_volumes_10.float(),
        }
        tumor_info['tumor_info_vector_organ_count_attenuation_malignancy'] = torch.cat([
            tumor_info['tumor_organ_name'],
            tumor_info['tumor_count'],
            tumor_info['known_tumor_count'],
            tumor_info['tumor_attenuation'],
            tumor_info['tumor_malignancy'],
            tumor_info['tumor_diameters'],
            tumor_info['tumor_volumes'],
        ], dim=0).float()

        if tumor_info['tumor_info_vector_organ_count_attenuation_malignancy'].shape[0] != 63:
            raise ValueError(
                f"tumor_info_vector has shape "
                f"{tumor_info['tumor_info_vector_organ_count_attenuation_malignancy'].shape}; "
                f"expected (63,) for {bdmap_id}/{organ_canonical}")
        return tumor_info
