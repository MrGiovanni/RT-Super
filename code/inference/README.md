# Inference quick reference

Entry point: `foundational/MedFormer/predict_abdomenatlas_teacher.py`.
Run from `foundational/MedFormer/`.

All four modes share these required args:

```bash
COMMON="
  --load <run-dir>/fold_0_epoch_50.pth
  --img_path /mnt/bodymaps/image_only/AbdomenAtlasPro/AbdomenAtlasPro/
  --class_list /home/psalvad2/R-Super/foundational/MedFormer/label_names_133.yaml
  --gpu <gpu> --organ_mask_on_lesion --malignancy_classification --cls_on_segmentation
  --train_mode teacher_decoder --num_inpt_ch 1
  --teacher_report_info_prob 0.9 --student_report_info_prob 0.0
  --reports_relaxed_malignancy_col malignancy
  --ids <list.csv> --meta <meta.csv> --reports <per-tumor-reports.csv>
  --parts <N> --current_part <K>"
```

For the longitudinal checkpoint also add:

```bash
LONG="--time_points 2 --use_time_consistency_loss
      --registration_mode image --initialization_registration unigradicon"
```

## 1. Teacher — normal (T_oplc)

Stage-1 sliding-window student over the full CT + stage-2 report-informed
teacher refining lesion channels inside per-organ crops (organ + 128³
windows on lesion-COMs that fall outside the organ crop).

```bash
python predict_abdomenatlas_teacher.py $COMMON $LONG \
  --use_teacher_eval --teacher_filter_laterality \
  --save_path ./result/<out-dir>/
```

## 2. Student — normal (S)

Plain stage-1 sliding-window student, no teacher pass.

```bash
python predict_abdomenatlas_teacher.py $COMMON $LONG \
  --save_path ./result/<out-dir>/
```

## 3. Teacher — fast (T_fast)

Skips full-volume SW. Stage 1 only runs on a per-organ ROI (union of GT
organ masks by default); stage 2 teacher then refines as in T_oplc.
~4.6× faster than T_oplc on Turkish-75; matches or beats DSC on every
organ.

```bash
python predict_abdomenatlas_teacher.py $COMMON $LONG \
  --use_teacher_eval --teacher_filter_laterality \
  --fast_teacher \
  --save_path ./result/<out-dir>/
```

ROI policy is controlled by `--fast_teacher_roi {full,report}` (default
`full` — wider ROI, better DSC; `report` is slightly faster but loses
~1.4 DSC).

## 4. Student — fast (S_fast)

Same SW window-skip applied to the student baseline (no teacher pass).
~4.3× faster than S; DSC within the 2 % budget on every organ.

```bash
python predict_abdomenatlas_teacher.py $COMMON $LONG \
  --fast_student \
  --save_path ./result/<out-dir>/
```

## Notes

- Drop the four `$LONG` flags for the non-longitudinal checkpoint.
- `--fast_teacher` requires `--use_teacher_eval`; `--fast_student` must
  not be combined with it (the launcher will refuse).
- `filter_already_predicted` runs **before** the `--parts` split — if
  the remaining-case count is smaller than `--parts`, each part rounds
  down to 0 cases. Either pass `--overwrite` or set `--parts <= remaining`.
- For long jobs that must survive SSH/Slurm teardown, launch via
  `systemd-run --user --no-block --unit=<name>` (see project root
  `CLAUDE.md`).
