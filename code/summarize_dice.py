#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gather_metrics.py
=================
Recursively scan `--source` for files whose name starts with "dice_nsd".
For each metrics CSV:

* read numeric columns **dice** and **nsd**
* ignore non‑numeric / NaN rows
* compute **mean** and **std (σ)** for both metrics
* emit one summary row:

    model , mean_dice , std_dice , mean_nsd , std_nsd

`model` = name of the folder that contains the CSV.

The combined table is written to

    <source>/average_metrics.csv
"""
import argparse, csv, sys
from pathlib import Path
import pandas as pd


# ------------------------------------------------------------------ #
def stats_from_csv(csv_path: Path) -> tuple[float, float, float, float]:
    """
    Return means and stds for dice and nsd, skipping NaNs.
    Raises ValueError if no numeric rows exist.
    """
    df = pd.read_csv(csv_path, usecols=["dice", "nsd"])
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    df["nsd"]  = pd.to_numeric(df["nsd"],  errors="coerce")

    if df["dice"].notna().sum() == 0 or df["nsd"].notna().sum() == 0:
        raise ValueError("no numeric rows")

    return (df["dice"].mean(skipna=True),
            df["dice"].std(skipna=True, ddof=0),
            df["nsd"].mean(skipna=True),
            df["nsd"].std(skipna=True,  ddof=0))


def gather(root: Path, pattern="dice_nsd*.csv") -> list[dict]:
    rows = []
    for csv_path in root.rglob(pattern):
        try:
            d_mean, d_std, n_mean, n_std = stats_from_csv(csv_path)
        except Exception as e:
            print(f"[skip] {csv_path}: {e}", file=sys.stderr)
            continue

        rows.append(dict(model=csv_path.parent.name,
                         mean_dice=f"{d_mean:.4f}",
                         std_dice=f"{d_std:.4f}",
                         mean_nsd=f"{n_mean:.4f}",
                         std_nsd=f"{n_std:.4f}"))
    return rows


# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", required=True,
                   help="Root directory that holds dice_nsd*.csv files")
    p.add_argument("--pattern", default="dice_nsd*.csv",
                   help="Glob pattern for metrics files")
    return p.parse_args()


def main():
    a = parse_args()
    root = Path(a.source).resolve()
    if not root.is_dir():
        sys.exit(f"source directory does not exist: {root}")

    rows = gather(root, a.pattern)
    if not rows:
        sys.exit(f"No '{a.pattern}' files found under {root}")

    out_path = root / "average_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "mean_dice", "std_dice", "mean_nsd", "std_nsd"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary for {len(rows)} models → {out_path}")


if __name__ == "__main__":
    main()