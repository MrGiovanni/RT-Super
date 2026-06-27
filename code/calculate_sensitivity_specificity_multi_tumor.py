import csv
import argparse
import pandas as pd

def evaluate_predictions(ground_truth_csv, predictions_csv, output_csv):
    """
    Compare ground truth (report-based) with predicted volumes, computing
    sensitivity and specificity at multiple volume thresholds.

    Ground truth CSV columns (relevant ones):
        - "BDMAP ID" (unique case identifier)
        - "number of <organ> lesion instances" for each organ

    Predictions CSV columns (from previous code):
        - "BDMAP_ID"  (unique case identifier, matching ground truth's "BDMAP ID")
        - "<organ> tumor volume predicted" for each organ

    We define ground_truth_label = 1 if #lesion_instances >=1, else 0.
    Then for each threshold T, predicted_label = 1 if volume_predicted >= T, else 0.

    We'll compute for each organ:
        - sensitivity = 100 * TP / (TP + FN)
        - specificity = 100 * TN / (TN + FP)

    Output CSV will have columns:
        threshold,
        <organ>_sensitivity, <organ>_specificity for each organ.
    Each row corresponds to a single threshold.
    """
    
    # --------------------------
    # 1) Read ground truth CSV
    # --------------------------
    gt_df = pd.read_csv(ground_truth_csv)
    # Rename column "BDMAP ID" -> "BDMAP_ID" for consistency
    if "BDMAP ID" in gt_df.columns:
        gt_df = gt_df.rename(columns={"BDMAP ID": "BDMAP_ID"})

    # -------------------------
    # 2) Read predictions CSV and determine organs
    # -------------------------
    pred_df = pd.read_csv(predictions_csv)
    # Extract organ names by checking columns that end with "tumor volume predicted"
    organs = [col.replace(" tumor volume predicted", "")
              for col in pred_df.columns if col.endswith("tumor volume predicted")]
    
    # Keep only the relevant columns in predictions DataFrame
    relevant_cols = ["BDMAP_ID"] + [f"{organ} tumor volume predicted" for organ in organs]
    pred_df = pred_df[relevant_cols]

    # -------------------------
    # 3) Create binary ground truth labels for each organ
    # -------------------------
    for organ in organs:
        if organ == 'any':
            continue
        gt_col = f"number of {organ.replace('adrenal gland','adrenal')} lesion instances"
        if gt_col in gt_df.columns:
            gt_df[f"gt_{organ}"] = gt_df[gt_col].apply(lambda x: 1 if x >= 1 else 0)
        else:
            raise ValueError(f"Ground truth CSV missing expected column: '{gt_col}'")
    
    # -------------------------
    # 4) Merge on BDMAP_ID
    # -------------------------
    df_merged = pd.merge(gt_df, pred_df, on="BDMAP_ID", how="inner")

    # Build mappings for easier reference
    organ_gt_map = { organ: f"gt_{organ}" for organ in organs }
    if args.prediction_used == 'lesion type':
        organ_pred_map = { organ: f"{organ} tumor volume predicted" for organ in organs if organ != 'any'}
    elif args.prediction_used == 'lesion':
        organ_pred_map = { organ: f"any tumor volume predicted" for organ in organs }
    else:
        raise ValueError('Lesion type not identified')

    # -------------------------
    # 5) Define thresholds
    # -------------------------
    thresholds = []
    for i in range(1, 10):
        thresholds.append(i)
    for i in range(1, 10):
        thresholds.append(i * 10)
    for i in range(10, 100):
        thresholds.append(i * 10)
    for i in range(1, 100):
        thresholds.append(i * 100)
    for i in range(1, 100):
        thresholds.append(i * 1000)

    # -------------------------
    # 6) Helper function for formatting metrics
    # -------------------------
    def format_metric(numer, denom):
        """
        Return "XX% (x/y)" or "N/A (0/0)" if denom is 0.
        """
        if denom == 0:
            return "N/A (0/0)"
        perc = 100.0 * numer / denom
        return f"{perc:.1f}% ({numer}/{denom})"

    # -------------------------
    # 7) Compute evaluation metrics at each threshold
    # -------------------------
    results = []
    for T in thresholds:
        row_data = {"threshold": T}
        for organ in organs:
            if organ == 'any':
                continue
            # Initialize confusion matrix counts
            TP, FP, TN, FN = 0, 0, 0, 0

            if args.prediction_used=='lesion':
                # For the current organ, filter out any row that shows lesion in any other organ.
                # Create a list of ground truth columns for all organs except the one under evaluation.
                cols_other = [organ_gt_map[o] for o in organs if ((o != organ) and (o!='any'))]
                # Filter df_merged so that only rows where the sum over these columns equals zero are kept.
                df_filtered = df_merged[df_merged[cols_other].sum(axis=1) == 0]
            else:
                df_filtered = df_merged

            # Iterate over all merged cases
            for _, row in df_filtered.iterrows():
                gt_label = row[organ_gt_map[organ]]  # Binary ground truth label
                pred_volume = row[organ_pred_map[organ]]
                pred_label = 1 if pred_volume >= T else 0

                if gt_label == 1 and pred_label == 1:
                    TP += 1
                elif gt_label == 1 and pred_label == 0:
                    FN += 1
                elif gt_label == 0 and pred_label == 1:
                    FP += 1
                elif gt_label == 0 and pred_label == 0:
                    TN += 1

            # Compute sensitivity and specificity for the organ
            sens_str = format_metric(TP, TP + FN)
            spec_str = format_metric(TN, TN + FP)
            row_data[f"{organ}_sensitivity"] = sens_str
            row_data[f"{organ}_specificity"] = spec_str

        results.append(row_data)

    # -------------------------
    # 8) Write evaluation results to output CSV
    # -------------------------
    # Construct fieldnames: threshold plus sensitivity and specificity for each organ
    fieldnames = ["threshold"]
    for organ in organs:
        fieldnames.append(f"{organ}_sensitivity")
        fieldnames.append(f"{organ}_specificity")

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_data in results:
            writer.writerow(row_data)

    print(f"Evaluation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against ground truth at multiple volume thresholds."
    )
    parser.add_argument("--ground_truth_csv", type=str, required=True,
                        help="Path to the ground truth CSV (report-based).")
    parser.add_argument("--predictions_csv", type=str, required=True,
                        help="Path to the predictions CSV (volumes).")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to output CSV where evaluation metrics will be saved.")
    parser.add_argument("--prediction_used", type=str, default='lesion type',
                        help="lesion / lesion type") 
    args = parser.parse_args()

    evaluate_predictions(
        ground_truth_csv=args.ground_truth_csv,
        predictions_csv=args.predictions_csv,
        output_csv=args.output_csv
    )