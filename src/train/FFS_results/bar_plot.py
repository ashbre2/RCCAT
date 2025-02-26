import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Map targets for clearer labeling
TARGET_LABELS = {
    "FC": "FCO2",
    "FCH4": "FCH4"
}

# Define model order explicitly
MODEL_ORDER = ["linear", "random_forest", "svm", "lightgbm", "xgboost", "gru", "lstm"]

# Dictionary to store results: (model, target) -> {r2, r, rmse}
results = {}

# Find all CSV files in the current directory
for filepath in glob.glob("*.csv"):
    filename = os.path.basename(filepath)
    
    # Parse filename, expecting format like "random_forest_FC_FFS.csv"
    parts = filename.split('_')
    if len(parts) < 2:
        continue  # Skip files that don’t match expected pattern
    
    # Handle models with underscores (e.g. "random_forest")
    # We assume the last two parts are [TARGET, "FFS.csv"] or similar
    model = "_".join(parts[:-2]).lower()
    target = parts[-2].upper()  # Expect "FC" or "FCH4"

    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Ensure necessary columns exist
    required_cols = {"r2", "r", "rmse", "Step"}
    if not required_cols.issubset(df.columns):
        print(f"Warning: {filename} is missing one of {required_cols}")
        continue
    
    # Sort by Step and pick the last row
    df_sorted = df.sort_values("Step")
    df_last = df_sorted.iloc[-1]  # this is the row for the largest Step number
    
    # Extract r2, r, rmse from the last step
    last_r2 = df_last["r2"]
    last_r = df_last["r"]
    last_rmse = df_last["rmse"]
    
    # Store results
    results[(model, target)] = {
        "r2": last_r2,
        "r": last_r,
        "rmse": last_rmse
    }

# Create a 3-row x 2-column figure (for r2, r, rmse across FCO2 & FCH4)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

metrics = ["r2", "r", "rmse"]
for row_idx, metric in enumerate(metrics):
    for col_idx, target_key in enumerate(["FC", "FCH4"]):
        
        ax = axes[row_idx, col_idx]
        
        # Gather values for the metric, ordered by MODEL_ORDER
        metric_values = []
        for m in MODEL_ORDER:
            val = np.nan
            if (m, target_key) in results:
                val = results[(m, target_key)][metric]
            metric_values.append(val)
        
        # Determine colors:
        # For r2 and r: lowest = red, highest = green
        # For rmse: lowest = green, highest = red
        min_val, max_val = np.nanmin(metric_values), np.nanmax(metric_values)
        if metric in ["r2", "r"]:
            bar_colors = [
                "red" if v == min_val else "green" if v == max_val else "blue"
                for v in metric_values
            ]
        else:  # rmse
            bar_colors = [
                "green" if v == min_val else "red" if v == max_val else "blue"
                for v in metric_values
            ]
        
        # Make the bar plot
        xvals = np.arange(len(MODEL_ORDER))
        ax.bar(xvals, metric_values, color=bar_colors, alpha=0.8)
        
        # Add a dashed reference line for the linear model if it exists
        if ("linear", target_key) in results:
            baseline = results[("linear", target_key)][metric]
            ax.axhline(baseline, color='red', linestyle='--', linewidth=1)
        
        # Axis limits: r2 and r are fixed to 0–1, rmse is auto-scaled
        if metric in ["r2", "r"]:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(None)  # let rmse auto-scale
        
        # Labeling
        ax.set_xticks(xvals)
        if row_idx == 2:  # only put model names on bottom row
            ax.set_xticklabels(MODEL_ORDER, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
        
        # y-axis label on the left column only
        if col_idx == 0:
            ax.set_ylabel(metric.upper())
        
        # Title for top row
        if row_idx == 0:
            plot_title = TARGET_LABELS.get(target_key, target_key)
            ax.set_title(plot_title)

plt.tight_layout()
plt.show()
