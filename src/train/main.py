# main.py

import pandas as pd
import numpy as np
import os

# For parallelization
from joblib import Parallel, delayed

# Custom modules
from loso import LOSO
from base_features import BASE_FEATURES
from stats import STATS
from ffs import FFS_COMBS
from data_loader import process_site_data

# -----------------------------------------
# 1. Experimental Parameters, 
# -----------------------------------------
model_type = 'random_forest' # linear, random_forest, svm, xgboost, lightgbm, gru, lstm
target_var = 'FCH4'       # e.g., "FC" or "FCH4"
site_list = ['US-Myb', 'US-Tw1', 'US-Tw4']

# -----------------------------------------
# 2. Load Data
# -----------------------------------------
feature_map = BASE_FEATURES()
all_feature_keys = list(feature_map.keys())

features_df, targets_df, site_labels = process_site_data(
    site_list,
    all_feature_keys,
    [target_var]  # pass as list
)

# -----------------------------------------
# 3. Forward Feature Selection (FFS)
# -----------------------------------------
max_steps = 5
tolerance = 0.01

selected_features = []
best_r2 = -np.inf

# We'll store step-by-step info here
progress_records = []

for step in range(1, max_steps + 1):
    print(f"\n=== FFS Step {step} ===")
    
    # Generate candidate combos adding exactly one new feature
    candidate_combos = FFS_COMBS(all_feature_keys, selected_features)
    if not candidate_combos:
        print("No candidates left. Stopping.")
        break

    # -----------------------------------------------------
    # PARALLELIZE the LOSO calls for all candidate combos
    # -----------------------------------------------------
    # 1) Run LOSO for each combo in parallel:
    loso_results = Parallel(n_jobs=-1)(
        delayed(LOSO)(
            features_df, 
            targets_df, 
            target_var, 
            feature_list=combo, 
            model_type=model_type
        )
        for combo in candidate_combos
    )

    # 2) Compute STATS for each LOSO result
    stats_list = [STATS(preds_df) for preds_df in loso_results]

    # Find the best combo for this step
    step_best_r2 = -np.inf
    step_best_r = None
    step_best_rmse = None
    step_best_combo = None

    # Loop over each combo and its corresponding stats
    for combo, stats_df in zip(candidate_combos, stats_list):
        r2_value = stats_df['R2'].iloc[0]
        r_value = stats_df['r'].iloc[0]
        rmse_value = stats_df['RMSE'].iloc[0]
        
        # Track the best combo for this single step
        if r2_value > step_best_r2:
            step_best_r2 = r2_value
            step_best_r = r_value
            step_best_rmse = rmse_value
            step_best_combo = combo

    # Check if there's a meaningful improvement over our global best
    improvement = step_best_r2 - best_r2
    if improvement > tolerance:
        # Accept the new combo
        selected_features = step_best_combo
        best_r2 = step_best_r2
        
        # Record the newly added feature (last element in the combo)
        newly_added = step_best_combo[-1] if step_best_combo else None
        
        # Store results for printing / CSV
        progress_records.append({
            'Model': model_type,
            'Target': target_var,
            'Step': step,
            'r2': round(step_best_r2, 3),
            'r': round(step_best_r, 3),
            'rmse': round(step_best_rmse, 3),
            'Features selected': newly_added
        })
        
        print(f"  Added feature: {newly_added}")
        print(f"  R² = {step_best_r2:.3f} (improvement = {improvement:.3f})")
    else:
        print("No significant improvement. Stopping early.")
        break

# -----------------------------------------
# 4. Show and Save the Step-by-Step Summary
# -----------------------------------------

progress_df = pd.DataFrame(progress_records, 
                           columns=['Model', 'Target', 'Step', 'r2', 'r', 'rmse', 'Features selected'])


print("\n=== FFS Progress Summary ===")
print(progress_df.to_string(index=False))



# Define the full path to save the file
output_dir = "FFS_results"
filename = f"{model_type}_{target_var}_FFS.csv"
filepath = os.path.join(output_dir, filename)

# Save the DataFrame as a CSV inside the FFS_results folder
progress_df.to_csv(filepath, index=False)

print(f"Saved to: {filepath}")


filename = f"{model_type}_{target_var}_FFS.csv"
progress_df.to_csv(filename, index=False)
 

# -----------------------------------------
# 5. (Optional) Final Check with Selected Features
# -----------------------------------------
print("\n=== Final Selected Features ===")
print(selected_features)

if selected_features:
    final_preds_df = LOSO(
        features_df, 
        targets_df, 
        target_var, 
        feature_list=selected_features, 
        model_type=model_type
    )
    final_stats_df = STATS(final_preds_df)
    print("\n=== Final Model Stats ===")
    print(final_stats_df)
