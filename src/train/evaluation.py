import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def compute_metrics(results_dict):
    """
    Computes per-site and overall metrics: MSE, RMSE, R², and correlation.
    Returns a DataFrame with the results.
    """
    metrics = []
    all_actual = []
    all_predicted = []

    for site, vals in results_dict.items():
        actual = np.array(vals["actual"])
        predicted = np.array(vals["predicted"])
        if len(actual) == 0:
            continue
        all_actual.extend(actual)
        all_predicted.extend(predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted) if len(np.unique(actual)) > 1 else float('nan')
        corr = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else float('nan')
        metrics.append([site, mse, rmse, r2, corr])
    
    if len(all_actual) == 0:
        return pd.DataFrame(columns=["Site", "MSE", "RMSE", "R²", "Correlation"]).set_index("Site")

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    overall_mse = mean_squared_error(all_actual, all_predicted)
    overall_rmse = np.sqrt(overall_mse)
    overall_r2 = r2_score(all_actual, all_predicted)
    overall_corr = np.corrcoef(all_actual, all_predicted)[0, 1]

    metrics.append(["Overall", overall_mse, overall_rmse, overall_r2, overall_corr])
    metrics_df = pd.DataFrame(metrics, columns=["Site", "MSE", "RMSE", "R²", "Correlation"]).set_index("Site")
    return metrics_df
