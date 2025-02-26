import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

def STATS(predictions_df):
    """
    Computes R², Pearson correlation (r), and RMSE from predictions.

    Expected format of predictions_df:
    - A DataFrame with a multi-index (datetime, site_label).
    - Must contain the columns: 'target' (actual values) and 'prediction' (model predictions).
    
    Returns:
    - A single-row DataFrame with R², r, and RMSE.
    """

    # Check if the required columns exist
    required_columns = {'target', 'prediction'}
    if not required_columns.issubset(predictions_df.columns):
        raise ValueError(f"predictions_df must contain columns {required_columns}")

    # Extract actual and predicted values
    y_true = predictions_df['target']
    y_pred = predictions_df['prediction']

    # Compute R² (coefficient of determination)
    r2 = r2_score(y_true, y_pred)

    # Compute Pearson correlation coefficient
    r_value, _ = pearsonr(y_true, y_pred)

    # Compute Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Store results in a DataFrame
    stats_df = pd.DataFrame({'R2': [r2], 'r': [r_value], 'RMSE': [rmse]})

    return stats_df
