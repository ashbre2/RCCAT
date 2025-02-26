import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks
from AI_model import get_model_for_target, build_neural_net

def LOSO(features_df, targets_df, target_variable, feature_list=None, model_type='linear'):
    """
    Perform Leave-One-Site-Out (LOSO) cross-validation with feature scaling applied to all models.

    Parameters
    ----------
    features_df : pd.DataFrame
        A DataFrame containing all features and a 'site_label' column (indexed by datetime).
    targets_df : pd.DataFrame
        Contains at least one target variable (target_variable) plus 'site_label' column
        (indexed by datetime).
    target_variable : str
        The target variable to predict (e.g., 'FCH4' or 'FC').
    feature_list : list, optional
        Columns to use for modeling. If None, all columns (except site_label) are used.
    model_type : str, optional
        One of ['linear', 'svm', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru'].

    Returns
    -------
    predictions_df : pd.DataFrame
        A DataFrame indexed by (datetime, site_label) with:
            - target: actual target value
            - prediction: model predictions
    """

    # ----------------------------------------------------
    # 1. Select Features
    # ----------------------------------------------------
    if feature_list is not None:
        cols_to_use = [col for col in feature_list if col in features_df.columns]
        if 'site_label' not in cols_to_use:
            cols_to_use.append('site_label')
        X_full = features_df[cols_to_use]
    else:
        cols_to_use = [c for c in features_df.columns if c != 'site_label']
        X_full = features_df[cols_to_use + ['site_label']]

    # ----------------------------------------------------
    # 2. Drop rows where target is NaN
    # ----------------------------------------------------
    valid_mask = ~targets_df[target_variable].isna()
    X = X_full.loc[valid_mask].copy()
    y = targets_df.loc[valid_mask, target_variable].copy()

    # ----------------------------------------------------
    # 3. Fit a single scaler across ALL data
    # ----------------------------------------------------
    numeric_cols = [c for c in X.columns if c != 'site_label']
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit scalers on ALL sites (to maintain consistency)
    scaler_X.fit(X[numeric_cols].values)
    scaler_y.fit(y.values.reshape(-1, 1))

    # Transform ALL data once (ensuring the same scaling across LOSO)
    X_scaled = X.copy()
    X_scaled.loc[:, numeric_cols] = scaler_X.transform(X_scaled[numeric_cols].values)
    y_scaled = pd.Series(
        data=scaler_y.transform(y.values.reshape(-1,1)).ravel(),
        index=y.index
    )

    # Get unique sites
    unique_sites = X['site_label'].unique()
    all_predictions = []

    # ----------------------------------------------------
    # 4. LOSO Cross-Validation
    # ----------------------------------------------------
    for holdout_site in unique_sites:
        # Create train/test splits
        train_mask = (X_scaled['site_label'] != holdout_site)
        test_mask  = (X_scaled['site_label'] == holdout_site)

        X_train_scaled = X_scaled.loc[train_mask, numeric_cols]
        y_train_scaled = y_scaled.loc[train_mask]

        X_test_scaled = X_scaled.loc[test_mask, numeric_cols]
        y_test_scaled = y_scaled.loc[test_mask]

        # ------------------------------------------------
        # 4a. Get the model
        # ------------------------------------------------
        model = get_model_for_target(target_variable, model_type)

        # ------------------------------------------------
        # 4b. If it's LSTM/GRU, build and train the Keras model
        # ------------------------------------------------
        if model_type in ['lstm', 'gru']:
            X_train_seq = np.expand_dims(X_train_scaled.values, axis=1)
            X_test_seq  = np.expand_dims(X_test_scaled.values, axis=1)
            input_dim = X_train_seq.shape[-1]

            nn_model = build_neural_net(model_type, input_dim)
            es = callbacks.EarlyStopping(patience=20, restore_best_weights=True)

            nn_model.fit(
                X_train_seq, y_train_scaled.values,
                validation_data=(X_test_seq, y_test_scaled.values),
                epochs=100, batch_size=512,
                callbacks=[es],
                verbose=0
            )

            test_preds_scaled = nn_model.predict(X_test_seq).flatten()
            test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1,1)).ravel()
            actuals = y.loc[test_mask].values  # unscaled actual
        
        else:
            # ------------------------------------------------
            # 4c. Scale all models, not just LSTMs
            # ------------------------------------------------
            model.fit(X_train_scaled, y_train_scaled)
            test_preds_scaled = model.predict(X_test_scaled)
            
            # Convert predictions back to original scale
            test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1,1)).ravel()
            actuals = y.loc[test_mask].values

        # ------------------------------------------------
        # 4d. Collect predictions
        # ------------------------------------------------
        fold_predictions = pd.DataFrame({
            'target': actuals,
            'prediction': test_preds,
            'site_label': X.loc[test_mask, 'site_label']  # unscaled site_label
        }, index=X.loc[test_mask].index)

        all_predictions.append(fold_predictions)

    # ----------------------------------------------------
    # 5. Concatenate and finalize results
    # ----------------------------------------------------
    predictions_df = pd.concat(all_predictions).sort_index()
    predictions_df.set_index('site_label', append=True, inplace=True)
    predictions_df.sort_index(inplace=True)

    return predictions_df
