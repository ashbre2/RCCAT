# README

## Overview
This project implements a Forward Feature Selection (FFS) workflow for predicting fluxes (e.g., `FC` or `FCH4`) at multiple sites. It uses a Leave-One-Site-Out (LOSO) cross-validation approach to evaluate model performance when each site is held out in turn.

## Repository Structure

1. **main.py**  
   - **Core driver script**.  
   - Performs Forward Feature Selection by adding features step-by-step and checking model improvement (using LOSO).  
   - Saves a CSV summary of the selection process in `FFS_results/` (and also in the local directory).  
   - Prints final selected features and final model performance metrics.

2. **AI_model.py**  
   - Defines `get_model_for_target(...)`, which returns an *unfitted* model (e.g., Linear Regression, Random Forest, XGBoost, LightGBM, SVM, LSTM, or GRU) based on user choice and target.  
   - Defines `build_neural_net(...)`, which constructs a simple LSTM or GRU neural network when `model_type` is `lstm` or `gru`.

3. **base_features.py**  
   - Returns a dictionary mapping each *feature name* (as used internally) to a more readable label (e.g., `{'Tair_f_tavg': 'Air Temperature'}`).

4. **data_loader.py**  
   - Loads site-specific datasets (feature and target NetCDF files) and merges them.  
   - Outputs combined DataFrames: one for the features and one for the targets, keeping track of each site.

5. **evaluation.py**  
   - Provides a function `compute_metrics(...)` to compute mean squared error (MSE), RMSE, R², and correlation for each site and overall.

6. **ffs.py**  
   - Implements the logic for generating new feature combinations for the Forward Feature Selection process.  
   - At each step, it takes the already-selected features and systematically tries adding one new feature at a time.

7. **loso.py**  
   - Executes the Leave-One-Site-Out cross-validation.  
   - For each site, it trains the model on the other sites and tests on the held-out site.  
   - Applies feature scaling and returns the predictions (and associated actual values).

---

## How to Run

1. **Install Dependencies**  
   Ensure you have the necessary Python libraries:
   - `pandas`, `numpy`
   - `scikit-learn`
   - `joblib`
   - `xgboost`
   - `lightgbm`
   - `tensorflow`
   - Any other dependencies you see in the `import` statements.

2. **Organize Your Data**  
   - Place NetCDF files for each site in the correct location (the code looks for them in `../../data/processed` by default).  
   - The filenames are expected to follow the pattern `[siteName]_WLDAS.nc`, `[siteName]_MCD15A3H.nc`, `[siteName]_MOD13Q1.nc`, `[siteName]_targets.nc`, etc.

3. **Configure `main.py`**  
   - In `main.py`, edit:
     - `model_type` (e.g., `"random_forest"`, `"linear"`, etc.).
     - `target_var` (e.g., `"FCH4"` or `"FC"`).
     - `site_list` (list of site names you want to run).

4. **Run `main.py`**  
   ```bash
   python main.py

