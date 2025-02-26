import os
import pandas as pd
from netcdf_handler import read_netcdf_file  # Ensure this function correctly reads NetCDF files

# Define the correct base path
BASE_DATA_PATH = r"..\..\data\processed"

def load_site_data(site_name):
    """
    Loads feature and target data for a given site from NetCDF files.
    Returns a tuple: (features_data, targets_data)
    """

    # Construct file paths (No Subdirectories)
    wldas_path = os.path.join(BASE_DATA_PATH, f"{site_name}_WLDAS.nc")
    modis_path1 = os.path.join(BASE_DATA_PATH, f"{site_name}_MCD15A3H.nc")
    modis_path2 = os.path.join(BASE_DATA_PATH, f"{site_name}_MOD13Q1.nc")
    targets_path = os.path.join(BASE_DATA_PATH, f"{site_name}_targets.nc")

    wldas_data = read_netcdf_file(wldas_path)
    modis_data1 = read_netcdf_file(modis_path1)
    modis_data2 = read_netcdf_file(modis_path2)
    targets_data = read_netcdf_file(targets_path)

    # Merge available feature datasets
    site_features = pd.DataFrame()
    if wldas_data is not None:
        site_features = wldas_data.copy()
    if modis_data1 is not None:
        site_features = pd.merge(site_features, modis_data1, left_index=True, right_index=True, how='inner') if not site_features.empty else modis_data1
    if modis_data2 is not None:
        site_features = pd.merge(site_features, modis_data2, left_index=True, right_index=True, how='inner') if not site_features.empty else modis_data2

    return site_features, targets_data

def merge_site_data(site_list):
    merged_features = pd.DataFrame()
    merged_targets = pd.DataFrame()
    site_labels = []
    
    for i, site in enumerate(site_list):
        features_data, targets_data = load_site_data(site)
        
        # Add site labels to the dataframes
        features_data['site_label'] = i
        targets_data['site_label'] = i
        
        # Record label info (e.g., "Site 0 (SiteName): X rows")
        site_labels.append(f"Site {i} ({site}): {features_data.shape[0]} rows")
        
        merged_features = pd.concat([merged_features, features_data], axis=0)
        merged_targets = pd.concat([merged_targets, targets_data], axis=0)
    
    return merged_features, merged_targets, site_labels



def process_site_data(site_list, selected_features, target_variables):
    """
    Processes data by merging sites and subsetting on the selected features and target variables.
    Returns separate DataFrames for features and targets along with the site labels.
    """
    features_df, targets_df, site_labels = merge_site_data(site_list)

    # Ensure 'site_label' is retained in the features DataFrame
    features_df = features_df[selected_features + ['site_label']]
    targets_df  = targets_df[target_variables   + ['site_label']]  # Only subset targets on target_variables

    return features_df, targets_df, site_labels


