from itertools import combinations

def FFS_COMBS(all_feature_keys, feature_set):
    """
    Generate feature combinations based on the given feature_set.
    
    - If feature_set is empty, return a list of single-feature combinations.
    - If feature_set has 1 feature, return all pairs with that feature and another feature.
    - If feature_set has 2 features, return all triplets with those features and another feature.
    - Generalized for N features in feature_set.
    """
    # Ensure all features in feature_set are valid
    feature_set = [f for f in feature_set if f in all_feature_keys]

    if not feature_set:
        # If feature_set is empty, return single feature combinations
        single_combinations = [[feature] for feature in all_feature_keys]
        return single_combinations
    
    # Determine remaining features
    remaining_features = [f for f in all_feature_keys if f not in feature_set]

    if not remaining_features:
        return []  # No possible new combinations if all features are already in feature_set
    
    # Generate new combinations
    num_new_features = 1  # We always add one new feature at a time
    additional_combinations = list(combinations(remaining_features, num_new_features))

    # Format the final list
    result = [feature_set + list(combo) for combo in additional_combinations]
    
    return result

# # Example usage:
# all_feature_keys = ['AvgSurfT_tavg', 'BareSoilT_tavg', 'CanopInt_tavg', 'ECanop_tavg']

# print("Single feature combinations:", generate_feature_combinations(all_feature_keys, []))
# print("Pairs with 'AvgSurfT_tavg':", generate_feature_combinations(all_feature_keys, ['AvgSurfT_tavg']))
# print("Triplets with ['AvgSurfT_tavg', 'BareSoilT_tavg']:", generate_feature_combinations(all_feature_keys, ['AvgSurfT_tavg', 'BareSoilT_tavg']))
