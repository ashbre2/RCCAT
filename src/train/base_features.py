def BASE_FEATURES():

    
    # Define the complete feature mapping (internal_name: human_readable_label)
    feature_map = {
        'AvgSurfT_tavg': 'Surface Temperature',
        'BareSoilT_tavg': 'Bare Soil Temperature',
        'CanopInt_tavg': 'Total Canopy Water Storage',
        'ECanop_tavg': 'Interception Evaporation',
        'ESoil_tavg': 'Bare Soil Evaporation',
        'Evap_tavg': 'Total Evapotranspiration',
        'LWdown_f_tavg': 'Surface Downwelling Longwave Flux',
        'Lwnet_tavg': 'Surface Net Downward Longwave Flux',
        'Psurf_f_tavg': 'Surface Pressure',
        'Qair_f_tavg': 'Specific Humidity',
        'Qg_tavg': 'Downward Heat Flux in Soil',
        'Qh_tavg': 'Surface Upward Sensible Heat Flux',
        'Qle_tavg': 'Surface Upward Latent Heat Flux',
        'Qs_tavg': 'Surface Runoff Amount',
        'Rainf_f_tavg': 'Rainfall Flux (Rain + Snow)',
        'Rainf_tavg': 'Precipitation Rate',
        'SWdown_f_tavg': 'Surface Downwelling Shortwave Flux',
        'SoilMoi00_10cm_tavg': 'Soil Moisture (0-200 cm), m³ m⁻³',
        'SoilTemp00_10cm_tavg': 'Soil Temperature (0-100 cm), K',
        'Swnet_tavg': 'Surface Net Downward Shortwave Flux',
        'TVeg_tavg': 'Vegetation Transpiration',
        'Tair_f_tavg': 'Air Temperature W',
        'VegT_tavg': 'Canopy Temperature',
        'Wind_f_tavg': 'Wind Speed W',
        'WT_tavg': 'Water in Aquifer and Saturated Soil',
        'WaterTableD_tavg': 'Water Table Depth',
        'LAI_filtered': 'Leaf Area Index (Filtered)',
        'Fpar_filtered': 'Fraction of Photosynthetically Active Radiation',
        'NDVI': 'Normalized Difference Vegetation Index',
        'EVI': 'Enhanced Vegetation Index',
        'MIR_reflectance': 'Mid-Infrared Reflectance',
        'NIR_reflectance': 'Near-Infrared Reflectance',
        'blue_reflectance': 'Blue Reflectance',
        'red_reflectance': 'Red Reflectance'
    }


    # Define the complete feature mapping (internal_name: human_readable_label)
    # feature_map = {
    #     'Tair_f_tavg': 'Air Temperature W'
    # }


    
    return feature_map



