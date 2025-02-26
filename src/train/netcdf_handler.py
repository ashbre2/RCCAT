import netCDF4 as nc
import pandas as pd
import numpy as np
import datetime
import os

def read_netcdf_file(file_path):
    """
    Reads a generic NetCDF file and returns a DataFrame.
    Expects the file to contain 'time' and 'values' variables,
    and an attribute 'variable_names' listing the column names.
    """
    with nc.Dataset(file_path, 'r') as ds:
        times = ds.variables['time'][:]
        values = ds.variables['values'][:]
        time_units = ds.variables['time'].units
        calendar = ds.variables['time'].calendar if 'calendar' in ds.variables['time'].ncattrs() else 'gregorian'
        var_names = ds.getncattr('variable_names').split(',')
    dates = nc.num2date(times, units=time_units, calendar=calendar)
    # Convert to Python datetime objects
    dates = [datetime.datetime(d.year, d.month, d.day) for d in dates]
    df = pd.DataFrame(values, index=dates, columns=var_names)
    return df

def save_dataframe_to_netcdf(df, file_path):
    """
    Saves a DataFrame to a NetCDF file.
    """
    folder = os.path.dirname(file_path)
    os.makedirs(folder, exist_ok=True)
    with nc.Dataset(file_path, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', len(df))
        ds.createDimension('variable', len(df.columns))
        times = ds.createVariable('time', 'f8', ('time',))
        values = ds.createVariable('values', 'f4', ('time', 'variable'))
        times[:] = nc.date2num(df.index.to_pydatetime(), units='days since 1970-01-01')
        values[:, :] = df.values
        times.units = 'days since 1970-01-01'
        times.calendar = 'gregorian'
        values.description = 'Processed data'
        ds.setncattr_string('variable_names', ','.join(df.columns))
    print(f"File saved as {file_path}")
