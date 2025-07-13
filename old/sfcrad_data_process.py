import sys
import xarray as xr
import numpy as np
from datetime import timedelta
from glob import glob
import cftime
import os
from future_wind_copy import combine_hemispheres
from datetime import datetime,timedelta



def process_files(model_dir):
    # Define the variables you are interested in
    variables = ['rsds', 'rsdsdiff', 'tas']
    
    # Get the list of files in the experiment subfolder
    files = sorted(os.listdir(model_dir))
    
    # Group files by time period
    file_groups = {}
    for file in files:
        for var in variables:
            if var in file:
                time_period = file.split('_')[-1].split('.')[0]
                if time_period not in file_groups:
                    file_groups[time_period] = {}
                file_groups[time_period][var] = os.path.join(files_dir, file)
    
    # Process each group of files
    for time_period, group in file_groups.items():
        if all(var in group for var in variables):
            # Open the datasets
            ds_rsds = xr.open_dataset(group['rsds'])
            ds_rsdsdiff = xr.open_dataset(group['rsdsdiff'])
            ds_tas = xr.open_dataset(group['tas'])
            
            # Apply your changes here
            # For example, let's just print the time period and the datasets
            print(f"Processing time period: {time_period}")
            print(ds_rsds)
            print(ds_rsdsdiff)
            print(ds_tas)
            
        else:
            print(f"Missing variables for time period: {time_period}")

# Example usage
files_dir = "/groups/FutureWind/AllEurope/CanESM5/historical/r1i1p2f1/"

process_files(files_dir)











