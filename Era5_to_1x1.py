import xarray as xr
import numpy as np
import os
import glob
import re
import xesmf as xe


def collect_files_era5(base_path):
    files_era5 = []
    file_pattern = os.path.join(base_path, "hourly_power_era5_*.nc")
    matched_files = glob.glob(file_pattern)
    files_era5.extend(matched_files)  # Fix: Use 'extend' to avoid nested lists
    return files_era5

def regrid(ds_in, ds_out, method='bilinear'):
    # Set up the regridder
    regridder = xe.Regridder(ds_in, ds_out, method=method, periodic=False)

    # Apply the regridder to all variables, preserving the time dimension
    regridded_ds = regridder(ds_in)
    return regridded_ds

def regrid_era5(files, standard_grid, output_path):
    for file in files:
        # Extract only the file name from the full path
        filename = os.path.basename(file)  # Correctly extract the file name
        output_file = os.path.join(output_path, filename)  # Combine with output path

        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"File already exists, skipping: {output_file}")
            continue

        # Open the file
        ds = xr.open_dataset(file, engine='netcdf4')

        # Debugging: Check shapes
        print(f"Input data shape: {ds.dims}")
        print(f"Standard grid shape: {standard_grid.dims}")

        # Get the regridding method for the current model
        method = "bilinear"  # Default to "bilinear" if model is not in the mapping

        # Apply the regridding function
        regridded_ds = regrid(ds, standard_grid, method=method)

        # Save the regridded dataset
        regridded_ds.to_netcdf(output_file)
        print(f"Regridded file saved to: {output_file}")
        
def main():
    base_path="/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power_Era5_h/"
    output_path="/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power_Era5_h_1x1/"
    files=collect_files_era5(base_path)
    # Load the standard grid
    standard_grid_path = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/europe_1x1.nc"
    standard_grid = xr.open_dataset(standard_grid_path)
    # Apply regridding
    regrid_era5(files, standard_grid, output_path)


if __name__ == "__main__":
    main()
