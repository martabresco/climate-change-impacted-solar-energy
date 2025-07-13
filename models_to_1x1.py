import xarray as xr
import numpy as np
import os
import glob
import re
import xesmf as xe

def collect_files(base_path, models, periods):
    files_model = {}

    # Define the year ranges for each period
    year_ranges = {
        "historical": range(1980, 2015),  # 2014 included
        "ssp585": range(2065, 2100)       # 2099 included
    }
    for model in models:
        model_files = {}
        for period in periods:
            # Construct the path
            search_path = os.path.join(base_path, model, period)
            # Match files with the desired pattern
            file_pattern = os.path.join(search_path, "aggregated_solar_power_seasons_*.nc")
            matched_files = glob.glob(file_pattern)

            # Filter files by year
            filtered_files = []
            for file_path in matched_files:
                filename = os.path.basename(file_path)
                # Extract year from filename
                match = re.search(r"(\d{4})", filename)
                if match:
                    year = int(match.group(1))
                    if year in year_ranges[period]:
                        filtered_files.append(file_path)

            model_files[period] = filtered_files
        files_model[model] = model_files

    return files_model


def regrid(ds_in, ds_out, method='conservative'):
    """
    Regrid a dataset while preserving the time dimension.

    Parameters:
    - ds_in: xarray.Dataset
        The input dataset to be regridded.
    - ds_out: xarray.Dataset
        The target grid dataset.
    - method: str
        The regridding method (e.g., 'bilinear', 'conservative').

    Returns:
    - regridded_ds: xarray.Dataset
        The regridded dataset with the time dimension preserved.
    """

    # Set up the regridder
    regridder = xe.Regridder(ds_in, ds_out, method=method, periodic=False)

    # Apply the regridder to all variables, preserving the time dimension
    regridded_ds = regridder(ds_in)


    return regridded_ds


def regrid_power(files, standard_grid):
    """
    Regrid power datasets for each model and period, preserving the time dimension.
    """
    # Define regridding methods for each model
    regridding_methods = {
        "ACCESS-CM2": "bilinear",
        "CanESM5": "bilinear",
        "CMCC-CM2-SR5": "conservative",
        "CMCC-ESM2": "conservative",
        "HadGEM3-GC31-LL": "bilinear",
        "HadGEM3-GC31-MM": "conservative",
        "MRI-ESM2-0": "bilinear"
    }

    for model, periods in files.items():
        for period, file_list in periods.items():
            # Create the output directory for the model and period
            output_dir = os.path.join(f"/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power_notemp/", model+"_1x1grid", period)

            for file_path in file_list:
                # Construct the output file path
                filename = os.path.basename(file_path)
                output_file = os.path.join(output_dir, filename)

                # Skip if the output file already exists
                if os.path.exists(output_file):
                    print(f"File already exists, skipping: {output_file}")
                    continue

                # Open the file
                ds = xr.open_dataset(file_path)

                # Debugging: Check shapes
                print(f"Input data shape: {ds.dims}")
                print(f"Standard grid shape: {standard_grid.dims}")

                # Get the regridding method for the current model
                method = regridding_methods.get(model, "bilinear")  # Default to "bilinear" if model is not in the mapping

                # Apply the regridding function
                regridded_ds = regrid(ds, standard_grid, method=method)

                # Save the regridded dataset
                regridded_ds.to_netcdf(output_file)

                print(f"Regridded file saved to: {output_file} using method: {method}")


def main():
    base_path = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power_notemp/"
    models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2","HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]  # Test with only one model
    periods = ["historical", "ssp585"]

    # Collect files
    files = collect_files(base_path, models, periods)

    # Load the standard grid
    standard_grid_path = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/europe_1x1.nc"
    standard_grid = xr.open_dataset(standard_grid_path)

    # Apply regridding
    regrid_power(files, standard_grid)


if __name__ == "__main__":
    main()