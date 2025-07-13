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
            #model+"_1x1grid"
            # Match files with the desired pattern
            file_pattern = os.path.join(search_path, "solar_power_*.nc")
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
def yearly_aggregation(files):
    """
    Aggregate yearly solar power for each file and save the result.
    """
    for model, periods in files.items():
        for period, file_list in periods.items():
            for file_path in file_list:
                file_name = os.path.basename(file_path)
                # Extract the year as a string for later
                year = int(re.search(r"(\d{4})", file_name).group(1))
                # Check whether the aggregated file already exists
                output_file = os.path.join(
                    os.path.dirname(file_path),
                    f"aggregated_solar_power_{year}.nc"
                )
                if os.path.exists(output_file):
                    print(f"Output for {year} already exists ({output_file}), skipping.")
                    continue

                # Open the dataset
                ds = xr.open_dataset(file_path)

                # Aggregate yearly solar power
                yearly_power = ds["specific generation"].sum(dim="time")

                # Save the aggregated dataset
                output_file = os.path.join(os.path.dirname(file_path), f"aggregated_solar_power_{year}.nc")
                yearly_power.to_netcdf(output_file)
                print(f"Yearly aggregated solar power saved to: {output_file}")

def seasonal_aggregation(files):
    """
    Aggregate seasonal solar power for each file and save the result.
    """
    for model, periods in files.items():
        for period, file_list in periods.items():
            for file_path in file_list:
                file_name = os.path.basename(file_path)
                # Extract the year as a string for later
                year = int(re.search(r"(\d{4})", file_name).group(1))
                # Open the dataset
                ds = xr.open_dataset(file_path)

                # Define seasons
                seasons = {
                    "Winter": [12, 1, 2],  # Winter
                    "Spring": [3, 4, 5],  # Spring
                    "Summer": [6, 7, 8],  # Summer
                    "Autumn": [9, 10, 11]  # Autumn
                }

                # Aggregate seasonal solar power
                seasonal_power = {}
                for season, months in seasons.items():
                    seasonal_power[season] = ds["specific generation"].where(
                        ds["time.month"].isin(months), drop=True
                    ).sum(dim="time")

                # Combine seasonal data into a single dataset
                seasonal_ds = xr.Dataset(seasonal_power)

                # Save the aggregated dataset
                output_file = os.path.join(os.path.dirname(file_path), f"aggregated_solar_power_seasons_{year}.nc")
                seasonal_ds.to_netcdf(output_file)
                print(f"Seasonal aggregated solar power saved to: {output_file}")


def main():
    base_path = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power_notemp/"
    models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2","HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]  # Test with only one model
    periods = ["historical", "ssp585"]

    # Collect files
    files = collect_files(base_path, models, periods)
    print(files)
    #yearly_aggregation(files)
    seasonal_aggregation(files)

if __name__ == "__main__":
    main()