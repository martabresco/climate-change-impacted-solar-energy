import os
import glob
import re
import atlite

def collect_files_era5(base_path):
    files_era5 = []

    # Define the year range
    year_range = range(1980, 2015)

    # Match files with the desired pattern
    file_pattern = os.path.join(base_path, "europe-*-era5.nc")
    matched_files = glob.glob(file_pattern)

    # Filter files by year
    for file_path in matched_files:
        filename = os.path.basename(file_path)
        # Extract year from filename
        match = re.search(r"(\d{4})", filename)
        if match:
            year = int(match.group(1))
            if year in year_range:
                files_era5.append(file_path)

    return files_era5


def main():
    base_path = "/groups/EXTREMES/cutouts/"
    saving_path = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power_Era5/"

    # Ensure the saving directory exists
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # Collect ERA5 files
    era5_BOC_files = collect_files_era5(base_path)
    print(era5_BOC_files)

    for file in era5_BOC_files:
        try:
            # Read in cutout instead of creating it
            cutout = atlite.Cutout(path=file)
            power_era5 = cutout.pv(
                panel='CSi',
                orientation='latitude_optimal',
                tracking=None,
            )

            # Extract the year from the file name
            match = re.search(r"(\d{4})", os.path.basename(file))
            if match:
                year = match.group(1)
                # Construct the output file name
                output_file = os.path.join(saving_path, f"power_era5_{year}.nc")
            else:
                raise ValueError(f"Could not extract year from file name: {file}")

            # Save the power data to a NetCDF file
            power_era5.to_netcdf(output_file)
            print(f"Saved power data to {output_file}")

        except Exception as e:
            print(f"Failed to process file {file}: {e}")


if __name__ == "__main__":
    main()
