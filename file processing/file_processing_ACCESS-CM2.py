import sys
import xarray as xr
import numpy as np
from datetime import timedelta
from glob import glob
import cftime
from future_wind_copy import combine_hemispheres 
from datetime import datetime
import os
import re
from collections import defaultdict



def cut_europe_and_interpolate(ds_rsds,ds_rsdsdiff,ds_tas):
    ds_rsds_europe = combine_hemispheres(ds_rsds,minlat=20.,maxlat=75.,minlon=330.,maxlon=50.)
    ds_rsdsdiff_europe = combine_hemispheres(ds_rsdsdiff,minlat=20.,maxlat=75.,minlon=330.,maxlon=50.)    
    ds_tas_europe = combine_hemispheres(ds_tas,minlat=20.,maxlat=75.,minlon=330.,maxlon=50.) 

    # Interpolate tas to match rsds time
    tas_interp= ds_tas_europe['tas'].interp(time=ds_rsds_europe['time'], method="linear")
    # Create a mask for the NaN values using .isnull()
    nan_mask = tas_interp.isnull()

    # For each time step, replace NaNs with the values from the next time step
    for t in range(len(tas_interp.time) - 1):  # Exclude the last time step
        #Use .isel() to ensure the correct alignment of coordinates
        tas_interp[t] = tas_interp[t].where(~nan_mask[t], tas_interp.isel(time=t + 1))
    
    ds_tas_europe['time'] = ds_rsds_europe['time']
    ds_tas_europe['tas'] = tas_interp


    return ds_rsds_europe, ds_rsdsdiff_europe, ds_tas_europe, tas_interp

def make_data_set(du,rsds,rsdsdiff,tas):
    """Creates xarray DataArray for netCDF write
    Args:
        du (dataset): sample dataset with attributes
        rsds (DataArray): wind speed 
        rsdsdiff (DataArray): wind direction
        tas (DataArray): surface air density

    Returns:
        xarray DataArray: DataArray for write
    """
    lat = xr.DataArray(
        data=rsds.lat.values.astype('float32'),
        dims=["lat"],
        coords=dict(
            lat=(["lat"], rsds.lat.values)
        ),
        attrs=dict(
        long_name="latitude",
        units="degrees_north",
        axis="Y"
        ),
    )
    lon = xr.DataArray(
        data=rsds.lon.values.astype('float32'),
        dims="lon",
        coords=dict(
            lon=(["lon"], rsds.lon.values)
        ),
        attrs=dict(
        long_name="longitude",
        units="degrees_east",
        axis="X"
        ),
    )
    
    ds = xr.Dataset(
        data_vars=dict(
            rsds = (
                ["time","lat","lon"],rsds.values.astype('float32'),
                dict(long_name = "rsds",
                units = "W/m2")),
            rsdsdiff = (
                ["time","lat","lon"],rsdsdiff.values.astype('float32'),
                dict(long_name = "rsdsdiff",
                units = "W/m2",
                vert_units = "W/m2")),
            tas = (
                ["time","lat","lon"],tas.values.astype('float32'),
                dict(long_name = "surface air density",
                units = "K",
                height = "surface")),
            ),
        coords=dict(
            lon=lon,
            lat=lat,
            time=rsds.time
            ),
        attrs=dict(
            data_source = "Processed data from CMIP6 runs",
            experiment = du.experiment_id,
            source = du.source_id,
            variant_label = du.variant_label,
            data_written = datetime.now().strftime("%d/%m/%Y %H:%M")
            )
    )   
    return ds


def main(): 
    # Folder containing the files
    folder_path = "/groups/FutureWind/SFCRAD/ACCESS-CM2/ssp585/r1i1p1f1/"  # Remember to change depending on the model

    # Regex to extract the period (last two date segments)
    pattern = re.compile(r"(\d{12})-(\d{12})\.nc$")

    # Desired period for filtering
    desired_start = datetime(2085, 1, 1, 0, 0)  # Start of the desired period
    desired_end = datetime(2095, 1, 1, 0, 0)    # End of the desired period

    # Organizing files by type and period
    file_dict = defaultdict(list)

    # List all netCDF files
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            start, end = match.groups()
            start_dt = datetime.strptime(start, "%Y%m%d%H%M")
            end_dt = datetime.strptime(end, "%Y%m%d%H%M")

            # Filter files to include only those overlapping with the desired period
            if end_dt >= desired_start and start_dt <= desired_end:
                # Determine file type and append to the dictionary
                if "tas" in filename:
                    file_dict["tas"].append((start_dt, end_dt, filename))
                elif "rsds_" in filename:  # Avoid matching "rsdsdiff"
                    file_dict["rsds"].append((start_dt, end_dt, filename))
                elif "rsdsdiff" in filename:
                    file_dict["rsdsdiff"].append((start_dt, end_dt, filename))

    # Function to find overlapping periods
    def find_overlaps(list1, list2):
        overlaps = []
        for start1, end1, file1 in list1:
            for start2, end2, file2 in list2:
                # Check if periods overlap
                if start1 <= end2 and start2 <= end1:
                    # Determine the overlapping period
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    overlaps.append((overlap_start, overlap_end, file1, file2))
        return overlaps

    # Find overlapping periods between tas and rsds files
    tas_rsds_overlaps = find_overlaps(file_dict["tas"], file_dict["rsds"])

    # Find overlapping periods between the overlaps above and rsdsdiff files
    final_overlaps = []
    for overlap_start, overlap_end, tas_file, rsds_file in tas_rsds_overlaps:
        for start3, end3, rsdsdiff_file in file_dict["rsdsdiff"]:
            if overlap_start <= end3 and start3 <= overlap_end:
                final_start = max(overlap_start, start3)
                final_end = min(overlap_end, end3)
                final_overlaps.append((final_start, final_end, tas_file, rsds_file, rsdsdiff_file))
    print(final_overlaps)

    # Process each set of overlapping files
    for overlap_start, overlap_end, tas_file, rsds_file, rsdsdiff_file in final_overlaps:
        print(f"Processing overlap from {overlap_start} to {overlap_end}:")
        print(f"  - TAS file: {tas_file}")
        print(f"  - RSDS file: {rsds_file}")
        print(f"  - RSDSDIFF file: {rsdsdiff_file}")

        # Open datasets
        tas_ds = xr.open_dataset(os.path.join(folder_path, tas_file))
        rsds_ds = xr.open_dataset(os.path.join(folder_path, rsds_file))
        rsdsdiff_ds = xr.open_dataset(os.path.join(folder_path, rsdsdiff_file))

        # Apply the first function: cut and interpolate
        ds_rsds_europe, ds_rsdsdiff_europe, ds_tas_europe, tas_interp = cut_europe_and_interpolate(
            rsds_ds, rsdsdiff_ds, tas_ds
        )
        rsds_data = ds_rsds_europe['rsds']
        rsdsdiff_data = ds_rsdsdiff_europe['rsdsdiff']
        tas_data = ds_tas_europe['tas']
        ds = make_data_set(rsds_ds, rsds_data, rsdsdiff_data, tas_data)

        # Ensure combined_dataset is defined and has a 'time' dimension
        if 'time' in ds.dims:
            unique_years = np.unique(ds.time.dt.year.values)

            for year in unique_years:
                # Generate the file path for the output file
                output_file = f"/groups/FutureWind/SFCRAD/ACCESS-CM2/ssp585/r1i1p1f1/rsds_rsdsdiff_tas_{year}.nc"  # Remember to change depending on model
                
                # Check if the file already exists
                if not os.path.exists(output_file):
                    # Extract data for this year using slicing
                    yearly_data = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

                    # Check if yearly_data is not empty
                    if yearly_data.time.size > 0:
                        # Save as .nc file
                        yearly_data.to_netcdf(output_file)
                        print(f"Saved data for year {year} to {output_file}")
                    else:
                        print(f"No data available for the year {year}")
                else:
                    print(f"File for year {year} already exists, skipping.")
        else:
            print("The dataset does not contain a 'time' dimension.")

        # Close datasets
        tas_ds.close()
        rsds_ds.close()
        rsdsdiff_ds.close()

    print("Processing complete.")

if __name__ == "__main__":
    main()

