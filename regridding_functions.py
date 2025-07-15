import numpy as np
import xesmf as xe
import numpy as np

import numpy as np
import xarray as xr
import xesmf as xe

import numpy as np
import xarray as xr
import xesmf as xe
from datetime import datetime
import pandas as pd

def read_and_average_era5_modified(field="influx_direct"):
    """Read a single NetCDF file containing multiple years and compute the long-term mean"""
    
    # Path to the single file containing data for multiple years
    #diri = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/"
    #file_2013 = "europe-2013.nc"  # Update this to your actual file name
    path="/groups/EXTREMES/cutouts/"
    files = [f'{path}europe-{year}-era5.nc' for year in range(1980, 2015)]

    # Print the generated file paths to verify correctness
    print(files)  # Check the paths generated (debugging step)
    
    # Open the dataset with xarray
    df = xr.open_mfdataset(files, combine="by_coords", join="inner")

    # Compute the long-term mean of the selected field over the time dimension
    return df[field].mean(dim="time")


def read_and_average_era5_4y(field="influx_direct"):
    """Read a single NetCDF file containing multiple years and compute the long-term mean"""
    
    # Path to the single file containing data for multiple years
    #diri = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/"
    #file_2013 = "europe-2013.nc"  # Update this to your actual file name
    path="/groups/EXTREMES/cutouts/"
    years = [1996, 2010, 2012, 2013]
    files = [f'{path}europe-{year}-era5.nc' for year in years]

    # Print the generated file paths to verify correctness
    print(files)  # Check the paths generated (debugging step)
    
    # Open the dataset with xarray
    df = xr.open_mfdataset(files, combine="by_coords", join="inner")

    # Compute the long-term mean of the selected field over the time dimension
    return df[field].mean(dim="time")



def read_and_average_sarah(field="influx_direct"):  # field is just a key to select a variable in the NetCDF file
    path = "/groups/EXTREMES/SARAH-3/"
    
    # List of years for which we have files (e.g., 2013)
    years = [1996, 2010, 2012, 2013]
    
    # Create the list of files without the influx_direct part in the file name
    files = [f'{path}europe-{year}-sarah3-era5.nc' for year in years]
    
    # Print the generated file paths to verify correctness
    print(files)  # Check the paths generated (debugging step)
    
    # Open the dataset with xarray
    df = xr.open_mfdataset(files, combine="by_coords")
    
    # Return the mean of the specified field over time
    return df[field].mean(dim="time")


def read_and_average_era5(field):
    """Read a range of years of ERA5 and compute the long-term mean"""
    
    diri = "/groups/EXTREMES/cutouts/"

    
    files = [f'{diri}europe-{year}-era5.nc' for year in range(1980, 2015)]
    print(files)
    df = xr.open_mfdataset(files,concat_dim="month",
                           combine="nested",decode_times=False)
    
    first_time = datetime(1980,1,15)
    end_time = datetime(2014,12,15)
    df = df.rename({'month':'time'})
    df['time'] = pd.date_range(first_time,end_time,
                               freq="1M")

    return df[field].mean(dim="time")


def read_and_average_cmip(diri,field="rsds"):
    path = "/groups/FutureWind/"+diri
    file = "rsds_rsdsdiff_tas"
    
    files = [f'{path+file}_{year}.nc' for year in range(1980, 2015)]
    print(files)
    df = xr.open_mfdataset(files,combine="by_coords")

    return df[field].mean(dim="time")


def regrid(ds_in, ds_out, method='conservative'):
    """Setup coordinates for esmf regridding"""

    # Get the longitude and latitude values as numpy arrays
    lon = ds_in.lon.values  # Convert to numpy array
    dlon = lon[1] - lon[0]
    lon_b = np.linspace(lon[0] - dlon/2., lon[-1] + dlon/2., len(lon) + 1)
    print(lon.size, lon_b.size)

    lat = ds_in.lat.values  # Convert to numpy array
    dlat = lat[1] - lat[0]
    lat_b = np.linspace(lat[0] - dlat/2., lat[-1] + dlat/2., len(lat) + 1)
    print(lat.size, lat_b.size)

    grid_in = {'lon': lon, 'lat': lat, 'lon_b': lon_b, 'lat_b': lat_b}

    # Repeat for output dataset
    lon = ds_out.lon.values  # Convert to numpy array
    dlon = lon[1] - lon[0]
    lon_b = np.linspace(lon[0] - dlon/2., lon[-1] + dlon/2., len(lon) + 1)
    print(lon.size, lon_b.size)

    lat = ds_out.lat.values  # Convert to numpy array
    dlat = lat[1] - lat[0]
    lat_b = np.linspace(lat[0] - dlat/2., lat[-1] + dlat/2., len(lat) + 1)
    print(lat.size, lat_b.size)

    grid_out = {'lon': lon, 'lat': lat, 'lon_b': lon_b, 'lat_b': lat_b}

    # Set up the regridder
    regridder = xe.Regridder(grid_in, grid_out, method, periodic=False)
    #regridder.clean_weight_file()

    return regridder

import numpy as np
import xesmf as xe





