#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:39:26 2019

@author: ahah
"""

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime,timedelta
import xesmf as xe

plt.rcParams.update({'font.size': 18,
                     'axes.labelsize': 18,
                     'axes.titlesize': 18,
                     'legend.fontsize': 18,
                     'xtick.labelsize': 12,
                     'ytick.labelsize': 12
                     })

#%%
def read_and_average_era5(field):
    """Read a range of years of ERA5 and compute the long-term mean"""
    
    diri = "/groups/FutureWind/ERA5/"
    file = "era5_mean"
    
    files = [f'{diri+file}_{year}.nc' for year in range(1979, 2015)]
    df = xr.open_mfdataset(files,concat_dim="month",
                           combine="nested",decode_times=False)
    
    first_time = datetime(1979,1,15)
    end_time = datetime(2015,1,15)
    df = df.rename({'month':'time'})
    df['time'] = pd.date_range(first_time,end_time,
                               freq="1M")

    return df[field].mean(dim="time")

def read_and_average_cmip(diri,field="wind_speed"):
    path = "/groups/FutureWind/"+diri
    file = "wspd_wdir_6hrLev"
    
    files = [f'{path+file}_{year}.nc' for year in range(1979, 2015)]
    #print(files)
    df = xr.open_mfdataset(files,combine="by_coords")

    return df[field].sel(level=100).mean(dim="time")

def regrid(ds_in, ds_out, method='conservative'):
    """Setup coordinates for esmf regridding"""

    lon = ds_in.lon   # centers
    dlon = lon[1] - lon[0]
    lon_b = np.linspace(lon[0]-dlon/2.,lon[-1]+dlon/2.,len(lon)+1)
    print(lon.size,lon_b.size)

    lat = ds_in.lat
    dlat = lat[1] - lat[0]
    lat_b = np.linspace(lat[0]-dlat/2.,lat[-1]+dlat/2.,len(lat)+1)
    print(lat.size,lat_b.size)
    
    grid_in = {'lon': lon, 'lat': lat,
               'lon_b': lon_b, 'lat_b': lat_b}

    lon = ds_out.lon   # centers
    dlon = lon[1] - lon[0]
    lon_b = np.linspace(lon[0]-dlon/2.,lon[-1]+dlon/2.,len(lon)+1)
    print(lon.size,lon_b.size)

    lat = ds_out.lat
    dlat = lat[1] - lat[0]
    lat_b = np.linspace(lat[0]-dlat/2.,lat[-1]+dlat/2.,len(lat)+1)
    print(lat.size,lat_b.size)

    grid_out = {'lon': lon, 'lat': lat,
                'lon_b': lon_b, 'lat_b': lat_b}

    regridder = xe.Regridder(grid_in, grid_out, method, periodic=False)
    regridder.clean_weight_file()
    return regridder

wmean = read_and_average_era5('wind_speed_6hr')

model = "NorESM2-LM" #CNRM-CM6-1"
experiment = "historical"
variant = "r1i1p1f1"
diri = model + "/" + experiment + "/" + variant + "/"
wmean_cmip = read_and_average_cmip(diri)
print(wmean_cmip)

regridder = regrid(wmean,wmean_cmip)
print(regridder)
wmean_interp = regridder(wmean)

f = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.Orthographic(10., 60.))
levs = np.arange(2.,12.,0.5)

wmean_interp.plot(ax=ax, transform=ccrs.PlateCarree(),
           levels=levs,cmap=plt.cm.rainbow)
plt.show()

#%%

#%%

# regridder = xe.Regridder(grid_in, grid_out, 'conservative')
# regridder.clean_weight_file()
# regridder

# data_out = regridder(wmean)

# f = plt.figure(figsize=(8, 8))
# ax = plt.axes(projection=ccrs.Orthographic(10., 60.))
# data_out.plot(ax=ax, transform=ccrs.PlateCarree(),
#               levels=levs,cmap=plt.cm.rainbow)
# plt.show()
