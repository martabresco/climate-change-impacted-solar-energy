#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:25:18 CEST 2020
Expanded grid and automatic download
Thu Feb  3 12:43:10 CET 2022

@author: ahah
"""

import sys
import xarray as xr
import numpy as np
from wrf import interplevel
from datetime import timedelta
from glob import glob
import cftime
import future_wind as fw

def read_and_interp(dt,dq,du,dv,times):
    
    lev = [50.,100.,200.]
    Rd = 287.058
    g = 9.80665
    levs = slice(1.0,0.9)

    t = fw.combine_hemispheres(dt.ta,time=times,lev=levs)
    ps = fw.combine_hemispheres(dt.ps,time=times)    
    q = fw.combine_hemispheres(dq.hus,time=times,lev=levs) 
    u = fw.combine_hemispheres(du.ua,time=times,lev=levs)
    v = fw.combine_hemispheres(dv.va,time=times,lev=levs) 

    print("Times read",u.time[0].dt.strftime("%Y-%m-%d_%H:%M").values,
          u.time[-1].dt.strftime("%Y-%m-%d_%H:%M").values)

    rho = fw.air_density(ps,t.isel(lev=0),q.isel(lev=0))
    
    tv = fw.virtual_temperature(t,q)
    ap = du.ap.sel(lev=levs)
    b = du.b.sel(lev=levs)

    p = ap + b * ps  # pressure at full model levels
    p = p.transpose('time', 'lev', 'lat', 'lon')
    p0 = p.isel(lev=0)
    
    z0 = -(Rd/g) * tv.isel(lev=0) * np.log(p0/ps)
    z = z0

    klev = len(p.lev)
    for k in range(1,klev):
        z1 = z0 - (Rd/g)*0.5*(tv.isel(lev=k)+tv.isel(lev=k-1)) * \
            np.log(p.isel(lev=k)/p.isel(lev=k-1))
        z = xr.concat([z,z1], 'lev')
        z0 = z1
    
    z = z.transpose('time', 'lev', 'lat', 'lon')

    wspd = np.sqrt(u*u + v*v) 
    aux = interplevel(wspd, np.log(z), np.log(lev), meta=False)
    ws_z = xr.DataArray(aux, \
        coords=[z.time,lev,z.lat,z.lon], 
        dims=['time','level','lat','lon'])
        
    u_z = interplevel(u, z, lev)
    v_z = interplevel(v, z, lev)
    wd_z = np.rad2deg(np.arctan2(u_z,v_z)) + 180.
    
    filename = "wspd_wdir_"+u.time[0].dt.strftime("%Y-%m-%d_%H").values+".nc"

    return ws_z,wd_z,rho,filename

def make_path_to_file(root_url,model,experiment,variant,table,var,grid,version,dates):

    root_file = "/".join((root_url,experiment,variant,table,var,grid,version))
    filename = "_".join((var,table,model,experiment,variant,grid,dates))+".nc"
    path_to_file = "/".join((root_file,filename))
    return(path_to_file)

def main():
    # model = "CanESM5"
    # root_url = "http://crd-esgf-drc.ec.gc.ca/thredds/dodsC/esgF_dataroot/AR6/CMIP6/ScenarioMIP/CCCma/CanESM5"
    # ssp585/r1i1p2f1/6hrLev/va/gn/v20190429/va_6hrLev_CanESM5_ssp585_r1i1p2f1_gn_205101010000-205112311800.nc"
    model = "CanESM5-1"
    root_url = "http://crd-esgf-drc.ec.gc.ca/thredds/dodsC/esgI_dataroot/AR6/CMIP6/ScenarioMIP/CCCma/CanESM5-1"
    # ssp585/r1i1p2f1/6hrLev/ua/gn/v20190429/ua_6hrLev_CanESM5-1_ssp585_r1i1p2f1_gn_205101010000-205112311800.nc"
    
    grid = "gn"
    version = "v20190429"
    
    calendar = 'noLeap'
    experiment = sys.argv[1]
    if (experiment == "historical"):
        year = 1980; last_year = 2014
    else:
        # year = 2015; last_year = 2050
        year = 2050; last_year = 2070
    variant = sys.argv[2]  
    print("Retrieve data for",\
        "\n model:  ",model,"\n experiment:",experiment,\
        "\n variant:",variant)

    # What filenames already exist in the directory
    filenames = "wspd_wdir_????-??-??_??.nc"
    old_files = sorted(glob(filenames))

    if not old_files:    # This is necessary for the scenario files that start at 00Z
        print("No previous files")
        month = 1
        date = cftime.datetime(year,1,1,6,calendar=calendar) # Files start at 06 not 00
    else:
        ff = xr.open_dataset(old_files[-1],decode_times=True,use_cftime=True)
        date = ff.time[-1] + timedelta(hours=6)
        print("Next date:",date.values)
        year = date.dt.year.values
        month = date.dt.month.values
        date = fw.datetime_to_cftime(date,calendar=calendar)

    print("year",year)
    last_date = cftime.datetime(last_year+1,1,1,0,calendar=calendar)
    print(date,last_date)

    while (year <= last_year):

        # What is the date string in the files, 1 year each
        start_date = cftime.datetime(year,1,1,0,calendar=calendar)
        end_date = cftime.datetime(year,12,31,18,calendar=calendar)
        dates = "-".join((start_date.strftime("%Y%m%d%H%M"),end_date.strftime("%Y%m%d%H%M")))
        print("Yearly file dates",dates)

        # Find the file where next_day is found 
        var = "ua"; table = "6hrLev"
        path_to_file = make_path_to_file(root_url,model,experiment,variant,table,var,grid,version,dates)
        print("file open:",path_to_file)
        du = xr.open_dataset(path_to_file,decode_times=True,use_cftime=True)
        
        var = "va"; table = "6hrLev"
        path_to_file = make_path_to_file(root_url,model,experiment,variant,table,var,grid,version,dates)
        print("file open:",path_to_file)
        dv = xr.open_dataset(path_to_file,decode_times=True,use_cftime=True)

        var = "ta"; table = "6hrLev"
        path_to_file = make_path_to_file(root_url,model,experiment,variant,table,var,grid,version,dates)
        print("file open:",path_to_file)
        dt = xr.open_dataset(path_to_file,decode_times=True,use_cftime=True)

        var = "hus"; table = "6hrLev"
        path_to_file = make_path_to_file(root_url,model,experiment,variant,table,var,grid,version,dates)
        print("file open:",path_to_file)
        dq = xr.open_dataset(path_to_file,decode_times=True,use_cftime=True)

        for i in range(12):
            date = cftime.datetime(year,month,1,0,calendar=calendar)
            year = year + month // 12
            month = month % 12 + 1
            print(year,month)
            date_end = cftime.datetime(year,month,1,0,calendar=calendar) - timedelta(hours=6)
            print(date,date_end)

            ws,wd,rho,filename = read_and_interp(
                dt,dq,du,dv,slice(date,date_end))

            ds = fw.make_data_set(du,ws,wd,rho)
            ds.to_netcdf(filename,mode="w",engine="netcdf4",
                        unlimited_dims='time')
            print(filename," written to disk")
                
        # date = cftime.datetime(year,1,1,6,calendar=calendar)
        # print("Next date:",date.strftime("%Y-%m-%d_%H"),"year:",year)

if __name__ == "__main__":
    main()
