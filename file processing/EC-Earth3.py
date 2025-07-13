#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:51:03 CEST 2022

@author: ahah
"""
import sys
import xarray as xr
import numpy as np
from wrf import interplevel
from datetime import timedelta
from glob import glob
import cftime

from pyesgf.search import SearchConnection
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

    p = ap + b * ps # pressure at full model levels
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

def main():

    # Different servers might be available at different 
    # times
    # server = 'https://esgf-node.llnl.gov/esg-search'
    server = 'https://esgf-data.dkrz.de/esg-search'
    conn = SearchConnection(server, distrib=True)

    # model = "EC-Earth3"
    model = "EC-Earth3-Veg"
    calendar = "proleptic_gregorian"
    experiment = sys.argv[1]
    if (experiment == "historical"):
        year = 1980; last_year = 2014
    else:
        year = 2015; last_year = 2050
    variant = sys.argv[2]  
    print("Retrieve data for",\
        "\n model:  ",model,"\n experiment:",experiment,\
        "\n variant:",variant)

    # What filenames already exist in the directory
    filenames = "wspd_wdir_????-??-??_??.nc"
    old_files = sorted(glob(filenames))

    if not old_files:    # This is necessary for the scenario files that start at 00Z
        print("No previous files")
        date = cftime.datetime(year,1,1,0,calendar=calendar)
    else:
        ff = xr.open_dataset(old_files[-1],decode_times=True,use_cftime=True)
        date = ff.time[-1] + timedelta(hours=6)
        print("Next date:",date.values)
        year = date.dt.year.values
        date = fw.datetime_to_cftime(date,calendar=calendar)

    last_date = cftime.datetime(last_year+1,1,1,0,calendar=calendar)
    nn = 31

    while (year <= last_year):

        u_file = fw.search_esgf(
            conn,"ua",model,experiment,variant,date=date)    

        # First start and end dates in file from the filename
        dates = u_file[0].split("_")[-1].split("-")[0][0:10]
        start_time_in_file = fw.date_to_cftime(dates,calendar='proleptic_gregorian')
        dates = u_file[0].split("_")[-1].split("-")[1][0:10]
        end_time_in_file = fw.date_to_cftime(dates,calendar='proleptic_gregorian')
        print("Dates in file:",start_time_in_file.strftime(),\
            end_time_in_file.strftime())

        print("Open U dataset",u_file)
        du = xr.open_dataset(u_file[0],decode_times=True,use_cftime=True)
        time = [fw.datetime_to_cftime(date) for date in du.time]

        v_file = fw.search_esgf(
            conn,"va",model,experiment,variant,date=date)
        print("Open V dataset",v_file[0])
        dv = xr.open_dataset(v_file[0],decode_times=True,use_cftime=True)

        t_file = fw.search_esgf(
            conn,"ta",model,experiment,variant,date=date)
        print("Open T dataset",t_file[0])
        dt = xr.open_dataset(t_file[0],decode_times=True,use_cftime=True)

        q_file = fw.search_esgf(
            conn,"hus",model,experiment,variant,date=date)
        print("Open Q dataset",q_file[0])
        dq = xr.open_dataset(q_file[0],decode_times=True,use_cftime=True)

        while (date <= min([end_time_in_file,last_date])):

            date_end = date + timedelta(days=nn) -timedelta(hours=6)
            date_end = min([date_end,end_time_in_file])

            ws,wd,rho,filename = read_and_interp(
                dt,dq,du,dv,slice(date,date_end))
            ds = fw.make_data_set(du,ws,wd,rho)
            ds.to_netcdf(filename,mode="w",engine="netcdf4",
                            unlimited_dims='time')
            print(filename," written to disk")
        
            date = date_end + timedelta(hours=6)
            print("Next date:",date)

        year = date.year
        print("Next date:",date.strftime("%Y-%m-%d_%H"),"year:",year)
        
if __name__ == "__main__":
    main()
