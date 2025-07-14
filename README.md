# Data
.nc files and .csv files are not shared in the Github repository, only in Sophia. 

- CMIP6 data after processing: "/groups/FutureWind/SFCRAD/" Europe selected (same area as for wind data), temperature interpolation, per-year split
- SARAH-3 data (4 years) for bias correction: "/groups/EXTREMES/SARAH-3/"
- Power CF: "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power/"
Contains 3h CF for each model and period, original grid and 1x1 in .nc. Also contains .csv for statistical results and differences BoC/EoC for solar CF (yearly, seasonal) and energy droughts. 
In the folder with regridded data, for each model and period there is: 3-h solar CF, seasonal CF and yearly .nc. Also one .nc fiile with solar drought events per period, a summary file (per country) and a .csv (location, year, season, start month, duration)
- temp_diff: "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/temp_diff/"
Contains temperature differences (abs, rel) per model in original grid resolution
- temp_diff: "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/temp_diff_1x1/"
Temperature differences regridded to 1x1
- bias factors: "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/bias_factors/"
Contains 
- power_Era5_h: "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power_Era5_h/"
Hourly solar CF from ERA5
- power_Era5_h_1x1: "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power_Era5_h_1x1/"
ERA5 hourly power regridded to 1x1
- power_Era5_yearly: "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power_Era5_yearly/"
ERA5 yearly solar power in 0.25 grid
- power_notemp: "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power_notemp/"
Solar CF from CMIP6 removing Temp effect, regridded to 1x1
- albedo
mean ERA5 albedo for historical period regridded to each model's original grid


# Codes
All codes used to generate the results, uploaded to Github repository

- regridding_functions.py: auxiliary functions to average ERA5, SARAH-3 and CMIP6 variables for bias correction, regridding function (based on Andrea's work), returns regridder
- bias_correction.py: to obtain bias factor per model and component in original grid. Includes the two step bias correction. Requires an .nc with 0.3 grid and auxiliary regridding functions 


- pv_functions.py: contains SolarPosition (change in time index handling), get_oritnation, make_latitude_optimal, surface_orientation, diffuseHorizontalIrrad, TiltedDiffuseIrrad, TiltedDirectIrrad, ToltedGroundIrrad (takes albedo regridded to each model's grid), TiltedIrrad (main change: redefining influx_toa, influx direct and influx diffuse). 
- power_calculation.py: code to transform from the CMIP6 yearly files to solar CF. Handles the calendar standarization and includes the multiplication by the bias factors. Uses the pv functions from atlite modified as necessary in pv_functions.py. after CF has been calculated, it is multiplied x3 to represent all power in the period. Results for every 3h and yearly aggregation are saved as .nc
- aggregated_CF_values.py: perform annual or seasonal aggregation of solar CF (with or without temp effect)
- power_era5_yearly: obtain yearly solar CF from ERA5
- power_era5_hourly.py: obtain solar CF from hourly ERA5 cutouts applying atlite functions (necessary for energy drought evaluation)




- compute_and_save_temp_diffs.py: compute absolute and relative differences in temperature between BoC and EoC


- plots.py: auxiliary plotting functions (ignore)
- models_to_1x1.py, Era5_to_1x1.py: to regrid all solar CF time series from model grid and ERA5 grid to 1x1 grid. Needs a .nc file with 1x1 grid. 