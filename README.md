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

- regridding_functions.py: 
- bias_correction.py: to obtain bias factor per model and component in original grid. Includes the two step bias correction. Requires an .nc with 0.3 grid and auxiliary regridding functions 
- 