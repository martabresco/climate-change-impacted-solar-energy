# Climate-Change-Impacted Solar Energy Generation
This project analyzes how solar energy generation across Europe is affected by climate change, using satellite data (SARAH-3), reanalysis (ERA5) and climate model data (CMIP6). It includes bias correction, capacity factor (CF) calculations, temperature effects, and energy drought statistics.

> **Note:** Data files (`.nc`, `.csv`) are not included in the GitHub repository. They are stored in Sophia ("/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/")

---

## Table of Contents

1. [Data](#data)
   - [CMIP6 data after processing](#cmip6-data-after-processing)
   - [SARAH-3 data (bias correction)](#sarah-3-data-bias-correction)
   - [Power CF](#power-cf)
   - [Temperature differences](#temperature-differences)
   - [Bias factors](#bias-factors)
   - [ERA5 data](#era5-data)
   - [Power without temperature effect](#power-without-temperature-effect)
   - [Albedo](#albedo)
2. [Code Overview](#code-overview)
   - [Core Processing](#core-processing)
   - [Bias Correction](#bias-correction)
   - [PV and Power Calculations](#pv-and-power-calculations)
   - [Aggregation and Statistics](#aggregation-and-statistics)
   - [Energy Drought Analysis](#energy-drought-analysis)
   - [ERA5 vs CMIP6 Comparison](#era5-vs-cmip6-comparison)
   - [Regridding Utilities](#regridding-utilities)
---

## Data

- **CMIP6 data after processing**:  
  `/groups/FutureWind/SFCRAD/`  
  Europe selected (same area as for wind data), temperature interpolation, per-year split

- **SARAH-3 data (4 years) for bias correction**:  
  `/groups/EXTREMES/SARAH-3/`

- **Power CF**:  
  `/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power/`  
  Contains 3h CF for each model and period, original grid and 1x1 in `.nc`. Also contains `.csv` files for statistical results and differences BoC/EoC for solar CF (yearly, seasonal) and energy droughts.  
  In the folder with regridded data, for each model and period there is: 3-hour solar CF, seasonal CF, and yearly `.nc`. Also one `.nc` file with solar drought events per period, a summary file (per country), and a `.csv` with event info (location, year, season, start month, duration).

- **temp_diff**:  
  `/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/temp_diff/`  
  Contains temperature differences (absolute, relative) per model in original grid resolution

- **temp_diff (regridded)**:  
  `/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/temp_diff_1x1/`  
  Temperature differences regridded to 1x1

- **bias factors**:  
  `/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/bias_factors/`  
  Contains per-model, per-component bias correction factors

- **power_Era5_h**:  
  `/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power_Era5_h/`  
  Hourly solar CF from ERA5

- **power_Era5_h_1x1**:  
  `/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power_Era5_h_1x1/`  
  ERA5 hourly power regridded to 1x1

- **power_Era5_yearly**:  
  `/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power_Era5_yearly/`  
  ERA5 yearly solar power on 0.25° grid

- **power_notemp**:  
  `/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/power_notemp/`  
  Solar CF from CMIP6 **removing temperature effect**, regridded to 1x1

- **albedo**:  
  Mean ERA5 albedo for historical period regridded to each model's original grid
---
## Codes

All codes used to generate the results, uploaded to Github repository

- **regridding_functions.py**: auxiliary functions to average ERA5, SARAH-3 and CMIP6 variables for bias correction, regridding function (based on Andrea's work), returns regridder  
- **bias_correction.py**: to obtain bias factor per model and component in original grid. Includes the two step bias correction. Requires a `.nc` with 0.3 grid and auxiliary regridding functions  

- **pv_functions.py**: contains `SolarPosition` (change in time index handling), `get_orientation`, `make_latitude_optimal`, `surface_orientation`, `diffuseHorizontalIrrad`, `TiltedDiffuseIrrad`, `TiltedDirectIrrad`, `TiltedGroundIrrad` (takes albedo regridded to each model's grid), `TiltedIrrad` (main change: redefining `influx_toa`, `influx direct` and `influx diffuse`)  
- **power_calculation.py**: code to transform from the CMIP6 yearly files to solar CF. Handles the calendar standarization and includes the multiplication by the bias factors. Uses the pv functions from atlite modified as necessary in `pv_functions.py`. After CF has been calculated, it is multiplied ×3 to represent all power in the period. Results for every 3h and yearly aggregation are saved as `.nc`  
- **aggregated_CF_values.py**: perform annual or seasonal aggregation of solar CF (with or without temp effect)  
- **power_era5_yearly.py**: obtain yearly solar CF from ERA5  
- **power_era5_hourly.py**: obtain solar CF from hourly ERA5 cutouts applying atlite functions (necessary for energy drought evaluation)  

- **compute_and_save_temp_diffs.py**: compute absolute and relative differences in temperature between BoC and EoC  

- **CF_statistics_yearly.ipynb**: Includes main results for changes in yearly solar CF:  
    - Per-model relative difference BoC/EoC  
    - Median relative difference across models BoC/EoC  
    - Sign and magnitude agreement  
    - Per country-aggregation  
    - Shapiro and t-test for each country and model. Results stored in two `.csv`:  
        - `solar_country_model_stats.csv` stores country, model, median yearly power for BoC and EoC, relative change, result of shapiro for BoC and EoC, result of t-test and significance (yes/no)  
        - `solar_stats_per_country.csv` contains summary per country. How many models find results significant and what is the median relative change  
    - KDE curves + histogram plots, Q-Q plots  
    - Summary plots of statistical significance and median relative change  

- **CF_statistics_seasonal.ipynb**: main results for changes in seasonal solar CF  
    - Results stored in two `.csv`:  
        - `solar_model_stats_per_season.csv` contains model, country, season, shapiro and t-test results, median BoC and EoC, relative difference  
        - `solar_stats_seasonal_summary.csv` returns results summarized at country level  

- **energy_droughts_frequency_CMIP6.ipynb**:  
    - Calculating daily solar power for each year for CMIP6 models and ERA5 and daily mean per week (`daily_mean_per_week.nc`) for both CMIP6 (per model, per period) and ERA5  
    - Evaluating solar drought events per model and period, recorded in `solar_drought_events_{period}.csv`. Includes for each event: model, period, year, season, lat and lon, start month, duration. This file is later on used for significance testing  
    - Evaluating solar drought events for ERA5, recorded in `solar_drought_events.csv` (ERA5 folder)  
    - Statistical testing of difference in frequency of yearly drought events between BoC/EoC. Results stored in `country_drought_stats_BOC_vs_EOC.csv`  
    - Statistical testing of difference in frequency of seasonal drought events between BoC/EoC. Results stored in `country_seasonal_drought_tests.csv`  

- **energy_drought_duration_CMIP6.ipynb**:  
    - Uses drought events stored in `solar_drought_events_{period}.csv` to evaluate P50, P75 and P90 durations  
    - Statistical tests applied to P50, P75 and P90 and saved in `.csv`  

- **energy_droughts_ERA5_CMIP6_comp.ipynb**:  
    - Evaluates differences between ERA5 and CMIP6 in number of drought days per year (stored in `country_era5_vs_boc_drought_days.csv`), plots annual drought days ERA5 and BoC. Uses the solar drought events information saved in `solar_drought_events.csv`, in `energy_droughts_frequency_CMIP6`  
    - Yearly differences in P90 duration (`country_era5_vs_boc_p90_tests.csv`)  
    - Seasonal analysis: `country_seasonal_era5_vs_boc_days.csv`  
    - Significance and magnitude per year and per season for each country  

- **change_in_solarCF_and_drought_duration.ipynb**:  
    - Takes statistical results from changes in seasonal and yearly solar CF and changes in seasonal and yearly drought frequency and evaluates how many models agree on both  

- **plots.py**: auxiliary plotting functions (ignore)  
- **models_to_1x1.py**, **Era5_to_1x1.py**: to regrid all solar CF time series from model grid and ERA5 grid to 1x1 grid. Needs a `.nc` file with 1x1 grid.  

