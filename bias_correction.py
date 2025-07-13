import xarray as xr
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO)
#import xesmf as xe
from regridding_functions import read_and_average_era5_4y
from regridding_functions import read_and_average_sarah
from regridding_functions import regrid
from regridding_functions import read_and_average_era5_marta
from regridding_functions import read_and_average_cmip
import os


def bias_factor_era5_sarah(var):
    #calculates the bias factor between era5 and sarah for the variable var, for now only 4 years data
    rsds_era5_mean_4y= read_and_average_era5_4y(var) #read and av the 4 years of era5 for bias correction with sarah
    rsds_sarah_mean= read_and_average_sarah(var) #same for sarah
    rsds_era5_mean_cut=rsds_era5_mean_4y.sel(x=slice(-12, 35), y=slice(33, 64.8)) #cut to the max latitude covered by sarah
    rsds_sarah_mean_cut=rsds_sarah_mean.sel(x=slice(-12, 35), y=slice(33, 64.8))
    regridder=regrid(rsds_era5_mean_cut, rsds_sarah_mean_cut, method='conservative')  #regrid era5 (0.25x0.25) to the sarah grid (0.3x0.3)
    rsds_era5_mean_interp_cut_4y=regridder(rsds_era5_mean_cut)
    denominator_era5_sarah= rsds_era5_mean_interp_cut_4y.values  # ERA5 dataset
    numerator_era5_sarah= rsds_sarah_mean_cut.values  # SARAH dataset
    # Ensure valid bias factor calculation
    mask_valid = (denominator_era5_sarah != 0) & (numerator_era5_sarah != 0) # Avoid division by zero and all values in sarah that have mean 0
    bias_factor_era5_sarah = np.where(mask_valid, numerator_era5_sarah / denominator_era5_sarah, np.nan)  # Replace invalid cases with NaN
    print(f"Shape of bias_factor_era5_sarah: {bias_factor_era5_sarah.shape}")
    return bias_factor_era5_sarah

#now the goal is that for each MODEL I calculate and print ALL bias factors
def bias_factor_era5_model(model, period, variant, direct_bias_factor_era5_sarah, diffuse_bias_factor_era5_sarah, output_dir):
    # Define the output file paths

    filepath_total = os.path.join(output_dir, f'total_bias_factor_{model}.nc')
    filepath_direct = os.path.join(output_dir, f'direct_bias_factor_{model}.nc')
    filepath_diffuse = os.path.join(output_dir, f'diffuse_bias_factor_{model}.nc')
    filepath_temp= os.path.join(output_dir, f'temp_bias_factor_{model}.nc')


    # Compute the required variables for total bias factor calculation
    rsds_era5_mean_BOC = read_and_average_era5_marta('influx_direct')  # mean of era5 historical period for each grid cell
    rsds_model_mean_BOC = read_and_average_cmip(f'SFCRAD/{model}/{period}/{variant}/', 'rsds')  # mean of model of historical period for each grid cell
    rsdsdiff_model_mean_BOC = read_and_average_cmip(f'SFCRAD/{model}/{period}/{variant}/', "rsdsdiff") 
    rsdsdiff_era5_mean_BOC = read_and_average_era5_marta("influx_diffuse") 
    temp_era5_mean_BOC = read_and_average_era5_marta("temperature")  # mean of era5 historical period for each grid cell
    temp_model_mean_BOC = read_and_average_cmip(f'SFCRAD//{model}/{period}/{variant}/', 'tas')  # mean of model of historical period for each grid cell

    rsds_era5_mean_BOC = rsds_era5_mean_BOC.sel(x=slice(-12, 35), y=slice(33, 64.8))
    rsds_model_mean_BOC = rsds_model_mean_BOC.sel(lon=slice(-12, 35), lat=slice(33, 64.8))
    rsdsdiff_model_mean_BOC = rsdsdiff_model_mean_BOC.sel(lon=slice(-12, 35), lat=slice(33, 64.8))
    rsdsdiff_era5_mean_BOC= rsdsdiff_era5_mean_BOC.sel(x=slice(-12, 35), y=slice(33, 64.8))
    temp_era5_mean_BOC = temp_era5_mean_BOC.sel(x=slice(-12, 35), y=slice(33, 64.8))
    temp_model_mean_BOC = temp_model_mean_BOC.sel(lon=slice(-12, 35), lat=slice(33, 64.8))

    ds_03 = xr.open_dataset('europe_03.nc')  # grid 0.3x0.3
    regridder_era5_direct = regrid(rsds_era5_mean_BOC, ds_03, method='conservative')  # regrid era5 to the 0.3x0.3º grid
    rsds_era5_03 = regridder_era5_direct(rsds_era5_mean_BOC)  # regridded historical mean from era5 to 0.3x0.3º grid
    regridder_era5_diffuse=regrid(rsdsdiff_era5_mean_BOC, ds_03, method='conservative')  # regrid era5 to the 0.3x0.3º grid
    rsdsdiff_era5_03 = regridder_era5_diffuse(rsdsdiff_era5_mean_BOC)


    rsds_era5_correct = rsds_era5_03.sel(lon=slice(-12, 35), lat=slice(33, 64.8)) * direct_bias_factor_era5_sarah  # apply bias factor to era5 rsds
    rsdsdiff_era5_correct = rsdsdiff_era5_03.sel(lon=slice(-12, 35), lat=slice(33, 64.8)) * diffuse_bias_factor_era5_sarah  # apply bias factor to era5 rsdsdiff
    regridder_era503_model_direct = regrid(rsds_era5_correct, rsds_model_mean_BOC, method='conservative')  # regrid corrected era5 to the model grid
    regridder_era503_model_diffuse = regrid(rsdsdiff_era5_correct, rsds_model_mean_BOC, method='conservative')  # regrid corrected era5 to the model grid
    rsds_era5_correct_model = regridder_era503_model_direct(rsds_era5_correct)  # regrid corrected era5 to the model grid
    rsdsdiff_era5_correct_model = regridder_era503_model_diffuse(rsdsdiff_era5_correct)  # regrid corrected era5 to the model grid
    regridder_era5_model_temp=regrid(temp_era5_mean_BOC, temp_model_mean_BOC, method='conservative')  # regrid era5 to the model grid
    temp_era5_correct_model=regridder_era5_model_temp(temp_era5_mean_BOC)  # regrid corrected era5 to the model grid

# Calculate and save total bias factor if missing
    if not os.path.exists(filepath_total):
        total_num_era5_model = rsds_era5_correct_model.values + rsdsdiff_era5_correct_model.values
        total_den_era5_model = rsds_model_mean_BOC.values
        mask_total = (total_num_era5_model != 0) & (total_den_era5_model != 0)
        total_bias_factor_era5_model = np.where(mask_total, total_num_era5_model / total_den_era5_model, np.nan)

        ds_total = xr.Dataset(
            {"bias_factor": (["lat", "lon"], total_bias_factor_era5_model)},
            coords={
                "lat": rsds_model_mean_BOC.lat,
                "lon": rsds_model_mean_BOC.lon,
            },
        )
        ds_total.to_netcdf(filepath_total)
        logging.info(f"Saved total bias factor to {filepath_total}")

    # Calculate and save direct bias factor if missing
    if not os.path.exists(filepath_direct):
        direct_num_era5_model = rsds_era5_correct_model.values
        direct_den_era5_model = rsds_model_mean_BOC.values - rsdsdiff_model_mean_BOC.values  # subtract diffuse from direct radiation
        mask_direct = (direct_num_era5_model != 0) & (direct_den_era5_model != 0)
        direct_bias_factor_era5_model = np.where(mask_direct, direct_num_era5_model / direct_den_era5_model, np.nan)

        ds_direct = xr.Dataset(
            {"bias_factor": (["lat", "lon"], direct_bias_factor_era5_model)},
            coords={
                "lat": rsds_model_mean_BOC.lat,
                "lon": rsds_model_mean_BOC.lon,
            },
        )
        ds_direct.to_netcdf(filepath_direct)
        logging.info(f"Saved direct bias factor to {filepath_direct}")

    # Calculate and save diffuse bias factor if missing
    if not os.path.exists(filepath_diffuse):
        diffuse_num_era5_model = rsdsdiff_era5_correct_model.values
        diffuse_den_era5_model = rsdsdiff_model_mean_BOC.values
        mask_diffuse = (diffuse_num_era5_model != 0) & (diffuse_den_era5_model != 0)
        diffuse_bias_factor_era5_model = np.where(mask_diffuse, diffuse_num_era5_model / diffuse_den_era5_model, np.nan)

        ds_diffuse = xr.Dataset(
            {"bias_factor": (["lat", "lon"], diffuse_bias_factor_era5_model)},
            coords={
                "lat": rsds_model_mean_BOC.lat,
                "lon": rsds_model_mean_BOC.lon,
            },
        )
        ds_diffuse.to_netcdf(filepath_diffuse)
        logging.info(f"Saved diffuse bias factor to {filepath_diffuse}")

    # Calculate and save temperature bias factor if missing
    if not os.path.exists(filepath_temp):
        temp_num_era5_model = temp_era5_correct_model.values
        temp_den_era5_model = temp_model_mean_BOC.values
        mask_temp = (temp_num_era5_model != 0) & (temp_den_era5_model != 0)
        temp_bias_factor_era5_model = np.where(mask_temp, temp_num_era5_model / temp_den_era5_model, np.nan)

        ds_temp = xr.Dataset(
            {"bias_factor": (["lat", "lon"], temp_bias_factor_era5_model)},
            coords={
                "lat": rsds_model_mean_BOC.lat,
                "lon": rsds_model_mean_BOC.lon,
            },
        )
        ds_temp.to_netcdf(filepath_temp)
        logging.info(f"Saved temperature bias factor to {filepath_temp}")
    
def main():
    models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]
    variants = ["r1i1p1f1", "r1i1p2f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f3", "r1i1p1f1"]
    period = "historical"
    output_dir = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/bias_factors/"
    os.makedirs(output_dir, exist_ok=True)
    direct_bias_factor_era5_sarah= bias_factor_era5_sarah("influx_direct")  # Calculate bias factor for direct radiation
    diffuse_bias_factor_era5_sarah= bias_factor_era5_sarah("influx_diffuse")  # Calculate bias factor for diffuse radiation
   
    for model, variant in zip(models, variants):
        bias_factor_era5_model(model, period, variant, direct_bias_factor_era5_sarah, diffuse_bias_factor_era5_sarah, output_dir)
        logging.info(f"Computed and saved bias factors for model {model}")


if __name__ == "__main__":
    main()