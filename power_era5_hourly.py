from era5_power_yearly import collect_files_era5
import re, os
import xarray as xr
from regridding_functions import regrid
from bias_correction import bias_factor_era5_sarah

from atlite.convert import convert_pv
from atlite.pv.orientation import get_orientation



    
def main():
    base_path = "/groups/EXTREMES/cutouts/"
    saving_path = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/power_Era5_h/"
    # Collect ERA5 files
    era5_BOC_files = collect_files_era5(base_path)
    print(era5_BOC_files)
    ds_sarah="/groups/EXTREMES/SARAH-3/europe-1996-sarah3-era5.nc"
    ds_sarah=xr.open_dataset(ds_sarah)

    vars=["influx_direct","influx_diffuse"]
    panel={
    "model": "huld",  # Model type
    "name": "CSi",  # Panel name
    "source": "Huld 2010",  # Source of the model

    # Used for calculating capacity per m2
    "efficiency": 0.1,  # Efficiency of the panel

    # Panel temperature coefficients
    "c_temp_amb": 1,  # Panel temperature coefficient of ambient temperature
    "c_temp_irrad": 0.035,  # Panel temperature coefficient of irradiance (K / (W/m2))

    # Reference conditions
    "r_tamb": 293,  # Reference ambient temperature (20 degC in Kelvin)
    "r_tmod": 298,  # Reference module temperature (25 degC in Kelvin)
    "r_irradiance": 1000,  # Reference irradiance (W/m^2)

    # Fitting parameters
    "k_1": -0.017162,
    "k_2": -0.040289,
    "k_3": -0.004681,
    "k_4": 0.000148,
    "k_5": 0.000169,
    "k_6": 0.000005,

    # Inverter efficiency
    "inverter_efficiency": 0.9
}
    orientation1='latitude_optimal'
    for file in era5_BOC_files:
        try:
            ds=xr.open_dataset(file)
            regridder=regrid(ds, ds_sarah, method='conservative')  #regrid era5 (0.25x0.25) to the sarah grid (0.3x0.3)
            ds=regridder(ds)
            for var in vars:
                # Read in cutout instead of creating it
                bias_factor_era5_sarah_var=bias_factor_era5_sarah(var)
                ds[var]=bias_factor_era5_sarah_var*ds[var].sel(lon=slice(-12, 35), lat=slice(33, 64.8))

            orientation = get_orientation(orientation1)

            power_era5=convert_pv(ds,
                panel=panel, 
                orientation=orientation,
                tracking=None,
                clearsky_model=None
            )

            # Extract the year from the file name
            match = re.search(r"(\d{4})", os.path.basename(file))
            if match:
                year = match.group(1)
                # Construct the output file name
                output_file = os.path.join(saving_path, f"hourly_power_era5_{year}.nc")
            else:
                raise ValueError(f"Could not extract year from file name: {file}")

            # Save the power data to a NetCDF file
            power_era5.to_netcdf(output_file)
            print(f"Saved power data to {output_file}")

        except Exception as e:
            print(f"Failed to process file {file}: {e}")
if __name__ == "__main__":
    main()