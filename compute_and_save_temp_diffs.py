import xarray as xr
import os
import xesmf as xe


def load_period_medians(model, variant, period, base_dir="/groups/FutureWind/SFCRAD"):
    """
    Load all .nc files for a given model/variant/period,
    apply bias factor, filter to Europe, and compute overall & seasonal medians of tas.

    Returns dict with:
      - 'median_all': DataArray (lat, lon)
      - 'median_seasonal': DataArray (season, lat, lon)
    """
    path = os.path.join(base_dir, model, period, variant)
    pattern = os.path.join(path, "rsds_rsdsdiff_tas_*.nc")
    print(pattern)

    ds = xr.open_mfdataset(pattern, combine="by_coords", engine="netcdf4")
    tas = ds["tas"]

    # apply bias correction
    bias_folder = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/bias_factors/"
    bias_factor_file = os.path.join(bias_folder, f"temp_bias_factor_{model}.nc")
    print(bias_factor_file)
    bias_factor = xr.open_dataset(bias_factor_file)["bias_factor"]

    # select Europe region and apply factor
    tas = tas.sel(lon=slice(-12, 35), lat=slice(33, 64))
    tas = tas * bias_factor

    # compute medians
    median_all = tas.median(dim="time")
    print("yearly median calculated")
    median_seasonal = tas.groupby("time.season").median(dim="time")
    print("seasonal median calculated")

    return {"median_all": median_all, "median_seasonal": median_seasonal}


def compute_and_save_differences(models, variants, periods, output_dir,
                                 base_dir="/groups/FutureWind/SFCRAD"):
    """
    For each model, load period medians, compute absolute & relative differences,
    and save to NetCDF.

    - models: list of model names
    - variants: list of corresponding variant strings
    - periods: tuple of two period names, e.g. ('historical','ssp585')
    - output_dir: directory to save <model>_diffs.nc files
    """
    os.makedirs(output_dir, exist_ok=True)

    for model, variant in zip(models, variants):
        p0, p1 = periods
        # load medians without years filter
        med0 = load_period_medians(model, variant, p0, base_dir)
        med1 = load_period_medians(model, variant, p1, base_dir)

        # absolute differences
        abs_all = med1['median_all'] - med0['median_all']
        abs_seasonal = med1['median_seasonal'] - med0['median_seasonal']
        print("calc abs differences")

        # relative differences
        rel_all = abs_all / med0['median_all']
        rel_seasonal = abs_seasonal / med0['median_seasonal']
        print("calculated relative differences")

        # assemble and save
        ds_out = xr.Dataset({
            f"abs_diff_all_{p1}_vs_{p0}": abs_all,
            f"rel_diff_all_{p1}_vs_{p0}": rel_all,
            f"abs_diff_seasonal_{p1}_vs_{p0}": abs_seasonal,
            f"rel_diff_seasonal_{p1}_vs_{p0}": rel_seasonal,
        })

        ds_out.attrs.update({
            'model': model,
            'variant': variant,
            'periods_compared': f"{p1} minus {p0}"
        })

        out_path = os.path.join(output_dir, f"{model}_diffs.nc")
        ds_out.to_netcdf(out_path)
        print(f"Saved differences for {model} to {out_path}")


def regrid_saved_differences(models, periods,
                             input_dir="/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/temp_diff/",
                             target_grid=None,
                             output_dir="/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/temp_diff_1x1/",
                             reuse_weights=False):
    """
    Load precomputed difference NetCDFs, choose regridding per model,
    regrid all diff fields to target grid, and save.
    """
    os.makedirs(output_dir, exist_ok=True)
    conservative_models = ("CMCC-CM2-SR5", "CMCC-ESM2", "HadGEM3-GC31-MM")

    # cache regridders
    regridder_cache = {}

    target = xr.open_dataset(target_grid)

    for model in models:
        in_file = os.path.join(input_dir, f"{model}_diffs.nc")
        ds_in = xr.open_dataset(in_file)

        method = 'conservative' if model in conservative_models else 'bilinear'
        ref_var = next(v for v in ds_in.data_vars if 'abs_diff_all' in v)

        cache_key = f"{model}_{method}"
        if reuse_weights and cache_key in regridder_cache:
            regridder = regridder_cache[cache_key]
        else:
            regridder = xe.Regridder(ds_in[ref_var], target, method,
                                     reuse_weights=reuse_weights)
            regridder_cache[cache_key] = regridder

        regridded = {}
        for var in ds_in.data_vars:
            da = ds_in[var]
            if 'season' in da.dims:
                stacked = da.stack(z=('season',))
                out = regridder(stacked).unstack('z')
            else:
                out = regridder(da)
            regridded[var] = out

        ds_out = xr.Dataset(regridded, attrs=ds_in.attrs)
        ds_out.attrs['regrid_method'] = method

        out_file = os.path.join(output_dir, f"{model}_diffs_regridded.nc")
        ds_out.to_netcdf(out_file)
        print(f"Regridded differences for {model} ({method}) saved to {out_file}")


def main():
    models = ["ACCESS-CM2", "CanESM5", "CMCC-CM2-SR5", "CMCC-ESM2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0"]
    variants = ["r1i1p1f1", "r1i1p2f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f3", "r1i1p1f1"]
    periods = ('historical', 'ssp585')
    temp_diff_dir = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/temp_diff/"
    compute_and_save_differences(models, variants, periods, temp_diff_dir)

    grid_file = "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation-backup/europe_1x1.nc"
    regrid_saved_differences(models, periods,
                             input_dir=temp_diff_dir,
                             target_grid=grid_file)

if __name__ == "__main__":
    main()
