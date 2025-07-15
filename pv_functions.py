from __future__ import annotations

import sys
import logging

import numpy as np
import pandas as pd
import xarray as xr

from dask.array import (
    absolute, arccos, arcsin, arctan, arctan2, cos, maximum, mod,
    radians, sin, sqrt
)


from regridding_functions import regrid

# Set up logging
logging.basicConfig(level=logging.INFO)





def SolarPosition(ds, time_shift="0H"):
    """
    Calculate solar position (altitude and azimuth) based on the dataset's time and coordinates.

    Parameters:
        ds (xarray.Dataset): The dataset containing time, latitude, and longitude.
        time_shift (str): Time shift to apply (e.g., "+30min").

    Returns:
        xarray.Dataset: A dataset containing solar altitude and azimuth.
    """
    """
    Removed this when I had the error with the C-models. But this worked for the H models
    time_shift = pd.to_timedelta(time_shift)

    # Handle CFTimeIndex or DatetimeIndex
    t = ds.indexes["time"]
    if isinstance(ds.indexes["time"], xr.cftimeindex.CFTimeIndex):
        t = ds.indexes["time"].to_datetimeindex()
    else:
        t = pd.to_datetime(ds.indexes["time"])
    """
    time_shift = pd.to_timedelta(time_shift)

    # Always safely convert time
    t = ds.indexes["time"]
    if hasattr(t, "to_datetimeindex"):
        t = t.to_datetimeindex()
    else:
        t = pd.to_datetime(t)
  # Convert CFTimeIndex to DatetimeIndex
    t = t + time_shift
    

    # Convert time to Julian date
    julian_date = t.to_julian_date()
    n = xr.DataArray(julian_date, coords=ds["time"].coords) - 2451545.0

    # Extract hour and minute
    hour = (ds["time"] + time_shift).dt.hour
    minute = (ds["time"] + time_shift).dt.minute

    # Operations make new DataArray eager; reconvert to lazy dask arrays
    chunks = ds.chunksizes.get("time", "auto")
    if isinstance(chunks, tuple):
        chunks = chunks[0]
    n = n.chunk(chunks)
    hour = hour.chunk(chunks)
    minute = minute.chunk(chunks)

    # Solar position calculations
    L = 280.460 + 0.9856474 * n  # mean longitude (deg)
    g = radians(357.528 + 0.9856003 * n)  # mean anomaly (rad)
    l = radians(L + 1.915 * sin(g) + 0.020 * sin(2 * g))  # ecliptic long. (rad)
    ep = radians(23.439 - 4e-7 * n)  # obliquity of the ecliptic (rad)

    ra = arctan2(cos(ep) * sin(l), cos(l))  # right ascension (rad)
    lmst = (6.697375 + (hour + minute / 60.0) + 0.0657098242 * n) * 15.0 + ds["lon"]
    h = (radians(lmst) - ra + np.pi) % (2 * np.pi) - np.pi  # hour angle (rad)

    dec = arcsin(sin(ep) * sin(l))  # declination (rad)

    # Altitude and azimuth calculations
    lat = radians(ds["lat"])
    alt = arcsin(
        (sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(h)).clip(min=-1.0, max=1.0)
    ).rename("altitude")
    alt.attrs["time shift"] = f"{time_shift}"
    alt.attrs["units"] = "rad"

    az = arccos(
        ((sin(dec) * cos(lat) - cos(dec) * sin(lat) * cos(h)) / cos(alt)).clip(
            min=-1.0, max=1.0
        )
    )
    az = az.where(h <= 0, 2 * np.pi - az).rename("azimuth")
    az.attrs["time shift"] = f"{time_shift}"
    az.attrs["units"] = "rad"

    vars = {da.name: da for da in [alt, az]}
    solar_position = xr.Dataset(vars)

    return solar_position

def get_orientation(name, **params):
    """
    Definitions:
    -`slope` is the angle between ground and panel.
    -`azimuth` is the clockwise angle from North.
        i.e. azimuth = 180 faces exactly South
    """
    if isinstance(name, dict):
        params = name
        name = params.pop("name", "constant")
    return getattr(sys.modules[__name__], f"make_{name}")(**params)


def make_latitude_optimal():
    """
    Returns an optimal tilt angle for the given ``lat``, assuming that the
    panel is facing towards the equator, using a simple method from [1].

    This method only works for latitudes between 0 and 50. For higher
    latitudes, a static 40 degree angle is returned.

    These results should be used with caution, but there is some
    evidence that tilt angle may not be that important [2].

    Function and documentation has been adapted from gsee [3].

    [1] http://www.solarpaneltilt.com/#fixed
    [2] http://dx.doi.org/10.1016/j.solener.2010.12.014
    [3] https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py

    Parameters
    ----------
    lat : float
        Latitude in degrees.

    """

    def latitude_optimal(lon, lat, solar_position):
        slope = np.empty_like(lat.values)

        below_25 = np.abs(lat.values) <= np.radians(25)
        below_50 = np.abs(lat.values) <= np.radians(50)

        slope[below_25] = 0.87 * np.abs(lat.values[below_25])
        slope[~below_25 & below_50] = 0.76 * np.abs(
            lat.values[~below_25 & below_50]
        ) + np.radians(0.31)
        slope[~below_50] = np.radians(40.0)

        # South orientation for panels on northern hemisphere and vice versa
        azimuth = np.where(lat.values < 0, 0, np.pi)
        return dict(
            slope=xr.DataArray(slope, coords=lat.coords),
            azimuth=xr.DataArray(azimuth, coords=lat.coords),
        )

    return latitude_optimal


def make_constant(slope, azimuth):
    slope = radians(slope)
    azimuth = radians(azimuth)

    def constant(lon, lat, solar_position):
        return dict(slope=slope, azimuth=azimuth)

    return constant


def make_latitude(azimuth=180):
    azimuth = radians(azimuth)

    def latitude(lon, lat, solar_position):
        return dict(slope=lat, azimuth=azimuth)

    return latitude


def SurfaceOrientation(ds, solar_position, orientation, tracking=None):
    """
    Compute cos(incidence) for slope and panel azimuth.

    References
    ----------
    [1] Sproul, A. B., Derivation of the solar geometric relationships using
    vector analysis, Renewable Energy, 32(7), 1187–1205 (2007).
    [2] Marion, William F., and Aron P. Dobos. Rotation angle for the optimum
    tracking of one-axis trackers. No. NREL/TP-6A20-58891. National Renewable
    Energy Lab.(NREL), Golden, CO (United States), 2013.

    """
    lon = radians(ds["lon"])
    lat = radians(ds["lat"])

    orientation = orientation(lon, lat, solar_position)
    surface_slope = orientation["slope"]
    surface_azimuth = orientation["azimuth"]

    sun_altitude = solar_position["altitude"]
    sun_azimuth = solar_position["azimuth"]

    if tracking is None:
        cosincidence = sin(surface_slope) * cos(sun_altitude) * cos(
            surface_azimuth - sun_azimuth
        ) + cos(surface_slope) * sin(sun_altitude)

    elif tracking == "horizontal":  # horizontal tracking with horizontal axis
        axis_azimuth = orientation[
            "azimuth"
        ]  # here orientation['azimuth'] refers to the azimuth of the tracker axis.
        rotation = arctan(
            (cos(sun_altitude) / sin(sun_altitude)) * sin(sun_azimuth - axis_azimuth)
        )
        surface_slope = abs(rotation)
        surface_azimuth = axis_azimuth + arcsin(
            sin(rotation / sin(surface_slope))
        )  # the 2nd part yields +/-1 and determines if the panel is facing east or west
        cosincidence = cos(surface_slope) * sin(sun_altitude) + sin(
            surface_slope
        ) * cos(sun_altitude) * cos(sun_azimuth - surface_azimuth)

    elif tracking == "tilted_horizontal":  # horizontal tracking with tilted axis'
        axis_tilt = orientation[
            "slope"
        ]  # here orientation['slope'] refers to the tilt of the tracker axis.

        rotation = arctan(
            (cos(sun_altitude) * sin(sun_azimuth - surface_azimuth))
            / (
                cos(sun_altitude) * cos(sun_azimuth - surface_azimuth) * sin(axis_tilt)
                + sin(sun_altitude) * cos(axis_tilt)
            )
        )

        surface_slope = arccos(cos(rotation) * cos(axis_tilt))

        azimuth_difference = sun_azimuth - surface_azimuth
        azimuth_difference = np.where(
            azimuth_difference > np.pi, 2 * np.pi - azimuth_difference, azimuth_difference
        )
        azimuth_difference = np.where(
            azimuth_difference < -np.pi, 2 * np.pi + azimuth_difference, azimuth_difference
        )
        rotation = np.where(
            np.logical_and(rotation < 0, azimuth_difference > 0),
            rotation + np.pi,
            rotation,
        )
        rotation = np.where(
            np.logical_and(rotation > 0, azimuth_difference < 0),
            rotation - np.pi,
            rotation,
        )

        cosincidence = cos(rotation) * (
            sin(axis_tilt) * cos(sun_altitude) * cos(sun_azimuth - surface_azimuth)
            + cos(axis_tilt) * sin(sun_altitude)
        ) + sin(rotation) * cos(sun_altitude) * sin(sun_azimuth - surface_azimuth)

    elif tracking == "vertical":  # vertical tracking, surface azimuth = sun_azimuth
        cosincidence = sin(surface_slope) * cos(sun_altitude) + cos(
            surface_slope
        ) * sin(sun_altitude)
    elif tracking == "dual":  # both vertical and horizontal tracking
        cosincidence = np.float64(1.0)
    else:
        assert False, (
            "Values describing tracking system must be None for no tracking,"
            + "'horizontal' for 1-axis horizontal tracking,"
            + "tilted_horizontal' for 1-axis horizontal tracking of tilted panle,"
            + "vertical' for 1-axis vertical tracking, or 'dual' for 2-axis tracking"
        )

    # fixup incidence angle: if the panel is badly oriented and the sun shines
    # on the back of the panel (incidence angle > 90degree), the irradiation
    # would be negative instead of 0; this is prevented here.
    cosincidence = cosincidence.clip(min=0)

    return xr.Dataset(
        {
            "cosincidence": cosincidence,
            "slope": surface_slope,
            "azimuth": surface_azimuth,
        }
    )


import logging

import numpy as np
from dask.array import cos, fmax, fmin, radians, sin, sqrt

logger = logging.getLogger(__name__)


def DiffuseHorizontalIrrad(ds, solar_position, clearsky_model, influx):
    # Clearsky model from Reindl 1990 to split downward radiation into direct
    # and diffuse contributions. Should switch to more up-to-date model, f.ex.
    # Ridley et al. (2010) http://dx.doi.org/10.1016/j.renene.2009.07.018 ,
    # Lauret et al. (2013):http://dx.doi.org/10.1016/j.renene.2012.01.049

    sinaltitude = sin(solar_position["altitude"])
    influx_toa = ds["influx_toa"]

    if clearsky_model is None:
        clearsky_model = (
            "enhanced" if "temperature" in ds and "humidity" in ds else "simple"
        )

    # Reindl 1990 clearsky model

    k = influx / influx_toa  # clearsky index
    # k.values[k.values > 1.0] = 1.0
    # k = k.rename('clearsky index')

    if clearsky_model == "simple":
        # Simple Reindl model without ambient air temperature and
        # relative humidity
        fraction = (
            ((k > 0.0) & (k <= 0.3))
            * fmin(1.0, 1.020 - 0.254 * k + 0.0123 * sinaltitude)
            + ((k > 0.3) & (k < 0.78))
            * fmin(0.97, fmax(0.1, 1.400 - 1.749 * k + 0.177 * sinaltitude))
            + (k >= 0.78) * fmax(0.1, 0.486 * k - 0.182 * sinaltitude)
        )
    elif clearsky_model == "enhanced":
        # Enhanced Reindl model with ambient air temperature and relative
        # humidity
        T = ds["tas"]
        rh = ds["humidity"]

        fraction = (
            ((k > 0.0) & (k <= 0.3))
            * fmin(
                1.0,
                1.000 - 0.232 * k + 0.0239 * sinaltitude - 0.000682 * T + 0.0195 * rh,
            )
            + ((k > 0.3) & (k < 0.78))
            * fmin(
                0.97,
                fmax(
                    0.1,
                    1.329 - 1.716 * k + 0.267 * sinaltitude - 0.00357 * T + 0.106 * rh,
                ),
            )
            + (k >= 0.78)
            * fmax(0.1, 0.426 * k - 0.256 * sinaltitude + 0.00349 * T + 0.0734 * rh)
        )
    else:
        raise KeyError("`clearsky model` must be chosen from 'simple' and 'enhanced'")

    # Set diffuse fraction to one when the sun isn't up
    # fraction = fraction.where(sinaltitude >= sin(radians(threshold))).fillna(1.0)
    # fraction = fraction.rename('fraction index')

    return (influx * fraction).rename("diffuse horizontal")


def TiltedDiffuseIrrad(ds, solar_position, surface_orientation, direct, diffuse):
    # Hay-Davies Model

    sinaltitude = sin(solar_position["altitude"])
    influx_toa = ds["influx_toa"]

    cosincidence = surface_orientation["cosincidence"]
    surface_slope = surface_orientation["slope"]

    influx = direct + diffuse

    with np.errstate(divide="ignore", invalid="ignore"):
        # brightening factor
        f = sqrt(direct / influx).fillna(0.0)

        # anisotropy factor
        A = direct / influx_toa

    # geometric factor
    R_b = cosincidence / sinaltitude

    diffuse_t = (
        (1.0 - A)
        * ((1 + cos(surface_slope)) / 2.0)
        * (1.0 + f * sin(surface_slope / 2.0) ** 3)
        + A * R_b
    ) * diffuse

    # fixup: clip all negative values (unclear why it gets negative)
    # note: REatlas does not do the fixup
    if logger.isEnabledFor(logging.WARNING):
        if ((diffuse_t < 0.0) & (sinaltitude > sin(radians(1.0)))).any():
            logger.warning(
                "diffuse_t exhibits negative values above altitude threshold."
            )

    with np.errstate(invalid="ignore"):
        diffuse_t = diffuse_t.clip(min=0).fillna(0)

    return diffuse_t.rename("diffuse tilted")


def TiltedDirectIrrad(solar_position, surface_orientation, direct):
    sinaltitude = sin(solar_position["altitude"])
    cosincidence = surface_orientation["cosincidence"]

    # geometric factor
    R_b = cosincidence / sinaltitude

    return (R_b * direct).rename("direct tilted")


def _albedo(mean_albedo):

    if mean_albedo is not None:
        albedo = mean_albedo
    else:
        raise ValueError("An external albedo DataArray must be provided.")

    return albedo


def TiltedGroundIrrad(ds, solar_position, surface_orientation, influx, mean_albedo):
    surface_slope = surface_orientation["slope"]
    ground_t = influx * _albedo(mean_albedo) * (1.0 - cos(surface_slope)) / 2.0
    return ground_t.rename("ground tilted")


def TiltedIrradiation(
    ds,
    mean_albedo,
    solar_position,
    surface_orientation,
    trigon_model,
    clearsky_model,
    bf_direct, 
    bf_diffuse,  
    bf_total,
    tracking=0,
    altitude_threshold=1.0,
    irradiation="total",
):
    """
    Calculate the irradiation on a tilted surface.

    Parameters
    ----------
    ds : xarray.Dataset
        Cutout data used for calculating the irradiation on a tilted surface.
    solar_position : xarray.Dataset
        Solar position calculated using atlite.pv.SolarPosition,
        containing a solar 'altitude' (in rad, 0 to pi/2) for the 'ds' dataset.
    surface_orientation : xarray.Dataset
        Surface orientation calculated using atlite.orientation.SurfaceOrientation.
    trigon_model : str
        Type of trigonometry model. Defaults to 'simple'if used via `convert_irradiation`.
    clearsky_model : str or None
        Either the 'simple' or the 'enhanced' Reindl clearsky
        model. The default choice of None will choose dependending on
        data availability, since the 'enhanced' model also
        incorporates ambient air temperature and relative humidity.
        NOTE: this option is only used if the used climate dataset
        doesn't provide direct and diffuse irradiation separately!
    altitude_threshold : float
        Threshold for solar altitude in degrees. Values in range (0, altitude_threshold]
        will be set to zero. Default value equals 1.0 degrees.
    irradiation : str
        The irradiation quantity to be returned. Defaults to "total" for total
        combined irradiation. Other options include "direct" for direct irradiation,
        "diffuse" for diffuse irradation, and "ground" for irradiation reflected
        by the ground via albedo. NOTE: "ground" irradiation is not calculated
        by all `trigon_model` options in the `convert_irradiation` method,
        so use with caution!

    Returns
    -------
    result : xarray.DataArray
        The desired irradiation quantity on the tilted surface.

    """
    influx_toa = ds["rsds"]*bf_total #Bias corrected total irradiance

    def clip(influx, influx_max):
        # use .data in clip due to dask-xarray incompatibilities
        return influx.clip(min=0, max=influx_max.transpose(*influx.dims).data)

    #if "influx" in ds:
    #    influx = clip(ds["influx"], influx_toa)
    #    diffuse = DiffuseHorizontalIrrad(ds, solar_position, clearsky_model, influx)
    #    direct = influx - diffuse

    if "rsds" in ds and "rsdsdiff" in ds:
        direct = clip((ds["rsds"]-ds['rsdsdiff'])*bf_direct, influx_toa - ds["rsdsdiff"]*bf_diffuse) #bias corrected direct irradiance
        diffuse = clip(ds["rsdsdiff"]*bf_diffuse, influx_toa - (ds["rsds"]-ds['rsdsdiff'])*bf_direct) #bias corrected diffuse irradiance
    else:
        raise AssertionError(
            "Need either influx or influx_direct and influx_diffuse in the "
            "dataset. Check your cutout and dataset module."
        )
    if trigon_model == "simple": #function only modified to use the simple model
        k = surface_orientation["cosincidence"] / sin(solar_position["altitude"])
        if tracking != "dual":
            cos_surface_slope = cos(surface_orientation["slope"])
        elif tracking == "dual":
            cos_surface_slope = sin(solar_position["altitude"])

        influx = direct + diffuse
        direct_t = k * direct
        diffuse_t = (1.0 + cos_surface_slope) / 2.0 * diffuse
        ground_t = _albedo(mean_albedo) * influx * ((1.0 - cos_surface_slope) / 2.0)

        total_t = direct_t.fillna(0.0) + diffuse_t.fillna(0.0) + ground_t.fillna(0.0)
    else:
        diffuse_t = TiltedDiffuseIrrad(
            ds, solar_position, surface_orientation, direct, diffuse
        )
        direct_t = TiltedDirectIrrad(solar_position, surface_orientation, direct)
        ground_t = TiltedGroundIrrad(
            ds, solar_position, surface_orientation, direct + diffuse
        )

        total_t = direct_t + diffuse_t + ground_t

    if irradiation == "total":
        result = total_t.rename("total tilted")
    elif irradiation == "direct":
        result = direct_t.rename("direct tilted")
    elif irradiation == "diffuse":
        result = diffuse_t.rename("diffuse tilted")
    elif irradiation == "ground":
        result = ground_t.rename("ground tilted")

    # The solar_position algorithms have a high error for small solar altitude
    # values, leading to big overall errors from the 1/sinaltitude factor.
    # => Suppress irradiation below solar altitudes of 1 deg.

    cap_alt = solar_position["altitude"] < radians(altitude_threshold)
    result = result.where(~(cap_alt | (direct + diffuse <= 0.01)), 0)
    result.attrs["units"] = "W m**-2"

    return result

def _power_huld(irradiance, t_amb, pc):
    """
    AC power per capacity predicted by Huld model, based on W/m2 irradiance.

    Maximum power point tracking is assumed.

    [1] Huld, T. et al., 2010. Mapping the performance of PV modules,
    effects of module type and data averaging. Solar Energy, 84(2),
    p.324-338. DOI: 10.1016/j.solener.2009.12.002
    """
    # normalized module temperature
    T_ = (pc["c_temp_amb"] * t_amb + pc["c_temp_irrad"] * irradiance) - pc["r_tmod"]

    # normalized irradiance
    G_ = irradiance / pc["r_irradiance"]

    log_G_ = np.log(G_.where(G_ > 0))
    # NB: np.log without base implies base e or ln
    eff = (
        1
        + pc["k_1"] * log_G_
        + pc["k_2"] * (log_G_) ** 2
        + T_ * (pc["k_3"] + pc["k_4"] * log_G_ + pc["k_5"] * log_G_**2)
        + pc["k_6"] * (T_**2)
    )

    eff = eff.fillna(0.0).clip(min=0)

    da = G_ * eff * pc.get("inverter_efficiency", 1.0)
    da.attrs["units"] = "kWh/kWp"
    da = da.rename("specific generation")

    return da


def _power_bofinger(irradiance, t_amb, pc):
    """
    AC power per capacity predicted by bofinger model, based on W/m2
    irradiance.

    Maximum power point tracking is assumed.

    [2] Hans Beyer, Gerd Heilscher and Stefan Bofinger, 2004. A robust
    model for the MPP performance of different types of PV-modules
    applied for the performance check of grid connected systems.
    """
    fraction = (pc["NOCT"] - pc["Tamb"]) / pc["Intc"]

    eta_ref = (
        pc["A"]
        + pc["B"] * irradiance
        + pc["C"] * np.log(irradiance.where(irradiance != 0))
    )
    eta = (
        eta_ref
        * (1.0 + pc["D"] * (fraction * irradiance + (t_amb - pc["Tstd"])))
        / (1.0 + pc["D"] * fraction / pc["ta"] * eta_ref * irradiance)
    ).fillna(0)

    capacity = (pc["A"] + pc["B"] * 1000.0 + pc["C"] * np.log(1000.0)) * 1e3
    power = irradiance * eta * (pc.get("inverter_efficiency", 1.0) / capacity)
    power = power.where(irradiance >= pc["threshold"], 0)
    return power.rename("AC power")


def SolarPanelModel(ds, irradiance, pc,bf_tas):
    model = pc.get("model", "huld")

    if model == "huld":
        ds['tas']=ds['tas']*bf_tas
        return _power_huld(irradiance, ds["tas"], pc)
    elif model == "bofinger":
        return _power_bofinger(irradiance, ds["tas"], pc)
    else:
        AssertionError(f"Unknown panel model: {model}")


