from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

def map_plots(variable,
              cmap='viridis',
              setnan=True,
              vmin=None,
              vmax=None,
              title=None,
              label=''):
    """
    Plot a 2D DataArray (x,y) on a PlateCarree map, masking zeros if requested.
    If title is falsy, no title is drawn and the axes are expanded
    symmetrically so the colorbar sits neatly inside.
    """
    # mask zeros→NaN and subset
    if setnan:
        variable = xr.where(variable != 0, variable, np.nan)
    variable = variable.sel(x=slice(-12, 35), y=slice(33, 64))

    lon = variable['x']
    lat = variable['y']

    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax.set_extent([-12, 35, 33, 64], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon, lat, variable,
        transform=ccrs.PlateCarree(),
        cmap=cmap, shading='auto',
        vmin=vmin, vmax=vmax
    )

    # map features
    ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    # only draw a title if non-empty/non-None
    if title:
        ax.set_title(title, fontsize=16)

    # gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5,
        color='gray', linestyle='--',
        x_inline=False, y_inline=False
    )
    gl.xlocator = mticker.FixedLocator(lon.values)
    gl.ylocator = mticker.FixedLocator(lat.values)
    gl.xformatter = mticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    gl.yformatter = mticker.FuncFormatter(lambda y, _: f"{y:.2f}")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 16, 'rotation': 45, 'ha': 'right'}
    gl.ylabel_style = {'fontsize': 16}

    # colorbar (shrink lowers its height)
    cbar = fig.colorbar(mesh, ax=ax,
                        orientation='vertical',
                        pad=0.02,
                        shrink=0.8)
    cbar.set_label(label, rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # if no title, expand axes symmetrically by δ
    if not title:
        left, bottom, width, height = ax.get_position().bounds
        δ = 0.05
        new_bottom = max(0, bottom - δ/2)
        ax.set_position([left, new_bottom, width, height + δ])

    plt.show()



import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point, box

def country_plots(
    variable: xr.DataArray,
    cmap: str = 'RdBu_r',
    vmin: float = None,
    vmax: float = None,
    title: str = '',
    label: str = '',
    missing_frac_thresh: float = 0.7  # fraction of missing cells to flag a country as unavailable
):
    """
    Take a DataArray on dims (y,x), compute per-country means, and plot.
    
    - variable: xarray.DataArray with coords .x/.y in degrees
    - cmap: matplotlib colormap name
    - vmin/vmax: if None→use full data min/max; else→clip colorbar
    - title: plot title
    - label: colorbar label
    - missing_frac_thresh: float in [0,1]; if a country has this fraction or more of its
      cells missing, it will be treated as unavailable and colored gray.
    """
    # 1) Subset region
    da = variable.sel(x=slice(-12, 35), y=slice(33, 64))
    
    # 2) Flatten to a Pandas table of (lon, lat, value), KEEPING NaNs
    xs, ys = da.x.values, da.y.values
    lon2d, lat2d = np.meshgrid(xs, ys)
    df = pd.DataFrame({
        'lon': lon2d.ravel(),
        'lat': lat2d.ravel(),
        'value': da.values.ravel()
    })
    
    # 3) Make a GeoDataFrame of points (including those with NaN value)
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df.lon, df.lat)],
        crs="EPSG:4326"
    )
    
    # 4) Load high-res countries and clip to the box
    ne_50m = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(ne_50m).to_crs("EPSG:4326")
    bbox = box(-12, 33, 35, 64)
    world = world.clip(bbox)
    
    # 5) Spatial join
    joined = gpd.sjoin(
        gdf_pts, 
        world[['NAME_LONG','geometry']],
        how='inner',
        predicate='within'
    )
    
    # 6) Compute availability stats per country
    stats = joined.groupby('NAME_LONG').agg(
        total_cells = ('value','size'),
        avail_cells = ('value', lambda x: x.notna().sum())
    )
    stats['missing_frac'] = 1 - stats['avail_cells'] / stats['total_cells']
    
    # 7) Identify countries with too many missing cells
    bad_countries = stats.index[stats['missing_frac'] >= missing_frac_thresh]
    
    # 8) Compute mean_diff only for sufficiently available countries
    valid = joined[~joined['NAME_LONG'].isin(bad_countries)]
    country_means = (
        valid
        .groupby('NAME_LONG')['value']
        .mean()
        .reset_index(name='mean_diff')
    )
    
    # 9) Merge means into world; others remain NaN
    world = world.merge(country_means, on='NAME_LONG', how='left')
    
    # 10) Decide on vmin/vmax
    if vmin is None and vmax is None:
        vmin_plot = world['mean_diff'].min()
        vmax_plot = world['mean_diff'].max()
    else:
        vmin_plot, vmax_plot = vmin, vmax
    norm = mpl.colors.Normalize(vmin=vmin_plot, vmax=vmax_plot)
    
    # 11) Plot
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    world.plot(
        column='mean_diff',
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        linewidth=0.5,
        edgecolor='black',
        ax=ax,
        missing_kwds={
            'color':'lightgrey',
            #'edgecolor':'red',
            #'hatch':'///',
            'label':'no data'
        }
    )
    
    # map styling
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.coastlines('10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-12, 35, 33, 64], ccrs.PlateCarree())
    ax.set_title(title, fontsize=16)
    
    # 1° gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        linestyle='--',
        xlocs=np.arange(-12, 36, 1),
        ylocs=np.arange(33, 65, 1)
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize':16,'rotation':45,'ha':'right'}
    gl.ylabel_style = {'fontsize':16}
    
    # colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    plt.show()

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box

def country_plots_weighted(
    variable: xr.DataArray,
    cmap: str = 'RdBu_r',
    vmin: float = None,
    vmax: float = None,
    title: str = '',
    label: str = '',
    missing_frac_thresh: float = 0.7,
    proj_crs: str = "EPSG:3035"
):
    """
    Plot per‐country means of `variable`, weighting each grid cell by the
    fraction of its area within the country.  
    Countries with ≥ missing_frac_thresh fraction of their total cell‐area
    missing — or which lie outside Europe — will be gray.
    """
    # 1) Subset region & extract coords + values
    da = variable.sel(x=slice(-12, 35), y=slice(33, 64))
    xs, ys = da.x.values, da.y.values
    vals = da.values  # shape (ny, nx)

    # 2) Build GeoDataFrame of grid‐cell polygons with a unique cell_id
    dx = np.diff(xs).mean()
    dy = np.diff(ys).mean()
    records = []
    cell_id = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            records.append({
                'cell_id': cell_id,
                'value': vals[j, i],
                'geometry': box(x - dx/2, y - dy/2, x + dx/2, y + dy/2)
            })
            cell_id += 1
    gdf_cells = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # 3) Load & filter to Europe only
    ne50 = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(ne50).to_crs("EPSG:4326")
    # keep only European continent
    world = world[world['CONTINENT'] == 'Europe']
    # clip to our bounding box
    region_box = box(-12, 33, 35, 64)
    world = world.clip(region_box)

    # 4) Compute intersections: each cell ∩ country
    inter = gpd.overlay(gdf_cells, world[['NAME_LONG','geometry']],
                        how='intersection')

    # 5) Project intersections & cells to an equal‐area CRS
    inter_proj = inter.to_crs(proj_crs)
    cells_proj = gdf_cells.to_crs(proj_crs)[['cell_id','geometry']]
    cells_proj['cell_area'] = cells_proj.geometry.area

    # 6) Compute overlap area & weight for each piece
    inter_proj['overlap_area'] = inter_proj.geometry.area
    inter_proj = inter_proj.merge(cells_proj[['cell_id','cell_area']], on='cell_id')
    inter_proj['weight'] = inter_proj['overlap_area'] / inter_proj['cell_area']

    # 7) Compute per-country missing‐area fraction
    grp          = inter_proj.groupby('NAME_LONG')
    total_weight = grp['weight'].sum()
    avail_weight = grp.apply(lambda g: (g['weight'] * g['value'].notna()).sum())
    avail_frac   = avail_weight / total_weight

    # 8) Identify “good” countries
   # keep only those where at least `missing_frac_thresh` of the AREA is available
    good  = avail_frac[avail_frac >= missing_frac_thresh].index
    valid = inter_proj[inter_proj['NAME_LONG'].isin(good)]


    # 9) Compute weighted mean per country
    weighted = valid.groupby('NAME_LONG').apply(
        lambda g: (g['weight'] * g['value']).sum() / g['weight'].sum()
    ).rename('mean_diff').reset_index()

    # 10) Merge into our Europe‐only world for plotting
    world = world.merge(weighted, on='NAME_LONG', how='left')

    # 11) Color‐limits
    if vmin is None and vmax is None:
        vmin_plot, vmax_plot = world['mean_diff'].min(), world['mean_diff'].max()
    else:
        vmin_plot, vmax_plot = vmin, vmax
    norm = mpl.colors.Normalize(vmin=vmin_plot, vmax=vmax_plot)

    # 12) Plot
    fig, ax = plt.subplots(figsize=(12, 8),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    world.plot(
        column='mean_diff',
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        linewidth=0.5,
        edgecolor='black',
        ax=ax,
        missing_kwds={'color':'lightgrey'}
    )
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.coastlines('10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-12, 35, 33, 64], ccrs.PlateCarree())
    ax.set_title(title, fontsize=16)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--',
                      xlocs=np.arange(-12, 36, 1), ylocs=np.arange(33, 65, 1))
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'fontsize':16,'rotation':45,'ha':'right'}
    gl.ylabel_style = {'fontsize':16}

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    plt.show()


def map_plots_discrete(variable, cmap='viridis', setnan=True, vmin=None, vmax=None, norm=None, title='', label=''):
    """
    Plot a map with discrete categories using a legend instead of a colorbar.

    Parameters:
    - variable: xarray.DataArray, the variable to plot.
    - cmap: str or Colormap, color map to use.
    - setnan: bool, whether to set 0 values as NaN.
    - vmin, vmax: float, value limits.
    - norm: matplotlib.colors.BoundaryNorm, required for discrete bins.
    - title: str, plot title.
    - label: str, legend label title.

    Returns:
    - None (displays the plot).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.cm as cm
    import xarray as xr
    import numpy as np

    # Handle masking if needed
    if setnan:
        variable = xr.where(variable != 0, variable, float('nan')).sel(x=slice(-12, 35), y=slice(33, 64))
    else:
        variable = variable.sel(x=slice(-12, 35), y=slice(33, 64))

    if variable.isnull().all():
        print("⚠️ All values are NaN — nothing to plot.")
        return

    # Extract grid
    lon = variable.x
    lat = variable.y

    # Ensure colormap object
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([-12, 35, 33, 64], crs=ccrs.PlateCarree())

    # pcolormesh plot
    c = ax.pcolormesh(
        lon, lat, variable,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading='auto',
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )

    # Map features
    ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.set_title(title, fontsize=16)

    # Gridlines
    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        linestyle='--',
        x_inline=False,
        y_inline=False
    )
    gridlines.xlocator = plt.FixedLocator(lon.values)
    gridlines.ylocator = plt.FixedLocator(lat.values)
    gridlines.xformatter = mticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    gridlines.yformatter = mticker.FuncFormatter(lambda y, _: f"{y:.2f}")
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {'fontsize': 12, 'rotation': 45, 'ha': 'right'}
    gridlines.ylabel_style = {'fontsize': 12}

    # Legend for discrete categories
    if norm is not None:
        levels = norm.boundaries[:-1]
        colors = [cmap(norm(level)) for level in levels]
        legend_patches = [mpatches.Patch(color=color, label=f"{int(level)}") for color, level in zip(colors, levels)]
        ax.legend(
            handles=legend_patches,
            title=label,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
            title_fontsize=12,
            frameon=True,
            edgecolor="black",
        )

    plt.show()


import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import numpy as np

def map_plots_mask(variable,
              cmap='viridis',
              setnan=True,
              vmin=None,
              vmax=None,
              title='',
              label='',
              robust_mask: xr.DataArray = None,
              hatch_style: str = '////',
              hatch_color: str = 'gray'):
    """
    Plot a map of `variable` on PlateCarree.  Optionally overlay a hatch
    on cells where `robust_mask` is False.

    Parameters
    ----------
    variable : xarray.DataArray
        Must have coords .x (lon) and .y (lat).
    cmap : str or Colormap, optional
        Main colormap for the variable.
    setnan : bool, optional
        If True, zero-values in `variable` become NaN.
    vmin, vmax : float, optional
        Color scale limits.
    title : str, optional
        Plot title.
    label : str, optional
        Colorbar label.
    robust_mask : xarray.DataArray of bool, optional
        Same dims as `variable`.  Wherever this is False, a hatch
        will be drawn on top of the cell.
    hatch_style : str, optional
        Matplotlib hatch pattern, e.g. '/', 'xx', '////'.
    hatch_color : str, optional
        Color of the hatch lines.

    Returns
    -------
    fig, ax : Matplotlib Figure and Axes
    """

    # 0) handle NaNs
    if setnan:
        variable = xr.where(variable != 0, variable, np.nan)

    # 1) slice to desired lon/lat box
    variable = variable.sel(x=slice(-12, 35), y=slice(33, 64))

    # 2) extract coords
    lon = variable.x
    lat = variable.y

    # 3) build figure + map
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax.set_extent([-12, 35, 33, 64], crs=ccrs.PlateCarree())

    # 4) main pcolormesh
    cmap_obj = mpl.cm.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    mesh = ax.pcolormesh(
        lon, lat, variable,
        transform=ccrs.PlateCarree(),
        cmap=cmap_obj,
        shading='auto',
        vmin=vmin, vmax=vmax
    )

    # 5) optional hatch overlay
    if robust_mask is not None:
        # align & slice the mask to same box
        rm = robust_mask.sel(x=slice(-12, 35), y=slice(33, 64))
        # inverse mask: 1 where we WANT a hatch
        inv = (~rm).astype(int)
        ax.contourf(
            lon, lat, inv,
            levels=[0.5, 1.5],
            colors='none',
            hatches=[hatch_style],
            transform=ccrs.PlateCarree(),
            alpha=0  # ensure we only see the hatch
        )

    # 6) coastlines & land/ocean
    ax.add_feature(cfeature.LAND,   facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN,  facecolor='white',     zorder=0)
    ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # 7) gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        linestyle='--',
        x_inline=False,
        y_inline=False
    )
    # place labels every 5°
    gl.xlocator = plt.FixedLocator(np.arange(-10, 36, 1))
    gl.ylocator = plt.FixedLocator(np.arange(34, 65, 1))
    gl.xformatter = mticker.FuncFormatter(lambda x, _: f"{x:.0f}°E")
    gl.yformatter = mticker.FuncFormatter(lambda y, _: f"{y:.0f}°N")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 12, 'rotation': 45, 'ha': 'right'}
    gl.ylabel_style = {'fontsize': 12}

    # 8) title
    ax.set_title(title, fontsize=16)

    # 9) colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(label, rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # 10) optional legend entry for the hatch
    if robust_mask is not None:
        from matplotlib.patches import Patch
        hatch_patch = Patch(
            facecolor='none',
            edgecolor=hatch_color,
            hatch=hatch_style,
            label='Masked out (robust_mask=False)'
        )
        ax.legend(
            handles=[hatch_patch],
            loc='lower left',
            framealpha=0.8,
            fontsize=12,
            title='Overlay'
        )

    plt.tight_layout()
    return fig, ax


def map_plots_lon(variable,
                  cmap='viridis',
                  setnan=True,
                  vmin=None,
                  vmax=None,
                  title=None,
                  label=''):
    """
    Plot a 2D DataArray (lat,lon) on a PlateCarree map, masking zeros if requested.
    If title is falsy, no title is drawn and the axes are expanded
    symmetrically so the colorbar sits neatly inside.
    """
    # mask zeros→NaN and subset
    if setnan:
        variable = xr.where(variable != 0, variable, np.nan)
    variable = variable.sel(lon=slice(-12, 35), lat=slice(33, 64))

    lon = variable['lon']
    lat = variable['lat']

    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax.set_extent([-12, 35, 33, 64], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon, lat, variable,
        transform=ccrs.PlateCarree(),
        cmap=cmap, shading='auto',
        vmin=vmin, vmax=vmax
    )

    # map features
    ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    # conditional title
    if title:
        ax.set_title(title, fontsize=16)

    # gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5,
        color='gray', linestyle='--',
        x_inline=False, y_inline=False
    )
    gl.xlocator = mticker.FixedLocator(lon.values)
    gl.ylocator = mticker.FixedLocator(lat.values)
    gl.xformatter = mticker.FuncFormatter(lambda x, _: f"{x:.1f}")
    gl.yformatter = mticker.FuncFormatter(lambda y, _: f"{y:.1f}")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 16, 'rotation': 45, 'ha': 'right'}
    gl.ylabel_style = {'fontsize': 16}

    # colorbar
    cbar = fig.colorbar(mesh, ax=ax,
                        orientation='vertical',
                        pad=0.02,
                        shrink=0.8)
    cbar.set_label(label, fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # symmetric lift if no title
    if not title:
        left, bottom, width, height = ax.get_position().bounds
        δ = 0.05
        new_bottom = max(0, bottom - δ/2)
        ax.set_position([left, new_bottom, width, height + δ])

    plt.show()