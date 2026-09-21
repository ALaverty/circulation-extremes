
# This script plots all figures for the compound-extremes analysis, main text and supplementary.
# Temperature follows the convention per figure: detrended for the regime characterization, the climatology row of figure 1,
# and the supplementary anomaly grid; raw everywhere else. --raw or --detr overrides it throughout.
# Usage: python figures.py all <season> [k] tmax

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.vectorized import contains
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/lab/singh/amanda/python_codes/era_updated'
NC = f'{BASE}/processed/compound_extremes'
PLOTS = f'{BASE}/plots'
CSV = f'{BASE}/csv'
FIGURES = f'{BASE}/figures'

EXTENT = [-170, -100, 20, 80]
EARLY_YEARS = (1941, 1970)
LATER_YEARS = (1991, 2020)
REFERENCE_PERIOD = (1991, 2020)
ALPHA = 0.10

TYPES = ['wet-cold', 'wet-warm', 'dry-warm', 'dry-cold']
TYPE_LABELS = {'wet-cold': 'Wet-Cold', 'wet-warm': 'Wet-Warm',
               'dry-warm': 'Dry-Warm', 'dry-cold': 'Dry-Cold'}
WET_TYPES = {'wet-cold', 'wet-warm'}
FILE_TYPE = {'wet-warm': 'warm_wet', 'wet-cold': 'cold_wet',
             'dry-warm': 'warm_dry', 'dry-cold': 'cold_dry'}

TYPE_COLORS = {'wet-cold': '#0B3D91', 'wet-warm': '#C0392B',
               'dry-warm': '#D68910', 'dry-cold': '#5DADE2'}
TYPE_COLORS_DARK = {'wet-cold': '#154360', 'wet-warm': '#7B241C',
                    'dry-warm': '#9C640C', 'dry-cold': '#1F618D'}
CLUSTER_COLORS = ['#3AA88B', '#F08A64', '#5B7FC4',
                  '#9BBF3F', '#F5CE3E', '#D9B48F']
COLOR_EARLY, COLOR_LATER = '#4A2C6E', '#F5C242'
GRAY = np.array([0.90, 0.90, 0.90])

LETTERS = 'abcdefghijklmnopqrstuvwxyz'
SEASON_MONTHS = {'winter': [12, 1, 2], 'spring': [3, 4, 5],
                 'summer': [6, 7, 8], 'fall': [9, 10, 11]}

def other_tag(var_tag, raw):
    return tag_for(var_tag.split('_')[0], raw=raw)

def tag_for(var='tmax', raw=False, thresh=None):
    t = f'{var}_raw' if raw else var
    return f'{t}_{thresh}' if thresh else t

def load_field(season, name, var_tag, cluster=None, kind='frequencies'):
    short = FILE_TYPE[name]
    stem = (f'{NC}/{season}_{short}_{var_tag}'
            + (f'_cluster{cluster}' if cluster is not None else ''))
    fp = f'{stem}_{kind}.nc'
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f'{fp} not found. Run: python compound_extremes.py freq '
            f'{season} <k> ...')
    return xr.open_dataarray(fp)

def load_all(season, var_tag, num_clusters, kind, cluster_range=True):
    return {c: {n: load_field(season, n, var_tag, c, kind) for n in TYPES}
            for c in range(num_clusters)}

def load_diff_fields(season, tag):
    fp = f'{NC}/{season}_{tag}_diff_fields.nc'
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f'{fp} not found. Run: python compound_extremes.py diff '
            f'{season} <k> ...')
    return xr.open_dataset(fp)

def load_annual_series(season, tag):
    return pd.read_csv(f'{NC}/{season}_{tag}_annual_series.csv')

def load_domain_tests(season, tag):
    return pd.read_csv(f'{NC}/{season}_{tag}_domain_mean_tests.csv')

def load_clusters(season):
    df = pd.read_csv(f'{BASE}/csv/{season}_all_cluster_assignments.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def attach_clusters(da, season):
    lookup = load_clusters(season).drop_duplicates('date').set_index(
        'date')['cluster']
    dates = pd.DatetimeIndex(pd.to_datetime(da.valid_time.values))
    return da.assign_coords(cluster=(
        'valid_time', lookup.reindex(dates).fillna(-1).astype(int).values))

def _pick_z(ds):
    da = ds['z'] if 'z' in ds.data_vars else ds[
        [v for v in ds.data_vars if len(ds[v].dims) >= 3][0]]
    for d in ('number', 'pressure_level'):
        if d in da.dims:
            da = da.squeeze(d, drop=True)
    return da

def load_z500(season, kind='anomaly'):
    if kind == 'anomaly':
        fp = f'{BASE}/processed/{season}_anomalies_for_clustering.nc'
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f'{fp} not found. Rerun z500_regimes.py for this season with '
                'SAVE_ANOMALIES = True.')
        return attach_clusters(_pick_z(xr.open_dataset(fp)), season)

    cache = f'{BASE}/processed/detrended_raw_data.nc'
    if not os.path.exists(cache):
        raise FileNotFoundError(f'{cache} not found; the height contours need it')
    da = _pick_z(xr.open_dataset(cache))
    months = pd.to_datetime(da.valid_time.values).month
    idx = np.flatnonzero(np.isin(months, SEASON_MONTHS[season]))
    return attach_clusters(da.isel(valid_time=idx), season)

def land_mask_for(da):
    lats, lons = da.latitude.values, da.longitude.values
    shp = shpreader.natural_earth(resolution='50m', category='physical',
                                  name='land')
    geoms = unary_union(list(shpreader.Reader(shp).geometries()))
    lon_mesh, lat_mesh = np.meshgrid(np.where(lons > 180, lons - 360, lons),
                                     lats)
    mask = contains(geoms, lon_mesh.ravel(),
                    lat_mesh.ravel()).reshape(lon_mesh.shape)
    return mask, xr.DataArray(mask, coords={'latitude': da.latitude,
                                            'longitude': da.longitude},
                              dims=['latitude', 'longitude'])

def wet_mask_from_thresholds(season, land_mask, period=None):
    if period is None:
        fp = (f'{BASE}/processed/precipitation_thresholds_gamma_pzero_mask/'
              f'precip_wet_threshold_gamma_{season}.nc')
    else:
        fp = (f'{BASE}/processed/precipitation_thresholds_gamma_periods/'
              f'precip_wet_threshold_gamma_{season}_{period}.nc')
    return np.isnan(xr.open_dataarray(fp).values) & land_mask

def grid_cell_areas(lats, lons):
    R = 6371.0
    dlat = np.abs(lats[1] - lats[0]) * np.pi / 180
    dlon = np.abs(lons[1] - lons[0]) * np.pi / 180
    areas = np.zeros((len(lats), len(lons)))
    for i, lat in enumerate(lats):
        r = lat * np.pi / 180
        areas[i, :] = R**2 * dlon * (np.sin(r + dlat / 2) - np.sin(r - dlat / 2))
    return areas

def mesh(da):
    lon_mesh, lat_mesh = np.meshgrid(da.longitude.values, da.latitude.values)
    return np.where(lon_mesh > 180, lon_mesh - 360, lon_mesh), lat_mesh

def setup_map(ax, labels=True, left=True, bottom=True, white_bg=True):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    if white_bg:
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)
    ax.coastlines(linewidth=0.7, color='black', zorder=4)
    ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor='gray', zorder=4)
    if labels:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.xlines = gl.ylines = False
        gl.top_labels = gl.right_labels = False
        gl.left_labels, gl.bottom_labels = left, bottom
        gl.xlabel_style = gl.ylabel_style = {'size': 8}
        return gl

def stipple(ax, lon_plot, lat_mesh, sig, density=3, size=1.0, color='black'):
    s = sig[::density, ::density]
    if s.any():
        ax.scatter(lon_plot[::density, ::density][s],
                   lat_mesh[::density, ::density][s], color=color, s=size,
                   alpha=0.6, marker='.', transform=ccrs.PlateCarree(),
                   zorder=7)

def mark_masked(ax, lon_plot, lat_mesh, mask, size=2.5):
    if mask is not None and mask.any():
        ax.scatter(lon_plot[mask], lat_mesh[mask], c='#cccccc', s=size,
                   alpha=0.85, marker='s', transform=ccrs.PlateCarree(),
                   zorder=3)

def panel_label(ax, text, boxed=True):
    kw = dict(bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                        edgecolor='none', alpha=0.8)) if boxed else {}
    ax.text(0.02, 0.97, text, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='left', zorder=10, **kw)

def freq_cmap(vmax=10.0, wet=True):
    if wet:
        low = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        bounds = sorted(set([0] + low + list(range(4, int(vmax) + 1))
                            + [vmax]))
    else:
        cand = [0, 0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0]
        bounds = sorted(set([b for b in cand if b <= vmax] + [vmax]))
    n = len(bounds) - 1
    base = plt.get_cmap('YlGnBu', max(n - 1, 1))
    cmap = mcolors.ListedColormap([np.array([1, 1, 1, 1])]
                                  + [base(i) for i in range(n - 1)])
    return cmap, mcolors.BoundaryNorm(bounds, ncolors=n), bounds

def tick_labels(bounds):
    return [f'{v:.2f}' if v < 1
            else f'{v:.1f}' if abs(v - round(v)) > 0.01
            else f'{int(round(v))}' for v in bounds]

def save_fig(fig, outpath, formats=('png',)):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    for ext in formats:
        fig.savefig(f'{outpath}.{ext}', dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print(f'  Saved: {outpath}.' + '/.'.join(formats))

def mw_test(early, later, min_n=5):
    e = np.asarray(early, float)
    l = np.asarray(later, float)
    e, l = e[np.isfinite(e)], l[np.isfinite(l)]
    if len(e) < min_n or len(l) < min_n:
        return np.nan, np.nan
    try:
        res = mannwhitneyu(e, l, alternative='two-sided')
        return float(res.pvalue), float(2.0 * res.statistic / (len(e) * len(l)) - 1.0)
    except Exception:
        return np.nan, np.nan

def annual_composites(z500, cluster, year_range, min_days=5):
    dates = pd.to_datetime(z500.valid_time.values)
    cmask = (z500.cluster == cluster).values
    vals = z500.values
    nlat, nlon = len(z500.latitude), len(z500.longitude)

    comps, years, n_days = [], [], []
    for yr in range(year_range[0], year_range[1] + 1):
        sel = cmask & (dates.year == yr)
        n = int(sel.sum())
        if n < min_days:
            continue
        comps.append(np.nanmean(vals[sel], axis=0))
        years.append(yr)
        n_days.append(n)
    if not comps:
        return (np.empty((0, nlat, nlon)), np.array([], int), np.array([], int))
    return (np.asarray(comps), np.asarray(years, int), np.asarray(n_days, int))

def extent_mask(lats, lons_plot, extent=EXTENT):
    lon_min, lon_max, lat_min, lat_max = extent
    lat_mesh, lon_mesh = np.meshgrid(lats, lons_plot, indexing='ij')
    return ((lon_mesh >= lon_min) & (lon_mesh <= lon_max)
            & (lat_mesh >= lat_min) & (lat_mesh <= lat_max))

def compare_periods(z500, cluster, min_days=5, alpha=ALPHA, cell_mask=None,
                    do_cell_map=True):
    comps_e, yrs_e, nd_e = annual_composites(z500, cluster, EARLY_YEARS, min_days)
    comps_l, yrs_l, nd_l = annual_composites(z500, cluster, LATER_YEARS, min_days)

    lats = z500.latitude.values
    nlat, nlon = len(lats), len(z500.longitude)
    area_w = np.broadcast_to(np.cos(np.deg2rad(lats))[:, None], (nlat, nlon))

    def amplitude(comp):
        finite = np.isfinite(comp)
        if cell_mask is not None:
            finite = finite & cell_mask
        return (float(np.sqrt(np.average(comp[finite] ** 2,
                                         weights=area_w[finite])))
                if finite.any() else np.nan)

    amp_e = np.array([amplitude(c) for c in comps_e])
    amp_l = np.array([amplitude(c) for c in comps_l])
    amp_p, amp_r = mw_test(amp_e, amp_l)

    comp_e = comps_e.mean(axis=0) if len(comps_e) else np.full((nlat, nlon), np.nan)
    comp_l = comps_l.mean(axis=0) if len(comps_l) else np.full((nlat, nlon), np.nan)

    pvals = np.full((nlat, nlon), np.nan)
    if do_cell_map and len(comps_e) >= 3 and len(comps_l) >= 3:
        for i in range(nlat):
            for j in range(nlon):
                if cell_mask is not None and not cell_mask[i, j]:
                    continue
                pvals[i, j] = mw_test(comps_e[:, i, j], comps_l[:, i, j])[0]

    valid = np.isfinite(pvals)
    sig = np.zeros((nlat, nlon), dtype=bool)
    if valid.sum() > 0:
        sig[valid] = multipletests(pvals[valid], alpha=alpha,
                                   method='fdr_bh')[0]

    return dict(comp_e=comp_e, comp_l=comp_l, diff=comp_l - comp_e, sig=sig,
                n_yrs_e=len(yrs_e), n_yrs_l=len(yrs_l),
                amp_e=amp_e, amp_l=amp_l, amp_p=amp_p, amp_r=amp_r,
                amp_sig=False, amp_p_adj=np.nan,
                n_cells_tested=int(valid.sum()), n_cells_sig=int(sig.sum()),
                mean_days_e=float(nd_e.mean()) if len(nd_e) else np.nan,
                mean_days_l=float(nd_l.mean()) if len(nd_l) else np.nan)

def amplitude_sig(results, num_clusters, alpha=ALPHA):
    for c in range(num_clusters):
        if c in results:
            p = results[c]['amp_p']
            results[c]['amp_sig'] = bool(np.isfinite(p) and p < alpha)
    return results


# Temperature  --raw or --detr flag on the command line overrides the defualts
TEMP_MODE = {
    '2': 'detr',              
    'anomaly_grid': 'detr',
}
TEMP_MODE_DEFAULT = 'raw'
FIG1_CLIM_DETRENDED = True

THRESH_MAPS = 'fixed'
THRESH_STORY = 'ownclim'     

MAP_VMAX = {'fixed': dict(wet=1.0, dry=4.0),
            'ownclim': dict(wet=1.0, dry=4.0)}

EVENTS = {'wet-cold': '2010-02-03', 'wet-warm': '2011-12-05',
          'dry-warm': '2012-01-04', 'dry-cold': '2022-12-19'}

FIG1_WET_VMAX, FIG1_DRY_VMAX = 4.0, 10.0     
FREQ_YMAX = {'wet': 2.5, 'dry': 9.0}          
EXT_YMAX = {'wet': 2.0, 'dry': 5.0}           
ROLLING = 10

FREQ_YLIM = {'wet-cold': (0, 5.0), 'wet-warm': (0, 5.0),
             'dry-warm': (0, 16.0), 'dry-cold': (0, 16.0)}
EXT_YLIM = {'wet-cold': (0, 1600), 'wet-warm': (0, 1600),
            'dry-warm': (0, 4500), 'dry-cold': (0, 4500)}

Z_TEMP_VMAX, Z_PRECIP_VMAX = 1.5, 100

FIG1_LAYOUT = dict(size=(19, 18.5),
                   heights=(1.75, 0.035, 0.09, 1.65), 
                   outer_hspace=0.05,
                   map_hspace=0.04,                   
                   series_hspace=0.40)               
FIG3_LAYOUT = dict(size=(18, 13),
                   heights=(1.0, 0.045, 0.14, 1.85), 
                   outer_hspace=0.10,
                   box_hspace=0.35)
DP_LIM = (-5.5, 5.5)

def trend(years, values):
    import pymannkendall as mk
    ok = np.isfinite(values)
    slope, intercept = stats.theilslopes(values[ok], years[ok])[:2]
    try:
        p = float(mk.hamed_rao_modification_test(values[ok], alpha=ALPHA).p)
    except Exception:
        p = np.nan
    return slope, intercept, p

def series_panel(ax, years, vals, color, ylabel, ymax, step, label,
                 legend=False, unit_fmt='%.2f'):
    ax.plot(years, vals, color=color, linewidth=1.6, alpha=0.4, label='Annual')
    roll = pd.Series(vals).rolling(ROLLING, center=True, min_periods=1).mean()
    ax.plot(years, roll, color='black', linewidth=2.5,
            label=f'{ROLLING}-yr rolling mean', zorder=5)

    slope, intercept, p = trend(years, vals)
    ax.plot(years, slope * years + intercept, color='black', linewidth=2.0,
            linestyle='--', alpha=0.7, zorder=4)
    ax.text(0.97, 0.97, f'{slope * 10:+.2f}/dec, p={p:.2f}',
            transform=ax.transAxes, fontsize=12, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                      edgecolor='none'))

    ax.set_title(f'{label})', fontsize=15, fontweight='bold', loc='left')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_xlim(1940, 2025)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 0.01, step))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
    ax.set_xticks(np.arange(1940, 2025, 20))
    ax.tick_params(axis='both', labelsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    if legend:
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

def figure1(season, k, var_tag):
    from compound_extremes import CompoundExtremes

    a = CompoundExtremes(season, k, BASE, var=var_tag.split('_')[0],
                         raw=var_tag.endswith('raw'))
    a.load_data()
    a.identify_extremes()
    z500 = load_z500(season, 'anomaly')
    lon_plot, lat_mesh = mesh(a.temp_anom)
    land, _ = land_mask_for(a.temp_anom)
    wet_masked = wet_mask_from_thresholds(season, land)
    areas_dates = pd.to_datetime(a.temp_anom.valid_time.values)

    clim_tag = other_tag(var_tag, raw=False) if FIG1_CLIM_DETRENDED else var_tag
    if clim_tag != var_tag:
        print(f'  climatology row uses {clim_tag}; the other rows use {var_tag}')
    ser = pd.read_csv(f'{NC}/{season}_{var_tag}_fullrecord_annual.csv')

    L = FIG1_LAYOUT
    fig = plt.figure(figsize=L['size'])
    outer = fig.add_gridspec(4, 1, height_ratios=L['heights'],
                             hspace=L['outer_hspace'], top=0.95, bottom=0.05,
                             left=0.05, right=0.98)
    g_map = outer[0].subgridspec(2, 4, hspace=L['map_hspace'], wspace=0.06)
    g_cb = outer[1].subgridspec(1, 2, wspace=0.15)
    g_ts = outer[3].subgridspec(2, 4, hspace=L['series_hspace'], wspace=0.18)
    axes = np.empty((4, 4), dtype=object)
    for r in range(2):
        for c in range(4):
            axes[r, c] = fig.add_subplot(g_map[r, c],
                                         projection=ccrs.PlateCarree())
            axes[r + 2, c] = fig.add_subplot(g_ts[r, c])

    # row 1: event days
    for j, name in enumerate(TYPES):
        ax = axes[0, j]
        target = pd.Timestamp(EVENTS[name])
        idx = int(np.argmin(np.abs(areas_dates - target)))
        date = areas_dates[idx]
        if date != target:
            print(f'    note: nearest day to {target.date()} is {date.date()}')

        setup_map(ax, labels=False, white_bg=False)
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f0', zorder=0)
        day = a.masks[name].isel(valid_time=idx).values
        ax.pcolormesh(lon_plot, lat_mesh,
                      np.where(land & day.astype(bool), 1.0, np.nan),
                      cmap=ListedColormap([TYPE_COLORS[name]]), vmin=0, vmax=1,
                      alpha=0.55, transform=ccrs.PlateCarree(), zorder=2)

        zi = int(np.argmin(np.abs(
            pd.to_datetime(z500.valid_time.values) - date)))
        zday = z500.isel(valid_time=zi).values
        vmax = float(np.nanmax(np.abs(zday)))
        cs = ax.contour(lon_plot, lat_mesh, zday,
                        levels=np.linspace(-vmax, vmax, 11), colors='black',
                        linewidths=1.0, transform=ccrs.PlateCarree(), zorder=3)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
        ax.set_title(f'{LETTERS[j]}) {TYPE_LABELS[name]}:\n'
                     f'{date:%B} {date.day}, {date.year}',
                     fontsize=12, fontweight='bold', loc='left')

    # row 2: 1991-2020 climatological frequency
    wet_cmap, wet_norm, wet_b = freq_cmap(FIG1_WET_VMAX, wet=True)
    dry_cmap, dry_norm, dry_b = freq_cmap(FIG1_DRY_VMAX, wet=False)
    cs_wet = cs_dry = None
    for j, name in enumerate(TYPES):
        ax = axes[1, j]
        is_wet = name in WET_TYPES
        cmap, norm, bounds = ((wet_cmap, wet_norm, wet_b) if is_wet
                              else (dry_cmap, dry_norm, dry_b))
        setup_map(ax, labels=True, left=(j == 0), bottom=(j == 0))
        data = load_field(season, name, clim_tag, kind='reference_frequencies')
        if is_wet:
            mark_masked(ax, lon_plot, lat_mesh, wet_masked, size=3)
            data = data.where(~xr.DataArray(
                wet_masked, coords=data.coords, dims=data.dims))
        cs = ax.contourf(lon_plot, lat_mesh, data, levels=bounds, cmap=cmap,
                         norm=norm, extend='max',
                         transform=ccrs.PlateCarree(), zorder=2)
        if is_wet:
            cs_wet = cs
        else:
            cs_dry = cs
        ax.set_title(f'{LETTERS[4 + j]})', fontsize=13, fontweight='bold',
                     loc='left')

    # rows 3 and 4: full-record series
    for j, name in enumerate(TYPES):
        sub = ser[ser['type'] == name].sort_values('year')
        yrs = sub['year'].values.astype(float)
        wet = name in WET_TYPES
        series_panel(axes[2, j], yrs, sub['frequency'].values,
                     TYPE_COLORS[name],
                     'Frequency (% of days)' if j == 0 else '',
                     FREQ_YMAX['wet' if wet else 'dry'],
                     0.5 if wet else 2.0, LETTERS[8 + j], legend=(j == 0))
        series_panel(axes[3, j], yrs, sub['extent'].values / 1000.0,
                     TYPE_COLORS[name],
                     'Max Daily Extent (10$^6$ km$^2$)' if j == 0 else '',
                     EXT_YMAX['wet' if wet else 'dry'],
                     0.5 if wet else 1.0, LETTERS[12 + j], legend=(j == 0))

    cb = fig.colorbar(cs_wet, cax=fig.add_subplot(g_cb[0]),
                      orientation='horizontal', extend='max')
    cb.set_ticks(wet_b)
    cb.set_ticklabels(tick_labels(wet_b))
    cb.set_label('Frequency (% of days)  |  Gray = masked', fontsize=11)
    cb = fig.colorbar(cs_dry, cax=fig.add_subplot(g_cb[1]),
                      orientation='horizontal', extend='max')
    cb.set_ticks(dry_b)
    cb.set_ticklabels(tick_labels(dry_b))
    cb.set_label('Frequency (% of days)', fontsize=11)

    fig.suptitle(f'{season.title()}: Recent Compound Extreme Events',
                 fontsize=14, fontweight='bold', y=0.985)
    save_fig(fig, f'{FIGURES}/{season}_fig1_events_climatology_series')

def figure2(season, k, var_tag):
    var = var_tag.split('_')[0]
    z_anom = load_z500(season, 'anomaly')
    z_raw = load_z500(season, 'height')

    tz = attach_clusters(xr.open_dataarray(
        f'{BASE}/processed/temperature/{var}_{season}_zscore.nc'), season)
    pa = attach_clusters(xr.open_dataarray(
        f'{BASE}/processed/precipitation_raw/'
        f'precip_raw_{season}_anomalies_absolute_mm.nc'), season)
    precip = xr.open_dataset(
        f'{BASE}/processed/precipitation_raw/precip_raw_{season}.nc')
    pvar = 'tp' if 'tp' in precip.data_vars else list(precip.data_vars)[0]
    pclim = precip[pvar].mean('valid_time').values

    land, land_da = land_mask_for(tz)
    lon_plot, lat_mesh = mesh(tz)
    clusters = load_clusters(season)
    n_total = len(clusters)

    fig, axes = plt.subplots(4, k, figsize=(2.6 * k, 9),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    z_vmax = np.ceil(max(float(np.abs(z_anom.isel(valid_time=(
        z_anom.cluster == c).values).mean('valid_time')).max())
        for c in range(k)) / 20) * 20
    z_levels = np.linspace(-z_vmax, z_vmax, 21)
    t_levels = np.linspace(-Z_TEMP_VMAX, Z_TEMP_VMAX, 21)
    p_levels = np.linspace(-Z_PRECIP_VMAX, Z_PRECIP_VMAX, 21)

    cs_z = cs_t = cs_p = None
    for c in range(k):
        n_days = int((clusters['cluster'] == c).sum())

        ax = axes[0, c]
        setup_map(ax, labels=False, white_bg=False)
        za = z_anom.isel(valid_time=(z_anom.cluster == c).values
                         ).mean('valid_time').values
        zr = z_raw.isel(valid_time=(z_raw.cluster == c).values
                        ).mean('valid_time').values
        cs_z = ax.contourf(lon_plot, lat_mesh, za, levels=z_levels,
                           cmap='RdBu_r', extend='both',
                           transform=ccrs.PlateCarree(), zorder=1)
        cl = ax.contour(lon_plot, lat_mesh, zr,
                        levels=np.linspace(zr.min(), zr.max(), 12),
                        colors='#555555', linewidths=0.5, alpha=0.6,
                        transform=ccrs.PlateCarree(), zorder=3)
        ax.clabel(cl, inline=True, fontsize=5, fmt='%.0f')
        ax.set_title(f'Cluster {c}:\n{n_days} days '
                     f'({n_days / n_total * 100:.1f}%)',
                     fontsize=10, fontweight='bold', loc='left')
        panel_label(ax, f'{LETTERS[c]})')

        ax = axes[1, c]
        setup_map(ax, labels=False, white_bg=False)
        cs_t = ax.contourf(lon_plot, lat_mesh,
                           tz.isel(valid_time=(tz.cluster == c).values
                                   ).mean('valid_time'), levels=t_levels,
                           cmap='bwr', extend='both',
                           transform=ccrs.PlateCarree(), zorder=1)
        panel_label(ax, f'{LETTERS[k + c]})')

        ax = axes[2, c]
        setup_map(ax, labels=False, white_bg=False)
        pm = pa.isel(valid_time=(pa.cluster == c).values
                     ).mean('valid_time').values
        cs_p = ax.contourf(lon_plot, lat_mesh,
                           np.where(pclim > 0.01, pm / pclim * 100, np.nan),
                           levels=p_levels, cmap='BrBG', extend='both',
                           transform=ccrs.PlateCarree(), zorder=1)
        panel_label(ax, f'{LETTERS[2 * k + c]})')

        # dominant enhanced type: cluster frequency minus all other days
        ax = axes[3, c]
        setup_map(ax, labels=False, white_bg=False)
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f0', zorder=0)
        anom = np.stack([
            (load_field(season, n, var_tag, c, 'frequencies').values
             - load_field(season, n, var_tag, c,
                          'noncluster_frequencies').values)
            for n in TYPES])
        elig = np.isfinite(anom) & (anom > 0)
        vals = np.where(elig, anom, -np.inf)
        best = np.argmax(vals, axis=0)
        margin = np.max(vals, axis=0) - np.sort(vals, axis=0)[-2]
        n_elig = elig.sum(axis=0)
        dominant = (n_elig == 1) | ((n_elig >= 2) & (margin >= 0.5))

        ax.pcolormesh(lon_plot, lat_mesh,
                      np.where(~dominant & land, 0.0, np.nan),
                      cmap=ListedColormap([GRAY]), vmin=-0.1, vmax=0.1,
                      alpha=0.55, transform=ccrs.PlateCarree(),
                      shading='auto', zorder=1)
        for t, name in enumerate(TYPES):
            m = dominant & (best == t) & land
            if m.any():
                ax.pcolormesh(lon_plot, lat_mesh, np.where(m, float(t), np.nan),
                              cmap=ListedColormap([TYPE_COLORS[name]]),
                              vmin=t - 0.1, vmax=t + 0.1, alpha=0.55,
                              transform=ccrs.PlateCarree(), shading='auto',
                              zorder=1)
        panel_label(ax, f'{LETTERS[3 * k + c]})')

    for row, label in enumerate(['Z500 (m)', 'Temp (z-scores)', 'Precip (%)',
                                 'Dominant Extremes']):
        axes[row, 0].text(-0.10, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=10, fontweight='bold', rotation=90,
                          va='center', ha='center')

    fig.subplots_adjust(right=0.90, top=0.90, bottom=0.06, hspace=0.09,
                        wspace=0.05)
    for i, (cs, ticks) in enumerate([
            (cs_z, None),
            (cs_t, np.arange(-1.5, 1.6, 0.5)),
            (cs_p, np.arange(-100, 101, 25))]):
        cax = fig.add_axes([0.91, 0.90 - (i + 1) * 0.17 - i * 0.05, 0.010, 0.17])
        cb = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
        if ticks is not None:
            cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=7)

    fig.legend(handles=[mpatches.Patch(facecolor=TYPE_COLORS[n],
                                       label=TYPE_LABELS[n], alpha=0.55)
                        for n in TYPES]
               + [mpatches.Patch(facecolor=GRAY, label='Mixed Event Type')],
               loc='center left', ncol=1, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.90, 0.15))
    fig.suptitle(f'{season.title()}: Circulation Regime Characterization',
                 fontsize=12, fontweight='bold', y=0.96)
    save_fig(fig, f'{FIGURES}/{season}_fig2_regime_characterization')

def figure3(season, k, var_tag, thresh=THRESH_MAPS):
    print(f'  thresholds: {thresh}')
    tag = f'{var_tag}_{thresh}'
    ds = load_diff_fields(season, tag)
    annual = load_annual_series(season, tag)
    tests = load_domain_tests(season, tag)

    template = ds['diff_all_wet_cold']
    land, land_da = land_mask_for(template)
    wet_masked = wet_mask_from_thresholds(season, land)
    lon_plot, lat_mesh = mesh(template)

    L = FIG3_LAYOUT
    fig = plt.figure(figsize=L['size'])
    outer = fig.add_gridspec(4, 1, height_ratios=L['heights'],
                             hspace=L['outer_hspace'], top=0.92, bottom=0.07,
                             left=0.06, right=0.98)
    g_map = outer[0].subgridspec(1, 4, wspace=0.06)
    g_cb = outer[1].subgridspec(1, 2, wspace=0.15)
    g_box = outer[3].subgridspec(2, 4, hspace=L['box_hspace'], wspace=0.18)
    axes_map = [fig.add_subplot(g_map[0, j], projection=ccrs.PlateCarree())
                for j in range(4)]
    axes_f = [fig.add_subplot(g_box[0, j]) for j in range(4)]
    axes_e = [fig.add_subplot(g_box[1, j]) for j in range(4)]

    im_wet = im_dry = None
    for j, name in enumerate(TYPES):
        ax = axes_map[j]
        key = f'all_{name.replace("-", "_")}'
        is_wet = name in WET_TYPES
        vmax = MAP_VMAX[thresh]['wet' if is_wet else 'dry']
        setup_map(ax, labels=(j == 0), left=(j == 0), bottom=(j == 0))
        im = ax.pcolormesh(template.longitude, template.latitude,
                           ds[f'diff_{key}'].clip(-vmax, vmax).where(land_da),
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')
        if is_wet:
            im_wet = im
            mark_masked(ax, lon_plot, lat_mesh, wet_masked, size=1.5)
        else:
            im_dry = im
        stipple(ax, lon_plot, lat_mesh,
                ds[f'sig_{key}'].values.astype(bool), density=5, size=2.0)
        ax.set_title(f'{LETTERS[j]}) {TYPE_LABELS[name]}', fontsize=13,
                     fontweight='bold', loc='left', pad=8)

    for row, (axs, metric, sig_col, ylims, ylabel, offset) in enumerate([
            (axes_f, 'frequency', 'freq_sig', FREQ_YLIM, 'Frequency (%)', 4),
            (axes_e, 'extent', 'ext_sig', EXT_YLIM,
             'Max Spatial Extent (\u00d710\u00b3 km\u00b2)', 8)]):
        for j, name in enumerate(TYPES):
            ax = axs[j]
            early, later, flags = [], [], []
            for c in range(k):
                sub = annual[(annual.cluster.astype(str) == str(c))
                             & (annual.type == name)
                             & (annual.metric == metric)]
                early.append(sub[sub.period == 'early']['value'].values)
                later.append(sub[sub.period == 'later']['value'].values)
                row_t = tests[(tests.cluster.astype(str) == str(c))
                              & (tests.type == name)]
                flags.append(bool(row_t[sig_col].iloc[0]) if len(row_t) else False)
            paired_box(ax, early, later, k, flags,
                       ylabel if j == 0 else '', ylims[name])
            ax.set_title(f'{LETTERS[offset + j]})', fontsize=13,
                         fontweight='bold', loc='left')

    axes_f[-1].legend(handles=[
        mpatches.Patch(facecolor=COLOR_EARLY, alpha=0.75,
                       label=f'{EARLY_YEARS[0]}-{EARLY_YEARS[1]}'),
        mpatches.Patch(facecolor=COLOR_LATER, alpha=0.75,
                       label=f'{LATER_YEARS[0]}-{LATER_YEARS[1]}')],
        loc='upper right', fontsize=11, framealpha=0.9)

    cb = fig.colorbar(im_wet, cax=fig.add_subplot(g_cb[0]),
                      orientation='horizontal', extend='both')
    cb.set_label('Change in frequency (% of days)\nGray = masked', fontsize=11)
    cb = fig.colorbar(im_dry, cax=fig.add_subplot(g_cb[1]),
                      orientation='horizontal', extend='both')
    cb.set_label('Change in frequency (% of days)', fontsize=11)

    fig.supxlabel('Circulation Regime', fontsize=14, y=0.03)
    label = ('1991-2020 thresholds' if thresh == 'fixed'
             else 'period-specific thresholds')
    fig.suptitle(f'{season.title()} Historical Change: '
                 f'({LATER_YEARS[0]}\u2013{LATER_YEARS[1]} vs. '
                 f'{EARLY_YEARS[0]}\u2013{EARLY_YEARS[1]}) | {label}',
                 fontsize=15, fontweight='bold', y=0.97)
    save_fig(fig, f'{FIGURES}/{season}_fig3_change_maps_boxplots_{thresh}')

def paired_box(ax, early, later, k, sig_flags, ylabel, ylim):
    w = 1.6
    for data, offset, color in ((early, -0.32, COLOR_EARLY),
                                (later, 0.32, COLOR_LATER)):
        clean = [d[np.isfinite(d)] if len(d) else np.array([np.nan])
                 for d in data]
        bp = ax.boxplot(clean, positions=[c * w + offset for c in range(k)],
                        patch_artist=True, widths=0.46, whis=1.5,
                        showfliers=True,
                        flierprops=dict(marker='o', markersize=2.5,
                                        markerfacecolor=color, alpha=0.5,
                                        linestyle='none'),
                        medianprops=dict(color='black', linewidth=2.0),
                        whiskerprops=dict(color=color, linewidth=1.2),
                        capprops=dict(color=color, linewidth=1.2),
                        boxprops=dict(linewidth=1.2))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(color)
            patch.set_alpha(0.75 if sig_flags[i] else 0.35)

    ax.set_xticks([c * w for c in range(k)])
    ax.set_xticklabels([f'C{c}' for c in range(k)], fontsize=11)
    ax.set_xlim(-0.9, (k - 1) * w + 0.9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_ylim(ylim)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linewidth=0.5, alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    
def later_contours(ax, lon_plot, lat_mesh, field, neg, pos):
    levels = np.concatenate([neg, pos])
    return ax.contour(lon_plot, lat_mesh, field, levels=levels, colors='black',
                      linewidths=0.8,
                      linestyles=['dashed'] * len(neg) + ['solid'] * len(pos),
                      transform=ccrs.PlateCarree(), zorder=3)

def peak_location(field, lon_plot, lat_mesh, sign='positive', min_mag=30):
    masked = np.where(field > 0, field, -np.inf) if sign == 'positive' \
        else np.where(field < 0, -field, -np.inf)
    if not np.isfinite(masked).any() or masked.max() < min_mag:
        return np.nan, np.nan
    idx = np.unravel_index(np.argmax(masked), masked.shape)
    return lon_plot[idx], lat_mesh[idx]

def figure4(season, k, var_tag):
    tag = f'{var_tag}_{THRESH_STORY}'
    ds = load_diff_fields(season, tag)
    z500 = load_z500(season, 'anomaly')
    lon_plot, lat_mesh = mesh(z500)
    cell_mask = extent_mask(z500.latitude.values,
                            np.where(z500.longitude.values > 180,
                                     z500.longitude.values - 360,
                                     z500.longitude.values))

    print('  Z500 composites per cluster')
    z = {c: compare_periods(z500, c, cell_mask=cell_mask) for c in range(k)}
    amplitude_sig(z, k)
    
    # Day counts per cluster and period, for the column headers
    dates = pd.to_datetime(z500.valid_time.values)
    in_e = (dates.year >= EARLY_YEARS[0]) & (dates.year <= EARLY_YEARS[1])
    in_l = (dates.year >= LATER_YEARS[0]) & (dates.year <= LATER_YEARS[1])
    cl = z500.cluster.values
    n_days = {c: (int(((cl == c) & in_e).sum()), int(((cl == c) & in_l).sum()))
              for c in range(k)}

    print(f'\n  {season} composite amplitude (area-weighted RMS Z500 anomaly)')
    print(f'  r > 0: early exceeds later.  {"clu":<5}{"n_e":>5}{"n_l":>5}'
          f'{"med_e":>9}{"med_l":>9}{"delta":>9}{"p_raw":>8}'
          f'{"r":>7}  sig')
    for c in range(k):
        r = z[c]
        me = np.nanmedian(r['amp_e']) if len(r['amp_e']) else np.nan
        ml = np.nanmedian(r['amp_l']) if len(r['amp_l']) else np.nan
        print(f'  C{c:<4}{r["n_yrs_e"]:>5}{r["n_yrs_l"]:>5}{me:>9.2f}'
              f'{ml:>9.2f}{ml - me:>+9.2f}{r["amp_p"]:>8.3f}'
              f'{r["amp_r"]:>7.3f}'
              f'  {"YES" if r["amp_sig"] else "no"}')
        print(f'       cells: {r["n_cells_sig"]:,} of {r["n_cells_tested"]:,} '
              f'significant')


    comp_vmax = max(max(np.abs(z[c]['comp_e']).max(),
                        np.abs(z[c]['comp_l']).max()) for c in range(k))
    diff_vmax = max(np.abs(z[c]['diff']).max() for c in range(k))
    comp_levels = np.linspace(-comp_vmax, comp_vmax, 21)
    diff_levels = np.linspace(-diff_vmax, diff_vmax, 21)

    step = comp_vmax / 10
    pos = step * np.arange(1, 11)
    neg = -pos[::-1]

    fig = plt.figure(figsize=(2.6 * k, 6.5))
    row0 = [fig.add_subplot(3, k, c + 1, projection=ccrs.PlateCarree())
            for c in range(k)]
    row1 = [fig.add_subplot(3, k, k + c + 1, projection=ccrs.PlateCarree())
            for c in range(k)]
    row2 = [fig.add_subplot(3, k, 2 * k + c + 1) for c in range(k)]

    cs0 = cs1 = None
    for c in range(k):
        d = z[c]
        ax = row0[c]
        setup_map(ax, labels=False, white_bg=False)
        cs0 = ax.contourf(lon_plot, lat_mesh, d['comp_e'], levels=comp_levels,
                          cmap='RdBu_r', extend='both',
                          transform=ccrs.PlateCarree(), zorder=1)
        later_contours(ax, lon_plot, lat_mesh, d['comp_l'], neg, pos)
        dom = ('positive' if d['comp_l'][d['comp_l'] > 0].sum()
               >= abs(d['comp_l'][d['comp_l'] < 0].sum()) else 'negative')
        for sign in (dom, 'negative' if dom == 'positive' else 'positive'):
            for field, color, zo in ((d['comp_e'], '#4A2C6E', 7),
                                     (d['comp_l'], '#F5C242', 6)):
                lo, la = peak_location(field, lon_plot, lat_mesh, sign)
                if np.isfinite(lo):
                    ax.plot(lo, la, marker='x', color=color, markersize=9,
                            markeredgewidth=2.2,
                            transform=ccrs.PlateCarree(), zorder=zo)
        ax.set_title(f'Cluster {c}\n(n={n_days[c][0]}/{n_days[c][1]})',
                     fontsize=10, fontweight='bold')
        panel_label(ax, f'{LETTERS[c]})')

        ax = row1[c]
        setup_map(ax, labels=False, white_bg=False)
        cs1 = ax.contourf(lon_plot, lat_mesh, d['diff'], levels=diff_levels,
                          cmap='coolwarm', extend='both',
                          transform=ccrs.PlateCarree(), zorder=1)
        later_contours(ax, lon_plot, lat_mesh, d['comp_l'], neg, pos)
        stipple(ax, lon_plot, lat_mesh, d['sig'], density=3, size=0.6,
                color='#555555')
        panel_label(ax, f'{LETTERS[k + c]})')

        # row 2: per-cell frequency change
        ax = row2[c]
        vals, positions, annots = [], [], []
        for t, name in enumerate(TYPES):
            key = f'c{c}_{name.replace("-", "_")}'
            dp = ds[f'diff_{key}'].values
            sig = ds[f'sig_{key}'].values.astype(bool)
            elig = np.isfinite(dp)
            v = dp[elig]
            if not len(v):
                continue
            lo, hi = np.percentile(v, [1, 99])
            vals.append(v[(v >= lo) & (v <= hi)])
            positions.append(t + 1)
            annots.append((t + 1, (sig & elig).sum() / elig.sum() * 100))

        if vals:
            bg = ax.violinplot(vals, positions=positions, showmedians=False,
                               showextrema=False, widths=0.7, bw_method=0.2)
            for pc in bg['bodies']:
                pc.set_facecolor('#f5f5f0')
                pc.set_alpha(1.0)
                pc.set_edgecolor('none')
                pc.set_zorder(1)
            parts = ax.violinplot(vals, positions=positions, showmedians=True,
                                  showextrema=False, widths=0.7, bw_method=0.2)
            for pc, name in zip(parts['bodies'], TYPES):
                pc.set_facecolor(TYPE_COLORS_DARK[name])
                pc.set_alpha(0.65)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.5)
                pc.set_zorder(2)
            if 'cmedians' in parts:
                parts['cmedians'].set_color('black')
                parts['cmedians'].set_linewidth(1.1)
                parts['cmedians'].set_zorder(3)
            ax.boxplot(vals, positions=positions, widths=0.15,
                       patch_artist=True, showfliers=False, showcaps=False,
                       whiskerprops=dict(linewidth=0),
                       medianprops=dict(color='black', linewidth=1.2),
                       boxprops=dict(facecolor='white', edgecolor='black',
                                     linewidth=1.0, alpha=0.9), zorder=5)
        ax.axhline(0, linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        for x_ann, pct in annots:      # not `pos`: that holds the contour levels
            if np.isfinite(pct) and pct >= 0.005:
                ax.text(x_ann, 0.02, f'{pct:.2f}%',
                        transform=ax.get_xaxis_transform(), ha='center',
                        va='bottom', fontsize=6, color='#444444')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
        ax.set_ylim(DP_LIM)
        ax.set_yticks([-5, -2.5, 0, 2.5, 5])
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', alpha=0.3, linewidth=0.4)
        if c:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
        panel_label(ax, f'{LETTERS[2 * k + c]})', boxed=False)

    for axs, label in zip([row0, row1, row2],
                          [f'{EARLY_YEARS[0]}-{EARLY_YEARS[1]} composite +\n '
                           f'{LATER_YEARS[0]}-{LATER_YEARS[1]} contours',
                           '\u0394Z500 + \n '
                           f'{LATER_YEARS[0]}-{LATER_YEARS[1]} contours',
                           '\u0394Frequency (%)\n']):
        axs[0].text(-0.20, 0.5, label, transform=axs[0].transAxes, fontsize=9,
                    fontweight='bold', rotation=90, va='center', ha='center')

    fig.subplots_adjust(right=0.90, top=0.90, bottom=0.06, hspace=0.04,
                        wspace=0.10)
    for cs, y_cb, label, tick_step in ((cs0, 0.68, "Z500\u2032 (m)", 50),
                                      (cs1, 0.40, "\u0394Z500\u2032 (m)", 10)):
        cax = fig.add_axes([0.91, y_cb, 0.015, 0.2])
        cb = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
        cb.set_label(label, fontsize=9)
        vmax = comp_vmax if tick_step == 50 else diff_vmax
        ticks = np.round(np.linspace(-vmax, vmax, 7) / tick_step) * tick_step
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{int(t)}' for t in ticks])
        cb.ax.tick_params(labelsize=7)

    fig.legend(handles=[mpatches.Patch(facecolor=TYPE_COLORS_DARK[n],
                                       label=TYPE_LABELS[n], alpha=0.7)
                        for n in TYPES],
               loc='center left', ncol=1, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.91, 0.20))
    fig.suptitle(f'{season.title()}: Circulation Change and Compound Extreme '
                 f'Response', fontsize=12, fontweight='bold', y=0.99)
    save_fig(fig, f'{FIGURES}/{season}_fig4_circulation_change_response')


# supplementary figures

ALL_SEASONS = ['winter', 'spring', 'summer', 'fall']
MIN_DAYS_PER_YEAR = 5          
MIN_YEARS_FOR_TREND = 10
ROLLING_REGIME = 5
YLIM_PCTL = (0.5, 99.0)

def _theil_sen_mk(years, values):
    import pymannkendall as mk
    ok = np.isfinite(values)
    if ok.sum() < MIN_YEARS_FOR_TREND:
        return dict(slope=np.nan, intercept=np.nan, p=np.nan,
                    test='insufficient_years', n_years=int(ok.sum()))
    slope, intercept = stats.theilslopes(values[ok], years[ok])[:2]
    try:
        p = float(mk.hamed_rao_modification_test(values[ok], alpha=ALPHA).p)
    except Exception:
        p = np.nan
    test = 'hamed_rao'
    if not np.isfinite(p):
        p = float(stats.kendalltau(np.arange(ok.sum()), values[ok])[1])
        test = 'kendalltau_uncorrected'
    return dict(slope=slope, intercept=intercept, p=p, test=test,
                n_years=int(ok.sum()))

def _season_year(dates, season):
    y = dates.year.values.astype(int)
    if season == 'winter':
        y = np.where(dates.month.values == 12, y + 1, y)
    return y

def _regime_duration(season, k):
    fp = f'{CSV}/{season}_pattern_persistence.csv'
    if not os.path.exists(fp):
        raise FileNotFoundError(f'{fp} not found')
    df = pd.read_csv(fp, parse_dates=['start_date', 'end_date'])
    df['plot_year'] = (df['season_year'] if season == 'winter'
                       else df['start_date'].dt.year)
    years = np.arange(int(df['plot_year'].min()), int(df['plot_year'].max()) + 1)

    out = {}
    for c in range(k):
        sub = df[df['cluster_id'] == c]
        series = sub.groupby('plot_year')['duration'].mean().reindex(years)
        per_year = sub.groupby('plot_year').size().reindex(years, fill_value=0)
        rec = dict(series=series, pooled=float(sub['duration'].mean())
                   if len(sub) else np.nan,
                   sparsity=float(np.median(per_year.values)),
                   unit='streaks/yr', n_obs=int(len(sub)))
        rec.update(_theil_sen_mk(years.astype(float), series.values))
        out[c] = rec
    return years, out

def _regime_self_transition(season, k):
    df = load_clusters(season).sort_values('date').reset_index(drop=True)
    df['season_year'] = _season_year(df['date'].dt, season)

    nxt_c, nxt_d = df['cluster'].shift(-1), df['date'].shift(-1)
    nxt_y = df['season_year'].shift(-1)
    df['valid'] = ((nxt_d - df['date']).dt.days == 1) & (nxt_y == df['season_year'])
    df['stays'] = df['valid'] & (nxt_c == df['cluster'])

    years = np.arange(int(df['season_year'].min()), int(df['season_year'].max()) + 1)
    to_days = lambda p: 1.0 / (1.0 - np.clip(p, 0.0, 0.95))

    out = {}
    for c in range(k):
        sub = df[df['cluster'] == c]
        den = sub.groupby('season_year')['valid'].sum().reindex(years, fill_value=0)
        num = sub.groupby('season_year')['stays'].sum().reindex(years, fill_value=0)
        p_ann = np.full(len(years), np.nan)
        ok = den.values >= MIN_DAYS_PER_YEAR
        p_ann[ok] = num.values[ok] / den.values[ok]
        series = pd.Series(to_days(p_ann), index=years)
        series[~np.isfinite(p_ann)] = np.nan
        total = int(den.values.sum())
        pooled_p = num.values.sum() / total if total else np.nan
        rec = dict(series=series, pooled_p=float(pooled_p),
                   pooled=float(to_days(pooled_p)),
                   sparsity=float(np.median(den.values)),
                   unit='regime-days/yr', n_obs=total)
        rec.update(_theil_sen_mk(years.astype(float), series.values))
        out[c] = rec
    return years, out

def _regime_frequency(season, k):
    df = load_clusters(season).sort_values('date').reset_index(drop=True)
    df['season_year'] = _season_year(df['date'].dt, season)
    years = np.arange(int(df['season_year'].min()), int(df['season_year'].max()) + 1)
    total = df.groupby('season_year').size().reindex(years, fill_value=0)

    out = {}
    for c in range(k):
        counts = df[df['cluster'] == c].groupby(
            'season_year').size().reindex(years, fill_value=0)
        pct = np.full(len(years), np.nan)
        ok = total.values > 0
        pct[ok] = counts.values[ok] / total.values[ok] * 100.0
        rec = dict(series=pd.Series(pct, index=years),
                   pooled=float(len(df[df['cluster'] == c]) / len(df) * 100),
                   sparsity=float(np.median(counts.values)), unit='days/yr',
                   n_obs=int((df['cluster'] == c).sum()))
        rec.update(_theil_sen_mk(years.astype(float), rec['series'].values))
        out[c] = rec
    return years, out

def _regime_trend_figure(data, seasons, ylabel, suptitle, outstem, k, label):
    allv = np.concatenate([d[c]['series'].values[
        np.isfinite(d[c]['series'].values)] for _, d in data.values()
        for c in range(k)])
    lo, hi = np.percentile(allv, YLIM_PCTL)
    pad = 0.04 * (hi - lo)
    lo, hi = max(0.0, lo - pad), hi + pad
    n_out = int(np.sum((allv < lo) | (allv > hi)))
    if n_out:
        print(f'    {label}: {n_out} of {len(allv)} annual values '
              f'({100 * n_out / len(allv):.1f}%) fall outside the y-range '
              'and are clipped off-scale')

    fig, axes = plt.subplots(len(seasons), 1,
                             figsize=(7.5, 2.5 * len(seasons) + 1),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax, season in zip(axes, seasons):
        years, d = data[season]
        for c in range(k):
            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            vals = d[c]['series'].values
            ok = np.isfinite(vals)
            ax.plot(years, vals, color=color, linewidth=1.0, alpha=0.20, zorder=2)
            smooth = np.full_like(vals, np.nan, dtype=float)
            if ok.sum() >= ROLLING_REGIME:
                smooth[ok] = pd.Series(vals[ok]).rolling(
                    ROLLING_REGIME, center=True, min_periods=1).mean().values
            ax.plot(years, smooth, color=color, linewidth=1.0, alpha=0.85,
                    zorder=3)
            if np.isfinite(d[c]['slope']) and np.isfinite(d[c]['intercept']):
                sig = np.isfinite(d[c]['p']) and d[c]['p'] < ALPHA
                ax.plot(years[ok], d[c]['slope'] * years[ok] + d[c]['intercept'],
                        color=color, linewidth=1.0,
                        linestyle='-' if sig else ':',
                        alpha=0.9 if sig else 0.5, zorder=4)
        ax.set_ylabel(f'{season.title()} {ylabel}', fontsize=10)
        ax.set_ylim(lo, hi)
        ax.set_xlim(years.min(), years.max())
        ax.grid(linewidth=0.5, alpha=0.28)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Season-year', fontsize=10)

    handles = [Line2D([], [], color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                      linewidth=1.8, label=f'Cluster {c}') for c in range(k)]
    handles += [Line2D([], [], color='0.4', linewidth=1.6, linestyle='-',
                       label=f'Trend, p<{ALPHA}'),
                Line2D([], [], color='0.4', linewidth=1.0, linestyle=':',
                       label='Trend, not significant')]
    fig.legend(handles=handles, loc='lower center', ncols=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(suptitle, fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.075, 1, 0.965])
    save_fig(fig, outstem)

def supp_persistence(season, k, var_tag):
    dur, self_t, freq = {}, {}, {}
    seasons, missing = [], []
    for s in ALL_SEASONS:
        try:
            dur[s] = _regime_duration(s, k)
        except FileNotFoundError:
            missing.append(s)
            continue
        self_t[s] = _regime_self_transition(s, k)
        freq[s] = _regime_frequency(s, k)
        seasons.append(s)
        print(f'    computed {s}')
    if missing:
        print(f'    skipped (no pattern_persistence.csv): {", ".join(missing)}')
    if not seasons:
        print('    no season has a persistence file; nothing to draw')
        return

    _regime_trend_figure(dur, seasons, 'Mean streak\nduration (days)',
                         'Circulation Regime Persistence, 1940\u20132024\n'
                         'Mean streak duration | 5-year rolling mean',
                         f'{FIGURES}/FigSx_regime_persistence', k, 'persistence')
    _regime_trend_figure(freq, seasons, 'Frequency\n(% of days)',
                         'Circulation Regime Frequency, 1940\u20132024\n'
                         'Percent of season-days | 5-year rolling mean',
                         f'{FIGURES}/FigSx_regime_frequency', k, 'frequency')

    rows = []
    for metric, data in (('self_transition', self_t), ('mean_duration', dur)):
        for s in seasons:
            _, d = data[s]
            for c in range(k):
                r = d[c]
                rows.append(dict(
                    metric=metric, season=s, cluster=c,
                    pooled_estimate_days=round(r['pooled'], 3),
                    pooled_self_transition_p=(round(r['pooled_p'], 4)
                                              if 'pooled_p' in r else ''),
                    sparsity=round(r['sparsity'], 1), sparsity_unit=r['unit'],
                    n_observations=r['n_obs'], n_years_tested=r['n_years'],
                    theil_sen_slope_per_decade=(round(r['slope'] * 10, 4)
                                                if np.isfinite(r['slope']) else ''),
                    mk_p_value=(round(r['p'], 4) if np.isfinite(r['p']) else ''),
                    mk_test=r['test']))
    tab = pd.DataFrame(rows)
    p = pd.to_numeric(tab['mk_p_value'], errors='coerce').values
    tab['significant'] = np.where(np.isfinite(p) & (p < ALPHA), 'yes', 'no')
    os.makedirs(f'{BASE}/tables', exist_ok=True)
    tab.to_csv(f'{BASE}/tables/TableSx_regime_persistence.csv', index=False)

    frows = []
    for s in seasons:
        _, d = freq[s]
        for c in range(k):
            r = d[c]
            frows.append(dict(
                season=s, cluster=c, pooled_frequency_pct=round(r['pooled'], 3),
                median_days_per_year=round(r['sparsity'], 1), n_days=r['n_obs'],
                n_years_tested=r['n_years'],
                theil_sen_slope_pct_per_decade=(round(r['slope'] * 10, 4)
                                                if np.isfinite(r['slope']) else ''),
                mk_p_value=(round(r['p'], 4) if np.isfinite(r['p']) else ''),
                mk_test=r['test']))
    ftab = pd.DataFrame(frows)
    p = pd.to_numeric(ftab['mk_p_value'], errors='coerce').values
    ftab['significant'] = np.where(np.isfinite(p) & (p < ALPHA), 'yes', 'no')
    ftab.to_csv(f'{BASE}/tables/TableSx_regime_frequency.csv', index=False)
    print(f'  wrote both supplementary tables to {BASE}/tables')

    for name, t in (('persistence', tab), ('frequency', ftab)):
        n_sig = (t['significant'] == 'yes').sum()
        print(f'  {name}: {n_sig} significant of {len(t)} tests '
              f'(p<{ALPHA}, uncorrected)')
        fb = t[t['mk_test'] != 'hamed_rao']
        if len(fb):
            print(f'    MK fallback used for {len(fb)} cell(s)')

def _type_by_cluster_grid(data, sig, wet_masked, k, col_labels, title,
                          outstem, wet_vmax, dry_vmax, cbar_label,
                          domain_sig=None):
    template = next(iter(data[0].values()))
    lon_plot, lat_mesh = mesh(template)
    wet_levels = np.linspace(-wet_vmax, wet_vmax, 17)
    dry_levels = np.linspace(-dry_vmax, dry_vmax, 17)

    fig, axes = plt.subplots(4, k, figsize=(3.75 * k, 12.5),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = np.asarray(axes).reshape(4, k)
    cs_wet = cs_dry = None
    panel = 0
    for j, name in enumerate(TYPES):
        is_wet = name in WET_TYPES
        for c in range(k):
            ax = axes[j, c]
            setup_map(ax, labels=False)
            cs = ax.contourf(lon_plot, lat_mesh, data[c][name],
                             levels=wet_levels if is_wet else dry_levels,
                             cmap='bwr', extend='both',
                             transform=ccrs.PlateCarree(), zorder=1)
            if is_wet:
                cs_wet = cs
                mark_masked(ax, lon_plot, lat_mesh, wet_masked, size=1.2)
            else:
                cs_dry = cs
            stipple(ax, lon_plot, lat_mesh, sig[c][name], density=3, size=0.7)

            flag = (' *' if domain_sig is not None
                    and domain_sig[c].get(name, False) else '')
            if j == 0:
                ax.set_title(col_labels[c], fontsize=9.5, fontweight='bold',
                             pad=6)
            if c == 0:
                ax.text(-0.14, 0.5, TYPE_LABELS[name], transform=ax.transAxes,
                        fontsize=11, fontweight='bold', rotation=90,
                        va='center', ha='center')
            panel_label(ax, f'({LETTERS[panel]}){flag}')
            panel += 1

    fig.subplots_adjust(left=0.07, right=0.90, top=0.91, bottom=0.05,
                        wspace=0.05, hspace=0.10)
    for cs, pos in ((cs_wet, 0.53), (cs_dry, 0.07)):
        cax = fig.add_axes([0.92, pos, 0.014, 0.36])
        cb = fig.colorbar(cs, cax=cax, extend='both')
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.985)
    save_fig(fig, outstem)

def supp_anomaly_grid(season, k, var_tag):
    anom = {c: {n: load_field(season, n, var_tag, c, 'anomaly') for n in TYPES}
            for c in range(k)}
    sig = {c: {n: load_field(season, n, var_tag, c,
                             'significance').values.astype(bool)
               for n in TYPES} for c in range(k)}
    land, _ = land_mask_for(anom[0][TYPES[0]])
    wet_masked = wet_mask_from_thresholds(season, land)

    clusters = load_clusters(season)
    in_ref = ((clusters['date'].dt.year >= REFERENCE_PERIOD[0])
              & (clusters['date'].dt.year <= REFERENCE_PERIOD[1]))
    labels = [f'Cluster {c} (n={int((in_ref & (clusters["cluster"] == c)).sum())})'
              for c in range(k)]

    _type_by_cluster_grid(
        anom, sig, wet_masked, k, labels,
        f'{season.title()} compound-extreme frequency anomaly by circulation '
        f'regime\nCluster days vs. rest of {REFERENCE_PERIOD[0]}\u2013'
        f'{REFERENCE_PERIOD[1]} (fixed thresholds)',
        f'{FIGURES}/{season}_figS_within_reference_anomaly',
        4.0, 10.0, 'Frequency anomaly (% points)')

def supp_diff_grid(season, k, var_tag, thresh=None):
    if thresh is None:
        for mode in ('ownclim', 'fixed'):
            supp_diff_grid(season, k, var_tag, mode)
        return
    tag = f'{var_tag}_{thresh}'
    ds = load_diff_fields(season, tag)
    tests = load_domain_tests(season, tag)

    key = lambda c, n: f'c{c}_{n.replace("-", "_")}'
    diffs = {c: {n: ds[f'diff_{key(c, n)}'] for n in TYPES} for c in range(k)}
    sig = {c: {n: ds[f'sig_{key(c, n)}'].values.astype(bool) for n in TYPES}
           for c in range(k)}
    dsig = {c: {n: bool(tests[(tests.cluster.astype(str) == str(c))
                              & (tests.type == n)]['freq_sig'].iloc[0])
                for n in TYPES} for c in range(k)}

    land, _ = land_mask_for(diffs[0][TYPES[0]])
    wet_masked = wet_mask_from_thresholds(
        season, land, None if thresh == 'fixed' else 'later')

    labels = []
    for c in range(k):
        row = tests[(tests.cluster.astype(str) == str(c))
                    & (tests.type == TYPES[0])].iloc[0]
        labels.append(f'Cluster {c}\n(n={int(row["n_days_early"])}/'
                      f'{int(row["n_days_later"])})')

    label = ('fixed 1991-2020 thresholds' if thresh == 'fixed'
             else 'period-specific (own climatology) thresholds')
    _type_by_cluster_grid(
        diffs, sig, wet_masked, k, labels,
        f'{season.title()} compound-extreme frequency change by circulation '
        f'regime\n{LATER_YEARS[0]}\u2013{LATER_YEARS[1]} minus '
        f'{EARLY_YEARS[0]}\u2013{EARLY_YEARS[1]}, {label}\n'
        '* = domain-mean change also significant',
        f'{FIGURES}/{season}_figS_diff_grid_{thresh}',
        3.0, 6.0, 'Frequency change (% points)', domain_sig=dsig)

def supp_change_ownclim(season, k, var_tag):
    figure3(season, k, var_tag, thresh='ownclim')
    
# regional time series and trends
ERA5 = '/data/lab/singh/data/ERA5_updated'
TREND_OUT = f'{BASE}/processed/trends'
TREND_VARS = {
    'precip': dict(sub='precip', pat='daily_{y}_precipsum.nc', names=('tp', 'precip'),
                   kind='precipitation', label='PRECIP (mm/day)',
                   unit='mm day$^{-1}$ yr$^{-1}$'),
    'tmax': dict(sub='tmax', pat='daily_{y}_tmax.nc', names=('t2m', 'tmax'),
                 kind='temperature', label='TMAX (\u00b0C)',
                 unit='\u00b0C yr$^{-1}$'),
}
TREND_SEASONS = [('winter', 'DJF'), ('spring', 'MAM'),
                 ('summer', 'JJA'), ('fall', 'SON')]

def _regional_daily(var):
    os.makedirs(TREND_OUT, exist_ok=True)
    fp = f'{TREND_OUT}/{var}_regional_daily.csv'
    if os.path.exists(fp):
        return pd.read_csv(fp, index_col=0, parse_dates=True)['v']

    cfg = TREND_VARS[var]
    root = f'{ERA5}/{cfg["sub"]}'
    years = sorted({int(m) for m in re.findall(
        r'daily_(\d{4})_', ' '.join(glob.glob(f'{root}/' + cfg['pat'].format(y='*'))))})
    print(f'    {var}: reducing {len(years)} years to a daily domain mean')
    parts = []
    for i, y in enumerate(years):
        if i % 10 == 0:
            print(f'      year {i + 1}/{len(years)}: {y}')
        with xr.open_dataset(f'{root}/' + cfg['pat'].format(y=y)) as ds:
            lat = ds.latitude.values
            sub = ds.sel(latitude=slice(80, 20) if lat[0] > lat[-1] else slice(20, 80),
                         longitude=slice(190, 260))
            da = sub[next(n for n in cfg['names'] if n in sub.data_vars)]
            units = str(da.attrs.get('units', '')).lower()
            s = da.weighted(np.cos(np.deg2rad(da.latitude))).mean(
                ('latitude', 'longitude')).load().to_series()
        if cfg['kind'] == 'temperature' and (units.startswith('k') or s.iloc[0] > 100):
            s = s - 273.15
        elif cfg['kind'] == 'precipitation' and (units in ('m', 'metres', 'meters')
                                                  or s.max() < 1):
            s = s * 1000.0
        parts.append(s)
    daily = pd.concat(parts).sort_index().rename('v')
    daily.to_frame().to_csv(fp)
    return daily

def _seasonal_means(daily):
    t = daily.index
    out = {'annual': daily.groupby(t.year).mean()}
    for key, months in (('spring', [3, 4, 5]), ('summer', [6, 7, 8]),
                        ('fall', [9, 10, 11])):
        sel = daily[t.month.isin(months)]
        out[key] = sel.groupby(sel.index.year).mean()
    djf = daily[t.month.isin([12, 1, 2])]
    m = djf.index.month.values
    sy = np.where(m == 12, djf.index.year.values + 1, djf.index.year.values)
    complete = sorted(set(sy[m == 12]) & set(sy[m != 12]))
    out['winter'] = djf.groupby(sy).mean().loc[complete]
    return out

def _fmt_p(p):
    return 'p<0.001' if p < 0.001 else f'p={p:.3f}'

def supp_trends(season, k, var_tag):
    fig, axes = plt.subplots(2, len(TREND_SEASONS),
                         figsize=(5 * len(TREND_SEASONS), 8))
    rows = []
    for r, var in enumerate(('precip', 'tmax')):
        cfg = TREND_VARS[var]
        means = _seasonal_means(_regional_daily(var))
        is_t = cfg['kind'] == 'temperature'
        for c, (key, label) in enumerate(TREND_SEASONS):
            ax = axes[r, c]
            s = means[key]
            yy, v = s.index.values.astype(float), s.values.astype(float)
            t = _theil_sen_mk(yy, v)
            icept = np.median(v - t['slope'] * yy)
            ax.plot(yy, v, 'o-', lw=1.6, ms=4, alpha=0.7, color='0.25')

            sig = np.isfinite(t['p']) and t['p'] < ALPHA
            if sig:
                ax.plot(yy, t['slope'] * yy + icept, 'r--', lw=2,
                        label=(f'Linear: {t["slope"]:+.4f} {cfg["unit"]}, '
                               f'{_fmt_p(t["p"])}' if is_t else 'Sen slope'))

            r2q = np.nan
            if is_t:
                x0 = yy - yy[0]
                quad = np.polyval(np.polyfit(x0, v, 2), x0)
                r2q = 1 - ((v - quad) ** 2).sum() / ((v - v.mean()) ** 2).sum()
                ax.plot(yy, quad, 'b-', lw=2, label=f'Quadratic (R\u00b2={r2q:.3f})')
                ax.legend(fontsize=8, loc='upper left')
            elif sig:
                ax.legend(fontsize=8, loc='upper right')
                ax.text(0.03, 0.97, f'{t["slope"]:+.4f} {cfg["unit"]}\n'
                                    f'{_fmt_p(t["p"])}',
                        transform=ax.transAxes, va='top', ha='left', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor='0.4', alpha=0.9))

            ax.set_title(label, fontsize=11)
            if c == 0:
                ax.set_ylabel(cfg['label'], fontsize=10)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda val, _, t_=is_t: f'{val:.1f}' if t_ else f'{val:.2f}'))
            ax.grid(True, alpha=0.3)

            rows.append(dict(variable=var, season=label, n_years=t['n_years'],
                             first_year=int(yy[0]), last_year=int(yy[-1]),
                             sen_slope_per_year=t['slope'], p=t['p'],
                             mk_test=t['test'], quadratic_r2=r2q,
                             total_change=t['slope'] * (yy[-1] - yy[0])))
    fig.supxlabel('Year', fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'{FIGURES}/figS_regional_trends_timeseries')
    summ = pd.DataFrame(rows)
    summ.to_csv(f'{TREND_OUT}/regional_trend_summary.csv', index=False)
    print('\n' + summ.to_string(index=False))
    fb = summ[summ['mk_test'] != 'hamed_rao']
    if len(fb):
        print(f'  WARNING: uncorrected Kendall fallback used for {len(fb)} panel(s)')

def supp_climrow(season, k, var_tag):
    land, _ = land_mask_for(load_field(season, TYPES[0], var_tag))
    wet_masked = wet_mask_from_thresholds(season, land)
    wet_cmap, wet_norm, wet_b = freq_cmap(FIG1_WET_VMAX, wet=True)
    dry_cmap, dry_norm, dry_b = freq_cmap(FIG1_DRY_VMAX, wet=False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    cs_wet = cs_dry = None
    for j, name in enumerate(TYPES):
        ax = axes[j]
        is_wet = name in WET_TYPES
        cmap, norm, bounds = ((wet_cmap, wet_norm, wet_b) if is_wet
                              else (dry_cmap, dry_norm, dry_b))
        setup_map(ax, labels=True, left=(j == 0), bottom=(j == 0))
        data = load_field(season, name, var_tag, kind='frequencies')
        lon_plot, lat_mesh = mesh(data)
        if is_wet:
            mark_masked(ax, lon_plot, lat_mesh, wet_masked, size=3)
            data = data.where(~xr.DataArray(wet_masked, coords=data.coords,
                                            dims=data.dims))
        cs = ax.contourf(lon_plot, lat_mesh, data, levels=bounds, cmap=cmap,
                         norm=norm, extend='max',
                         transform=ccrs.PlateCarree(), zorder=2)
        if is_wet:
            cs_wet = cs
        else:
            cs_dry = cs
        ax.text(0.0, 1.02, f'{LETTERS[j]})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='left')

    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.20,
                        wspace=0.06)
    for cs, x, bounds, label in ((cs_wet, 0.05, wet_b,
                                  'Frequency (% of days)  |  Gray = masked'),
                                 (cs_dry, 0.555, dry_b,
                                  'Frequency (% of days)')):
        cax = fig.add_axes([x, 0.13, 0.44, 0.035])
        cb = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='max')
        cb.set_ticks(bounds)
        cb.set_ticklabels(tick_labels(bounds))
        cb.set_label(label, fontsize=14)
        cb.ax.tick_params(labelsize=12)

    fig.suptitle('Climatological Frequency | 1940\u20132024', fontsize=16,
                 fontweight='bold', y=0.96)
    save_fig(fig, f'{FIGURES}/{season}_figS_climatology_row_full')

REGISTRY = {
    '1': figure1, '2': figure2, '3': figure3, '4': figure4,
    'persistence': supp_persistence,
    'anomaly_grid': supp_anomaly_grid,
    'diff_grid': supp_diff_grid,
    'change_ownclim': supp_change_ownclim,
    'trends': supp_trends,
    'climrow': supp_climrow,
}
GROUPS = {
    'main': ['1', '2', '3', '4'],
    'supp': ['persistence', 'anomaly_grid', 'diff_grid', 'change_ownclim', 'trends',
             'climrow'],
}
GROUPS['all'] = GROUPS['main'] + GROUPS['supp']

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print('  figures: ' + ', '.join(REGISTRY))
        print('  groups : ' + ', '.join(GROUPS))
        sys.exit(1)
    which, season = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    k = next((int(x) for x in args if x.isdigit()), 6)
    var = next((x for x in args if x in ('tmax', 'tmin')), 'tmax')
    override = ('raw' if '--raw' in args
                else 'detr' if '--detr' in args else None)

    todo = GROUPS.get(which, [which])
    unknown = [t for t in todo if t not in REGISTRY]
    if unknown:
        print(f'unknown figure(s): {unknown}')
        print('  choose from: ' + ', '.join(REGISTRY))
        sys.exit(1)

    os.makedirs(FIGURES, exist_ok=True)
    for name in todo:
        mode = override or TEMP_MODE.get(name, TEMP_MODE_DEFAULT)
        var_tag = tag_for(var, raw=(mode == 'raw'))
        print(f'\n{"=" * 70}\n{name}: {season}, k={k}, {var_tag}'
              + ('  (overridden)' if override else '') + f'\n{"=" * 70}')
        REGISTRY[name](season, k, var_tag)
    print('\nComplete.')


if __name__ == '__main__':
    main()
