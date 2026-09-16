# This script processes z500 geopotential height data, detrends using a quadratic fit per season to the area-weighted domain-mean seasonal series.
# It creates a climatology (1991-2020 with a 31-day moving average) and anomalies (detrended data - climatology). 
# Finally, it uses k-means clustering on seasonal data to identify the typical circulation regimes by season.
# Usage: python z500_regimes.py <season> [k]    # season: 0=winter 1=spring 2=summer 3=fall

import gc
import glob
import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PATH = '/data/lab/singh/amanda/python_codes/era_updated'
ERA5 = '/data/lab/singh/data/ERA5_updated/Z500'

LAT, LON = 'latitude', 'longitude'
LAT_MIN, LAT_MAX = 20, 80            # regional trend domain
LON_MIN, LON_MAX = 190, 260
HEMI_LAT_MIN, HEMI_LAT_MAX = 0, 90   # hemispheric trend domain
BASELINE_YEAR, TREND_DEGREE = 1940, 2
WINDOW_SIZE = 31
CLIM_START, CLIM_END = 1991, 2020
MIN_DAYS_PER_SEASON = 80             # a complete season is ~90 days
BLOCK = 500                          # days in memory during the anomaly calculations
SAVE_COMPOSITES = True               # nc/<season>_cluster<k>_composite.nc
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
SEASON_OF_MONTH = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                   6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
SEASON_KEYS = {'winter': 'DJF', 'spring': 'MAM', 'summer': 'JJA', 'fall': 'SON'}
MONTHS = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
SCALE = {'winter': 280, 'spring': 180, 'summer': 140, 'fall': 200}
SEASON_COLOR = {'DJF': '#4C72B0', 'MAM': '#55A868', 'JJA': '#C44E52', 'SON': '#DD8452'}

REG_TREND = f'{PATH}/processed/regional_trend_data.nc'
HEMI_TREND = f'{PATH}/processed/hemispheric_trend_data_hemi.nc'
DETRENDED = f'{PATH}/processed/detrended_raw_data.nc'
CLIM = f'{PATH}/processed/climatology.nc'
COMPARE_FIG = f'{PATH}/plots/seasonal_detrending_timeseries.png'
COMPARE_CSV = f'{PATH}/csv/seasonal_detrending_stats.csv'

for sub in ['processed', 'nc', 'csv', 'plots']:
    os.makedirs(f'{PATH}/{sub}', exist_ok=True)
  
def open_da(path):
    ds = xr.open_dataset(path)
    if 'z' in ds.data_vars:
        return ds['z']
    cand = [v for v in ds.data_vars if len(ds[v].dims) >= 2]
    if not cand:
        raise KeyError(f'no data variable found in {path}')
    return ds[cand[0]]

def load_z(year, lat_rng, lon_rng=None):
    sel = {LAT: slice(lat_rng[1], lat_rng[0])}
    if lon_rng is not None:
        sel[LON] = slice(lon_rng[0], lon_rng[1])
    with xr.open_dataset(f'{ERA5}/daily_{year}_Z500.nc') as full:
        ds = full.sel(sel)
        if ds.sizes.get('pressure_level') == 1:
            ds = ds.squeeze('pressure_level')
        return (ds['z'] / 9.80665).load()   

def discover_years():
    files = ' '.join(glob.glob(f'{ERA5}/daily_*_Z500.nc'))
    years = sorted(set(re.findall(r'daily_(\d{4})_Z500\.nc', files)))
    if not years:
        raise FileNotFoundError(f'no ERA5 files matched in {ERA5}')
    print(f'Found {len(years)} years: {years[0]}-{years[-1]}')
    return years

def domain_mean(years, lat_rng, lon_rng=None):
    out = []
    for i, y in enumerate(years):
        if i % 10 == 0:
            print(f'  year {i + 1}/{len(years)}: {y}')
        z = load_z(y, lat_rng, lon_rng)
        w = np.cos(np.deg2rad(z[LAT]))
        out.append(z.weighted(w).mean([LAT, LON]))
        del z
    return xr.concat(out, dim='valid_time').sortby('valid_time')

def season_frame(daily):
    df = pd.DataFrame({'time': pd.to_datetime(daily.valid_time.values),
                       'v': np.asarray(daily.values, dtype=float)})
    df['month'] = df['time'].dt.month
    df['season'] = df['month'].map(SEASON_OF_MONTH)
    df['season_year'] = np.where(df['month'] == 12, df['time'].dt.year + 1,
                                 df['time'].dt.year)
    return df.sort_values('time').reset_index(drop=True)

def fit_seasonal_trends(df):
    fits, offsets = {}, {}
    for s in SEASONS:
        g = df[df['season'] == s].groupby('season_year')['v'].agg(['mean', 'size'])
        complete = g[g['size'] >= MIN_DAYS_PER_SEASON]
        partial = list(g.index[g['size'] < MIN_DAYS_PER_SEASON].astype(int))
        x = complete.index.values.astype(float) - BASELINE_YEAR
        y = complete['mean'].values.astype(float)
        coeffs = np.polyfit(x, y, TREND_DEGREE)
        base = float(np.polyval(coeffs, 0.0))
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - np.sum((y - np.polyval(coeffs, x)) ** 2) / ss_tot) if ss_tot else np.nan

        for sy in g.index.values:
            offsets[(s, int(sy))] = float(np.polyval(coeffs, sy - BASELINE_YEAR) - base)
        fits[s] = dict(coeffs=coeffs, base=base, r2=r2, means=g['mean'].values,
                       season_years=g.index.values.astype(int))
        last = int(g.index.values[-1])
        note = f', extrapolated for incomplete {partial}' if partial else ''
        print(f'  {s}: n = {len(complete)} seasons, R2 = {r2:.4f}, '
              f'offset at {last} = {offsets[(s, last)]:+.2f} m{note}')
    return fits, offsets

def write_trend_file(path, varname, daily, fits, offsets, attrs):
    data, coords = {varname: daily}, {}
    for s in SEASONS:
        f, dim = fits[s], f'season_year_{s}'
        data[f'seasonal_means_{s}'] = ((dim,), f['means'])
        data[f'offset_{s}'] = ((dim,), np.array([offsets[(s, int(y))]
                                                 for y in f['season_years']]))
        data[f'coeffs_{s}'] = ((f'poly_{s}',), f['coeffs'])
        coords[dim] = f['season_years']
        coords[f'poly_{s}'] = np.arange(TREND_DEGREE + 1)
        attrs[f'trend_r2_{s}'] = f['r2']
        attrs[f'baseline_value_{s}'] = f['base']
    attrs['coeff_order'] = ('numpy.polyfit order: highest power first; '
                            f'x = season_year - {BASELINE_YEAR}')
    attrs['created'] = pd.Timestamp.now().isoformat()
    xr.Dataset(data, coords=coords).assign_attrs(attrs).to_netcdf(path)

def per_day_offsets(trend_file, varname):
    ds = xr.open_dataset(trend_file)
    df = season_frame(ds[varname])
    off = {}
    for s in SEASONS:
        c = np.asarray(ds[f'coeffs_{s}'].values, dtype=float)
        base = float(np.polyval(c, 0.0))
        for sy in df.loc[df['season'] == s, 'season_year'].unique():
            off[(s, int(sy))] = float(np.polyval(c, sy - BASELINE_YEAR) - base)
    ds.close()
    vals = [off[(s, int(sy))] for s, sy in zip(df['season'], df['season_year'])]
    return pd.Series(vals, index=df['time'])

def build_trends(years):
    if not os.path.exists(REG_TREND):
        daily = domain_mean(years, (LAT_MIN, LAT_MAX), (LON_MIN, LON_MAX))
        fits, offsets = fit_seasonal_trends(season_frame(daily))
        write_trend_file(REG_TREND, 'regional_means_daily', daily, fits, offsets,
                         {'method': 'seasonal_regional_mean_detrending',
                          'baseline_year': BASELINE_YEAR,
                          'trend_degree': TREND_DEGREE,
                          'trend_domain': f'lat {LAT_MIN}-{LAT_MAX}N, '
                                          f'lon {LON_MIN}-{LON_MAX}E'})
        del daily
        gc.collect()

    if not os.path.exists(HEMI_TREND):
        daily = domain_mean(years, (HEMI_LAT_MIN, HEMI_LAT_MAX))
        fits, offsets = fit_seasonal_trends(season_frame(daily))
        write_trend_file(HEMI_TREND, 'hemispheric_means_daily', daily, fits, offsets,
                         {'method': 'seasonal_hemispheric_mean_detrending',
                          'baseline_year': BASELINE_YEAR,
                          'trend_degree': TREND_DEGREE,
                          'trend_domain': f'Northern Hemisphere (lat {HEMI_LAT_MIN}-'
                                          f'{HEMI_LAT_MAX}N, all longitudes)',
                          'output_domain': f'Regional (lat {LAT_MIN}-{LAT_MAX}N, '
                                           f'lon {LON_MIN}-{LON_MAX}E)',
                          'note': 'fitted for the comparison figure only; '
                                  'not used to detrend any field'})
        del daily
        gc.collect()

def build_detrended(years):
    if os.path.exists(DETRENDED):
        return
    offsets = per_day_offsets(REG_TREND, 'regional_means_daily')
    print(f'  per-day offsets: {len(offsets)} days, '
          f'{offsets.min():+.2f} to {offsets.max():+.2f} m')
    tmp = []
    for i, y in enumerate(years):
        if i % 10 == 0:
            print(f'  year {i + 1}/{len(years)}: {y}')
        z = load_z(y, (LAT_MIN, LAT_MAX), (LON_MIN, LON_MAX))
        o = offsets.reindex(pd.to_datetime(z.valid_time.values)).values
        if np.isnan(o).any():
            raise RuntimeError(f'{int(np.isnan(o).sum())} days in {y} have no offset')
        off = xr.DataArray(o, dims=['valid_time'], coords={'valid_time': z.valid_time})
        f = f'{PATH}/processed/_tmp_detrended_{y}.nc'
        (z - off).rename('z').to_netcdf(f)
        tmp.append(f)
        del z, off
        gc.collect()
    parts = [xr.open_dataarray(f) for f in tmp]
    combined = xr.concat(parts, dim='valid_time').sortby('valid_time').rename('z')
    combined.attrs.update({
        'title': 'Detrended Z500',
        'method': 'seasonal_regional_mean_detrending',
        'baseline_year': BASELINE_YEAR,
        'trend_degree': TREND_DEGREE,
        'description': ('Z500 with a spatially uniform, season-specific offset '
                        'removed; offset from a quadratic fit to the area-weighted '
                        f'domain-mean seasonal series, referenced to {BASELINE_YEAR}'),
        'created': pd.Timestamp.now().isoformat()})
    combined.to_netcdf(DETRENDED)
    for p in parts:
        p.close()
    for f in tmp:
        os.remove(f)
    del combined, parts
    gc.collect()

def build_climatology():
    if os.path.exists(CLIM):
        return
    print(f'\nCLIMATOLOGY ({CLIM_START}-{CLIM_END}, 29 Feb excluded)')
    det = open_da(DETRENDED).sel(valid_time=slice(f'{CLIM_START}-01-01',
                                                  f'{CLIM_END}-12-31'))
    is_feb29 = ((det.valid_time.dt.month == 2) & (det.valid_time.dt.day == 29)).values
    det = det.isel(valid_time=np.flatnonzero(~is_feb29))
    md = (det.valid_time.dt.month * 100 + det.valid_time.dt.day).values
    det = det.assign_coords(monthday=('valid_time', md))
    print(f'  excluded {int(is_feb29.sum())} leap days; '
          f'{len(det.valid_time)} days in')
    day_sum = det.groupby('monthday').sum('valid_time')
    counts = pd.Series(md).value_counts().sort_index()
    if not np.array_equal(counts.index.values, day_sum.monthday.values):
        raise RuntimeError('calendar-day counts do not align with the grouped sums')
    if counts.nunique() != 1:
        raise RuntimeError('calendar days have unequal sample sizes '
                           f'({counts.min()}-{counts.max()}); the pooled mean below '
                           'assumes they are equal')
    n_per_day = int(counts.iloc[0])
    print(f'  {len(counts)} calendar days x {n_per_day} years')
    h = WINDOW_SIZE // 2
    pad = xr.concat([day_sum.isel(monthday=slice(-h, None)), day_sum,
                     day_sum.isel(monthday=slice(0, h))], dim='monthday')
    clim = (pad.rolling(monthday=WINDOW_SIZE, center=True).sum()
               .isel(monthday=slice(h, h + day_sum.sizes['monthday']))
            / float(WINDOW_SIZE * n_per_day))
    clim = clim.assign_coords(monthday=day_sum.monthday.values)
    # 29 Feb takes the smoothed 28 Feb field so downstream lookups resolve.
    clim = xr.concat([clim, clim.sel(monthday=228).assign_coords(monthday=229)],
                     dim='monthday').sortby('monthday').rename('z')
    clim.attrs.update({
        'title': 'Detrended Z500 Climatology',
        'climatology_period': f'{CLIM_START}-{CLIM_END}',
        'smoothing_window': WINDOW_SIZE,
        'smoothing_method': ('pooled centred 31-day moving average of daily values, '
                             'wrapped circularly across the year boundary'),
        'leap_day_handling': ('29 Feb excluded; monthday 229 assigned the smoothed '
                              '28 Feb field'),
        'created': pd.Timestamp.now().isoformat()})
    clim.to_netcdf(CLIM)
    del det, day_sum, pad, clim
    gc.collect()
    print(f'Saved: {CLIM}')

def linear_fit(x, y):
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - np.polyval(coeffs, x)) ** 2) / ss_tot if ss_tot else np.nan
    return float(coeffs[0]), float(np.sqrt(cov[0, 0])), float(r2), coeffs

def comparison_figure():
    if os.path.exists(COMPARE_FIG) and os.path.exists(COMPARE_CSV):
        return
    print('\nCOMPARISON FIGURE (regional vs hemispheric detrending)')
    reg, hemi = xr.open_dataset(REG_TREND), xr.open_dataset(HEMI_TREND)
    df = season_frame(reg['regional_means_daily'])
  
    def offsets_from(ds, s, yrs):
        c = np.asarray(ds[f'coeffs_{s}'].values, dtype=float)
        return np.polyval(c, yrs - BASELINE_YEAR) - float(np.polyval(c, 0.0)), c
    rows, panels = [], {}
    for s in SEASONS:
        g = df[df['season'] == s].groupby('season_year')['v'].agg(['mean', 'size'])
        # A December-only "winter mean" is not comparable to a full one.
        complete = g[g['size'] >= MIN_DAYS_PER_SEASON]
        yrs = complete.index.values.astype(float)
        raw = complete['mean'].values.astype(float)
        span = yrs.max() - yrs.min()
        reg_off, reg_c = offsets_from(reg, s, yrs)
        hemi_off, _ = offsets_from(hemi, s, yrs)
        reg_fit = np.polyval(reg_c, yrs - BASELINE_YEAR)
        reg_det, hemi_det = raw - reg_off, raw - hemi_off
        ss_tot = np.sum((raw - raw.mean()) ** 2)
        poly_r2 = float(1 - np.sum((raw - reg_fit) ** 2) / ss_tot) if ss_tot else np.nan
        slope, se, lin_r2, lin_c = linear_fit(yrs, raw)
        rows.append({
            'season': s,
            'regional_removed_m': float(reg_off[-1] - reg_off[0]),
            'hemispheric_removed_m': float(hemi_off[-1] - hemi_off[0]),
            'difference_m': float((reg_off[-1] - reg_off[0]) - (hemi_off[-1] - hemi_off[0])),
            'linear_rate_m_per_year': slope, 'linear_rate_se': se,
            'poly_r2': poly_r2, 'linear_r2': lin_r2,
            'regional_residual_m': float(linear_fit(yrs, reg_det)[0] * span),
            'hemispheric_residual_m': float(linear_fit(yrs, hemi_det)[0] * span),
            'detrended_correlation': float(np.corrcoef(reg_det, hemi_det)[0, 1]),
            'n_seasons': int(len(complete)),
            'excluded_season_years': ';'.join(
                str(d) for d in g.index[g['size'] < MIN_DAYS_PER_SEASON].astype(int))})
        panels[s] = dict(yrs=yrs, raw=raw, reg_det=reg_det, hemi_det=hemi_det,
                         reg_fit=reg_fit, lin_fit=np.polyval(lin_c, yrs))
        print(f'  {s}: regional {rows[-1]["regional_removed_m"]:+.1f} m, '
              f'hemispheric {rows[-1]["hemispheric_removed_m"]:+.1f} m, '
              f'r = {rows[-1]["detrended_correlation"]:.4f}')
    stats = pd.DataFrame(rows)
    stats.to_csv(COMPARE_CSV, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, s in zip(axes.ravel(), SEASONS):
        d, r = panels[s], stats[stats['season'] == s].iloc[0]
        ax.plot(d['yrs'], d['raw'], color='black', lw=1.8, label='Raw Data', zorder=4)
        ax.plot(d['yrs'], d['hemi_det'], color='blue', lw=1.5,
                label='Hemispheric Detrended', zorder=3)
        ax.plot(d['yrs'], d['reg_det'], color='green', lw=1.5,
                label='Regional Detrended', zorder=3)
        ax.plot(d['yrs'], d['reg_fit'], color='red', ls='--', lw=1.8,
                label='Polynomial Trend', zorder=2)
        ax.plot(d['yrs'], d['lin_fit'], color='magenta', ls=':', lw=1.8,
                label='Linear Trend', zorder=2)
        ax.text(0.015, 0.985,
                f'Climate Change Signal:\n'
                f'\u2022 Linear Rate: {r["linear_rate_m_per_year"]:+.3f} '
                f'\u00b1 {r["linear_rate_se"]:.3f} m/yr\n'
                f'\u2022 Polynomial Fit: R\u00b2 = {r["poly_r2"]:.3f}\n'
                f'\u2022 Linear Fit: R\u00b2 = {r["linear_r2"]:.3f}\n\n'
                f'Detrending:\n'
                f'\u2022 Hemispheric Removed: {r["hemispheric_removed_m"]:.1f} m\n'
                f'\u2022 Regional Removed: {r["regional_removed_m"]:.1f} m\n'
                f'\u2022 Difference: {r["difference_m"]:+.1f} m\n\n',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                fontfamily='monospace',
                bbox=dict(boxstyle='square,pad=0.5', facecolor='white',
                          edgecolor='black', linewidth=1.0))
        ax.set_title(s, fontsize=13, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Seasonal Mean Z500 (m)', fontsize=11)
        ax.grid(True, color='0.88', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    fig.suptitle('Domain-Wide Seasonal Time Series', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(COMPARE_FIG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    reg.close()
    hemi.close()

def seasonal_anomalies(key):
    name = SEASON_KEYS[key]
    det, clim = open_da(DETRENDED), open_da(CLIM)
    times = pd.to_datetime(det.valid_time.values)
    idx = np.flatnonzero(np.isin(times.month, MONTHS[name]))
    if idx.size == 0:
        raise RuntimeError(f'no {name} days found in {DETRENDED}')
    season_da = det.isel(valid_time=idx)
    t = times[idx]
    md = (t.month * 100 + t.day).values
    nlat, nlon = season_da.sizes[LAT], season_da.sizes[LON]
    X = np.empty((idx.size, nlat * nlon))
    print(f'Computing {name} anomalies for {idx.size} days '
          f'({t[0].date()} to {t[-1].date()})...')
    for i in range(0, idx.size, BLOCK):
        sl = slice(i, min(i + BLOCK, idx.size))
        blk = season_da.isel(valid_time=sl).values - clim.sel(monthday=md[sl]).values
        X[sl] = blk.reshape(blk.shape[0], -1)
    np.nan_to_num(X, copy=False)
    return X, season_da, t, (nlat, nlon)

def apply_remap(labels, key, k):
    # Renumber clusters to match an archived run, if remap_clusters.py wrote one
    path = f'{PATH}/csv/{key}_cluster_remap.csv'
    if not os.path.exists(path):
        return labels
    lut = pd.read_csv(path, comment='#').set_index('new')['reference'].to_dict()
    if sorted(lut) != list(range(k)) or sorted(lut.values()) != list(range(k)):
        raise RuntimeError(f'{path} is not a permutation of 0..{k - 1}; '
                           'it was derived for a different run')
    print(f'Applying cluster remap from {path}: {lut}')
    return np.array([lut[int(l)] for l in labels])

def plot_cluster(ax, anom, raw, title, scale):
    import cartopy.crs as ccrs
    from cartopy.io import shapereader
    from cartopy.feature import ShapelyFeature
    from matplotlib.colors import TwoSlopeNorm
    from shapely.geometry import box

    lon0, lon1 = LON_MIN - 360, LON_MAX - 360
    ax.set_extent([lon0, lon1, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    bounds = box(lon0, LAT_MIN, lon1, LAT_MAX)
    for shp, cat, color, w in [('admin_1_states_provinces_lakes', 'cultural', 'gray', 0.4),
                               ('coastline', 'physical', 'black', 0.8)]:
        reader = shapereader.Reader(shapereader.natural_earth(resolution='50m',
                                                              category=cat, name=shp))
        geoms = [g.intersection(bounds) for g in reader.geometries()
                 if g.intersects(bounds) and not g.intersection(bounds).is_empty]
        if geoms:
            ax.add_feature(ShapelyFeature(geoms, ccrs.PlateCarree(), facecolor='none',
                                          edgecolor=color, linewidth=w), zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 8}

    lon_mesh, lat_mesh = np.meshgrid(anom[LON], anom[LAT])
    lon_mesh = np.where(lon_mesh > 180, lon_mesh - 360, lon_mesh)
    levels = np.linspace(-scale, scale, 21)
    cs = ax.contourf(lon_mesh, lat_mesh, anom, levels=levels,
                     transform=ccrs.PlateCarree(), cmap='RdBu_r',
                     norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale),
                     extend='both')
    step = 75
    raw_levels = np.arange(np.floor(float(raw.min()) / step) * step,
                           np.ceil(float(raw.max()) / step) * step + step, step)
    cl = ax.contour(lon_mesh, lat_mesh, raw, levels=raw_levels, colors='lightslategray',
                    linewidths=0.6, transform=ccrs.PlateCarree(), alpha=0.8)
    ax.clabel(cl, inline=True, fontsize=7, fmt='%1.0f', colors='lightslategray')
    ax.set_title(title, fontsize=16, pad=10)
    return cs

def cluster_season(key, k):
    import cartopy.crs as ccrs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    t0 = time.time()
    name, scale = SEASON_KEYS[key], SCALE[key]
    X, season_da, times, (nlat, nlon) = seasonal_anomalies(key)
    print(f'K-means: {k} clusters on {X.shape[0]} days x {X.shape[1]} gridpoints')
    labels = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300).fit(X).labels_
    labels = apply_remap(labels, key, k)
    assign = pd.DataFrame({'date': times, 'year': times.year, 'month': times.month,
                           'season': key, 'cluster': labels})
    assign.to_csv(f'{PATH}/csv/{key}_all_cluster_assignments.csv', index=False)
    grid = lambda a: xr.DataArray(a.reshape(nlat, nlon), dims=(LAT, LON),
                                  coords={LAT: season_da[LAT], LON: season_da[LON]})
    global_mean = X.mean(axis=0)
    total_ss = np.sum((X - global_mean) ** 2)
    info, composites = [], {}
    for c in range(k):
        members = np.flatnonzero(labels == c)
        if members.size == 0:
            print(f'  cluster {c}: empty')
            continue
        print(f'  cluster {c}: {members.size} days')
        anom = grid(X[members].mean(axis=0))
        raw = season_da.isel(valid_time=members).mean('valid_time')
        composites[c] = (anom, raw)
        if SAVE_COMPOSITES:
            # 'Z500_raw' is detrended-but-not-anomalised, as in the original script
            out = xr.Dataset({'Z500_detrended_anomaly': anom, 'Z500_raw': raw})
            out.attrs['n_members'] = int(members.size)
            out.to_netcdf(f'{PATH}/nc/{key}_cluster{c}_composite.nc')
        pd.DataFrame({'Dates': times[members], 'Year': times[members].year,
                      'Month': times[members].month}).to_csv(
            f'{PATH}/csv/{key}_cluster{c}_dates.csv', index=False)
        between_ss = members.size * np.sum((X[members].mean(axis=0) - global_mean) ** 2)
        info.append({'Cluster': f'Cluster {c}', 'N_Members': int(members.size),
                     'Percentage': members.size / X.shape[0] * 100,
                     'Variance_Contribution': between_ss / total_ss, '_c': c})
    variance = pd.DataFrame(info)[['Cluster', 'Variance_Contribution',
                                   'N_Members', 'Percentage']]
    variance.to_csv(f'{PATH}/csv/{key}_variance_metrics.csv', index=False)
    # composite figure, one row
    fig = plt.figure(figsize=(4 * len(info), 5))
    cs = None
    for i, row in enumerate(info):
        c = row['_c']
        ax = fig.add_subplot(1, len(info), i + 1, projection=ccrs.PlateCarree())
        cs = plot_cluster(ax, *composites[c],
                          f'Cluster {c}: {row["N_Members"]} days '
                          f'({row["Percentage"]:.1f}%)', scale)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97], w_pad=3.0, h_pad=0.5)
    cbar = fig.colorbar(cs, cax=fig.add_axes([0.1, 0.01, 0.8, 0.04]),
                        orientation='horizontal', extend='both')
    cbar.set_label('Z500 Anomaly (m)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    ticks = np.round(np.linspace(-scale, scale, 11) / 10) * 10
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{int(t)}' for t in ticks])
    fig.suptitle(f'{key.capitalize()} Season Z500 Clusters - '
                 'Regionally Detrended Anomalies\n', fontsize=16, y=0.99)
    figname = f'{PATH}/plots/{key}_all_clusters_composite_{k}clusters.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.05,
                facecolor='white')
    plt.close()
    total_var = variance['Variance_Contribution'].sum()
    print(f'\n{"="*60}\n{name} | {k} clusters | {X.shape[0]} days | '
          f'{time.time() - t0:.1f}s\n{"="*60}')
    print(variance.to_string(index=False))
    print(f'\nTotal variance explained: {total_var:.4f} ({total_var * 100:.2f}%)')

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else '0'
    if arg == 'prep':
        years = discover_years()
        build_trends(years)
        build_detrended(years)
        build_climatology()
        comparison_figure()
        print('\nPrep complete.')
        return
    missing = [f for f in (REG_TREND, DETRENDED, CLIM) if not os.path.exists(f)]
    if missing:
        sys.exit('Missing prep output:\n  ' + '\n  '.join(missing)
                 + '\nRun: python z500_regimes.py prep')
    key = list(SEASON_KEYS)[int(arg)]
    cluster_season(key, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
if __name__ == '__main__':
    main()
