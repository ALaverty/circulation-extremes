# This script processes all the ERA5 temperature data, executing a seasonal quadratic detrending by grid cell, climatology calculations (1991-2020), and anomalies and zscores calculations.
# This is for both raw and detrended temperature
# Usage: python temperature_processing.py all tmax OR python temperature_processing.py all tmax --raw

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
warnings.filterwarnings('ignore')

BASE = '/data/lab/singh/amanda/python_codes/era_updated'
ERA5 = '/data/lab/singh/data/ERA5_updated'
OUT = f'{BASE}/processed/temperature'
RAW_OUT = f'{BASE}/processed/temperature_raw'

LAT, LON = 'latitude', 'longitude'
LAT_MIN, LAT_MAX = 20, 80
LON_MIN, LON_MAX = 190, 260

FIT_DEGREE = 2
CLIM_START, CLIM_END = 1991, 2020
WINDOW = 31
MIN_STD = 0.1                 
MIN_YEARS_FOR_FIT = 10
MIN_DAYS_PER_SEASON = 80   # a complete season is ~90 days

# DJF as Dec(y-1)+Jan(y)+Feb(y)
DJF_SHIFT_DECEMBER = True

SEASONS = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
           'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
SEASON_KEY = {'DJF': 'winter', 'MAM': 'spring', 'JJA': 'summer', 'SON': 'fall'}
MONTH_SEASON = {m: s for s, ms in SEASONS.items() for m in ms}

def paths(var, raw):
    root = RAW_OUT if raw else OUT
    tag = '_raw' if raw else ''
    return {'root': root,
            'clim': f'{root}/{var}{tag}_climatology.nc',
            'std': f'{root}/{var}{tag}_std_dayofyear.nc',
            'anom': f'{root}/{var}{tag}_{{season}}_anomalies.nc',
            'zscore': f'{root}/{var}{tag}_{{season}}_zscore.nc'}

def discover_years(var, root=None, suffix=''):
    pat = (f'{root or f"{ERA5}/{var}"}/daily_*_{var}{suffix}.nc')
    return sorted({int(m) for m in
                   re.findall(r'daily_(\d{4})_', ' '.join(glob.glob(pat)))})

def _subset(ds):
    lat = ds[LAT].values
    sl = slice(LAT_MAX, LAT_MIN) if lat[0] > lat[-1] else slice(LAT_MIN, LAT_MAX)
    return ds.sel({LAT: sl, LON: slice(LON_MIN, LON_MAX)})

def _pick(sub, var):
    for nm in ('t2m', var):
        if nm in sub.data_vars:
            return sub[nm]
    cands = [v for v in sub.data_vars if len(sub[v].dims) == 3]
    return sub[cands[0]] if cands else None

def load_year(var, year, detrended=False):
    fp = (f'{OUT}/{var}/daily_{year}_{var}_detrended.nc' if detrended
          else f'{ERA5}/{var}/daily_{year}_{var}.nc')
    if not os.path.exists(fp):
        return None
    with xr.open_dataset(fp) as ds:
        da = _pick(_subset(ds), var)
        if da is None:
            return None
        units = str(da.attrs.get('units', '')).lower()
        da = da.load()
    # Prefer the units attribute; fall back to one slice, not the whole year.
    if units.startswith('k') or float(da.isel(valid_time=0).mean()) > 100:
        da = da - 273.15
    return da

def season_labels(times):
    months, years = times.month.values, times.year.values.astype(int)
    seasons = np.array([MONTH_SEASON[m] for m in months])
    syears = years.copy()
    if DJF_SHIFT_DECEMBER:
        syears[months == 12] += 1
    return seasons, syears

def monthday_of(times):
    return (times.month * 100 + times.day).values

def window_sum(arr, window=WINDOW):
    h = window // 2
    out = np.zeros_like(arr, dtype=float)
    for k in range(-h, h + 1):
        out += np.roll(arr, -k, axis=0)
    return out

def align(field, monthday, template):
    return field.sel(monthday=xr.DataArray(
        monthday, dims='valid_time', coords={'valid_time': template.valid_time}))

def stage_detrend(var):
    out_dir = f'{OUT}/{var}'
    years = discover_years(var)
    if not years:
        print(f'  no raw files under {ERA5}/{var}/ - skipped')
        return
    if len(discover_years(var, out_dir, '_detrended')) == len(years):
        print(f'  detrended files present ({len(years)} years) - skipped')
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f'  {len(years)} years: {years[0]}-{years[-1]}')

    acc, lats, lons = {s: {} for s in SEASONS}, None, None
    for k, yr in enumerate(years):
        if k % 10 == 0:
            print(f'    means, year {k + 1}/{len(years)} ({yr})')
        da = load_year(var, yr)
        if da is None:
            continue
        if lats is None:
            lats, lons = da[LAT].values, da[LON].values
        seasons, syears = season_labels(pd.to_datetime(da.valid_time.values))
        vals = da.values
        for s in SEASONS:
            sel = seasons == s
            for sy in np.unique(syears[sel]) if sel.any() else []:
                m = sel & (syears == sy)
                tot, cnt = np.nansum(vals[m], axis=0), int(m.sum())
                if sy in acc[s]:
                    acc[s][sy][0] += tot
                    acc[s][sy][1] += cnt
                else:
                    acc[s][sy] = [tot, cnt]
        del da, vals
        gc.collect()

    fits = {}
    for s in SEASONS:
        sy_all = np.array(sorted(acc[s]), dtype=int)
        days = np.array([acc[s][y][1] for y in sy_all], dtype=int)
        complete = days >= MIN_DAYS_PER_SEASON
        sy_fit = sy_all[complete]
        if len(sy_fit) < MIN_YEARS_FOR_FIT:
            print(f'    {s}: only {len(sy_fit)} complete season-years '
                  '- not fitted')
            continue

        stack = np.array([acc[s][y][0] / acc[s][y][1] for y in sy_fit],
                         dtype=float)
        nlat, nlon = stack.shape[1:]
        x_fit = (sy_fit - sy_fit[0]).astype(float)
        y2 = stack.reshape(len(sy_fit), -1)
        coeffs = np.polyfit(x_fit, y2, FIT_DEGREE)

        fitted = np.polyval(coeffs, x_fit[:, None])
        ss_res = np.nansum((y2 - fitted) ** 2, axis=0)
        ss_tot = np.nansum((y2 - np.nanmean(y2, axis=0)) ** 2, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = 1.0 - ss_res / ss_tot

        x_all = (sy_all - sy_fit[0]).astype(float)
        rel = (np.polyval(coeffs, x_all[:, None])
               - np.polyval(coeffs, 0.0)).reshape(len(sy_all), nlat, nlon)

        fits[s] = {'idx': {int(y): i for i, y in enumerate(sy_all)},
                   'rel': rel.astype(np.float32), 'first': int(sy_fit[0])}
        dropped = [(int(y), int(d)) for y, d in zip(sy_all[~complete],
                                                    days[~complete])]
        print(f'    {s}: fitted on {len(sy_fit)}/{len(sy_all)} season-years, '
              f'R2 mean={np.nanmean(r2):.4f} median={np.nanmedian(r2):.4f} | '
              f'correction at {sy_all[-1]}: {np.nanmin(-rel[-1]):+.2f} to '
              f'{np.nanmax(-rel[-1]):+.2f} C')
        if dropped:
            print(f'      incomplete, extrapolated from the curve '
                  f'(< {MIN_DAYS_PER_SEASON} days): {dropped}')
    if not fits:
        print('    no seasons fitted - aborting')
        return

    ref = min(f['first'] for f in fits.values())
    uncorrected = 0
    for k, yr in enumerate(years):
        if k % 10 == 0:
            print(f'    applying, year {k + 1}/{len(years)} ({yr})')
        da = load_year(var, yr)
        if da is None:
            continue
        seasons, syears = season_labels(pd.to_datetime(da.valid_time.values))
        vals = da.values.copy()
        for s, f in fits.items():
            sel = seasons == s
            for sy in np.unique(syears[sel]) if sel.any() else []:
                m = sel & (syears == sy)
                i = f['idx'].get(int(sy))
                if i is None:
                    uncorrected += int(m.sum())
                else:
                    vals[m] -= f['rel'][i]
        ds = xr.DataArray(vals, coords=da.coords, dims=da.dims,
                          name='t2m').to_dataset(name='t2m')
        ds.attrs.update({
            'title': f'Season-detrended {var.upper()} {yr}',
            'method': 'seasonal_quadratic_detrending',
            'trend_degree': FIT_DEGREE,
            'djf_definition': ('Dec(y-1)+Jan(y)+Feb(y)' if DJF_SHIFT_DECEMBER
                               else 'Jan(y)+Feb(y)+Dec(y), same calendar year'),
            'reference_year': ref,
            'units': 'degrees_Celsius',
            'source': 'ERA5',
            'created': pd.Timestamp.now().isoformat()})
        ds.to_netcdf(f'{out_dir}/daily_{yr}_{var}_detrended.nc')
        del da, vals, ds
        gc.collect()
    if uncorrected:
        print(f'    WARNING: {uncorrected:,} days had no matching season-year '
              'fit and were left uncorrected')

def stage_clim(var, raw=False):
    pth = paths(var, raw)
    os.makedirs(pth['root'], exist_ok=True)
    clim_fp, std_fp = pth['clim'], pth['std']
    if os.path.exists(clim_fp) and os.path.exists(std_fp):
        print('  climatology and std present - skipped')
        return
    print(f'  accumulating {CLIM_START}-{CLIM_END} (29 Feb excluded)')

    md_list, s1, s2, cnt, lats, lons = None, None, None, None, None, None
    for yr in range(CLIM_START, CLIM_END + 1):
        da = load_year(var, yr, detrended=not raw)
        if da is None:
            print(f'    WARNING: {yr} missing')
            continue
        times = pd.to_datetime(da.valid_time.values)
        keep = ~((times.month == 2) & (times.day == 29))
        da, times = da.isel(valid_time=np.flatnonzero(keep)), times[keep]
        md = monthday_of(times)
        if md_list is None:
            lats, lons = da[LAT].values, da[LON].values
            md_list = np.array(sorted(set(md)))
            idx = {int(m): i for i, m in enumerate(md_list)}
            shape = (len(md_list), len(lats), len(lons))
            s1, s2 = np.zeros(shape), np.zeros(shape)
            cnt = np.zeros(len(md_list))
        vals = da.values.astype(np.float64)
        for j, m in enumerate(md):
            i = idx[int(m)]
            s1[i] += vals[j]
            s2[i] += vals[j] ** 2
            cnt[i] += 1
        del da, vals
        gc.collect()

    if md_list is None:
        print('    no detrended data found - run the detrend stage first')
        return
    print(f'    {len(md_list)} calendar days, {int(cnt.min())}-{int(cnt.max())} '
          'years each')
    if cnt.min() != cnt.max():
        raise RuntimeError('calendar days have unequal sample sizes; the '
                           'pooled recipe assumes they are equal')

    n_win = window_sum(cnt)
    clim = window_sum(s1) / n_win[:, None, None]

    a1 = window_sum(s1 - cnt[:, None, None] * clim)
    a2 = window_sum(s2 - 2 * clim * s1 + cnt[:, None, None] * clim ** 2)
    var_ = (a2 - a1 ** 2 / n_win[:, None, None]) / (n_win[:, None, None] - 1)
    std = np.sqrt(np.clip(var_, 0, None))
    n_low = int(np.sum(std < MIN_STD))
    if n_low:
        print(f'    {n_low:,} (monthday, cell) std values below {MIN_STD} C '
              '- raised to the floor')
        std = np.maximum(std, MIN_STD)

    def to_da(arr, name, attrs):
        da = xr.DataArray(arr.astype(np.float32),
                          coords={'monthday': md_list, LAT: lats, LON: lons},
                          dims=['monthday', LAT, LON], name=name)
        # 29 Feb takes the smoothed 28 Feb field so daily lookups resolve.
        da = xr.concat([da, da.sel(monthday=228).assign_coords(monthday=229)],
                       dim='monthday').sortby('monthday')
        da.attrs.update(attrs)
        return da

    common = {'climatology_period': f'{CLIM_START}-{CLIM_END}',
              'smoothing_window': WINDOW,
              'smoothing_method': ('pooled centred 31-day moving average, '
                                   'wrapped circularly'),
              'leap_day_handling': ('29 Feb excluded; monthday 229 assigned '
                                    'the smoothed 28 Feb field'),
              'units': 'degrees_Celsius',
              'created': pd.Timestamp.now().isoformat()}
    to_da(clim, 'clim', dict(common, title=f'Detrended {var.upper()} climatology',
                             data_source='raw' if raw else 'season-detrended')
          ).to_netcdf(clim_fp)
    to_da(std, 'std_dev', dict(common, min_std_floor=MIN_STD,
                               title=f'Day-of-year std of {var.upper()} anomalies',
                               long_name='pooled day-of-year standard deviation')
          ).to_netcdf(std_fp)
    print(f'    saved: {os.path.basename(clim_fp)}, {os.path.basename(std_fp)}')

    w = np.cos(np.deg2rad(lats))[:, None]
    print('    area-weighted mean std by season (C):')
    for s, months in SEASONS.items():
        sel = np.isin(md_list // 100, months)
        f = np.nanmean(std[sel], axis=0)
        fin = np.isfinite(f)
        print(f'      {s}: '
              f'{float(np.average(f[fin], weights=np.broadcast_to(w, f.shape)[fin])):.2f}')

def stage_anom(var, raw=False):
    pth = paths(var, raw)
    os.makedirs(pth['root'], exist_ok=True)
    outs = [pth[k].format(season=SEASON_KEY[s])
            for s in SEASONS for k in ('anom', 'zscore')]
    if all(os.path.exists(f) for f in outs):
        print('  seasonal anomaly and z-score files present - skipped')
        return
    clim_fp, std_fp = pth['clim'], pth['std']
    if not (os.path.exists(clim_fp) and os.path.exists(std_fp)):
        print('  climatology or std missing - run the clim stage first')
        return
    clim = xr.open_dataarray(clim_fp)
    std = xr.open_dataarray(std_fp)

    years = (discover_years(var) if raw
             else discover_years(var, f'{OUT}/{var}', '_detrended'))
    print(f'  {len(years)} years: {years[0]}-{years[-1]}')
    tmp = []
    for k, yr in enumerate(years):
        if k % 10 == 0:
            print(f'    year {k + 1}/{len(years)} ({yr})')
        da = load_year(var, yr, detrended=not raw)
        if da is None:
            continue
        md = monthday_of(pd.to_datetime(da.valid_time.values))
        anom = (da - align(clim, md, da)).drop_vars('monthday', errors='ignore')
        z = (anom / align(std, md, da)).drop_vars('monthday', errors='ignore')
        fp = f"{pth['root']}/_tmp_{var}_{yr}.nc"
        xr.Dataset({'anomalies': anom, 'zscore': z}).to_netcdf(fp)
        tmp.append(fp)
        del da, anom, z
        gc.collect()

    for s, months in SEASONS.items():
        key = SEASON_KEY[s]
        parts = [xr.open_dataset(f) for f in tmp]
        sel = [part.isel(valid_time=np.flatnonzero(
            np.isin(pd.to_datetime(part.valid_time.values).month, months)))
            for part in parts]
        combined = xr.concat([d for d in sel if d.sizes['valid_time']],
                             dim='valid_time').sortby('valid_time')
        times = pd.to_datetime(combined.valid_time.values)
        _, syears = season_labels(times)
        combined = combined.assign_coords(season_year=('valid_time', syears))

        for kind, unit in (('anomalies', 'degrees_Celsius'),
                           ('zscore', 'standard_deviations')):
            da = combined[kind]
            da.attrs.update({
                'temperature_type': var, 'season': s, 'months': months,
                'title': f'{s} {var.upper()} {kind}',
                'method': ('detrended minus day-of-year climatology'
                           if kind == 'anomalies' else
                           '(detrended - climatology) / day-of-year std'),
                'detrending': ('none (raw)' if raw else
                               'seasonal quadratic, per grid cell'),
                'season_year_definition': ('Dec(y-1)+Jan(y)+Feb(y)'
                                           if DJF_SHIFT_DECEMBER
                                           else 'calendar year'),
                'climatology_period': f'{CLIM_START}-{CLIM_END}',
                'units': unit, 'created': pd.Timestamp.now().isoformat()})
            da.to_netcdf(pth['anom' if kind == 'anomalies' else 'zscore']
                         .format(season=key))

        ref = (times.year >= CLIM_START) & (times.year <= CLIM_END)
        zr = combined['zscore'].isel(
            valid_time=np.flatnonzero(ref)).astype('float64')
        cell_std = zr.std('valid_time')
        print(f'    {key:7s} {s}: {combined.sizes["valid_time"]:,} days | '
              f'z in reference period: mean={float(zr.mean()):+.3f}, '
              f'per-cell std median={float(cell_std.median()):.3f} '
              f'(p25 {float(cell_std.quantile(0.25)):.3f}, '
              f'p75 {float(cell_std.quantile(0.75)):.3f})  (expect ~0 and ~1)')
        for part in parts:
            part.close()
        del parts, sel, combined
        gc.collect()

    for f in tmp:
        try:
            os.remove(f)
        except OSError:
            pass
        
STAGES = {'detrend': stage_detrend, 'clim': stage_clim, 'anom': stage_anom}
RAW_STAGES = ['clim', 'anom']   # nothing to detrend on the raw branch

def main():
    t0 = time.time()
    args = sys.argv[1:]
    raw = '--raw' in args
    args = [a for a in args if a != '--raw']
    stage = args[0] if args and args[0] in list(STAGES) + ['all'] else 'all'
    variables = [a for a in args if a in ('tmax', 'tmin')] or ['tmax', 'tmin']
    todo = (RAW_STAGES if raw else list(STAGES)) if stage == 'all' else [stage]
    if raw and 'detrend' in todo:
        print('  note: the raw branch has nothing to detrend; skipping that stage')
        todo = [t for t in todo if t != 'detrend']

    os.makedirs(OUT, exist_ok=True)
    print('=' * 66)
    print(f'  TEMPERATURE PIPELINE  |  stages: {", ".join(todo)}  |  '
          f'variables: {", ".join(variables)}  |  '
          f'{"RAW (not detrended)" if raw else "detrended"}')
    print(f'  DJF: {"Dec(y-1)+Jan(y)+Feb(y)" if DJF_SHIFT_DECEMBER else "Jan+Feb+Dec, same calendar year"}')
    print('=' * 66)

    for var in variables:
        for name in todo:
            print(f'\n{"-" * 66}\n{var.upper()} - {name}'
                  f'{" (raw)" if raw else ""}\n{"-" * 66}')
            STAGES[name](var) if name == 'detrend' else STAGES[name](var, raw)

    print(f'\n{"=" * 66}')
    print(f'  COMPLETE - {(time.time() - t0) / 60:.1f} minutes')
    print('=' * 66)


if __name__ == '__main__':
    main()
