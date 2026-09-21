# This script processes ERA5 precipitation data, creating seasonal files without detrending, calculating the climatology (1991-2020) and anomalies
# This also applies gamma thresholds (all nonzero days >0mm of reference period); 90th/10th percentile wet/dr
# This script masks out grid cells where more than 70% of days had zero precipitation and where the calculated wet threshold for extreme precipitation falls below 1mm per day
# Usage: python precipitation.py all

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
from scipy.stats import gamma
warnings.filterwarnings('ignore')

BASE = '/data/lab/singh/amanda/python_codes/era_updated'
ERA5 = '/data/lab/singh/data/ERA5_updated/precip'
RAW_OUT = f'{BASE}/processed/precipitation_raw'
THRESH_OUT = f'{BASE}/processed/precipitation_thresholds_gamma_pzero_mask'
PERIOD_OUT = f'{BASE}/processed/precipitation_thresholds_gamma_periods'

LAT, LON = 'latitude', 'longitude'
LAT_MIN, LAT_MAX = 20, 80
LON_MIN, LON_MAX = 190, 260

CLIM_START, CLIM_END = 1991, 2020
WINDOW = 31
PZERO_MASK = 0.70         
MIN_WET_THRESH = 1.0       

PERIODS = {'early': (1941, 1970), 'later': (1991, 2020)}

SEASONS = {'winter': [12, 1, 2], 'spring': [3, 4, 5],
           'summer': [6, 7, 8], 'fall': [9, 10, 11]}
SEASON_NAME = {'winter': 'DJF', 'spring': 'MAM', 'summer': 'JJA', 'fall': 'SON'}

def seasonal_file(season):
    return f'{RAW_OUT}/precip_raw_{season}.nc'

def load_year(year):
    fp = f'{ERA5}/daily_{year}_precipsum.nc'
    if not os.path.exists(fp):
        return None
    with xr.open_dataset(fp) as ds:
        lat = ds[LAT].values
        sl = slice(LAT_MAX, LAT_MIN) if lat[0] > lat[-1] else slice(LAT_MIN, LAT_MAX)
        sub = ds.sel({LAT: sl, LON: slice(LON_MIN, LON_MAX)})
        da = sub['tp'] if 'tp' in sub.data_vars else None
        if da is None:
            cands = [v for v in sub.data_vars
                     if 'precip' in v.lower() or 'tp' in v.lower()]
            if not cands:
                cands = [v for v in sub.data_vars if len(sub[v].dims) == 3]
            if not cands:
                return None
            da = sub[cands[0]]
        units = str(da.attrs.get('units', '')).lower()
        da = da.load()
    # Prefer the units attribute; fall back to one slice, not the whole year.
    if units in ('m', 'metres', 'meters') or float(da.isel(valid_time=0).max()) < 1:
        da = da * 1000.0
    return da


def window_sum(arr, window=WINDOW):
    h = window // 2
    out = np.zeros_like(arr, dtype=float)
    for k in range(-h, h + 1):
        out += np.roll(arr, -k, axis=0)
    return out


def monthday_of(times):
    return (times.month * 100 + times.day).values


def open_seasonal(season):
    da = xr.open_dataarray(seasonal_file(season))
    return da, pd.to_datetime(da.valid_time.values)

def stage_seasonal():
    if all(os.path.exists(seasonal_file(s)) for s in SEASONS):
        print('  seasonal raw files present - skipped')
        return
    os.makedirs(RAW_OUT, exist_ok=True)
    years = sorted({int(m) for m in re.findall(
        r'daily_(\d{4})_precipsum', ' '.join(glob.glob(f'{ERA5}/daily_*_precipsum.nc')))})
    if not years:
        print(f'  no files under {ERA5} - aborting')
        return
    print(f'  {len(years)} years: {years[0]}-{years[-1]}')

    tmp = []
    for k, yr in enumerate(years):
        if k % 10 == 0:
            print(f'    year {k + 1}/{len(years)} ({yr})')
        da = load_year(yr)
        if da is None:
            print(f'    WARNING: {yr} missing')
            continue
        fp = f'{RAW_OUT}/_tmp_precip_{yr}.nc'
        da.rename('tp').to_netcdf(fp)
        tmp.append(fp)
        del da
        gc.collect()

    for season, months in SEASONS.items():
        parts = [xr.open_dataarray(f) for f in tmp]
        sel = [p.isel(valid_time=np.flatnonzero(
            np.isin(pd.to_datetime(p.valid_time.values).month, months)))
            for p in parts]
        combined = xr.concat([p for p in sel if p.sizes['valid_time']],
                             dim='valid_time').sortby('valid_time')
        combined.attrs.update({
            'title': f'Raw {SEASON_NAME[season]} precipitation',
            'season': SEASON_NAME[season], 'months': months,
            'units': 'mm/day', 'detrending': 'none',
            'created': pd.Timestamp.now().isoformat()})
        combined.to_netcdf(seasonal_file(season))
        print(f'    {season:7s} {SEASON_NAME[season]}: '
              f'{combined.sizes["valid_time"]:,} days')
        for p in parts:
            p.close()
        del parts, sel, combined
        gc.collect()

    for f in tmp:
        try:
            os.remove(f)
        except OSError:
            pass

def stage_clim():
    clim_fp = f'{RAW_OUT}/precipsum_raw_climatology.nc'
    if os.path.exists(clim_fp):
        print('  climatology present - skipped')
        return
    print(f'  accumulating {CLIM_START}-{CLIM_END} (29 Feb excluded, '
          'no boundary months)')

    s1 = cnt = md_list = lats = lons = None
    for season in SEASONS:
        da, times = open_seasonal(season)
        keep = ((times.year >= CLIM_START) & (times.year <= CLIM_END)
                & ~((times.month == 2) & (times.day == 29)))
        da = da.isel(valid_time=np.flatnonzero(keep))
        md = monthday_of(times[keep])
        if md_list is None:
            lats, lons = da[LAT].values, da[LON].values
            md_list = np.array(sorted({m for mm in SEASONS.values()
                                       for m in _monthdays_for(mm)}))
            idx = {int(m): i for i, m in enumerate(md_list)}
            s1 = np.zeros((len(md_list), len(lats), len(lons)))
            cnt = np.zeros(len(md_list))
        vals = da.values.astype(np.float64)
        for j, m in enumerate(md):
            i = idx[int(m)]
            s1[i] += vals[j]
            cnt[i] += 1
        print(f'    {season:7s}: {len(md):,} days')
        da.close()
        del da, vals
        gc.collect()

    print(f'    {len(md_list)} calendar days, {int(cnt.min())}-{int(cnt.max())} '
          'years each')
    if cnt.min() != cnt.max():
        raise RuntimeError('calendar days have unequal sample sizes; the pooled '
                           'recipe assumes they are equal')

    clim = window_sum(s1) / window_sum(cnt)[:, None, None]
    da = xr.DataArray(clim.astype(np.float32),
                      coords={'monthday': md_list, LAT: lats, LON: lons},
                      dims=['monthday', LAT, LON], name='tp')
    # 29 Feb takes the smoothed 28 Feb field
    da = xr.concat([da, da.sel(monthday=228).assign_coords(monthday=229)],
                   dim='monthday').sortby('monthday')
    da.attrs.update({
        'title': 'Raw Precipitation Climatology',
        'climatology_period': f'{CLIM_START}-{CLIM_END}',
        'smoothing_window': WINDOW,
        'smoothing_method': ('pooled centred 31-day moving average, '
                             'wrapped circularly'),
        'leap_day_handling': ('29 Feb excluded; monthday 229 assigned the '
                              'smoothed 28 Feb field'),
        'data_source': 'raw_precipitation', 'detrending': 'none',
        'units': 'mm_per_day',
        'created': pd.Timestamp.now().isoformat()})
    da.to_netcdf(clim_fp)
    print(f'  saved: {clim_fp}')

def _monthdays_for(months):
    """Every monthday in the given months, on a non-leap calendar."""
    out = []
    for m in months:
        n = pd.Timestamp(2001, m, 1).days_in_month
        out += [m * 100 + d for d in range(1, n + 1)]
    return out


def stage_anom():
    outs = {s: f'{RAW_OUT}/precip_raw_{s}_anomalies_absolute_mm.nc'
            for s in SEASONS}
    if all(os.path.exists(f) for f in outs.values()):
        print('  seasonal anomaly files present - skipped')
        return
    clim_fp = f'{RAW_OUT}/precipsum_raw_climatology.nc'
    if not os.path.exists(clim_fp):
        print('  climatology missing - run the clim stage first')
        return
    clim = xr.open_dataarray(clim_fp)

    for season in SEASONS:
        da, times = open_seasonal(season)
        md = monthday_of(times)
        clim_aligned = clim.sel(monthday=xr.DataArray(
            md, dims='valid_time', coords={'valid_time': da.valid_time}))
        anom = (da - clim_aligned).drop_vars('monthday', errors='ignore')
        anom = anom.rename('precip_anomaly_mm')
        anom.attrs.update({
            'variable': 'precipitation', 'season': SEASON_NAME[season],
            'months': SEASONS[season],
            'title': f'{SEASON_NAME[season]} precipitation absolute anomalies (raw)',
            'method': 'raw - climatology (no detrending)', 'units': 'mm',
            'climatology_period': f'{CLIM_START}-{CLIM_END}',
            'note': ('negative = below-normal (dry), positive = above-normal '
                     '(wet); not detrended'),
            'created': pd.Timestamp.now().isoformat()})
        anom.to_netcdf(outs[season])
        print(f'    {season:7s} {SEASON_NAME[season]}: '
              f'{anom.sizes["valid_time"]:,} days | '
              f'mean={float(anom.mean()):+.3f} mm (expect ~0), '
              f'range {float(anom.min()):.1f} to {float(anom.max()):.1f} mm')
        da.close()
        del da, anom, clim_aligned
        gc.collect()

def fit_cell(ts):
    data = ts[~np.isnan(ts)]
    n_total = len(data)
    if n_total == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    p_zero = float(np.sum(data == 0) / n_total)
    nonzero = data[data > 0]
    if len(nonzero) < 5 or np.std(nonzero) < 0.01:
        return np.nan, np.nan, p_zero, np.nan, np.nan

    try:
        shape, _, scale = gamma.fit(nonzero, floc=0)
    except Exception:
        return np.nan, np.nan, p_zero, np.nan, np.nan
    if shape < 0.1 or shape > 100:
        return np.nan, np.nan, p_zero, np.nan, np.nan

    wet = gamma.ppf(0.90, a=shape, scale=scale)
    dry = gamma.ppf(0.10, a=shape, scale=scale)
    if not np.isfinite(wet) or wet > 500:
        wet = np.nan
    if not np.isfinite(dry):
        dry = np.nan
    return wet, dry, p_zero, float(shape), float(scale)

def wet_mask(wet_raw, pzero):
    m = np.isnan(wet_raw) | (pzero > PZERO_MASK)
    if MIN_WET_THRESH is not None:
        m |= (wet_raw < MIN_WET_THRESH)
    return m

def fit_thresholds(season, years, label):
    da, times = open_seasonal(season)
    sel = np.flatnonzero((times.year >= years[0]) & (times.year <= years[1]))
    if sel.size == 0:
        print(f'    {season}: no days in {years[0]}-{years[1]} - skipped')
        return None
    arr = da.isel(valid_time=sel).values
    lats, lons = da[LAT].values, da[LON].values
    da.close()
    nlat, nlon = len(lats), len(lons)
    print(f'    {season:7s} {label}: {arr.shape[0]:,} days, '
          f'{nlat * nlon:,} cells')

    maps = {k: np.full((nlat, nlon), np.nan) for k in
            ('wet', 'dry', 'pzero', 'shape', 'scale')}
    t0 = time.time()
    for i in range(nlat):
        if i % 40 == 0 and i:
            done = i / nlat
            print(f'      row {i}/{nlat} ({done:.0%}, '
                  f'{(time.time() - t0) / done / 60:.1f} min projected)')
        for j in range(nlon):
            (maps['wet'][i, j], maps['dry'][i, j], maps['pzero'][i, j],
             maps['shape'][i, j], maps['scale'][i, j]) = fit_cell(arr[:, i, j])

    total = nlat * nlon
    mask = wet_mask(maps['wet'], maps['pzero'])
    arid = maps['pzero'] > PZERO_MASK
    n_pz = int((arid & ~np.isnan(maps['wet'])).sum())
    n_low = 0 if MIN_WET_THRESH is None else int(
        ((maps['wet'] < MIN_WET_THRESH) & ~arid).sum())
    n_wet = int((~mask).sum())
    n_dry = int(np.isfinite(maps['dry']).sum())
    print(f'      wet usable {n_wet:,}/{total:,} ({100 * n_wet / total:.1f}%)  '
          f'masked: p_zero {n_pz:,}, sub-{MIN_WET_THRESH}mm {n_low:,}  '
          f'dry {n_dry:,}/{total:,} ({100 * n_dry / total:.1f}%)  '
          f'[{(time.time() - t0) / 60:.1f} min]')
    if n_wet:
        w = maps['wet'][~mask]
        d = maps['dry'][np.isfinite(maps['dry'])]
        print(f'      wet {w.min():.2f}-{w.max():.2f} mean {w.mean():.2f} mm/day | '
              f'dry {d.min():.4f}-{d.max():.2f} mean {d.mean():.4f} mm/day')

    coords = {LAT: lats, LON: lons}
    common = {'season': season, 'reference_period': f'{years[0]}-{years[1]}',
              'fitting_data': 'all nonzero precipitation (>0mm)',
              'n_ref_days': int(arr.shape[0]),
              'created': pd.Timestamp.now().isoformat()}
    out = {
        'wet': (maps['wet'], dict(common,
                long_name='Wet extreme threshold (90th percentile)',
                units='mm/day', percentile=90,
                method='gamma_90th_percentile_nonzero_pzero_masked',
                masking=(f'NaN where p_zero > {PZERO_MASK}'
                         + ('' if MIN_WET_THRESH is None else
                            f' or fitted threshold < {MIN_WET_THRESH} mm/day')),
                n_cells_masked=int(mask.sum()))),
        'dry': (maps['dry'], dict(common,
                long_name='Dry extreme threshold (10th percentile)',
                units='mm/day', percentile=10,
                method='gamma_10th_percentile_nonzero',
                masking='None - computed everywhere')),
        'wet_unmasked': (maps['wet'].copy(), dict(common,
                         long_name='Wet extreme threshold before masking',
                         units='mm/day', percentile=90,
                         method='gamma_90th_percentile_nonzero',
                         masking='none; NaN here means the fit itself failed')),
        'pzero': (maps['pzero'], {'long_name': 'Fraction of zero-precipitation days',
                                  'units': 'fraction'}),
        'shape': (maps['shape'], {'long_name': 'Gamma shape parameter'}),
        'scale': (maps['scale'], {'long_name': 'Gamma scale parameter',
                                  'units': 'mm/day'}),
    }
    das = {k: xr.DataArray(v, coords=coords, dims=[LAT, LON]).assign_attrs(a)
           for k, (v, a) in out.items()}
    return das, mask

def apply_wet_mask(das, mask):
    das['wet'] = das['wet'].where(~xr.DataArray(
        mask, coords=das['wet'].coords, dims=das['wet'].dims))
    return das

NAMES = {'wet': 'precip_wet_threshold_gamma', 'dry': 'precip_dry_threshold_gamma',
         'pzero': 'precip_pzero', 'shape': 'precip_gamma_shape',
         'scale': 'precip_gamma_scale',
         'wet_unmasked': 'precip_wet_threshold_gamma_unmasked'}

def stage_thresh():
    os.makedirs(THRESH_OUT, exist_ok=True)
    for season in SEASONS:
        outs = {k: f'{THRESH_OUT}/{NAMES[k]}_{season}.nc' for k in NAMES}
        if all(os.path.exists(f) for f in outs.values()):
            print(f'    {season}: present - skipped')
            continue
        res = fit_thresholds(season, (CLIM_START, CLIM_END), 'fixed')
        if res:
            das, mask = res
            for k, da in apply_wet_mask(das, mask).items():
                da.to_netcdf(outs[k])

def stage_periods():
    os.makedirs(PERIOD_OUT, exist_ok=True)
    for season in SEASONS:
        outs = {p: {k: f'{PERIOD_OUT}/{NAMES[k]}_{season}_{p}.nc' for k in NAMES}
                for p in PERIODS}
        if all(os.path.exists(f) for o in outs.values() for f in o.values()):
            print(f'    {season}: present - skipped')
            continue

        fits = {}
        for period, years in PERIODS.items():
            res = fit_thresholds(season, years, period)
            if res:
                fits[period] = res
        if len(fits) != len(PERIODS):
            print(f'    {season}: not all periods fitted - nothing written')
            continue

        union = np.logical_or.reduce([m for _, m in fits.values()])
        extra = ', '.join(f'{int((union & ~m).sum()):,} to {p}'
                          for p, (_, m) in fits.items())
        print(f'    {season}: union wet mask {int(union.sum()):,} cells (adds {extra})')
        for period, (das, _) in fits.items():
            for k, da in apply_wet_mask(das, union).items():
                da.assign_attrs(
                    period=period,
                    masking_note='wet mask is the union across periods'
                ).to_netcdf(outs[period][k])

STAGES = {'seasonal': stage_seasonal, 'clim': stage_clim, 'anom': stage_anom,
          'thresh': stage_thresh, 'periods': stage_periods}
DEFAULT = ['seasonal', 'clim', 'anom', 'thresh']

def main():
    t0 = time.time()
    args = [a for a in sys.argv[1:] if a in list(STAGES) + ['all']]
    todo = list(STAGES) if 'all' in args else (args or DEFAULT)

    print('=' * 66)
    print(f'  PRECIPITATION PIPELINE  |  stages: {", ".join(todo)}')
    print(f'  raw (not detrended) | clim {CLIM_START}-{CLIM_END} | '
          f'p_zero mask {PZERO_MASK:.0%}')
    print('=' * 66)
    for name in todo:
        print(f'\n{"-" * 66}\n{name}\n{"-" * 66}')
        STAGES[name]()
    print(f'\n{"=" * 66}')
    print(f'  COMPLETE - {(time.time() - t0) / 60:.1f} minutes')
    print('=' * 66)

if __name__ == '__main__':
    main()
