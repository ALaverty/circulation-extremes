# This script generates the compound temperature-precipitation extreme frequencies: climatological and per-cluster frequencies, cluster anomalies against the rest of the reference period, per-cell significance
# It also generates the differences between periods: 1991-2020 against 1941-1970, per-cell and domain-mean, for frequency and maximum single-day extent
# Usage: python compound_extremes.py freq <season> <k> tmax
#        python compound_extremes.py freq <season> <k> tmax --raw
#        python compound_extremes.py series <season> <k> tmax --raw
#        python compound_extremes.py events <season> <k> tmax --raw --n=25 --since=2000
#        python compound_extremes.py diff <season> <k> tmax fixed --raw
#        python compound_extremes.py diff <season> <k> tmax ownclim --raw

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.vectorized import contains
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/lab/singh/amanda/python_codes/era_updated'
OUT_NC = f'{BASE}/processed/compound_extremes'
OUT_CSV = f'{BASE}/csv'

REFERENCE_PERIOD = (1991, 2020)
EARLY_YEARS = (1941, 1970)
LATER_YEARS = (1991, 2020)

TYPES = ['wet-cold', 'wet-warm', 'dry-warm', 'dry-cold']
WET_TYPES = {'wet-cold', 'wet-warm'}
FILE_TYPE = {'wet-warm': 'warm_wet', 'wet-cold': 'cold_wet',
             'dry-warm': 'warm_dry', 'dry-cold': 'cold_dry'}

ALPHA = 0.10               # Mann-Whitney tests run at the 10% level
MIN_DAYS_PER_YEAR = 5      # cluster days for a season-year to contribute
MIN_YEARS = 10             # annual values needed in each sample
MIN_NONZERO_YEARS = 5      # non-zero years needed across both samples
MIN_DOMAIN_YEARS = 5       # years needed for a domain-mean test

SERIES_EXCLUDE_WET_MASK = True

def annual_freq(mask_da, cluster_arr, cluster, min_days=MIN_DAYS_PER_YEAR,
                years_range=None, with_other=False):
    dates = pd.to_datetime(mask_da.valid_time.values)
    in_c = np.asarray(cluster_arr == cluster)
    in_o = np.asarray(cluster_arr >= 0) & ~in_c      # unassigned days excluded
    vals = mask_da.values

    years_all = np.unique(dates.year)
    if years_range is not None:
        years_all = years_all[(years_all >= years_range[0])
                              & (years_all <= years_range[1])]

    fc, fo, years = [], [], []
    for yr in years_all:
        yr_sel = dates.year == yr
        sel_c = in_c & yr_sel
        n_c = int(sel_c.sum())
        if n_c < min_days:
            continue
        if with_other:
            sel_o = in_o & yr_sel
            n_o = int(sel_o.sum())
            if n_o < min_days:
                continue
            fo.append(vals[sel_o].sum(axis=0) / n_o * 100.0)
        fc.append(vals[sel_c].sum(axis=0) / n_c * 100.0)
        years.append(int(yr))

    years = np.asarray(years, dtype=int)
    cube = np.stack(fc).astype(np.float64) if fc else None
    if not with_other:
        return cube, years
    return cube, (np.stack(fo).astype(np.float64) if fo else None), years


def mw_cells(cube_a, cube_b, domain, alpha=ALPHA, min_years=MIN_YEARS,
             min_nonzero=MIN_NONZERO_YEARS):
    nlat, nlon = domain.shape
    pvals = np.full((nlat, nlon), np.nan)
    effect = np.full((nlat, nlon), np.nan)
    sig = np.zeros((nlat, nlon), dtype=bool)
    none = np.zeros((nlat, nlon), dtype=bool)

    if cube_a is None or cube_b is None:
        return sig, none, pvals, effect
    if cube_a.shape[0] < min_years or cube_b.shape[0] < min_years:
        return sig, none, pvals, effect

    nonzero = (cube_a > 0).sum(axis=0) + (cube_b > 0).sum(axis=0)
    testable = domain & (nonzero >= min_nonzero)

    if testable.any():
        a, b = cube_a[:, testable], cube_b[:, testable]
        res = mannwhitneyu(a, b, alternative='two-sided', axis=0)
        pvals[testable] = np.asarray(res.pvalue, dtype=float)
        effect[testable] = 1.0 - 2.0 * np.asarray(
            res.statistic, dtype=float) / (a.shape[0] * b.shape[0])

    tested = np.isfinite(pvals) & domain
    effect = np.where(tested, effect, np.nan)
    if tested.sum() > 0:
        rejected, _, _, _ = multipletests(pvals[tested], alpha=alpha,
                                          method='fdr_bh')
        sig[tested] = rejected
    return sig, tested, pvals, effect

def domain_mean_series(cube, tested, weights):
    if cube is None or not tested.any():
        return np.array([])
    w = weights[tested]
    vals = cube[:, tested]
    good = np.isfinite(vals)
    out = np.full(vals.shape[0], np.nan)
    for k in range(vals.shape[0]):
        if good[k].any():
            out[k] = np.average(vals[k][good[k]], weights=w[good[k]])
    return out[np.isfinite(out)]

def build_land_mask(lats, lons):
    shp = shpreader.natural_earth(resolution='50m', category='physical',
                                  name='land')
    geoms = unary_union(list(shpreader.Reader(shp).geometries()))
    lon_mesh, lat_mesh = np.meshgrid(np.where(lons > 180, lons - 360, lons), lats)
    mask = contains(geoms, lon_mesh.ravel(),
                    lat_mesh.ravel()).reshape(lon_mesh.shape)
    print(f'  land mask: {mask.sum():,} of {mask.size:,} cells')
    return mask

def grid_cell_areas(lats, lons):
    R = 6371.0
    dlat = np.abs(lats[1] - lats[0]) * np.pi / 180
    dlon = np.abs(lons[1] - lons[0]) * np.pi / 180
    areas = np.zeros((len(lats), len(lons)))
    for i, lat in enumerate(lats):
        r = lat * np.pi / 180
        areas[i, :] = R**2 * dlon * (np.sin(r + dlat / 2) - np.sin(r - dlat / 2))
    return areas

def temp_file(base, var, season, raw):
    return (f'{base}/processed/temperature_raw/{var}_raw_{season}_anomalies.nc'
            if raw else
            f'{base}/processed/temperature/{var}_{season}_anomalies.nc')

def open_precip(path):
    ds = xr.open_dataset(path)
    for name in ('tp', 'precip', 'precipitation'):
        if name in ds.data_vars:
            da = ds[name]
            break
    else:
        da = ds[[v for v in ds.data_vars if len(ds[v].dims) == 3][0]]
    units = str(da.attrs.get('units', '')).lower()
    if units in ('m', 'metres', 'meters') or float(da.isel(valid_time=0).max()) < 1:
        da = da * 1000.0
    return da


class CompoundExtremes:

    def __init__(self, season, num_clusters, base=BASE, var='tmax', raw=False):
        self.season = season
        self.num_clusters = num_clusters
        self.base = base
        self.var = var
        self.raw = raw
        self.significance = None

    def load_data(self):
        tf = temp_file(self.base, self.var, self.season, self.raw)
        print(f'  temperature ({"raw" if self.raw else "detrended"}): {tf}')
        self.temp_anom = xr.open_dataarray(tf)

        pf = f'{self.base}/processed/precipitation_raw/precip_raw_{self.season}.nc'
        print(f'  precipitation: {pf}')
        self.precip = open_precip(pf)

        cdf = pd.read_csv(
            f'{self.base}/csv/{self.season}_all_cluster_assignments.csv')
        cdf['date'] = pd.to_datetime(cdf['date'])
        lookup = cdf.drop_duplicates('date').set_index('date')['cluster']
        dates = pd.DatetimeIndex(pd.to_datetime(self.temp_anom.valid_time.values))
        self.cluster_arr = lookup.reindex(dates).fillna(-1).astype(int).values
        n_un = int((self.cluster_arr < 0).sum())
        if n_un:
            print(f'  WARNING: {n_un:,} days have no cluster assignment')

        self.temp_anom = self.temp_anom.assign_coords(
            cluster=('valid_time', self.cluster_arr))
        self.precip = self.precip.assign_coords(
            cluster=('valid_time', self.cluster_arr))

        self.land_mask = build_land_mask(self.temp_anom.latitude.values,
                                         self.temp_anom.longitude.values)
        self.land_da = xr.DataArray(
            self.land_mask,
            coords={'latitude': self.temp_anom.latitude,
                    'longitude': self.temp_anom.longitude},
            dims=['latitude', 'longitude'])
        print(f'  {len(self.temp_anom.valid_time):,} days, '
              f'{len(self.temp_anom.latitude)} x '
              f'{len(self.temp_anom.longitude)} grid')

    def _thresholds(self, temp, period_label=None):
        warm = temp.quantile(0.90, dim='valid_time')
        cold = temp.quantile(0.10, dim='valid_time')
        if period_label is None:
            d = f'{self.base}/processed/precipitation_thresholds_gamma_pzero_mask'
            suffix = ''
        else:
            d = f'{self.base}/processed/precipitation_thresholds_gamma_periods'
            suffix = f'_{period_label}'
        wet = xr.open_dataarray(
            f'{d}/precip_wet_threshold_gamma_{self.season}{suffix}.nc')
        dry = xr.open_dataarray(
            f'{d}/precip_dry_threshold_gamma_{self.season}{suffix}.nc')
        return warm, cold, wet, dry, np.isnan(wet.values) & self.land_mask

    @staticmethod
    def _combine(temp, precip, warm, cold, wet, dry):
        w, c = temp > warm, temp < cold
        is_wet = precip > wet                       # NaN threshold -> False
        is_dry = ((dry == 0) & (precip == 0)) | ((dry > 0) & (precip < dry))
        return {'wet-cold': is_wet & c, 'wet-warm': is_wet & w,
                'dry-warm': is_dry & w, 'dry-cold': is_dry & c}

    def identify_extremes(self):
        dates = pd.to_datetime(self.temp_anom.valid_time.values)
        ref = ((dates.year >= REFERENCE_PERIOD[0])
               & (dates.year <= REFERENCE_PERIOD[1]))
        print(f'  reference period: {int(ref.sum()):,} days')
        warm, cold, wet, dry, wet_masked = self._thresholds(
            self.temp_anom.isel(valid_time=ref))
        self.wet_masked = wet_masked
        print(f'  wet threshold masked: {int(wet_masked.sum()):,} of '
              f'{int(self.land_mask.sum()):,} land cells')
        self.masks = self._combine(self.temp_anom, self.precip,
                                   warm, cold, wet, dry)
        for name, m in self.masks.items():
            n_days = int((m.sum(dim=['latitude', 'longitude']) > 0).sum())
            print(f'    {name:9s}: {int(m.sum()):,} cell-days, '
                  f'{n_days:,} days with at least one cell')

    def period_masks(self, period_years, period_label):
        """Extremes defined from one period's own thresholds."""
        dates = pd.to_datetime(self.temp_anom.valid_time.values)
        sel = ((dates.year >= period_years[0]) & (dates.year <= period_years[1]))
        temp_p = self.temp_anom.isel(valid_time=sel)
        warm, cold, wet, dry, wet_masked = self._thresholds(temp_p, period_label)
        masks = self._combine(temp_p, self.precip.isel(valid_time=sel),
                              warm, cold, wet, dry)
        return masks, self.cluster_arr[sel], int(sel.sum()), wet_masked

    def _weighted_mean(self, da):
        w = np.cos(np.deg2rad(da.latitude))
        return float(da.weighted(w).mean(('latitude', 'longitude'), skipna=True))

    def _freq_over(self, name, idx, n=None):
        """Frequency (%) of one type over the given day indices, land only."""
        n = len(idx) if n is None else n
        if not n:
            return self.masks[name].isel(valid_time=[]).sum(
                dim='valid_time').where(self.land_da) * np.nan
        return (self.masks[name].isel(valid_time=idx).sum(dim='valid_time')
                / n * 100).where(self.land_da)

    def calculate_frequencies(self):
        n_all = len(self.temp_anom.valid_time)
        dates = pd.to_datetime(self.temp_anom.valid_time.values)
        ref = ((dates.year >= REFERENCE_PERIOD[0])
               & (dates.year <= REFERENCE_PERIOD[1]))
        ref_idx = np.flatnonzero(ref)

        self.clim_freq, self.ref_freq = {}, {}
        print(f'\n  full record ({n_all:,} days) and reference '
              f'({len(ref_idx):,} days):')
        for name in TYPES:
            self.clim_freq[name] = self._freq_over(name, np.arange(n_all))
            self.ref_freq[name] = self._freq_over(name, ref_idx)
            print(f'    {name:9s}: full {self._weighted_mean(self.clim_freq[name]):5.2f}%'
                  f'   ref {self._weighted_mean(self.ref_freq[name]):5.2f}%')

        print('\n  per-cluster frequency (full record):')
        self.cluster_freq = {}
        for c in range(self.num_clusters):
            idx = np.flatnonzero(self.cluster_arr == c)
            self.cluster_freq[c] = {n: self._freq_over(n, idx) for n in TYPES}
            print(f'    cluster {c}: {len(idx):,} days  ' + '  '.join(
                f'{n[:1]}{n.split("-")[1][:1]}='
                f'{self._weighted_mean(self.cluster_freq[c][n]):.2f}%'
                for n in TYPES))

        self.noncluster_freq = {}
        for c in range(self.num_clusters):
            other = np.flatnonzero((self.cluster_arr != c)
                                   & (self.cluster_arr >= 0))
            self.noncluster_freq[c] = {n: self._freq_over(n, other)
                                       for n in TYPES}

        self.ref_cluster_freq, self.ref_anomaly = {}, {}
        for c in range(self.num_clusters):
            sel = np.flatnonzero(ref & (self.cluster_arr == c))
            rest = np.flatnonzero(ref & (self.cluster_arr != c)
                                  & (self.cluster_arr >= 0))
            self.ref_cluster_freq[c] = {n: self._freq_over(n, sel) for n in TYPES}
            self.ref_anomaly[c] = {n: self.ref_cluster_freq[c][n]
                                   - self._freq_over(n, rest) for n in TYPES}

    def get_significance(self, alpha=ALPHA):
        if self.significance is None:
            self.significance = self._significance(alpha)
        return self.significance

    def _significance(self, alpha=ALPHA):
        sig_all, self.effect_maps = {}, {}
        print(f'\n  per-cell Mann-Whitney, cluster vs rest of '
              f'{REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]}:')
        for c in range(self.num_clusters):
            sig_all[c], self.effect_maps[c] = {}, {}
            n_s = n_t = 0
            for name in TYPES:
                m = self.masks[name].transpose('valid_time', 'latitude',
                                               'longitude')
                cube_c, cube_o, years = annual_freq(
                    m, self.cluster_arr, c, MIN_DAYS_PER_YEAR,
                    REFERENCE_PERIOD, with_other=True)
                # (rest, cluster) so positive effect means cluster exceeds rest
                sig, tested, _, eff = mw_cells(cube_o, cube_c, self.land_mask,
                                               alpha=alpha)
                sig_all[c][name], self.effect_maps[c][name] = sig, eff
                n_s += int(sig.sum())
                n_t += int(tested.sum())
            print(f'    cluster {c}: {len(years)} years, {n_s:,} significant '
                  f'of {n_t:,} tested (4 types)')
        return sig_all

    def regime_associations(self, alpha=ALPHA):
        sig = self.get_significance(alpha)
        w2d = np.tile(np.cos(np.deg2rad(self.temp_anom.latitude.values))[:, None],
                      (1, len(self.temp_anom.longitude)))
        rows = []
        for c in range(self.num_clusters):
            for name in TYPES:
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = (self.ref_anomaly[c][name] / self.ref_freq[name] * 100)
                rel = rel.where(self.land_da).values.copy()
                rel[~sig[c][name]] = np.nan
                ok = np.isfinite(rel)
                rows.append(dict(
                    cluster=c, type=name,
                    pct_significant_area=(sig[c][name] & self.land_mask).sum()
                    / self.land_mask.sum() * 100,
                    sig_weighted_mean_relative_anomaly_pct=(
                        np.nansum(rel[ok] * w2d[ok]) / np.nansum(w2d[ok])
                        if ok.any() else np.nan)))
        df = pd.DataFrame(rows)
        df['rank_score'] = (df['pct_significant_area'] / 100
                            * df['sig_weighted_mean_relative_anomaly_pct'].abs())
        return df.sort_values('rank_score', ascending=False)

    def save(self):
        os.makedirs(OUT_NC, exist_ok=True)
        os.makedirs(OUT_CSV, exist_ok=True)
        tag = f'{self.var}_raw' if self.raw else self.var
        sig = self.get_significance()

        for name, short in FILE_TYPE.items():
            self.clim_freq[name].to_netcdf(
                f'{OUT_NC}/{self.season}_{short}_{tag}_frequencies.nc')
            self.ref_freq[name].to_netcdf(
                f'{OUT_NC}/{self.season}_{short}_{tag}_reference_frequencies.nc')
            for c in range(self.num_clusters):
                stem = f'{OUT_NC}/{self.season}_{short}_{tag}_cluster{c}'
                self.cluster_freq[c][name].to_netcdf(f'{stem}_frequencies.nc')
                self.noncluster_freq[c][name].to_netcdf(
                    f'{stem}_noncluster_frequencies.nc')
                self.ref_anomaly[c][name].to_netcdf(f'{stem}_anomaly.nc')
                xr.DataArray(sig[c][name], coords=self.land_da.coords,
                             dims=self.land_da.dims).to_netcdf(
                                 f'{stem}_significance.nc')
                xr.DataArray(
                    self.effect_maps[c][name], coords=self.land_da.coords,
                    dims=self.land_da.dims
                ).assign_attrs(description=(
                    'rank-biserial; positive = cluster exceeds the rest of the '
                    'reference period')).to_netcdf(f'{stem}_effect.nc')
        print(f'  wrote frequency, anomaly, significance and effect fields to '
              f'{OUT_NC}')

        rows = []
        for c in range(self.num_clusters):
            idx = np.flatnonzero(self.cluster_arr == c)
            row = {'cluster': c, 'n_days': len(idx)}
            row.update({n: int(self.masks[n].isel(valid_time=idx).sum())
                        for n in TYPES})
            rows.append(row)
        stats = pd.DataFrame(rows)
        stats.to_csv(f'{OUT_CSV}/{self.season}_compound_stats_{tag}.csv',
                     index=False)
        print('\n' + stats.to_string(index=False))

        assoc = self.regime_associations()
        assoc.to_csv(f'{OUT_CSV}/{self.season}_regime_associations_{tag}.csv',
                     index=False)
        print('\n  top regime-type associations:')
        print(assoc.head(8).to_string(index=False))

def stage_series(a, areas, tag):
    os.makedirs(OUT_NC, exist_ok=True)
    land = a.land_mask
    nlat, nlon = land.shape
    weights = np.broadcast_to(
        np.cos(np.deg2rad(a.temp_anom.latitude.values))[:, None], (nlat, nlon))
    zeros = np.zeros(len(a.temp_anom.valid_time), dtype=int)

    dates = pd.to_datetime(a.temp_anom.valid_time.values)
    rows = []
    for name in TYPES:
        mask = a.masks[name].transpose('valid_time', 'latitude', 'longitude')
        dom = (valid_domain(land, a.wet_masked & land, name)
               if SERIES_EXCLUDE_WET_MASK else land)
        cube, years = annual_freq(mask, zeros, 0, MIN_DAYS_PER_YEAR)
        freq = domain_mean_series(cube, dom, weights)
        ext = annual_max_extent(mask, zeros, 0, dom, areas)

        daily = (mask.values[:, dom].astype(np.float32) * areas[dom]).sum(axis=1)
        ext_date = {}
        for y in years:
            idx = np.flatnonzero(dates.year == y)
            ext_date[y] = dates[idx[np.argmax(daily[idx])]].strftime('%Y-%m-%d')

        if not (len(years) == len(freq) == len(ext)):
            raise RuntimeError(f'{name}: series lengths disagree '
                               f'({len(years)}/{len(freq)}/{len(ext)})')
        rows += [dict(year=int(y), type=name, frequency=float(f),
                      extent=float(e), extent_date=ext_date[y])
                 for y, f, e in zip(years, freq, ext)]
        print(f'    {name:9s}: {len(years)} years, frequency '
              f'{freq.mean():.2f}% mean, extent {ext.mean():.0f} x10^3 km2 '
              f'mean, record {ext.max():.0f} on '
              f'{ext_date[years[int(np.argmax(ext))]]}')

    fp = f'{OUT_NC}/{a.season}_{tag}_fullrecord_annual.csv'
    pd.DataFrame(rows).to_csv(fp, index=False)
    print(f'  wrote {fp}')


def stage_events(a, areas, tag, n_events=10, year_min=2000):
    os.makedirs(OUT_CSV, exist_ok=True)
    land = a.land_mask
    dates = pd.to_datetime(a.temp_anom.valid_time.values)
    sel = np.flatnonzero(dates.year >= year_min)
    if sel.size == 0:
        print(f'  no days at or after {year_min}')
        return
    print(f'  {len(sel):,} days from {year_min} onward')

    streaks = None
    fp_streak = f'{OUT_CSV}/{a.season}_pattern_persistence.csv'
    if os.path.exists(fp_streak):
        streaks = pd.read_csv(fp_streak, parse_dates=['start_date', 'end_date'])
    else:
        print(f'  note: {os.path.basename(fp_streak)} not found; '
              'streak columns left empty')

    rows = []
    for name in TYPES:
        mask = a.masks[name].transpose('valid_time', 'latitude', 'longitude')
        dom = valid_domain(land, a.wet_masked & land, name)
        daily = (mask.values[sel][:, dom].astype(np.float32)
                 * areas[dom]).sum(axis=1)
        order = np.argsort(daily)[-n_events:][::-1]

        print(f'    {name}:')
        for rank, i in enumerate(order, 1):
            d = dates[sel][i]
            cluster = int(a.cluster_arr[sel][i])
            dur = np.nan
            if streaks is not None:
                m = streaks[(streaks['start_date'] <= d)
                            & (streaks['end_date'] >= d)]
                if len(m):
                    dur = int(m.iloc[0]['duration'])
            rows.append(dict(season=a.season, type=name, rank=rank,
                             date=d.strftime('%Y-%m-%d'),
                             extent_km2=round(float(daily[i]), 0),
                             cluster=cluster if cluster >= 0 else np.nan,
                             streak_duration=dur))
            if rank <= 3:
                print(f'      {rank}. {d:%Y-%m-%d}  {daily[i]:>12,.0f} km2  '
                      f'cluster {cluster}'
                      + (f'  streak {dur} d' if np.isfinite(dur) else ''))

    fp = f'{OUT_CSV}/{a.season}_{tag}_top_events.csv'
    pd.DataFrame(rows).to_csv(fp, index=False)
    print(f'  wrote {fp}')

def annual_max_extent(mask_da, cluster_arr, cluster, domain, areas,
                      min_days=MIN_DAYS_PER_YEAR):
    dates = pd.to_datetime(mask_da.valid_time.values)
    in_c = np.asarray(cluster_arr == cluster)
    vals = mask_da.values
    areas_dom = areas[domain]

    out = []
    for yr in np.unique(dates.year):
        sel = in_c & (dates.year == yr)
        if int(sel.sum()) < min_days:
            continue
        daily = (vals[sel][:, domain].astype(float) * areas_dom).sum(axis=1)
        out.append(float(daily.max()) / 1000.0)
    return np.asarray(out)

def valid_domain(land, wet_excl, name):
    return land & ~wet_excl if name in WET_TYPES else land

def build_masks(a, use_fixed):
    land = a.land_mask
    if use_fixed:
        a.identify_extremes()
        dates = pd.to_datetime(a.temp_anom.valid_time.values)
        sel_e = np.asarray((dates.year >= EARLY_YEARS[0])
                           & (dates.year <= EARLY_YEARS[1]))
        sel_l = np.asarray((dates.year >= LATER_YEARS[0])
                           & (dates.year <= LATER_YEARS[1]))
        me = {n: a.masks[n].isel(valid_time=sel_e) for n in TYPES}
        ml = {n: a.masks[n].isel(valid_time=sel_l) for n in TYPES}
        excl = a.wet_masked & land
        print(f'  wet exclusion (fixed): {int(excl.sum()):,} of '
              f'{int(land.sum()):,} land cells')
        print(f'  early {int(sel_e.sum()):,} days, later {int(sel_l.sum()):,} days')
        return (me, a.cluster_arr[sel_e], ml, a.cluster_arr[sel_l],
                excl, excl, excl, 'fixed 1991-2020 thresholds')

    me, ce, n_e, excl_e = a.period_masks(EARLY_YEARS, 'early')
    ml, cl, n_l, excl_l = a.period_masks(LATER_YEARS, 'later')
    excl_e, excl_l = excl_e & land, excl_l & land
    excl_test = excl_e | excl_l
    print(f'  wet exclusion: early {int(excl_e.sum()):,}, '
          f'later {int(excl_l.sum()):,}, union {int(excl_test.sum()):,} of '
          f'{int(land.sum()):,} land cells')
    print(f'  early {n_e:,} days, later {n_l:,} days')
    return (me, ce, ml, cl, excl_test, excl_e, excl_l,
            'period-specific (own-climatology) thresholds')

def domain_mean_test(results, num_clusters, early_key, later_key, prefix,
                     alpha=ALPHA):
    for name in TYPES:
        for c in results:
            r = results[c][name]
            e, l = r[early_key], r[later_key]
            p, eff = np.nan, np.nan
            if len(e) >= MIN_DOMAIN_YEARS and len(l) >= MIN_DOMAIN_YEARS:
                try:
                    res = mannwhitneyu(e, l, alternative='two-sided')
                    p = float(res.pvalue)
                    eff = 1.0 - 2.0 * res.statistic / (len(e) * len(l))
                except Exception:
                    pass
            r[f'{prefix}_p'] = p
            r[f'{prefix}_sig'] = bool(np.isfinite(p) and p < alpha)
            r[f'{prefix}_effect_r'] = eff
    return results

def compute_all(a, num_clusters, me, ce, ml, cl, excl_test, excl_e, excl_l,
                areas):
    land = a.land_mask
    nlat, nlon = land.shape
    weights = np.broadcast_to(
        np.cos(np.deg2rad(a.temp_anom.latitude.values))[:, None], (nlat, nlon))

    results = {}
    for c in list(range(num_clusters)) + ['all']:
        results[c] = {}
        if c == 'all':
            ce_c, cl_c, cid = np.zeros_like(ce), np.zeros_like(cl), 0
        else:
            ce_c, cl_c, cid = ce, cl, c
        in_e, in_l = (ce_c == cid), (cl_c == cid)
        n_e, n_l = int(in_e.sum()), int(in_l.sum())
        print(f'\n  cluster {c}: {n_e}/{n_l} days')

        for name in TYPES:
            mask_e = me[name].transpose('valid_time', 'latitude', 'longitude')
            mask_l = ml[name].transpose('valid_time', 'latitude', 'longitude')
            k_e = mask_e.isel(valid_time=in_e).sum(dim='valid_time').values
            k_l = mask_l.isel(valid_time=in_l).sum(dim='valid_time').values
            f_e = k_e / n_e * 100 if n_e else np.full((nlat, nlon), np.nan)
            f_l = k_l / n_l * 100 if n_l else np.full((nlat, nlon), np.nan)

            dom_test = valid_domain(land, excl_test, name)
            dom_e = valid_domain(land, excl_e, name)
            dom_l = valid_domain(land, excl_l, name)

            cube_e, yrs_e = annual_freq(mask_e, ce_c, cid, MIN_DAYS_PER_YEAR)
            cube_l, yrs_l = annual_freq(mask_l, cl_c, cid, MIN_DAYS_PER_YEAR)

            sig, tested, pvals, effect = mw_cells(cube_e, cube_l, dom_test)

            results[c][name] = dict(
                diff=np.where(dom_test, f_l - f_e, np.nan),
                sig=sig, tested=tested, domain=dom_test, pvals=pvals,
                effect_map=effect, n_e=n_e, n_l=n_l,
                annual_early=domain_mean_series(cube_e, dom_e, weights),
                annual_later=domain_mean_series(cube_l, dom_l, weights),
                extent_early=annual_max_extent(mask_e, ce_c, cid, dom_e, areas),
                extent_later=annual_max_extent(mask_l, cl_c, cid, dom_l, areas),
                n_cube_yrs_e=len(yrs_e), n_cube_yrs_l=len(yrs_l))

            n_dom, n_t = int(dom_test.sum()), int(tested.sum())
            print(f'    {name:9s}: {int(sig.sum()):,} sig of {n_t:,} tested '
                  f'({int(sig.sum()) / n_dom * 100 if n_dom else np.nan:.1f}% '
                  f'of {n_dom:,} in-domain; {n_dom - n_t:,} gated out)')
    return results

def _summary_row(c, name, r):
    med = lambda x: float(np.median(x)) if len(x) else np.nan
    n_t, n_d = int(r['tested'].sum()), int(r['domain'].sum())
    n_s = int(r['sig'].sum())
    row = dict(cluster=c, type=name, n_days_early=r['n_e'], n_days_later=r['n_l'],
               n_years_early=r['n_cube_yrs_e'], n_years_later=r['n_cube_yrs_l'],
               n_domain=n_d, n_tested=n_t, n_sig=n_s,
               pct_sig_of_tested=n_s / n_t * 100 if n_t else np.nan,
               pct_sig_of_domain=n_s / n_d * 100 if n_d else np.nan,
               median_cell_effect_r=float(np.nanmedian(r['effect_map']))
               if np.isfinite(r['effect_map']).any() else np.nan)
    for m, ek, lk in (('freq', 'annual_early', 'annual_later'),
                      ('ext', 'extent_early', 'extent_later')):
        row[f'{m}_n_years_early'] = len(r[ek])
        row[f'{m}_n_years_later'] = len(r[lk])
        row[f'{m}_median_early'] = med(r[ek])
        row[f'{m}_median_later'] = med(r[lk])
        row[f'{m}_p'] = r[f'{m}_p']
        row[f'{m}_sig'] = r[f'{m}_sig']
        row[f'{m}_effect_r'] = r[f'{m}_effect_r']
    return row

def save_diff(a, results, num_clusters, season, tag, thresh_label):
    os.makedirs(OUT_NC, exist_ok=True)
    lats = a.temp_anom.latitude.values
    lons = a.temp_anom.longitude.values

    dv = {}
    for c in results:
        for name in TYPES:
            r = results[c][name]
            key = ('all' if c == 'all' else f'c{c}') + f'_{name.replace("-", "_")}'
            dv[f'diff_{key}'] = (('latitude', 'longitude'), r['diff'])
            dv[f'sig_{key}'] = (('latitude', 'longitude'), r['sig'].astype(np.int8))
            dv[f'tested_{key}'] = (('latitude', 'longitude'),
                                   r['tested'].astype(np.int8))
            dv[f'pval_{key}'] = (('latitude', 'longitude'), r['pvals'])
            dv[f'effect_{key}'] = (('latitude', 'longitude'), r['effect_map'])
    xr.Dataset(dv, coords={'latitude': lats, 'longitude': lons}).assign_attrs({
        'description': ('compound-extreme frequency change, per-cell '
                        'Mann-Whitney U on annual frequencies'),
        'thresholds': thresh_label,
        'temperature': 'raw' if a.raw else 'detrended',
        'min_days_per_year': MIN_DAYS_PER_YEAR,
        'min_years_per_period': MIN_YEARS,
        'min_nonzero_years': MIN_NONZERO_YEARS,
        'effect_size': 'rank-biserial; positive = later exceeds early',
        'alpha': ALPHA,
        'multiple_testing': 'Benjamini-Hochberg FDR across tested cells',
        'early_period': f'{EARLY_YEARS[0]}-{EARLY_YEARS[1]}',
        'later_period': f'{LATER_YEARS[0]}-{LATER_YEARS[1]}',
        'created': pd.Timestamp.now().isoformat(),
    }).to_netcdf(f'{OUT_NC}/{season}_{tag}_diff_fields.nc')

    df = pd.DataFrame([_summary_row(c, n, results[c][n])
                       for c in results for n in TYPES])
    df.to_csv(f'{OUT_NC}/{season}_{tag}_domain_mean_tests.csv', index=False)

    annual = [dict(cluster=c, type=n, period=p, metric=metric, value=float(v))
              for c in results for n in TYPES
              for p, metric, arr in (
                  ('early', 'frequency', results[c][n]['annual_early']),
                  ('later', 'frequency', results[c][n]['annual_later']),
                  ('early', 'extent', results[c][n]['extent_early']),
                  ('later', 'extent', results[c][n]['extent_later']))
              for v in arr]
    pd.DataFrame(annual).to_csv(
        f'{OUT_NC}/{season}_{tag}_annual_series.csv', index=False)
    print(f'  wrote diff fields, domain-mean tests and annual series to {OUT_NC}')

    print(f'\n  {season} domain-mean change ({thresh_label})')
    print(f'  {"clu":<5}{"type":<10}{"f_yrs":>8}{"f_diff":>9}{"f_p":>8}{"fsig":>6}'
          f'{"e_yrs":>8}{"e_diff":>9}{"e_p":>8}{"esig":>6}')
    for _, x in df.iterrows():
        print(f'  {x["cluster"]:<5}{x["type"]:<10}'
              f'{x["freq_n_years_early"]:>3}/{x["freq_n_years_later"]:<4}'
              f'{x["freq_median_later"] - x["freq_median_early"]:>+9.3f}'
              f'{x["freq_p"]:>8.3f}{"YES" if x["freq_sig"] else "no":>6}'
              f'{x["ext_n_years_early"]:>3}/{x["ext_n_years_later"]:<4}'
              f'{x["ext_median_later"] - x["ext_median_early"]:>+9.1f}'
              f'{x["ext_p"]:>8.3f}{"YES" if x["ext_sig"] else "no":>6}')

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    mode, season, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    args = sys.argv[4:]
    raw = '--raw' in args
    var = next((x for x in args if x in ('tmax', 'tmin')), 'tmax')
    thresh = next((x for x in args if x in ('fixed', 'ownclim')), 'ownclim')
    n_events = next((int(x.split('=')[1]) for x in args
                     if x.startswith('--n=')), 10)
    year_min = next((int(x.split('=')[1]) for x in args
                     if x.startswith('--since=')), 2000)

    print('=' * 78)
    print(f'  COMPOUND EXTREMES [{mode}]  {season}  k={k}  {var}  '
          f'{"raw" if raw else "detrended"}'
          + (f'  {thresh}' if mode == 'diff' else ''))
    print('=' * 78)

    a = CompoundExtremes(season, k, BASE, var=var, raw=raw)
    a.load_data()

    if mode == 'freq':
        a.identify_extremes()
        a.calculate_frequencies()
        a.save()
    elif mode == 'series':
        a.identify_extremes()
        print('\n  full-record annual series')
        stage_series(a, grid_cell_areas(a.temp_anom.latitude.values,
                                        a.temp_anom.longitude.values),
                     f'{var}{"_raw" if raw else ""}')
    elif mode == 'events':
        a.identify_extremes()
        print('\n  most extensive event days')
        stage_events(a, grid_cell_areas(a.temp_anom.latitude.values,
                                        a.temp_anom.longitude.values),
                     f'{var}{"_raw" if raw else ""}', n_events, year_min)
    elif mode == 'diff':
        areas = grid_cell_areas(a.temp_anom.latitude.values,
                                a.temp_anom.longitude.values)
        print('\n  building period masks')
        me, ce, ml, cl, ex_t, ex_e, ex_l, label = build_masks(a, thresh == 'fixed')
        print('\n  per-cell tests and annual series')
        results = compute_all(a, k, me, ce, ml, cl, ex_t, ex_e, ex_l, areas)
        results = domain_mean_test(results, k, 'annual_early', 'annual_later',
                                   'freq')
        results = domain_mean_test(results, k, 'extent_early', 'extent_later',
                                   'ext')
        save_diff(a, results, k, season,
                  f'{var}{"_raw" if raw else ""}_{thresh}', label)
    else:
        print(f"mode must be 'freq', 'series', 'events' or 'diff', "
              f"got '{mode}'")
        sys.exit(1)

    print('\n' + '=' * 78)
    print('  COMPLETE')
    print('=' * 78)

if __name__ == '__main__':
    main()
