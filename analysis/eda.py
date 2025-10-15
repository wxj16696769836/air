#!/usr/bin/env python3
"""
EDA for environmental monitoring dataset (PM2.5, SO2, NO2, etc.).
- Loads CSV with pandas if available; falls back to pure-Python CSV.
- Computes schema, missingness, coordinate checks, pollutant stats, PM2.5 exceedances,
  temporal aggregates (season/hour/month/year), and pollutant correlations.
- Writes human-readable report and CSV summaries to an output directory.

Usage:
  python3 analysis/eda.py --input china_air_quality_raw.csv --outdir outputs --pm25-threshold 150

Outputs (default outdir=analysis_outputs):
  - report.txt: text summary of all metrics
  - pollutant_summary.csv: count/mean/p50/p95/min/max per pollutant
  - pm25_exceedances_by_city.csv
  - pm25_exceedances_by_season.csv
  - pm25_exceedances_by_hour.csv
  - pm25_means_by_season.csv
  - pm25_means_by_hour.csv
  - pm25_means_by_month.csv
  - pm25_means_by_year.csv
  - pollutant_correlation.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        return pd, np
    except Exception:
        return None, None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def write_csv(path: str, rows: List[List[object]], header: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def eda_with_pandas(input_path: str, outdir: str, pm25_threshold: float) -> str:
    pd, np = try_import_pandas()
    assert pd is not None

    df = pd.read_csv(input_path)

    # Basic info
    n_rows, n_cols = df.shape
    columns = list(df.columns)

    report_lines: List[str] = []
    report_lines.append('=== BASIC INFO ===')
    report_lines.append(f'Rows: {n_rows}  Cols: {n_cols}')
    report_lines.append(f'Columns: {columns}')

    # Missingness
    report_lines.append('\n=== MISSING VALUES PER COLUMN ===')
    na_counts = df.isna().sum().sort_values(ascending=False)
    report_lines.append(na_counts.to_string())

    # Coordinates sanity
    lat_col, lon_col = 'Latitude', 'Longitude'
    if lat_col in df.columns and lon_col in df.columns:
        lat = pd.to_numeric(df[lat_col], errors='coerce')
        lon = pd.to_numeric(df[lon_col], errors='coerce')
        lat_min, lat_max, lon_min, lon_max = 18.0, 54.0, 73.0, 135.0
        outside_mask = (lat < lat_min) | (lat > lat_max) | (lon < lon_min) | (lon > lon_max)
        outside_count = int(outside_mask.sum())
        outside_pct = outside_count / len(df) * 100 if len(df) else 0.0
        report_lines.append('\n=== COORDINATE BOUNDS CHECK (China bbox) ===')
        report_lines.append(f'Outside bbox count: {outside_count} ({outside_pct:.2f}%)')
        report_lines.append(f'Latitude range: {lat.min():.4f} to {lat.max():.4f}')
        report_lines.append(f'Longitude range: {lon.min():.4f} to {lon.max():.4f}')

    # City distribution
    if 'City' in df.columns:
        report_lines.append('\n=== TOP 10 CITIES BY ROW COUNT ===')
        report_lines.append(df['City'].value_counts().head(10).to_string())

    # Time coverage
    report_lines.append('\n=== TIME COVERAGE ===')
    for col in ['Year', 'Month', 'Hour', 'Season', 'Day of Week']:
        if col in df.columns:
            uni = sorted(pd.unique(df[col]))
            report_lines.append(f'{col}: unique={len(uni)} sample={uni[:10]}')

    # Pollutant stats
    pollutant_keywords = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    pollutant_cols = [c for c in df.columns if any(k in c for k in pollutant_keywords)]

    report_lines.append('\n=== POLLUTANT SUMMARY (count, mean, p50, p95, min, max) ===')
    rows: List[List[object]] = []
    for c in pollutant_cols:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() == 0:
            continue
        stats = {
            'count': int(s.count()),
            'mean': float(s.mean()),
            'p50': float(s.median()),
            'p95': float(s.quantile(0.95)),
            'min': float(s.min()),
            'max': float(s.max()),
        }
        report_lines.append(
            f"{c}: " + ', '.join([f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in stats.items()])
        )
        rows.append([c, stats['count'], stats['mean'], stats['p50'], stats['p95'], stats['min'], stats['max']])

    write_csv(os.path.join(outdir, 'pollutant_summary.csv'), rows,
              ['pollutant', 'count', 'mean', 'p50', 'p95', 'min', 'max'])

    # PM2.5 exceedances & temporal
    pm25_col = next((c for c in df.columns if 'PM2.5' in c), None)
    if pm25_col is not None:
        s = pd.to_numeric(df[pm25_col], errors='coerce')
        mask = s > pm25_threshold
        cnt = int(mask.sum())
        rate = cnt / len(df) * 100 if len(df) else 0.0
        report_lines.append(f"\n=== PM2.5 EXCEEDANCES > {pm25_threshold} ===")
        report_lines.append(f'Count: {cnt}  Share: {rate:.2f}%')

        if 'City' in df.columns and cnt:
            by_city = df.loc[mask, 'City'].value_counts()
            report_lines.append('\nTop 5 cities by exceedances:')
            report_lines.append(by_city.head(5).to_string())
            write_csv(os.path.join(outdir, 'pm25_exceedances_by_city.csv'),
                      [[k, int(v)] for k, v in by_city.items()], ['city', 'exceedance_count'])

        if 'Season' in df.columns and cnt:
            by_season = df.loc[mask, 'Season'].value_counts()
            report_lines.append('\nExceedances by Season:')
            report_lines.append(by_season.to_string())
            write_csv(os.path.join(outdir, 'pm25_exceedances_by_season.csv'),
                      [[k, int(v)] for k, v in by_season.items()], ['season', 'exceedance_count'])

        if 'Hour' in df.columns and cnt:
            by_hour = df.loc[mask].groupby('Hour')[pm25_col].size().sort_values(ascending=False)
            report_lines.append('\nTop hours by exceedance count:')
            report_lines.append(by_hour.head(10).to_string())
            write_csv(os.path.join(outdir, 'pm25_exceedances_by_hour.csv'),
                      [[int(k), int(v)] for k, v in by_hour.items()], ['hour', 'exceedance_count'])

        # Temporal averages
        report_lines.append('\n=== PM2.5 TEMPORAL AVERAGES ===')
        if 'Season' in df.columns:
            by = df.groupby('Season')[pm25_col].mean().sort_values(ascending=False).round(2)
            report_lines.append('By Season (mean):')
            report_lines.append(by.to_string())
            write_csv(os.path.join(outdir, 'pm25_means_by_season.csv'),
                      [[k, float(v)] for k, v in by.items()], ['season', 'pm25_mean'])
        if 'Hour' in df.columns:
            by = df.groupby('Hour')[pm25_col].mean().sort_values(ascending=False).round(2)
            report_lines.append('\nBy Hour (mean):')
            report_lines.append(by.head(10).to_string())
            write_csv(os.path.join(outdir, 'pm25_means_by_hour.csv'),
                      [[int(k), float(v)] for k, v in by.items()], ['hour', 'pm25_mean'])
        if 'Month' in df.columns:
            by = df.groupby('Month')[pm25_col].mean().round(2)
            report_lines.append('\nBy Month (mean):')
            report_lines.append(by.to_string())
            write_csv(os.path.join(outdir, 'pm25_means_by_month.csv'),
                      [[int(k), float(v)] for k, v in by.items()], ['month', 'pm25_mean'])
        if 'Year' in df.columns:
            by = df.groupby('Year')[pm25_col].mean().round(2)
            report_lines.append('\nBy Year (mean):')
            report_lines.append(by.to_string())
            write_csv(os.path.join(outdir, 'pm25_means_by_year.csv'),
                      [[int(k), float(v)] for k, v in by.items()], ['year', 'pm25_mean'])

    # Correlation (Spearman for robustness)
    if pollutant_cols:
        poll_df = df[pollutant_cols].apply(pd.to_numeric, errors='coerce')
        corr = poll_df.corr(method='spearman')
        report_lines.append('\n=== SPEARMAN CORRELATION (pollutants) ===')
        report_lines.append(corr.round(2).to_string())
        # Save as CSV matrix
        corr.to_csv(os.path.join(outdir, 'pollutant_correlation.csv'))

    report_path = os.path.join(outdir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    return report_path


def eda_pure_python(input_path: str, outdir: str, pm25_threshold: float) -> str:
    # Read CSV basic
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = header
        col_index = {name: i for i, name in enumerate(columns)}
        n_cols = len(columns)
        n_rows = 0
        missing_counts = [0] * n_cols

        pollutant_keywords = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        pollutant_cols = [c for c in columns if any(k in c for k in pollutant_keywords)]
        pm25_key = next((c for c in columns if 'PM2.5' in c), None)

        lat_col, lon_col = 'Latitude', 'Longitude'
        lat_min, lat_max, lon_min, lon_max = 18.0, 54.0, 73.0, 135.0
        lat_values: List[float] = []
        lon_values: List[float] = []
        outside_count = 0

        city_col = 'City'
        season_col = 'Season'
        hour_col = 'Hour'
        month_col = 'Month'
        year_col = 'Year'

        # Collectors
        numeric_collectors: Dict[str, List[float]] = defaultdict(list)
        city_counts: Dict[str, int] = defaultdict(int)
        exceed_city_counts: Dict[str, int] = defaultdict(int)
        exceed_season_counts: Dict[str, int] = defaultdict(int)
        exceed_hour_counts: Dict[str, int] = defaultdict(int)
        unique_values: Dict[str, set] = {k: set() for k in [year_col, month_col, hour_col, season_col, 'Day of Week'] if k in col_index}

        def to_float(x: str) -> float:
            try:
                return float(x)
            except Exception:
                return float('nan')

        for row in reader:
            if not row:
                continue
            n_rows += 1
            # Missingness
            for i, val in enumerate(row):
                if val is None or val == '' or val.strip() == '':
                    missing_counts[i] += 1

            # Unique values
            for k in unique_values.keys():
                v = row[col_index[k]]
                unique_values[k].add(v)

            # Coordinates
            if lat_col in col_index and lon_col in col_index:
                lat = to_float(row[col_index[lat_col]])
                lon = to_float(row[col_index[lon_col]])
                lat_values.append(lat)
                lon_values.append(lon)
                if not (math.isnan(lat) or math.isnan(lon)):
                    if (lat < lat_min) or (lat > lat_max) or (lon < lon_min) or (lon > lon_max):
                        outside_count += 1

            # City count
            if city_col in col_index:
                city_counts[row[col_index[city_col]]] += 1

            # Pollutants
            for c in pollutant_cols:
                val = to_float(row[col_index[c]])
                if not math.isnan(val):
                    numeric_collectors[c].append(val)

            # Exceedances
            if pm25_key is not None:
                pm25 = to_float(row[col_index[pm25_key]])
                if not math.isnan(pm25) and pm25 > pm25_threshold:
                    if city_col in col_index:
                        exceed_city_counts[row[col_index[city_col]]] += 1
                    if season_col in col_index:
                        exceed_season_counts[row[col_index[season_col]]] += 1
                    if hour_col in col_index:
                        exceed_hour_counts[row[col_index[hour_col]]] += 1

    # Build report
    report_lines: List[str] = []
    report_lines.append('=== BASIC INFO ===')
    report_lines.append(f'Rows: {n_rows}  Cols: {n_cols}')
    report_lines.append(f'Columns: {columns}')

    report_lines.append('\n=== MISSING VALUES PER COLUMN ===')
    for i, name in enumerate(columns):
        report_lines.append(f'{name}: {missing_counts[i]}')

    if lat_values and lon_values:
        lat_clean = [x for x in lat_values if not math.isnan(x)]
        lon_clean = [x for x in lon_values if not math.isnan(x)]
        report_lines.append('\n=== COORDINATE BOUNDS CHECK (China bbox) ===')
        pct = (outside_count / n_rows * 100) if n_rows else 0.0
        report_lines.append(f'Outside bbox count: {outside_count} ({pct:.2f}%)')
        if lat_clean:
            report_lines.append(f'Latitude range: {min(lat_clean):.4f} to {max(lat_clean):.4f}')
        if lon_clean:
            report_lines.append(f'Longitude range: {min(lon_clean):.4f} to {max(lon_clean):.4f}')

    if city_counts:
        report_lines.append('\n=== TOP 10 CITIES BY ROW COUNT ===')
        for city, cnt in sorted(city_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            report_lines.append(f'{city}: {cnt}')

    report_lines.append('\n=== TIME COVERAGE ===')
    for k, s in unique_values.items():
        uni = sorted(s)
        report_lines.append(f'{k}: unique={len(uni)} sample={uni[:10]}')

    # Pollutant summary CSV
    poll_rows: List[List[object]] = []
    report_lines.append('\n=== POLLUTANT SUMMARY (count, mean, p50, p95, min, max) ===')
    def quantile(a: List[float], q: float) -> float:
        if not a:
            return float('nan')
        a_sorted = sorted(a)
        i = (len(a_sorted) - 1) * q
        lo = int(math.floor(i))
        hi = int(math.ceil(i))
        if lo == hi:
            return a_sorted[lo]
        return a_sorted[lo] * (hi - i) + a_sorted[hi] * (i - lo)

    for c, arr in numeric_collectors.items():
        if not arr:
            continue
        arr_sorted = sorted(arr)
        count = len(arr_sorted)
        mean = sum(arr_sorted) / count
        p50 = arr_sorted[count//2] if count % 2 == 1 else (arr_sorted[count//2 - 1] + arr_sorted[count//2]) / 2
        p95 = quantile(arr_sorted, 0.95)
        mn = arr_sorted[0]
        mx = arr_sorted[-1]
        report_lines.append(f'{c}: count={count}, mean={mean:.3f}, p50={p50:.3f}, p95={p95:.3f}, min={mn:.3f}, max={mx:.3f}')
        poll_rows.append([c, count, mean, p50, p95, mn, mx])

    write_csv(os.path.join(outdir, 'pollutant_summary.csv'), poll_rows,
              ['pollutant', 'count', 'mean', 'p50', 'p95', 'min', 'max'])

    # Exceedances aggregated exports
    if pm25_key is not None:
        exc_city_rows = sorted(exceed_city_counts.items(), key=lambda kv: kv[1], reverse=True)
        write_csv(os.path.join(outdir, 'pm25_exceedances_by_city.csv'),
                  [[k, v] for k, v in exc_city_rows], ['city', 'exceedance_count'])
        exc_season_rows = sorted(exceed_season_counts.items(), key=lambda kv: kv[1], reverse=True)
        write_csv(os.path.join(outdir, 'pm25_exceedances_by_season.csv'),
                  [[k, v] for k, v in exc_season_rows], ['season', 'exceedance_count'])
        exc_hour_rows = sorted(exceed_hour_counts.items(), key=lambda kv: kv[1], reverse=True)
        write_csv(os.path.join(outdir, 'pm25_exceedances_by_hour.csv'),
                  [[k, v] for k, v in exc_hour_rows], ['hour', 'exceedance_count'])

        # Temporal means (second pass)
        groups = {
            'Season': defaultdict(lambda: [0.0, 0]),
            'Hour': defaultdict(lambda: [0.0, 0]),
            'Month': defaultdict(lambda: [0.0, 0]),
            'Year': defaultdict(lambda: [0.0, 0]),
        }
        with open(input_path, 'r', encoding='utf-8') as f2:
            r2 = csv.reader(f2)
            header2 = next(r2)
            cidx = {name: i for i, name in enumerate(header2)}
            def to_float2(x: str) -> float:
                try:
                    return float(x)
                except Exception:
                    return float('nan')
            for row2 in r2:
                pmv = to_float2(row2[cidx[pm25_key]])
                if math.isnan(pmv):
                    continue
                for k in groups.keys():
                    if k in cidx:
                        key = row2[cidx[k]]
                        acc = groups[k][key]
                        acc[0] += pmv
                        acc[1] += 1
        # Save group means
        for gname, acc_map in groups.items():
            rows_out = []
            for k, (s, c) in acc_map.items():
                if c > 0:
                    rows_out.append([k, s / c])
            rows_out.sort(key=lambda kv: kv[1], reverse=True)
            fname = f'pm25_means_by_{gname.lower()}.csv'
            write_csv(os.path.join(outdir, fname), rows_out, [gname.lower(), 'pm25_mean'])

    # Approx Pearson correlation matrix (pure Python)
    keys = pollutant_cols
    data: Dict[str, List[float]] = {k: [] for k in keys}
    with open(input_path, 'r', encoding='utf-8') as f3:
        r3 = csv.DictReader(f3)
        for row in r3:
            vals = []
            ok = True
            for k in keys:
                try:
                    vals.append(float(row[k]))
                except Exception:
                    ok = False
                    break
            if ok:
                for i, k in enumerate(keys):
                    data[k].append(vals[i])

    def pearson(x: List[float], y: List[float]) -> float:
        n = len(x)
        if n == 0:
            return float('nan')
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if denx == 0 or deny == 0:
            return float('nan')
        return num / (denx * deny)

    corr_keys = keys
    corr_rows: List[List[object]] = []
    corr_header = [''] + corr_keys
    for k1 in corr_keys:
        row_out: List[object] = [k1]
        for k2 in corr_keys:
            r = pearson(data[k1], data[k2])
            row_out.append('' if math.isnan(r) else f'{r:.2f}')
        corr_rows.append(row_out)
    write_csv(os.path.join(outdir, 'pollutant_correlation.csv'), corr_rows, corr_header)

    # Save report
    report_path = os.path.join(outdir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    return report_path


def run_eda(input_path: str, outdir: str, pm25_threshold: float) -> str:
    ensure_dir(outdir)
    pd, _ = try_import_pandas()
    if pd is not None:
        try:
            return eda_with_pandas(input_path, outdir, pm25_threshold)
        except Exception as e:
            # Fallback to pure Python if pandas path fails
            return eda_pure_python(input_path, outdir, pm25_threshold)
    else:
        return eda_pure_python(input_path, outdir, pm25_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description='EDA for environmental monitoring dataset')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--outdir', default='analysis_outputs', help='Output directory for reports and CSVs')
    parser.add_argument('--pm25-threshold', type=float, default=150.0, help='PM2.5 exceedance threshold')
    args = parser.parse_args()

    report_path = run_eda(args.input, args.outdir, args.pm25_threshold)
    print(f'Report written to: {report_path}')


if __name__ == '__main__':
    main()

