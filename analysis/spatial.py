#!/usr/bin/env python3
"""
Spatial visualizations for environmental monitoring data.
- Tries GeoPandas for geospatial handling if available.
- Falls back to matplotlib scatter/hexbin/KDE-like visuals when GeoPandas is unavailable.

Usage:
  python3 analysis/spatial.py --input china_air_quality_raw.csv --outdir maps --pm25-column "PM2.5 (µg/m³)" --city-filter ""
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def try_imports():
    pd = None
    gpd = None
    np = None
    plt = None
    try:
        import pandas as _pd  # type: ignore
        pd = _pd
    except Exception:
        pd = None
    try:
        import geopandas as _gpd  # type: ignore
        gpd = _gpd
    except Exception:
        gpd = None
    try:
        import numpy as _np  # type: ignore
        np = _np
    except Exception:
        np = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt  # type: ignore
        plt = _plt
    except Exception:
        plt = None
    return pd, gpd, np, plt


def load_data(input_path: str, pd) -> Tuple[List[float], List[float], List[float], List[str]]:
    latitudes: List[float] = []
    longitudes: List[float] = []
    pm25_values: List[float] = []
    cities: List[str] = []

    if pd is not None:
        try:
            df = pd.read_csv(input_path)
            latitudes = pd.to_numeric(df['Latitude'], errors='coerce').tolist()
            longitudes = pd.to_numeric(df['Longitude'], errors='coerce').tolist()
            pm25_values = pd.to_numeric(df['PM2.5 (µg/m³)'], errors='coerce').tolist()
            cities = df['City'].astype(str).tolist() if 'City' in df.columns else [''] * len(df)
            return latitudes, longitudes, pm25_values, cities
        except Exception:
            pass

    # Fallback pure CSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                latitudes.append(float(row['Latitude']))
            except Exception:
                latitudes.append(float('nan'))
            try:
                longitudes.append(float(row['Longitude']))
            except Exception:
                longitudes.append(float('nan'))
            try:
                pm25_values.append(float(row['PM2.5 (µg/m³)']))
            except Exception:
                pm25_values.append(float('nan'))
            cities.append(row.get('City', ''))
    return latitudes, longitudes, pm25_values, cities


def filter_by_city(lat: List[float], lon: List[float], pm25: List[float], city: List[str], city_filter: str):
    if not city_filter:
        return lat, lon, pm25, city
    lat2: List[float] = []
    lon2: List[float] = []
    pm2: List[float] = []
    city2: List[str] = []
    for a, b, c, d in zip(lat, lon, pm25, city):
        if (d or '').lower() == city_filter.lower():
            lat2.append(a); lon2.append(b); pm2.append(c); city2.append(d)
    return lat2, lon2, pm2, city2


def save_scatter(plt, outdir: str, lon: List[float], lat: List[float], pm25: List[float]) -> Optional[str]:
    if plt is None:
        return None
    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(lon, lat, c=pm25, cmap='inferno', s=10, alpha=0.7)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('PM2.5 scatter (colored by concentration)')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('PM2.5 (µg/m³)')
    outpath = os.path.join(outdir, 'pm25_scatter.png')
    fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def save_hexbin(plt, np, outdir: str, lon: List[float], lat: List[float], pm25: List[float]) -> Optional[str]:
    if plt is None:
        return None
    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(8, 6))
    c = pm25 if np is None else np.array(pm25)
    hb = ax.hexbin(lon, lat, C=c, gridsize=35, reduce_C_function=(None if np is None else np.mean), cmap='inferno')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('PM2.5 hexbin (mean in cell)')
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('PM2.5 (µg/m³)')
    outpath = os.path.join(outdir, 'pm25_hexbin.png')
    fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def save_city_mean_scatter(plt, outdir: str, lon: List[float], lat: List[float], pm25: List[float], city: List[str]) -> Optional[str]:
    if plt is None:
        return None
    ensure_dir(outdir)
    sums: Dict[str, Tuple[float, float, float, int]] = {}
    for x, y, v, c in zip(lon, lat, pm25, city):
        if math.isnan(x) or math.isnan(y) or math.isnan(v):
            continue
        sx, sy, sv, n = sums.get(c, (0.0, 0.0, 0.0, 0))
        sums[c] = (sx + x, sy + y, sv + v, n + 1)
    xs: List[float] = []; ys: List[float] = []; vs: List[float] = []
    for c, (sx, sy, sv, n) in sums.items():
        if n > 0:
            xs.append(sx / n); ys.append(sy / n); vs.append(sv / n)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xs, ys, c=vs, cmap='inferno', s=120, edgecolor='k')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('City mean PM2.5')
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('PM2.5 mean (µg/m³)')
    outpath = os.path.join(outdir, 'pm25_city_mean.png')
    fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def save_top_percentile_overlay(plt, np, outdir: str, lon: List[float], lat: List[float], pm25: List[float], percentile: float = 95.0) -> Optional[str]:
    if plt is None:
        return None
    ensure_dir(outdir)
    arr = pm25 if np is None else np.array(pm25)
    if np is None:
        # simple percentile
        sorted_vals = sorted([v for v in arr if not math.isnan(v)])
        if not sorted_vals:
            return None
        k = int(math.ceil(len(sorted_vals) * percentile / 100.0)) - 1
        thr = sorted_vals[max(0, min(k, len(sorted_vals) - 1))]
    else:
        thr = float(np.nanpercentile(arr, percentile))
    xs_hot: List[float] = []; ys_hot: List[float] = []
    xs_cold: List[float] = []; ys_cold: List[float] = []
    for x, y, v in zip(lon, lat, pm25):
        if math.isnan(x) or math.isnan(y) or math.isnan(v):
            continue
        if v >= thr:
            xs_hot.append(x); ys_hot.append(y)
        else:
            xs_cold.append(x); ys_cold.append(y)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs_cold, ys_cold, c='#bbbbbb', s=8, alpha=0.3)
    ax.scatter(xs_hot, ys_hot, c='red', s=12, alpha=0.7)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude');
    ax.set_title(f'PM2.5 hotspots (>= P{percentile:.0f})')
    outpath = os.path.join(outdir, 'pm25_hotspots.png')
    fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def generate_maps(input_path: str, outdir: str, city_filter: str = '') -> List[str]:
    pd, gpd, np, plt = try_imports()
    lon, lat, pm25, city = [], [], [], []
    lat, lon, pm25, city = load_data(input_path, pd)
    if len(lat) == 0:
        return []
    lat, lon, pm25, city = filter_by_city(lat, lon, pm25, city, city_filter)

    outputs: List[str] = []
    if plt is None:
        note_path = os.path.join(outdir, 'NO_MATPLOTLIB.txt')
        ensure_dir(outdir)
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write('matplotlib not available; spatial plots skipped.')
        return [note_path]

    p1 = save_scatter(plt, outdir, lon, lat, pm25)
    if p1: outputs.append(p1)
    p2 = save_hexbin(plt, np, outdir, lon, lat, pm25)
    if p2: outputs.append(p2)
    p3 = save_city_mean_scatter(plt, outdir, lon, lat, pm25, city)
    if p3: outputs.append(p3)
    p4 = save_top_percentile_overlay(plt, np, outdir, lon, lat, pm25, percentile=95.0)
    if p4: outputs.append(p4)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate spatial PM2.5 visualizations')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--outdir', default='maps', help='Output directory for map images')
    parser.add_argument('--city-filter', default='', help='Optional city name to filter')
    args = parser.parse_args()

    ensure_dir(args.outdir)
    outputs = generate_maps(args.input, args.outdir, args.city_filter)
    if outputs:
        print('Generated:', '\n'.join(outputs))
    else:
        print('No outputs generated')


if __name__ == '__main__':
    main()
