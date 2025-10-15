#!/usr/bin/env python3
"""
Generate PM2.5 temporal trend plots (seasonal, diurnal, monthly, yearly).

Usage:
  python3 analysis/plots.py --input china_air_quality_raw.csv --outdir figures
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def generate_plots(input_path: str, outdir: str) -> List[str]:
    ensure_dir(outdir)
    plt = try_import_matplotlib()
    if plt is None:
        note = os.path.join(outdir, 'NO_MATPLOTLIB.txt')
        with open(note, 'w', encoding='utf-8') as f:
            f.write('matplotlib not available; plots skipped.')
        return [note]

    pd = try_import_pandas()
    outputs: List[str] = []

    if pd is not None:
        try:
            df = pd.read_csv(input_path)
            pm25_col = 'PM2.5 (µg/m³)'
            # Season
            if 'Season' in df.columns:
                s = df.groupby('Season')[pm25_col].mean().reindex(['Spring','Summer','Autumn','Winter']).dropna()
                fig, ax = plt.subplots(figsize=(6,4))
                s.plot(kind='bar', color='#c85108', ax=ax)
                ax.set_ylabel('PM2.5 mean (µg/m³)'); ax.set_title('PM2.5 by Season')
                fig.tight_layout(); p = os.path.join(outdir, 'pm25_by_season.png'); fig.savefig(p, dpi=150); outputs.append(p); plt.close(fig)
            # Hour
            if 'Hour' in df.columns:
                h = df.groupby('Hour')[pm25_col].mean().sort_index()
                fig, ax = plt.subplots(figsize=(7,4))
                h.plot(kind='line', marker='o', color='#375a7f', ax=ax)
                ax.set_xlabel('Hour'); ax.set_ylabel('PM2.5 mean (µg/m³)'); ax.set_title('PM2.5 by Hour')
                fig.tight_layout(); p = os.path.join(outdir, 'pm25_by_hour.png'); fig.savefig(p, dpi=150); outputs.append(p); plt.close(fig)
            # Month
            if 'Month' in df.columns:
                m = df.groupby('Month')[pm25_col].mean().sort_index()
                fig, ax = plt.subplots(figsize=(7,4))
                m.plot(kind='line', marker='o', color='#2ca02c', ax=ax)
                ax.set_xlabel('Month'); ax.set_ylabel('PM2.5 mean (µg/m³)'); ax.set_title('PM2.5 by Month')
                fig.tight_layout(); p = os.path.join(outdir, 'pm25_by_month.png'); fig.savefig(p, dpi=150); outputs.append(p); plt.close(fig)
            # Year
            if 'Year' in df.columns:
                y = df.groupby('Year')[pm25_col].mean().sort_index()
                fig, ax = plt.subplots(figsize=(7,4))
                y.plot(kind='bar', color='#9467bd', ax=ax)
                ax.set_ylabel('PM2.5 mean (µg/m³)'); ax.set_title('PM2.5 by Year')
                fig.tight_layout(); p = os.path.join(outdir, 'pm25_by_year.png'); fig.savefig(p, dpi=150); outputs.append(p); plt.close(fig)
            return outputs
        except Exception:
            pass

    # Pure CSV fallback
    pm25_key = 'PM2.5 (µg/m³)'
    groups = { 'Season': defaultdict(lambda: [0.0, 0]), 'Hour': defaultdict(lambda: [0.0, 0]),
               'Month': defaultdict(lambda: [0.0, 0]), 'Year': defaultdict(lambda: [0.0, 0]) }
    with open(input_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                v = float(row[pm25_key])
            except Exception:
                continue
            for k in groups.keys():
                if k in row and row[k] != '':
                    s, c = groups[k][row[k]]
                    groups[k][row[k]] = [s + v, c + 1]

    def plot_dict(d: Dict[str, List[float]], title: str, xlabel: str, filename: str, sort_numeric: bool = False):
        keys = list(d.keys())
        if sort_numeric:
            try:
                keys = sorted(keys, key=lambda x: int(x))
            except Exception:
                keys = sorted(keys)
        else:
            keys = sorted(keys)
        vals = [d[k][0] / d[k][1] if d[k][1] > 0 else float('nan') for k in keys]
        fig, ax = plt.subplots(figsize=(7,4))
        if sort_numeric or title.endswith('Year'):
            ax.plot(keys, vals, marker='o')
        else:
            ax.bar(keys, vals)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel('PM2.5 mean (µg/m³)')
        fig.tight_layout(); p = os.path.join(outdir, filename); fig.savefig(p, dpi=150); plt.close(fig)
        return p

    outputs.append(plot_dict(groups['Season'], 'PM2.5 by Season', 'Season', 'pm25_by_season.png'))
    outputs.append(plot_dict(groups['Hour'], 'PM2.5 by Hour', 'Hour', 'pm25_by_hour.png', sort_numeric=True))
    outputs.append(plot_dict(groups['Month'], 'PM2.5 by Month', 'Month', 'pm25_by_month.png', sort_numeric=True))
    outputs.append(plot_dict(groups['Year'], 'PM2.5 by Year', 'Year', 'pm25_by_year.png', sort_numeric=True))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate PM2.5 trend plots')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--outdir', default='figures', help='Output directory for plots')
    args = parser.parse_args()

    outputs = generate_plots(args.input, args.outdir)
    print('Generated:', '\n'.join(outputs))


if __name__ == '__main__':
    main()
