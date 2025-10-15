#!/usr/bin/env python3
"""
Generate threshold-based alerts for PM2.5 exceedances.

Usage:
  python3 analysis/alerts.py --input china_air_quality_raw.csv --outdir alerts --threshold 150
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def classify_severity(pm25: float, threshold: float) -> str:
    if math.isnan(pm25):
        return 'unknown'
    if pm25 > max(250.0, threshold * 1.3):
        return 'hazardous'
    if pm25 > threshold:
        return 'severe'
    if pm25 > 115.0:
        return 'heavy'
    if pm25 > 75.0:
        return 'moderate'
    return 'good'


def generate_alerts(input_path: str, outdir: str, threshold: float) -> str:
    ensure_dir(outdir)
    alerts_csv = os.path.join(outdir, 'alerts.csv')
    alerts_jsonl = os.path.join(outdir, 'alerts.jsonl')
    summary_txt = os.path.join(outdir, 'summary.txt')

    fields = [
        'City','Latitude','Longitude','Season','Hour','Month','Year','Station ID',
        'PM2.5 (µg/m³)','Severity','Threshold'
    ]

    by_city: Dict[str, int] = {}
    by_season: Dict[str, int] = {}

    with open(input_path, 'r', encoding='utf-8') as f, \
         open(alerts_csv, 'w', newline='', encoding='utf-8') as fc, \
         open(alerts_jsonl, 'w', encoding='utf-8') as fj:
        reader = csv.DictReader(f)
        writer = csv.writer(fc)
        writer.writerow(fields)
        for row in reader:
            try:
                pm25 = float(row['PM2.5 (µg/m³)'])
            except Exception:
                pm25 = float('nan')
            if math.isnan(pm25) or pm25 <= threshold:
                continue
            sev = classify_severity(pm25, threshold)
            out_row = [
                row.get('City',''), row.get('Latitude',''), row.get('Longitude',''), row.get('Season',''),
                row.get('Hour',''), row.get('Month',''), row.get('Year',''), row.get('Station ID',''),
                pm25, sev, threshold
            ]
            writer.writerow(out_row)
            record = {
                'city': row.get('City',''), 'latitude': row.get('Latitude',''), 'longitude': row.get('Longitude',''),
                'season': row.get('Season',''), 'hour': row.get('Hour',''), 'month': row.get('Month',''), 'year': row.get('Year',''),
                'station_id': row.get('Station ID',''), 'pm25': pm25, 'severity': sev, 'threshold': threshold
            }
            fj.write(json.dumps(record, ensure_ascii=False) + '\n')
            c = row.get('City','')
            if c:
                by_city[c] = by_city.get(c, 0) + 1
            s = row.get('Season','')
            if s:
                by_season[s] = by_season.get(s, 0) + 1

    with open(summary_txt, 'w', encoding='utf-8') as fs:
        fs.write('Alerts by city (desc):\n')
        for k, v in sorted(by_city.items(), key=lambda kv: kv[1], reverse=True):
            fs.write(f'{k}: {v}\n')
        fs.write('\nAlerts by season (desc):\n')
        for k, v in sorted(by_season.items(), key=lambda kv: kv[1], reverse=True):
            fs.write(f'{k}: {v}\n')

    return alerts_csv


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate PM2.5 threshold alerts')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--outdir', default='alerts', help='Output directory for alerts')
    parser.add_argument('--threshold', type=float, default=150.0, help='PM2.5 threshold')
    args = parser.parse_args()

    out = generate_alerts(args.input, args.outdir, args.threshold)
    print(f'Alerts written to: {out}')


if __name__ == '__main__':
    main()

