#!/usr/bin/env python3
"""
Orchestrate the full pipeline: EDA -> maps -> alerts -> plots -> HTML report.

Usage:
  python3 analysis/run_all.py --input china_air_quality_raw.csv --outdir outputs --threshold 150

Outputs structure (under outdir):
  - analysis_outputs/
  - maps/
  - figures/
  - alerts/
  - report/report.html
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow importing the sibling modules when executed directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.eda import run_eda  # type: ignore
from analysis.spatial import generate_maps  # type: ignore
from analysis.alerts import generate_alerts  # type: ignore
from analysis.plots import generate_plots  # type: ignore
from analysis.generate_report import generate_html  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description='Run full environmental monitoring analysis pipeline')
    parser.add_argument('--input', default='china_air_quality_raw.csv', help='Path to input CSV')
    parser.add_argument('--outdir', default='outputs', help='Base output directory')
    parser.add_argument('--threshold', type=float, default=150.0, help='PM2.5 threshold')
    parser.add_argument('--city-filter', default='', help='Optional city name to filter maps')
    args = parser.parse_args()

    base = os.path.abspath(args.outdir)
    os.makedirs(base, exist_ok=True)
    analysis_dir = os.path.join(base, 'analysis_outputs')
    maps_dir = os.path.join(base, 'maps')
    figures_dir = os.path.join(base, 'figures')
    alerts_dir = os.path.join(base, 'alerts')
    report_dir = os.path.join(base, 'report')
    for d in [analysis_dir, maps_dir, figures_dir, alerts_dir, report_dir]:
        os.makedirs(d, exist_ok=True)

    print('Running EDA...')
    report_txt = run_eda(args.input, analysis_dir, args.threshold)
    print('EDA report ->', report_txt)

    print('Generating spatial maps...')
    maps = generate_maps(args.input, maps_dir, city_filter=args.city_filter)
    print('Maps ->', maps)

    print('Generating alerts...')
    alerts_csv = generate_alerts(args.input, alerts_dir, args.threshold)
    print('Alerts ->', alerts_csv)

    print('Generating plots...')
    figs = generate_plots(args.input, figures_dir)
    print('Plots ->', figs)

    print('Building HTML report...')
    html = generate_html(analysis_dir, maps_dir, figures_dir, alerts_dir, report_dir)
    print('Report ->', html)

    print('DONE')


if __name__ == '__main__':
    main()
