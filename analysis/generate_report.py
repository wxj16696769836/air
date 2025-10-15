#!/usr/bin/env python3
"""
Generate an HTML environmental quality report with figures and alerts.

Usage:
  python3 analysis/generate_report.py --analysis-dir analysis_outputs --maps-dir maps --figures-dir figures --alerts-dir alerts --outdir report
"""
from __future__ import annotations

import argparse
import csv
import html
import os
from typing import List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str, max_chars: int = 20000) -> str:
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read(max_chars)
    return s


def read_csv_preview(path: str, limit: int = 50) -> List[List[str]]:
    rows: List[List[str]] = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            rows.append(row)
            if i >= limit:
                break
    return rows


def generate_html(analysis_dir: str, maps_dir: str, figures_dir: str, alerts_dir: str, outdir: str) -> str:
    ensure_dir(outdir)
    report_txt = os.path.join(analysis_dir, 'report.txt')
    pollutant_summary_csv = os.path.join(analysis_dir, 'pollutant_summary.csv')
    alerts_csv = os.path.join(alerts_dir, 'alerts.csv')

    report_text = html.escape(read_text(report_txt))
    pollutant_preview = read_csv_preview(pollutant_summary_csv, 100)
    alerts_preview = read_csv_preview(alerts_csv, 100)

    # Gather images if present
    images = []
    for d in [maps_dir, figures_dir]:
        if not os.path.exists(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith('.png'):
                images.append((d, name))

    html_path = os.path.join(outdir, 'report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('<!DOCTYPE html><html><head><meta charset="utf-8"/>')
        f.write('<title>Environmental Quality Report</title>')
        f.write('<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;}table{border-collapse:collapse;margin:12px 0;}td,th{border:1px solid #ccc;padding:6px 8px;}h2{margin-top:28px;}pre{background:#f7f7f7;padding:12px;white-space:pre-wrap;}</style>')
        f.write('</head><body>')
        f.write('<h1>Environmental Quality Report</h1>')

        f.write('<h2>Summary (EDA)</h2>')
        f.write('<pre>')
        f.write(report_text)
        f.write('</pre>')

        if pollutant_preview:
            f.write('<h2>Pollutant Summary (preview)</h2><table>')
            for i, row in enumerate(pollutant_preview):
                tag = 'th' if i == 0 else 'td'
                f.write('<tr>')
                for cell in row:
                    f.write(f'<{tag}>{html.escape(cell)}</{tag}>')
                f.write('</tr>')
            f.write('</table>')

        if alerts_preview:
            f.write('<h2>Alerts (preview)</h2><table>')
            for i, row in enumerate(alerts_preview):
                tag = 'th' if i == 0 else 'td'
                f.write('<tr>')
                for cell in row:
                    f.write(f'<{tag}>{html.escape(cell)}</{tag}>')
                f.write('</tr>')
            f.write('</table>')

        if images:
            f.write('<h2>Figures</h2>')
            for d, name in images:
                rel = os.path.relpath(os.path.join(d, name), start=outdir)
                f.write(f'<div style="margin:12px 0"><img src="{html.escape(rel)}" alt="{html.escape(name)}" style="max-width:900px;width:100%"/></div>')

        f.write('</body></html>')

    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate HTML environmental quality report')
    parser.add_argument('--analysis-dir', default='analysis_outputs')
    parser.add_argument('--maps-dir', default='maps')
    parser.add_argument('--figures-dir', default='figures')
    parser.add_argument('--alerts-dir', default='alerts')
    parser.add_argument('--outdir', default='report')
    args = parser.parse_args()

    out = generate_html(args.analysis_dir, args.maps_dir, args.figures_dir, args.alerts_dir, args.outdir)
    print(f'Report HTML written to: {out}')


if __name__ == '__main__':
    main()

