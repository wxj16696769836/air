import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Config:
    input_csv: str = "china_air_quality_raw.csv"
    output_dir: str = "outputs"
    city: Optional[str] = None  # e.g., "Beijing"
    pm25_column: str = "PM2.5 (µg/m³)"
    lon_column: str = "Longitude"
    lat_column: str = "Latitude"
    timestamp_columns: Tuple[str, str, str, str] = ("Year", "Month", "Day of Week", "Hour")
    season_column: str = "Season"
    grid_size_m: int = 5000  # 5km grid for heatmap
    pm25_heavy_threshold: float = 150.0


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataframe(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv)
    # Basic cleaning: drop rows without coordinates or PM2.5
    df = df.dropna(subset=[cfg.lat_column, cfg.lon_column, cfg.pm25_column])
    # Coerce numeric columns just in case
    for col in [cfg.pm25_column, cfg.lat_column, cfg.lon_column]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[cfg.pm25_column, cfg.lat_column, cfg.lon_column])
    return df


def to_geodataframe(df: pd.DataFrame, cfg: Config) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[cfg.lon_column], df[cfg.lat_column]),
        crs="EPSG:4326",
    )
    return gdf


def project_to_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Use World Mercator for approximate meter units
    return gdf.to_crs(epsg=3857)


def aggregate_grid_heatmap(gdf_wm: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    # Determine grid extent
    bounds = gdf_wm.total_bounds  # minx, miny, maxx, maxy
    minx, miny, maxx, maxy = bounds
    grid_size = cfg.grid_size_m

    xs = np.arange(minx, maxx + grid_size, grid_size)
    ys = np.arange(miny, maxy + grid_size, grid_size)

    polygons = []
    for x in xs[:-1]:
        for y in ys[:-1]:
            polygons.append(Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]))

    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=gdf_wm.crs)

    # Spatial join points to grid cells then aggregate PM2.5
    join = gpd.sjoin(gdf_wm[[cfg.pm25_column, "geometry"]], grid, predicate="within", how="left")
    agg = join.groupby("index_right", as_index=False)[cfg.pm25_column].mean()
    grid[cfg.pm25_column] = agg.set_index("index_right")[cfg.pm25_column]
    return grid


def plot_pm25_heatmap(grid_wm: gpd.GeoDataFrame, cfg: Config, city_name: str) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    grid_wgs84 = grid_wm.to_crs(epsg=4326)
    grid_wgs84.plot(column=cfg.pm25_column, ax=ax, cmap="inferno", legend=True, missing_kwds={"color": "lightgrey"})
    ax.set_title(f"PM2.5 Heatmap - {city_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    out_path = os.path.join(cfg.output_dir, f"pm25_heatmap_{city_name}.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def filter_city(gdf: gpd.GeoDataFrame, cfg: Config) -> Tuple[gpd.GeoDataFrame, str]:
    if cfg.city and "City" in gdf.columns:
        city_gdf = gdf[gdf["City"].str.lower() == cfg.city.lower()]
        if not city_gdf.empty:
            return city_gdf, cfg.city
    # fallback: choose the most frequent city
    if "City" in gdf.columns and gdf["City"].notna().any():
        top_city = gdf["City"].value_counts().idxmax()
        return gdf[gdf["City"] == top_city], top_city
    # if no city column, just use all points and label "All"
    return gdf, "All"


def extract_time_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Columns: Year, Month, Day of Week, Hour already exist per sample header
    # Ensure types
    for col in cfg.timestamp_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def generate_pm25_alerts(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    alerts = df[df[cfg.pm25_column] > cfg.pm25_heavy_threshold].copy()
    if "City" in alerts.columns:
        alerts = alerts.sort_values(["City", cfg.pm25_column], ascending=[True, False])
    else:
        alerts = alerts.sort_values(cfg.pm25_column, ascending=False)
    return alerts


def plot_temporal_trends(df: pd.DataFrame, cfg: Config, city_name: str) -> Tuple[str, str, str]:
    out_hourly = os.path.join(cfg.output_dir, f"pm25_hourly_{city_name}.png")
    out_monthly = os.path.join(cfg.output_dir, f"pm25_monthly_{city_name}.png")
    out_season = os.path.join(cfg.output_dir, f"pm25_season_{city_name}.png")

    # Hourly
    if "Hour" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=df, x="Hour", y=cfg.pm25_column, estimator="mean", errorbar=("se", 1), ax=ax)
        ax.set_title(f"PM2.5 Hourly Trend - {city_name}")
        fig.savefig(out_hourly, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Monthly
    if "Month" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=df, x="Month", y=cfg.pm25_column, estimator="mean", errorbar=("se", 1), ax=ax)
        ax.set_title(f"PM2.5 Monthly Trend - {city_name}")
        fig.savefig(out_monthly, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Season
    if cfg.season_column in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        order = ["Spring", "Summer", "Autumn", "Winter"]
        sns.barplot(data=df, x=cfg.season_column, y=cfg.pm25_column, order=[s for s in order if s in df[cfg.season_column].unique()], ax=ax)
        ax.set_title(f"PM2.5 Seasonal Differences - {city_name}")
        fig.savefig(out_season, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return out_hourly, out_monthly, out_season


def main():
    cfg = Config()
    ensure_output_dir(cfg.output_dir)

    df = load_dataframe(cfg)
    df = extract_time_features(df, cfg)

    gdf = to_geodataframe(df, cfg)
    city_gdf, city_name = filter_city(gdf, cfg)

    # Project to meters and aggregate for heatmap
    gdf_wm = project_to_meters(city_gdf)
    grid = aggregate_grid_heatmap(gdf_wm, cfg)
    heatmap_path = plot_pm25_heatmap(grid, cfg, city_name)

    # Alerts
    alerts_df = generate_pm25_alerts(df, cfg)
    alerts_out = os.path.join(cfg.output_dir, f"pm25_alerts_threshold_{int(cfg.pm25_heavy_threshold)}.csv")
    alerts_df.to_csv(alerts_out, index=False)

    # Temporal trends (city filtered where possible)
    trend_df = city_gdf.drop(columns=["geometry"]) if "geometry" in city_gdf.columns else df
    plot_temporal_trends(trend_df, cfg, city_name)

    print("Generated outputs in:", os.path.abspath(cfg.output_dir))
    print("Heatmap:", heatmap_path)
    print("Alerts:", alerts_out)


if __name__ == "__main__":
    main()
