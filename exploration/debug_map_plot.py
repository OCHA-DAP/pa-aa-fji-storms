"""Debug script: generate the bubbles/swaths PDF using TEST_FORECAST_BLOB_NAME."""
import os
from io import BytesIO

import geopandas as gpd
import ocha_stratus as stratus
import pandas as pd
from dotenv import load_dotenv

from src.blob import PROJECT_PREFIX
from src.constants import FJI_CRS, LAU2, ROTUMA2
from src.datasources import codab, worldpop
from src.datasources.fms import (
    calculate_fms_buffers_gdf,
    calculate_shifted_exposures,
    calculate_uncertainty_cone,
    fji_time_str,
    parse_fms_forecast,
)
from src.exposure_calc import calculate_multi_adm_exposure
from src.plotting import plot_bubbles_and_swaths

load_dotenv()

os.makedirs("temp", exist_ok=True)

blob_name = os.environ["TEST_FORECAST_BLOB_NAME"]
print(f"Loading {blob_name}")
decoded_csv = BytesIO(stratus.load_blob_data(blob_name))

gdf_forecast = parse_fms_forecast(decoded_csv)
gdf_forecast = gdf_forecast.rename(columns={"forecast_time": "valid_time"})

row = gdf_forecast.iloc[0]
cyclone_name = row["cyclone_name"]
season = row["season"]
issued_time = row["base_time"]
forecast_display_str = fji_time_str(issued_time)
forecast_id = (
    f"{cyclone_name.lower().replace(' ', '_')}_{season}_fcast_"
    f"{issued_time:%Y%m%dT%H%MZ}"
)
print(f"Cyclone: {cyclone_name}, forecast ID: {forecast_id}")

gdf_readiness = gdf_forecast[gdf_forecast["leadtime"] <= 120].copy()

print("Calculating buffers...")
gdf_buffers_readiness = calculate_fms_buffers_gdf(gdf_readiness)
gdf_buffers_action = calculate_fms_buffers_gdf(
    gdf_forecast[gdf_forecast["leadtime"] <= 72].copy()
)

print("Loading worldpop + adm3...")
adm3 = codab.load_codab_from_blob(admin_level=3).to_crs(FJI_CRS)
da_wp = worldpop.load_worldpop_from_blob()
da_wp = da_wp.assign_coords({"x": ((da_wp.x + 360) % 360)}).sortby("x")
da_wp_clip = da_wp.rio.clip(adm3.geometry)

print("Calculating shifted exposures (perturbed tracks)...")
(
    df_exp_shift,
    gdf_shift_buffers,
    gdf_shift_tracks,
) = calculate_shifted_exposures(gdf_readiness, da_wp_clip)
df_exp_shift = df_exp_shift.sort_values(
    [f"exp_{x}" for x in [64, 50, 34]], ascending=False
)
worst_row = df_exp_shift.iloc[0]
best_row = df_exp_shift.iloc[-1]
if (best_row[[f"exp_{s}" for s in [34, 50, 64]]] == 0).all():
    opposite_deg = (worst_row["shift_deg"] + 180) % 360
    best_row = df_exp_shift.loc[
        (df_exp_shift["shift_deg"] - opposite_deg).abs().idxmin()
    ]

worst_buffers = gdf_shift_buffers[
    gdf_shift_buffers["shift_deg"] == worst_row["shift_deg"]
]
best_buffers = gdf_shift_buffers[
    gdf_shift_buffers["shift_deg"] == best_row["shift_deg"]
]

print("Calculating ADM3 exposures...")
df_exp_adm3_mostlikely = calculate_multi_adm_exposure(
    gdf_buffers_readiness, da_wp_clip, adm3
)
df_exp_adm3_worst = calculate_multi_adm_exposure(
    worst_buffers, da_wp_clip, adm3
)
df_exp_adm3_best = calculate_multi_adm_exposure(best_buffers, da_wp_clip, adm3)

df_exp_adm3_best["limit"] = "best"
df_exp_adm3_worst["limit"] = "worst"
df_exp_adm3_mostlikely["limit"] = "middle"
df_exp_adm3 = pd.concat(
    [df_exp_adm3_best, df_exp_adm3_mostlikely, df_exp_adm3_worst],
    ignore_index=True,
)

adm3_simple_template = stratus.load_parquet_from_blob(
    f"{PROJECT_PREFIX}/processed/plotting/adm3_simple_template.parquet"
)

adm3_no_rotuma_lau = adm3[~adm3["ADM2_PCODE"].isin([ROTUMA2, LAU2])]

uncertainty_cone = calculate_uncertainty_cone(gdf_readiness)


def _track_gdf(gdf):
    return gdf.set_geometry(
        gpd.points_from_xy(gdf["Longitude"], gdf["Latitude"])
    ).set_crs(FJI_CRS)


worst_track = gdf_shift_tracks[
    gdf_shift_tracks["shift_deg"] == worst_row["shift_deg"]
]
best_track = gdf_shift_tracks[
    gdf_shift_tracks["shift_deg"] == best_row["shift_deg"]
]
gdf_tracks_plot = [
    _track_gdf(gdf_readiness),
    _track_gdf(worst_track),
    _track_gdf(best_track),
]

print("Plotting...")
fig, axs, filepath = plot_bubbles_and_swaths(
    gdf_mostlikely_buffers=gdf_buffers_readiness,
    gdf_worst_buffers=worst_buffers,
    gdf_best_buffers=best_buffers,
    gdf_adm3_swath_plot=adm3_no_rotuma_lau,
    df_adm3_template=adm3_simple_template,
    gdf_adm3=adm3,
    df_exp_adm3=df_exp_adm3,
    cyclone_name=cyclone_name,
    forecast_display_str=forecast_display_str,
    forecast_id=forecast_id,
    save_local=True,
    gdf_tracks_plot=gdf_tracks_plot,
    uncertainty_cone=uncertainty_cone,
    forecast_label=[
        "Most likely track",
        "Worst case track",
        "Best case track",
    ],
)

print(f"Saved to {filepath}")
