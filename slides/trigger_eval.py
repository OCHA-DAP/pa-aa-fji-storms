"""Evaluate FMS forecast CSVs with the live trigger code, for the slide decks.

Uses the same functions as pipelines/monitor_forecast_trigger.py so every
number on the slides is what the monitoring system computes (or would have
computed) for that forecast.
"""

from functools import lru_cache
from pathlib import Path

import geopandas as gpd

from src.constants import EXP_THRESHOLD_64_KNOTS, FJI_CRS
from src.datasources import codab, worldpop
from src.datasources.fms import (
    calculate_fms_buffers_gdf,
    calculate_shifted_exposures,
    calculate_uncertainty_cone,
    parse_fms_forecast,
    to_fji_time,
)
from src.exposure_calc import calculate_single_adm_exposure


@lru_cache
def load_static():
    adm3 = codab.load_codab_from_blob(admin_level=3).to_crs(FJI_CRS)
    da_wp = worldpop.load_worldpop_from_blob()
    da_wp = da_wp.assign_coords({"x": ((da_wp.x + 360) % 360)}).sortby("x")
    da_wp_clip = da_wp.rio.clip(adm3.geometry)
    return adm3, da_wp_clip


def evaluate_forecast(path: Path, with_shifts: bool = False) -> dict:
    adm3, da_wp = load_static()
    gdf = parse_fms_forecast(path)
    gdf = gdf.rename(columns={"forecast_time": "valid_time"})
    gdf["Longitude"] = (gdf["Longitude"] + 360) % 360
    gdf = gdf[gdf["leadtime"] <= 120].copy()
    gdf_action = gdf[gdf["leadtime"] <= 72].copy()

    buf_r = calculate_fms_buffers_gdf(gdf.copy())
    buf_a = calculate_fms_buffers_gdf(gdf_action.copy())
    exp_r = calculate_single_adm_exposure(buf_r, da_wp)
    exp_a = calculate_single_adm_exposure(buf_a, da_wp)
    exp_r = exp_r.set_index("buffer_speed")["pop_exposed"].to_dict()
    exp_a = exp_a.set_index("buffer_speed")["pop_exposed"].to_dict()

    out = {
        "name": gdf.iloc[0]["cyclone_name"],
        "issued_utc": gdf.iloc[0]["base_time"],
        "issued_fji": to_fji_time(gdf.iloc[0]["base_time"]),
        "track": gdf,
        "buffers": buf_r,
        "buffers_action": buf_a,
        "exp_readiness": exp_r,
        "exp_action": exp_a,
        "readiness": exp_r.get(64, 0) >= EXP_THRESHOLD_64_KNOTS,
        "action": exp_a.get(64, 0) >= EXP_THRESHOLD_64_KNOTS,
    }
    if "Uncertainty" in gdf and gdf["Uncertainty"].notna().any():
        out["cone"] = calculate_uncertainty_cone(gdf.copy())
    if with_shifts:
        df_shift, _, _ = calculate_shifted_exposures(gdf.copy(), da_wp)
        out["worst_64"] = int(df_shift["exp_64"].max())
    # closest approach of the 64-kt zone to Fiji's land
    land = adm3.to_crs(3832).union_all()
    b64 = buf_r[buf_r["buffer_speed"] == 64].to_crs(3832).geometry.iloc[0]
    out["dist_64_km"] = (
        None if b64 is None or b64.is_empty else land.distance(b64) / 1000
    )
    centre = gpd.GeoSeries(gdf.geometry, crs=FJI_CRS).to_crs(3832)
    out["dist_centre_km"] = min(land.distance(p) for p in centre) / 1000
    return out
