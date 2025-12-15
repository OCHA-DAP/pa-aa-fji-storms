import base64
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import unary_union
from shapely.affinity import translate

from src.constants import FJI_CRS, NM_TO_M, QUADS
from src.datasources import ibtracs

SPEED2WORD = {34: "Gale", 50: "Storm", 64: "Hurricane"}
SPEEDS = [34, 50, 64]
QUAD_COL_RENAME = {
    f"{x.upper()}{speedword}Radius": f"quadrant_radius_{speed}_{x}"
    for speed, speedword in SPEED2WORD.items()
    for x in QUADS
}


def datetime_to_season(dt: datetime) -> int:
    year = dt.year
    if dt.month >= 7:
        season = year + 1
    else:
        season = year
    return season


def decode_b64_string(csv: str) -> StringIO:
    """Decodes encoded string of CSV.

    Parameters
    ----------
    csv: str
        String of CSV (received as command line argument of script)

    Returns
    -------
    StringIO
        StringIO of CSV, to be used in process_fms_forecast()
    """
    bytes_str = csv.encode("ascii") + b"=="
    converted_bytes = base64.b64decode(bytes_str)
    csv_str = converted_bytes.decode("ascii")
    str_out = StringIO(csv_str)
    return str_out


def parse_fms_forecast(
    path: Path | StringIO | BytesIO, best_track: bool = False
) -> gpd.GeoDataFrame:
    """Loads FMS raw forecast in default CSV export format from FMS cyclone
    forecast software.
    Parameters
    ----------
    path: Path | StringIO
        Path to raw forecast CSV. Path can be a StringIO
        (so CSV can be passed as an encoded string from Power Automate)

    Returns
    -------
    DataFrame of processed forecast
    """

    df_date = pd.read_csv(path, header=None, nrows=3)
    if not best_track:
        date_str = df_date.iloc[0, 1].removeprefix("baseTime=")
        base_time = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    cyclone_name = (
        df_date.iloc[2, 0].removeprefix("# CycloneName=").capitalize()
    )
    if isinstance(path, StringIO) or isinstance(path, BytesIO):
        path.seek(0)
    df_data = pd.read_csv(path, skiprows=range(6))
    df_data = df_data.drop([0])
    df_data = df_data.rename(
        columns={"Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']": "forecast_time"}
    )
    df_data["forecast_time"] = pd.to_datetime(
        df_data["forecast_time"]
    ).dt.tz_localize(None)
    if best_track:
        df_data = df_data.rename(columns={"forecast_time": "valid_time"})
    df_data["cyclone_name"] = cyclone_name
    if not best_track:
        df_data["base_time"] = base_time
        df_data["season"] = datetime_to_season(base_time)
    else:
        df_data["season"] = datetime_to_season(df_data["valid_time"].iloc[0])
    df_data["Name Season"] = (
        df_data["cyclone_name"] + " " + df_data["season"].astype(str)
    )
    if not best_track:
        df_data["leadtime"] = df_data["forecast_time"] - df_data["base_time"]
        df_data["leadtime"] = (
            df_data["leadtime"].dt.days * 24
            + df_data["leadtime"].dt.seconds / 3600  # noqa
        ).astype(int)
    df_data["Category"] = df_data["Category"].fillna(0)
    df_data["Category"] = df_data["Category"].astype(int, errors="ignore")

    gdf = gpd.GeoDataFrame(
        df_data,
        geometry=gpd.points_from_xy(df_data["Longitude"], df_data["Latitude"]),
    )
    gdf = gdf.set_crs(FJI_CRS)
    return gdf


def calculate_fms_buffers(gdf, best_track: bool = False):
    gdf["Longitude"] = gdf["Longitude"].apply(lambda x: (x + 360) % 360)
    name_season = gdf.iloc[0]["Name Season"]
    if not best_track:
        issued_time = gdf.iloc[0]["base_time"]
    for speed in SPEEDS:
        speedword = SPEED2WORD[speed]
        gdf = gdf.rename(
            columns={
                f"{x.upper()}{speedword}Radius": f"quadrant_radius_{speed}_{x}"
                for x in QUADS
            }
        )
    cols = ["valid_time", "Latitude", "Longitude", "Category", "MeanWind"] + [
        f"quadrant_radius_{speed}_{x}" for speed in SPEEDS for x in QUADS
    ]
    df = gdf[cols]
    df_interp = ibtracs.interpolate_track(
        df, time_col="valid_time", lat_col="Latitude", lon_col="Longitude"
    )
    gdf_interp = gpd.GeoDataFrame(
        data=df_interp,
        geometry=gpd.points_from_xy(
            df_interp["Longitude"], df_interp["Latitude"]
        ),
        crs=FJI_CRS,
    ).to_crs(3832)

    dicts = []
    geoms = []

    def convert_nm_to_m(value_nm: float):
        return np.nan_to_num(value_nm * NM_TO_M, nan=0)

    n_points = 360

    for speed in SPEEDS:
        polys = []
        for _, row in gdf_interp.iterrows():
            ne_col, se_col, sw_col, nw_col = [
                f"quadrant_radius_{speed}_{x}"
                for x in ["ne", "se", "sw", "nw"]
            ]
            if row[[ne_col, se_col, sw_col, nw_col]].isna().all():
                continue

            poly = ibtracs.make_quadrant_disk(
                (row.geometry.x, row.geometry.y),
                ne=convert_nm_to_m(row[ne_col]),
                se=convert_nm_to_m(row[se_col]),
                sw=convert_nm_to_m(row[sw_col]),
                nw=convert_nm_to_m(row[nw_col]),
                n_points=n_points,
            )
            polys.append(poly)
        gdf = gpd.GeoDataFrame(geometry=polys, crs=3832)
        merged = unary_union(gdf.geometry.values)
        if not best_track:
            dicts.append(
                {
                    "buffer_speed": speed,
                    "name_season": name_season,
                    "issued_time": issued_time,
                }
            )
        else:
            dicts.append(
                {
                    "buffer_speed": speed,
                    "name_season": name_season,
                }
            )
        geoms.append(merged)
    return geoms, dicts


def calculate_fms_buffers_gdf(
    gdf: gpd.GeoDataFrame, best_track: bool = False
) -> gpd.GeoDataFrame:
    geoms, dicts = calculate_fms_buffers(gdf, best_track=best_track)
    gdf_out = gpd.GeoDataFrame(dicts, geometry=geoms, crs=3832)
    gdf_out = gdf_out.to_crs(FJI_CRS)
    return gdf_out


def shift_gdf_points(
    gdf: gpd.GeoDataFrame,
    azimuth_deg: float,
    distance_col: str = "uncertainty_m",
    geographic_crs: str = FJI_CRS,
    projected_crs: str = "EPSG:3857",
    longitude_col: str = "Longitude",
    latitude_col: str = "Latitude",
):
    gdf = gdf.to_crs(projected_crs)
    angle_rad = np.deg2rad(azimuth_deg)
    dx = gdf[distance_col] * np.sin(angle_rad)
    dy = gdf[distance_col] * np.cos(angle_rad)
    new_geometry = [
        translate(geom, xoff=x, yoff=y)
        for geom, x, y in zip(gdf.geometry, dx, dy)
    ]
    df_out = gdf.drop(columns="geometry")
    df_out["shift_deg"] = azimuth_deg
    df_out["shift_distance_m"] = gdf[distance_col]
    gdf_out = gpd.GeoDataFrame(df_out, geometry=new_geometry, crs=gdf.crs)
    gdf_out = gdf_out.to_crs(geographic_crs)
    gdf_out[longitude_col] = gdf_out.geometry.x
    gdf_out[latitude_col] = gdf_out.geometry.y
    return gdf_out


def get_forecast_id(
    gdf: gpd.GeoDataFrame,
) -> str:
    row = gdf.iloc[0]
    return f"{row['cyclone_name'].lower().replace(' ', '_')}_{row['season']}_{row['base_time']:%Y%m%dT%H%MZ}"


def to_fji_time(dt):
    dt_utc = dt.replace(tzinfo=ZoneInfo("UTC"))
    dt_fiji = dt_utc.astimezone(ZoneInfo("Pacific/Fiji"))
    return dt_fiji


def fji_time_str(dt):
    dt_fiji = to_fji_time(dt)
    return f"{dt_fiji:%Y-%m-%d %H:%M} (Fiji time)"
