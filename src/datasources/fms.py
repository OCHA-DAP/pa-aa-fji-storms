from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.constants import FJI_CRS


def datetime_to_season(dt: datetime) -> int:
    year = dt.year
    if dt.month >= 7:
        season = year + 1
    else:
        season = year
    return season


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
