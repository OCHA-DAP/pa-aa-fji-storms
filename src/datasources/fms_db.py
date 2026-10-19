"""Load an FMS forecast from the storms database, shaped like the CSV parse.

ds-storms-pipeline (``src/pipelines/fms.py``) writes the forecast CSVs that
RSMC Nadi emails into ``storms.fms_forecasts`` / ``storms.fms_tracks_geo``.
``load_fms_forecast_from_db`` rebuilds the frame ``parse_fms_forecast``
returns for the same CSV (FMS column names, 0-360 longitudes, the derived
columns, index from 1), so ``monitor_forecast_trigger.py`` runs unchanged on
either source.
"""

import json
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.constants import FJI_CRS
from src.datasources.fms import datetime_to_season

OFFICIAL_TRACK = "Official Forecast Track"

# The FMS export's data columns after the time column, in file order.
FMS_COLUMNS = [
    "Latitude",
    "Longitude",
    "Symbol",
    "Category",
    "Pressure",
    "PressureOCI",
    "RadiusOCI",
    "Radius1000hPa",
    "RadiusMaxWinds",
    "MeanWind",
    "WindGust",
    "VerticalExtent",
    "Uncertainty",
    "FinalT",
    "CurrentIntensity",
    "GaleRadius",
    "StrongGaleRadius",
    "StormRadius",
    "HurricaneRadius",
    "NEGaleRadius",
    "SEGaleRadius",
    "SWGaleRadius",
    "NWGaleRadius",
    "NEStrongGaleRadius",
    "SEStrongGaleRadius",
    "SWStrongGaleRadius",
    "NWStrongGaleRadius",
    "NEStormRadius",
    "SEStormRadius",
    "SWStormRadius",
    "NWStormRadius",
    "NEHurricaneRadius",
    "SEHurricaneRadius",
    "SWHurricaneRadius",
    "NWHurricaneRadius",
    "P5Wind",
    "DataSource",
    "Land/Water",
    "CycloneStatus",
    "HowLocation",
    "DataTNoDvorak",
    "ModelTNoDvorak",
    "PatternTNoDvorak",
    "UncPressure",
    "HowPressure",
    "P/W_Method",
    "HowMaxWindRadius",
    "HowGaleRadius",
    "HowStormRadius",
    "HowHurricaneRadius",
    "UncMaxWindSpeed",
    "HowMaxWindSpeed",
    "HowGust",
    "EyeRadius",
    "UncEyeRadius",
    "HowEyeRadius",
]

# FMS column -> fms_tracks_geo column (mirrors ds-storms-pipeline's writer).
SCALAR_COLS = {
    "MeanWind": "wind_speed",
    "WindGust": "gust_speed",
    "Pressure": "pressure",
    "RadiusMaxWinds": "max_wind_radius",
    "RadiusOCI": "last_closed_isobar_radius",
    "PressureOCI": "last_closed_isobar_pressure",
    "Category": "category",
    "Uncertainty": "uncertainty",
}
RADIUS_WORDS = {34: "Gale", 48: "StrongGale", 50: "Storm", 64: "Hurricane"}
QUAD_ORDER = ("NE", "SE", "SW", "NW")

_DB_COLUMNS = [
    *SCALAR_COLS.values(),
    *(f"radius_{speed}" for speed in RADIUS_WORDS),
    *(f"quadrant_radius_{speed}" for speed in RADIUS_WORDS),
]

_QUERY = text(
    "SELECT f.name_as_issued, t.valid_time, "
    + ", ".join(f"t.{c}" for c in _DB_COLUMNS)
    + ", t.raw_fields::text AS raw_fields,"
    " ST_X(t.geometry) AS lon, ST_Y(t.geometry) AS lat"
    " FROM storms.fms_tracks_geo t"
    " JOIN storms.fms_forecasts f"
    " USING (disturbance_id, track_type, issued_time)"
    " WHERE t.disturbance_id = :disturbance_id"
    " AND t.track_type = :track_type"
    " AND t.issued_time = :issued_time"
    " ORDER BY t.valid_time"
)


def _float(value) -> float:
    return np.nan if value is None else float(value)


def _source_values(rows: pd.DataFrame) -> dict[str, list]:
    """Each FMS column's values, rebuilt from the database columns."""
    raw = [
        (json.loads(r) if isinstance(r, str) else r or {})
        for r in rows["raw_fields"]
    ]
    values = {
        col: [_float(v) for v in rows[db_col]]
        for col, db_col in SCALAR_COLS.items()
    }
    for speed, word in RADIUS_WORDS.items():
        values[f"{word}Radius"] = [_float(v) for v in rows[f"radius_{speed}"]]
        quads = [
            json.loads(q) if q else [None] * 4
            for q in rows[f"quadrant_radius_{speed}"]
        ]
        for i, quad in enumerate(QUAD_ORDER):
            values[f"{quad}{word}Radius"] = [_float(q[i]) for q in quads]
    values["Latitude"] = [float(v) for v in rows["lat"]]
    # The database holds [-180, 180); FMS writes east of the dateline as
    # 180-360, which is what the CSV parse (and so the monitor) sees.
    values["Longitude"] = [
        float(v) + 360 if v < 0 else float(v) for v in rows["lon"]
    ]
    for col in FMS_COLUMNS:
        if col not in values:
            values[col] = [r.get(col, np.nan) for r in raw]
    return values


def rows_to_forecast(
    rows: pd.DataFrame, issued_time: datetime
) -> gpd.GeoDataFrame:
    """Turn the query's rows into ``parse_fms_forecast``'s output frame."""
    if rows.empty:
        raise ValueError(
            f"No track points for the forecast issued {issued_time}"
        )
    names = rows["name_as_issued"].unique()
    if len(names) != 1:
        raise ValueError(f"Expected one forecast, got names {list(names)}")

    values = _source_values(rows)
    df = pd.DataFrame(index=pd.RangeIndex(1, len(rows) + 1))
    df["forecast_time"] = pd.to_datetime(rows["valid_time"]).to_numpy()
    for col in FMS_COLUMNS:
        df[col] = values[col]

    # Same derived columns, computed the same way, as parse_fms_forecast.
    base_time = pd.Timestamp(issued_time).to_pydatetime()
    cyclone_name = (names[0] or "").capitalize()
    df["cyclone_name"] = cyclone_name
    df["base_time"] = base_time
    df["season"] = datetime_to_season(base_time)
    df["Name Season"] = df["cyclone_name"] + " " + df["season"].astype(str)
    df["leadtime"] = df["forecast_time"] - df["base_time"]
    df["leadtime"] = (
        df["leadtime"].dt.days * 24 + df["leadtime"].dt.seconds / 3600
    ).astype(int)
    df["Category"] = df["Category"].fillna(0)
    df["Category"] = df["Category"].astype(int, errors="ignore")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    )
    return gdf.set_crs(FJI_CRS)


def load_fms_forecast_from_db(
    engine,
    disturbance_id: str,
    issued_time: str | datetime,
    track_type: str = OFFICIAL_TRACK,
) -> gpd.GeoDataFrame:
    """One forecast from storms.fms_*, as ``parse_fms_forecast`` returns it."""
    issued = pd.Timestamp(issued_time)
    if issued.tzinfo is not None:
        issued = issued.tz_convert(None)
    params = {
        "disturbance_id": disturbance_id,
        "track_type": track_type,
        "issued_time": issued.to_pydatetime(),
    }
    with engine.connect() as conn:
        rows = pd.read_sql(_QUERY, conn, params=params)
    return rows_to_forecast(rows, issued.to_pydatetime())
