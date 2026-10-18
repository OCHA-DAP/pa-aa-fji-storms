"""The database loader must rebuild exactly what parse_fms_forecast gives.

The CSV is synthetic (FMS export layout, made-up values). ``to_db_rows``
plays ds-storms-pipeline's writer (src/pipelines/fms.py): it maps a parsed
CSV row to what storms.fms_tracks_geo holds and the loader's query returns.
"""

import json
import math
from io import BytesIO

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.datasources.fms import parse_fms_forecast
from src.datasources.fms_db import (
    FMS_COLUMNS,
    QUAD_ORDER,
    RADIUS_WORDS,
    SCALAR_COLS,
    rows_to_forecast,
)

TIME_COL = "Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']"

ROWS = [
    {
        TIME_COL: "2026-04-07T00:00:00Z",
        "Latitude": -18.1,
        "Longitude": 174.8,
        "Symbol": 4.0,
        "Category": 3.0,
        "MeanWind": 85.0,
        "WindGust": 120.0,
        "Uncertainty": 50.0,
        "NEGaleRadius": 110.0,
        "SEGaleRadius": 110.0,
        "SWGaleRadius": 70.0,
        "NWGaleRadius": 70.0,
        "StormRadius": 40.0,
        "HurricaneRadius": 25.0,
        "P5Wind": 105.0,
    },
    {
        TIME_COL: "2026-04-07T06:00:00Z",
        "Latitude": -19.83333,
        "Longitude": 181.18333,
        "Category": 3.0,
        "MeanWind": 80.0,
        "Uncertainty": 65.0,
        "NEGaleRadius": 100.0,
        "SWGaleRadius": 60.0,
        "HurricaneRadius": 20.0,
    },
    {
        TIME_COL: "2026-04-07T12:00:00Z",
        "Latitude": -21.0,
        "Longitude": 199.5,
        "Uncertainty": 80.0,
    },
]


def make_csv(name="RAE [RAY]") -> bytes:
    lines = [
        "# trackType=Official Forecast Track,baseTime=2026-04-07T00:00:00Z",
        "# DisturbanceId=2526-11F",
        f"# CycloneName={name}",
        "# MathType of Data",
        "(Time->(Latitude,Longitude,...))",
        "# Column Headings",
        ",".join([TIME_COL, *FMS_COLUMNS]),
        "# Start of data",
    ]
    for row in ROWS:
        lines.append(
            ",".join(str(row.get(c, "NaN")) for c in [TIME_COL, *FMS_COLUMNS])
        )
    return ("\n".join(lines) + "\n").encode()


def _num(v):
    return (
        None
        if v is None or (isinstance(v, float) and math.isnan(v))
        else float(v)
    )


def to_db_rows(csv: bytes, name_as_issued: str) -> pd.DataFrame:
    raw = pd.read_csv(BytesIO(csv), skiprows=range(6)).drop([0])
    consumed = {TIME_COL, "Latitude", "Longitude", *SCALAR_COLS}
    for word in RADIUS_WORDS.values():
        consumed.add(f"{word}Radius")
        consumed.update(f"{q}{word}Radius" for q in QUAD_ORDER)
    out = []
    for _, r in raw.iterrows():
        lon = float(r["Longitude"])
        lon = lon - 360 if lon >= 180 else lon
        row = {
            "name_as_issued": name_as_issued,
            "valid_time": pd.Timestamp(r[TIME_COL]).tz_convert(None),
            "lat": float(r["Latitude"]),
            "lon": lon,
        }
        for col, db_col in SCALAR_COLS.items():
            row[db_col] = _num(r[col])
        for speed, word in RADIUS_WORDS.items():
            row[f"radius_{speed}"] = _num(r[f"{word}Radius"])
            quads = [_num(r[f"{q}{word}Radius"]) for q in QUAD_ORDER]
            row[f"quadrant_radius_{speed}"] = (
                None if all(q is None for q in quads) else json.dumps(quads)
            )
        extra = {
            c: float(v)
            for c, v in r.items()
            if c not in consumed and _num(v) is not None
        }
        row["raw_fields"] = json.dumps(extra) if extra else None
        out.append(row)
    return pd.DataFrame(out)


def test_db_rows_rebuild_the_csv_parse():
    csv = make_csv()
    expected = parse_fms_forecast(BytesIO(csv))
    got = rows_to_forecast(
        to_db_rows(csv, "RAE [RAY]"), pd.Timestamp("2026-04-07T00:00")
    )

    assert list(got.columns) == list(expected.columns)
    assert_frame_equal(
        pd.DataFrame(got.drop(columns="geometry")),
        pd.DataFrame(expected.drop(columns="geometry")),
        check_exact=False,
        rtol=1e-12,
    )
    assert got.crs == expected.crs
    assert np.allclose(got.geometry.x, expected.geometry.x, rtol=1e-12)
    assert np.allclose(got.geometry.y, expected.geometry.y, rtol=1e-12)


def test_names_and_derived_columns():
    got = rows_to_forecast(
        to_db_rows(make_csv(), "RAE [RAY]"), pd.Timestamp("2026-04-07T00:00")
    )
    assert got["cyclone_name"].iloc[0] == "Rae [ray]"
    assert got["Name Season"].iloc[0] == "Rae [ray] 2026"
    assert got["leadtime"].tolist() == [0, 6, 12]
    # East of the dateline comes back in FMS's 180-360 convention.
    assert got["Longitude"].tolist()[2] == 199.5
    assert got["Category"].tolist() == [3, 3, 0]


def test_empty_forecast_fails():
    rows = to_db_rows(make_csv(), "X").iloc[0:0]
    try:
        rows_to_forecast(rows, pd.Timestamp("2026-04-07T00:00"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
