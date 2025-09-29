import logging
import uuid

import geopandas as gpd
import numpy as np
import ocha_stratus as stratus
import pandas as pd
import pandera.pandas as pa
import xarray as xr
from ocha_lens.datasources.ibtracs import (
    _convert_string_columns,
    _convert_track_column_types,
    _normalize_longitude,
    _to_gdf,
    get_storms,
    normalize_radii,
)
from ocha_lens.utils.validation import (
    check_coordinate_bounds,
    check_crs,
    check_quadrant_list,
)
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_storms(basin: str = None):
    query = """
    SELECT * FROM storms.storms
    """
    if basin:
        query += f" WHERE genesis_basin = '{basin}'"
    df = pd.read_sql(query, stratus.get_engine("dev"))
    return df


def load_tracks(basin: str = None):
    query = """
    SELECT * FROM storms.observed_tracks
    """
    if basin:
        query += f" WHERE basin = '{basin}'"
    df = pd.read_sql(query, stratus.get_engine("dev"))
    return df


TRACK_SCHEMA = pa.DataFrameSchema(
    {
        # TODO: Investigate condition where wind speed is -1
        "wind_speed": pa.Column(
            "Int64", pa.Check.between(-1, 300), nullable=True
        ),
        "pressure": pa.Column(
            "Int64", pa.Check.between(800, 1100), nullable=True
        ),
        "max_wind_radius": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "last_closed_isobar_radius": pa.Column(
            "Int64", pa.Check.ge(0), nullable=True
        ),
        "last_closed_isobar_pressure": pa.Column(
            "Int64", pa.Check.between(800, 1100), nullable=True
        ),
        "gust_speed": pa.Column(
            "Int64", pa.Check.between(0, 400), nullable=True
        ),
        "sid": pa.Column(str, nullable=False),
        "provider": pa.Column(str, nullable=False),
        "usa_agency": pa.Column(str, nullable=False),
        "basin": pa.Column(str, nullable=False),
        "nature": pa.Column(str, nullable=True),
        "valid_time": pa.Column(pd.Timestamp, nullable=False),
        "quadrant_radius_34": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=False
        ),
        "quadrant_radius_50": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=False
        ),
        "quadrant_radius_64": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=True
        ),
        "point_id": pa.Column(str, nullable=False),
        "storm_id": pa.Column(str, nullable=True),
        "geometry": pa.Column(gpd.array.GeometryDtype, nullable=False),
    },
    strict=True,
    coerce=True,
    unique=["sid", "storm_id", "valid_time"],
    report_duplicates="all",
    checks=[
        pa.Check(
            lambda gdf: check_crs(gdf, "EPSG:4326"),
            error="CRS must be EPSG:4326",
        ),
        pa.Check(
            lambda gdf: check_coordinate_bounds(gdf),
            error="All coordinates must be within valid lat/lon bounds",
        ),
    ],
)


def get_best_tracks_usa_only(ds: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Extract the "best" storm tracks from the IBTrACS dataset.

    Extracts the main tracks that have been assigned a wmo_agency and
    are not marked as PROVISIONAL. These are the official, quality-controlled
    tracks used for most analyses.

    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset containing storm track data

    Returns
    -------
    pandas.DataFrame
        DataFrame containing best track data with standardized column names

    Notes
    -----
    The function handles agency-specific variables by prioritizing data from the
    agency designated as the WMO agency for each storm. For agencies that are part
    of the USA system, the function uses the corresponding USA data.
    """
    variables = [
        "wind",
        "gust",
        "pres",
        "rmw",
        "roci",
        "poci",
        "lat",
        "lon",
        "r34",
        "r50",
        "r64",
    ]

    base_cols = [
        "sid",
        "wmo_agency",
        "time",
        "quadrant",
        "basin",
        "nature",
        "track_type",
        "usa_agency",
    ]

    string_cols = [
        "sid",
        "wmo_agency",
        "basin",
        "nature",
        "track_type",
        "usa_agency",
    ]

    base_ds = ds[base_cols]
    df = base_ds.to_dataframe().reset_index()

    # Seems like checking the track_type is redundant here since provisional tracks
    # also don't have an assigned wmo_agency, but still good to be sure
    df = df[(df["wmo_agency"] != b"") & (df["track_type"] != b"PROVISIONAL")]
    if len(df) == 0:
        return pd.DataFrame(columns=list(TRACK_SCHEMA.columns.keys()))

    df = _convert_string_columns(df, string_cols)

    dff = df[base_cols].copy()
    usa_agencies = [x.decode("utf-8") for x in np.unique(ds.usa_agency.values)]
    usa_agencies = [x for x in usa_agencies if x]
    # print("USA agencies in dataset:", usa_agencies)

    for var_suffix in tqdm(variables):
        matching_vars = [
            v for v in ds.data_vars if v.endswith(f"_{var_suffix}")
        ]
        dff[var_suffix] = np.nan

        for var in matching_vars:
            # # We assume that vars are named `agency`_`var`
            # agency = var.split("_")[0]
            # if agency == "usa":
            #     # The USA agencies don't each have columns in their own right
            #     # This data is under the usa_* prefix
            #     mask = df["wmo_agency"].isin(usa_agencies)
            # else:
            #     mask = df["wmo_agency"].str.lower() == agency

            # just take USA data
            # print(df["usa_agency"].unique())
            # print(df["wmo_agency"].unique())
            mask = df["usa_agency"].isin(usa_agencies)
            # print(mask)

            if not mask.any():
                continue

            # Get the data from the original dataset and add to result
            # Need to map back to original indices for selection
            for idx in df[mask].index:
                storm_val = df.loc[idx, "storm"]
                dt_val = df.loc[idx, "date_time"]

                # Only the radii variables are also indexed by quadrant
                if any(substr in var for substr in ["r34", "r50", "r64"]):
                    quadrant_val = df.loc[idx, "quadrant"]
                    value = (
                        ds[var]
                        .sel(
                            storm=storm_val,
                            date_time=dt_val,
                            quadrant=quadrant_val,
                        )
                        .values.item()
                    )
                else:
                    value = (
                        ds[var]
                        .sel(storm=storm_val, date_time=dt_val)
                        .values.item()
                    )
                if not np.isnan(value):
                    dff.loc[idx, var_suffix] = value

    result_df = normalize_radii(dff)
    result_df.rename(
        columns={
            "time": "valid_time",
            "lat": "latitude",
            "lon": "longitude",
            "wind": "wind_speed",
            "gust": "gust_speed",
            "pres": "pressure",
            "rmw": "max_wind_radius",
            "roci": "last_closed_isobar_radius",
            "poci": "last_closed_isobar_pressure",
            "wmo_agency": "provider",
            "r34": "quadrant_radius_34",
            "r50": "quadrant_radius_50",
            "r64": "quadrant_radius_64",
        },
        inplace=True,
    )
    result_df = result_df.drop(columns=["track_type"])
    result_df["point_id"] = [str(uuid.uuid4()) for _ in range(len(result_df))]
    # Drop points that have null values in non-nullable columns
    result_df = result_df.dropna(
        subset=["latitude", "longitude", "valid_time", "sid"]
    )

    # Need to get the storms df to join in the storm_id that we created
    storms = get_storms(ds)
    merged_df = result_df.merge(storms[["sid", "storm_id"]], how="left")
    assert len(merged_df) == len(result_df)

    df = _convert_track_column_types(merged_df)
    df = _normalize_longitude(df)
    gdf = _to_gdf(df)
    if len(gdf) == 0:
        logger.warning("Returning empty geodataframe of best tracks")
        return gdf
    return TRACK_SCHEMA.validate(gdf)
