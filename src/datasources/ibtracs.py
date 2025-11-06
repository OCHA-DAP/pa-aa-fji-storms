import logging
import uuid
from typing import Literal, Tuple

import geopandas as gpd
import numpy as np
import ocha_stratus as stratus
import pandas as pd
import pandera.pandas as pa
import shapely
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
from rioxarray.exceptions import NoDataInBounds
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
)
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from src.constants import FJI_CRS, NM_TO_M, QUADS

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


def interpolate_track(
    df: pd.DataFrame,
    time_col: str = "valid_time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    freq: str = "30min",
    method: Literal["pchip", "akima", "cubic", "linear"] = "pchip",
    include_ends: bool = True,
) -> pd.DataFrame:
    """
    Resample a (time, lat, lon, ...) track to a regular grid (default 30 min).
    - Lat/lon use chosen spline method (default 'pchip').
    - All other numeric columns are interpolated linearly.
    - Assumes longitude already in [0, 360) and keeps it in [0, 360).
    - No extrapolation beyond the observed time span.
    - If only one point is available, return that point (same output schema).
    """

    # --- Prep ---
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.sort_values(time_col).drop_duplicates(
        subset=[time_col], keep="first"
    )
    work = work.dropna(subset=[lat_col, lon_col])

    n = len(work)
    if n == 0:
        # Nothing usable
        return pd.DataFrame(columns=[time_col, lat_col, lon_col]).astype(
            {time_col: "datetime64[ns, UTC]"}
        )

    # If exactly one point, return it in the same format (reset index, include numeric cols)
    if n == 1:
        row = work.iloc[0]
        out = pd.DataFrame(
            {
                time_col: [row[time_col]],
                lat_col: [float(row[lat_col])],
                lon_col: [float(row[lon_col]) % 360.0],
            }
        )
        # carry other numeric columns as-is
        other_cols = work.select_dtypes(
            include=[np.number]
        ).columns.difference([lat_col, lon_col])
        for col in other_cols:
            out[col] = float(row[col])
        return out.reset_index(drop=True)

    # --- target time grid
    tmin, tmax = work[time_col].min(), work[time_col].max()
    start = tmin.floor(freq) if include_ends else tmin.ceil(freq)
    end = tmax.ceil(freq) if include_ends else tmax.floor(freq)
    target = pd.date_range(start, end, freq=freq, tz="UTC")
    target = target[(target >= tmin) & (target <= tmax)]
    if target.empty:
        target = pd.DatetimeIndex([tmin, tmax])

    # --- time axis
    t0 = work[time_col].iloc[0]
    x = (work[time_col] - t0).dt.total_seconds().to_numpy()
    x_new = (pd.Series(target) - t0).dt.total_seconds().to_numpy()

    # --- lat/lon interpolation ---
    y_lat = work[lat_col].to_numpy(float)
    y_lon = work[lon_col].to_numpy(float)

    if method == "linear" or (method in ("akima", "cubic") and n < 3):
        interp_lat = lambda xv: np.interp(xv, x, y_lat)
        interp_lon = lambda xv: np.mod(np.interp(xv, x, y_lon), 360.0)
    elif method == "pchip":
        interp_lat = PchipInterpolator(x, y_lat)
        interp_lon = lambda xv: np.mod(PchipInterpolator(x, y_lon)(xv), 360.0)
    elif method == "akima":
        interp_lat = Akima1DInterpolator(x, y_lat)
        interp_lon = lambda xv: np.mod(
            Akima1DInterpolator(x, y_lon)(xv), 360.0
        )
    elif method == "cubic":
        interp_lat = CubicSpline(x, y_lat, bc_type="natural")
        interp_lon = lambda xv: np.mod(
            CubicSpline(x, y_lon, bc_type="natural")(xv), 360.0
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    lat_new = interp_lat(x_new)
    lon_new = interp_lon(x_new)

    # --- other numeric columns (linear only) ---
    other_cols = work.select_dtypes(include=[np.number]).columns.difference(
        [lat_col, lon_col]
    )
    out = pd.DataFrame(index=target)
    out[lat_col] = lat_new
    out[lon_col] = lon_new
    for col in other_cols:
        y = work[col].to_numpy(float)
        out[col] = np.interp(x_new, x, y)

    out.index.name = time_col
    out = out.reset_index()
    return out


def expand_quad_col(df, col):
    if f"{col}_ne" in df:
        print(f"already done for {col}")
        return df
    df_expanded = (
        df[col]
        .apply(pd.Series)
        .rename(
            columns={
                0: f"{col}_ne",
                1: f"{col}_nw",
                2: f"{col}_se",
                3: f"{col}_sw",
            }
        )
    )
    return df.join(df_expanded)


def _radius_from_quadrants(
    theta_deg: np.ndarray, ne: float, se: float, sw: float, nw: float
) -> np.ndarray:
    """
    Return radius for each angle by linearly interpolating between the
    four quadrant control points defined at bearings:
        45°  -> NE
        135° -> NW
        225° -> SW
        315° -> SE
    Bearing convention: 0° = East, 90° = North (mathematical).
    """
    # Control bearings (deg) and radii, with wrap-around point to close the loop
    bearings = np.array([45, 135, 225, 315, 405], dtype=float)
    radii = np.array([ne, nw, sw, se, ne], dtype=float)

    # Map all thetas into [0, 360) and also allow values up to 405 for interpolation
    t = (theta_deg % 360).astype(float)
    # For values in [0,45), make an equivalent in [360,405) to interpolate to NE nicely
    t_wrap = t.copy()
    t_wrap[t < 45] += 360

    # Interpolate and then map back (the interpolation function is periodic due to control duplication)
    r = np.interp(t_wrap, bearings, radii)
    return r


def make_quadrant_disk(
    center_xy: Tuple[float, float],
    ne: float,
    se: float,
    sw: float,
    nw: float,
    n_points: int = 360,
) -> Polygon:
    """
    Build a smooth polygon around (x, y) using quadrant radii. Units assumed meters.
    - center_xy: (x, y) in EPSG:3832
    - ne, se, sw, nw: radii for quadrants (meters)
    - n_points: angular resolution
    Bearing convention: 0°=East, 90°=North; polygon traced counter-clockwise.
    """
    x0, y0 = center_xy
    theta = np.linspace(0, 360, n_points, endpoint=False)  # degrees
    r = _radius_from_quadrants(theta, ne, se, sw, nw)

    # Convert polar -> Cartesian
    th = np.deg2rad(theta)
    xs = x0 + r * np.cos(th)
    ys = y0 + r * np.sin(th)

    # Ensure valid ring: close the polygon
    coords = np.column_stack([xs, ys])
    return Polygon(coords)


def calculate_wind_buffers_gdf(
    df: pd.DataFrame,
    quad_cols_format: str = "quadrant_radius_{speed}_{quad}",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    valid_time_col: str = "valid_time",
):
    """
    Calculate wind buffer polygons for given wind speed quadrants.
    Note that this function interpolates the storm track to a regular
    30-minute interval before calculating the wind buffers.
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with storm track data including quadrant radius columns
    quad_cols_format: str = 'quadrant_radius_{speed}_{quad}'
        Format string for quadrant radius columns, with placeholders for
        speed and quad (e.g., 'quadrant_radius_{speed}_{quad}')
    lon_col: str = 'Longitude'
        Name of the longitude column in df
    lat_col: str = 'Latitude'
        Name of the latitude column in df
    valid_time_col: str = 'valid_time'
        Name of the valid time column in df

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with wind buffer polygons for each speed

    """
    all_quad_cols = [
        quad_cols_format.format(speed=speed, quad=x)
        for speed in [34, 50, 64]
        for x in QUADS
    ]
    df = df[[lon_col, lat_col, valid_time_col] + all_quad_cols].copy()
    df[lon_col] = df[lon_col].apply(lambda x: (x + 360) % 360)
    df_interp = interpolate_track(
        df,
        time_col=valid_time_col,
        lat_col=lat_col,
        lon_col=lon_col,
        freq="30min",
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
    for speed in [34, 50, 64]:
        speed_quad_cols = tuple(
            quad_cols_format.format(speed=speed, quad=x) for x in QUADS
        )
        geoms.append(build_merged_wind_buffer(gdf_interp, speed_quad_cols))
        dicts.append({"speed": speed})
    return gpd.GeoDataFrame(dicts, geometry=geoms, crs=3832)


def build_merged_wind_buffer(
    gdf: gpd.GeoDataFrame,
    quad_cols: Tuple[str, str, str, str],
):
    """
    Build a merged wind buffer polygon from quadrant radii columns.
    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with point geometries and quadrant radius columns
    quad_cols: Tuple[str, str, str, str]
        Names of the four quadrant radius columns in order:
        (ne_col, se_col, sw_col, nw_col)

    Returns
    -------
    gpd.GeoSeries or None
        Merged polygon of wind buffers, or None if all radius values are NaN

    """
    ne_col, se_col, sw_col, nw_col = quad_cols
    polys = []
    gdf[[ne_col, se_col, sw_col, nw_col]] = (
        gdf[[ne_col, se_col, sw_col, nw_col]].fillna(0) * NM_TO_M
    )
    for _, row in gdf.iterrows():
        if row[[ne_col, se_col, sw_col, nw_col]].isna().all():
            return None

        poly = make_quadrant_disk(
            (row.geometry.x, row.geometry.y),
            row[ne_col],
            row[se_col],
            row[sw_col],
            row[nw_col],
        )
        polys.append(poly)
    return gpd.GeoSeries(polys).union_all()


def calculate_adm_exposure(
    da_wp_clip_adm: xr.DataArray,
    buffer_geometry: shapely.geometry.Polygon,
):
    if buffer_geometry is None:
        return 0
    # check that da longitude is in [0, 360)
    if np.any(da_wp_clip_adm["lon"] < 0):
        raise ValueError("Longitude must be in [0, 360) for exposure calc")
    # check that buffer_geometry is in [0, 360)
    if buffer_geometry.bounds[0] < 0:
        raise ValueError(
            "Buffer geometry must be in [0, 360) for exposure calc"
        )
    try:
        _da_clip = da_wp_clip_adm.rio.clip([buffer_geometry])
        return int(_da_clip.sum())
    except NoDataInBounds:
        return 0
