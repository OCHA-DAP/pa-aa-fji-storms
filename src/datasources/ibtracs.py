import logging
from typing import Literal, Tuple

import geopandas as gpd
import numpy as np
import ocha_stratus as stratus
import pandas as pd
import shapely
import xarray as xr
from rioxarray.exceptions import NoDataInBounds
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
)
from shapely.geometry import Polygon

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
