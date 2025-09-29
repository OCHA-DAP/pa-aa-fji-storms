---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: pa-aa-fji-storms
    language: python
    name: pa-aa-fji-storms
---

# Wind buffers
<!-- markdownlint-disable MD013 -->

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import ocha_lens as lens
import ocha_stratus as stratus
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.interpolate import (
    PchipInterpolator,
    Akima1DInterpolator,
    CubicSpline,
)
from typing import Literal, Iterable, Tuple
from tqdm.auto import tqdm

from src.datasources import ibtracs, codab
from src.constants import FJI_CRS
from src.blob import PROJECT_PREFIX
```

```python
adm0 = codab.load_codab_from_blob(admin_level=0).to_crs(FJI_CRS)
```

```python
basin = "SP"
```

```python
ds = xr.load_dataset("temp/IBTrACS.SP.v04r01.nc")
```

```python
ds
```

```python
df_sp = lens.ibtracs.get_tracks(ds)
```

```python
df_usa = ibtracs.get_best_tracks_usa_only(ds)
```

```python
# df_usa = df_usa.rename(
#     columns={
#         x: f"{x}_usa" for x in df_usa if "quadrant" in x and "usa" not in x
#     }
# )
```

```python
df_usa = df_usa.rename(
    columns={x: x.removesuffix("_usa") for x in df_usa if "quadrant" in x}
)
```

```python
df_usa.dtypes
```

```python
df_usa
```

```python
df_usa["longitude"] = df_usa.geometry.x
df_usa["latitude"] = df_usa.geometry.y
```

```python
df_storms = ibtracs.load_storms(basin=basin)
```

```python
# df_tracks = ibtracs.load_tracks(basin=basin)
df_tracks = df_usa.copy()
```

```python
# df_tracks["valid_time"] = df_tracks["valid_time"].dt.round("s")
```

```python
# index_cols = ["sid", "valid_time"]
# usa_cols = [x for x in df_usa if "usa" in x]
```

```python
# df_tracks = df_tracks.merge(df_storms).merge(df_usa[index_cols + usa_cols])
```

```python
df_tracks = df_tracks.drop(columns="storm_id").merge(df_storms)
```

```python
df_tracks.columns
```

```python
df_tracks
```

```python
df_tracks["quadrant_radius_34"].astype(str).unique()
```

```python
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
```

```python
buffer_speeds = [34, 50, 64]
```

```python
for buffer_speed in buffer_speeds:
    df_tracks = expand_quad_col(df_tracks, f"quadrant_radius_{buffer_speed}")
    # df_tracks = expand_quad_col(
    #     df_tracks, f"quadrant_radius_{buffer_speed}_usa"
    # )
```

```python
df_tracks.dtypes
```

```python
df_tracks["longitude"] = df_tracks["longitude"].apply(
    lambda x: (x + 360) % 360
)
```

```python
df_tracks = df_tracks.sort_values("valid_time")
```

```python
df_tracks
```

```python
NAN_LIST = "[nan, nan, nan, nan]"
```

```python
df_tracks.groupby("season")["sid"].nunique()
```

```python
df_tracks[df_tracks["quadrant_radius_34"].astype(str) != NAN_LIST].groupby(
    "season"
)["sid"].nunique()
```

```python
quads = ["ne", "nw", "se", "sw"]
```

```python
y = [f"quadrant_radius_{buffer_speed}_{quad}" for quad in quads]
```

```python
df_tracks.set_index("sid").loc["2025129S08138"]
```

```python
df_tracks["wind_speed"].unique()
```

```python
df_tracks[df_tracks.duplicated(subset=["valid_time", "sid"], keep=False)]
```

```python
WINSTON_SID = "2016041S14170"
```

```python
# test_group = df_tracks.set_index("sid").loc[WINSTON_SID]
test_group = df_tracks.set_index("sid").loc["2009346S10172"]
```

```python
test_group
```

```python
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
```

```python
test_group_interp = interpolate_track(test_group)
```

```python
test_group_interp[~test_group_interp["quadrant_radius_64_ne"].isnull()]
```

```python
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
```

```python
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
```

```python
buffer_speed = 34
x_col = "longitude"
y_col = "latitude"
ne_col, se_col, sw_col, nw_col = [
    f"quadrant_radius_{buffer_speed}_{x}" for x in ["ne", "se", "sw", "nw"]
]
n_points = 360
```

```python
NM_TO_M = 1.852 * 1000
```

```python
def convert_nm_to_m(value_nm: float):
    return np.nan_to_num(value_nm * NM_TO_M, nan=0)
```

```python
buffer_speeds
```

```python
ne_col, se_col, sw_col, nw_col = [
    f"quadrant_radius_34_{x}" for x in ["ne", "se", "sw", "nw"]
]
test_row = (
    df_tracks.to_crs(3832)
    .set_index("sid")
    .loc[WINSTON_SID]
    .iloc[20][
        [
            "valid_time",
            "quadrant_radius_34",
            ne_col,
            se_col,
            sw_col,
            nw_col,
            "geometry",
        ]
    ]
)
```

```python
df_tracks.set_index("sid").loc[WINSTON_SID].iloc[20]
```

```python
test_row
```

```python
poly = make_quadrant_disk(
    (test_row.geometry.x, test_row.geometry.y),
    ne=convert_nm_to_m(test_row[ne_col]),
    se=convert_nm_to_m(test_row[se_col]),
    sw=convert_nm_to_m(test_row[sw_col]),
    nw=convert_nm_to_m(test_row[nw_col]),
    n_points=n_points,
)
test_gdf = gpd.GeoDataFrame(geometry=[poly], crs=3832)
```

```python
(150 + 80) / 2
```

```python
test_gdf.to_crs(FJI_CRS).plot()
```

```python
minx, miny, maxx, maxy = test_gdf.to_crs(FJI_CRS).total_bounds
```

```python
maxy - miny
```

```python
top_m = (
    convert_nm_to_m(test_row[ne_col]) + convert_nm_to_m(test_row[nw_col])
) / 2
```

```python
bottom_m = (
    convert_nm_to_m(test_row[se_col]) + convert_nm_to_m(test_row[sw_col])
) / 2
```

```python
(bottom_m + top_m) / 1000 / 111
```

```python
(group["quadrant_radius_34"].astype(str) == NAN_LIST).all()
```

```python
group["quadrant_radius_34_ne"].dtype == float
```

```python
quad_cols = [
    x
    for x in df_tracks.columns
    if "quadrant" in x and df_tracks[x].dtype == float
]
quad_cols
```

```python
group[quad_cols].fillna(0)
```

```python
break_all = False
disable_inner_tqdm = True

# test_sid = WINSTON_SID
test_sid = "2009346S10172"

dicts = []
geoms = []
for sid, group in tqdm(df_tracks.groupby("sid")):
    # if sid != test_sid:
    #     continue
    if (group["quadrant_radius_34"].astype(str) == NAN_LIST).all():
        # skip since there are no buffers
        continue
    group[quad_cols] = group[quad_cols].fillna(0)
    if len(group) == 1:
        group_interp = group.copy()
    else:
        group_interp = interpolate_track(group)
    gdf_interp = gpd.GeoDataFrame(
        data=group_interp,
        geometry=gpd.points_from_xy(
            group_interp["longitude"], group_interp["latitude"]
        ),
        crs=FJI_CRS,
    ).to_crs(3832)

    for buffer_speed in buffer_speeds:
        polys = []
        for _, row in tqdm(
            gdf_interp.iterrows(),
            total=len(gdf_interp),
            disable=disable_inner_tqdm,
        ):
            ne_col, se_col, sw_col, nw_col = [
                f"quadrant_radius_{buffer_speed}_{x}"
                for x in ["ne", "se", "sw", "nw"]
            ]
            if row[[ne_col, se_col, sw_col, nw_col]].isna().all():
                continue

            poly = make_quadrant_disk(
                (row.geometry.x, row.geometry.y),
                ne=convert_nm_to_m(row[ne_col]),
                se=convert_nm_to_m(row[se_col]),
                sw=convert_nm_to_m(row[sw_col]),
                nw=convert_nm_to_m(row[nw_col]),
                n_points=n_points,
            )
            polys.append(poly)
        gdf = gpd.GeoDataFrame(geometry=polys, crs=df_tracks.crs)
        merged = unary_union(gdf.geometry.values)
        merged_gs = gpd.GeoSeries([merged], crs=gdf.crs)
        dicts.append({"sid": sid, "buffer_speed": buffer_speed})
        geoms.append(merged)
```

```python
gdf_buffers = gpd.GeoDataFrame(data=dicts, geometry=geoms, crs=3832)
```

```python
gdf_buffers.set_index("sid").loc[WINSTON_SID].plot()
```

```python
def plot_test_sid(sid):
    test_group_interp = interpolate_track(
        df_tracks.set_index("sid").loc[sid],
    )
    test_gdf_interp = gpd.GeoDataFrame(
        data=test_group_interp,
        geometry=gpd.points_from_xy(
            test_group_interp["longitude"], test_group_interp["latitude"]
        ),
        crs=FJI_CRS,
    ).to_crs(3832)

    fig, ax = plt.subplots(dpi=200)
    adm0.to_crs(3832).boundary.plot(ax=ax, color="k", linewidth=0.5)
    df_tracks.set_index("sid").loc[sid].plot(ax=ax, markersize=2, color="red")
    test_gdf_interp.plot(ax=ax, markersize=0.5, alpha=0.5)
    gdf_buffers.set_index("sid").loc[sid].plot(
        ax=ax, linewidth=0.5, column="buffer_speed", alpha=0.2
    )
    ax.axis("off")
```

```python
plot_test_sid(WINSTON_SID)
```

```python
# Yasa
plot_test_sid("2020346S13168")
```

```python
# Mal
plot_test_sid("2023316S09167")
```

```python
# Mick
plot_test_sid("2009346S10172")
```

```python
# Harold
plot_test_sid("2020092S09155")
```

```python
# Cody
plot_test_sid("2022008S17173")
```

```python
# gdf_buffers.rename(columns={"buffer_speed": "buff_speed"})
```

```python
# blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/wind_buffers.shp"
```

```python
# stratus.upload_shp_to_blob(
#     gdf_buffers.rename(columns={"buffer_speed": "buff_speed"}), blob_name
# )
```

```python
import io

buf = io.BytesIO()
gdf_buffers.to_parquet(buf, index=False)  # writes GeoParquet metadata
buf.seek(0)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/wind_buffers.parquet"
stratus.upload_blob_data(data=buf.getvalue(), blob_name=blob_name)
```

```python

```
