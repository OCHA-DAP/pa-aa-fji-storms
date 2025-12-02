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

# FMS historical forecast buffers

Processing historical forecast buffers to compare them against the observed track buffers. The main one we need to check is Mal 2024, since this is the edge case for the wind exposure part of the trigger (we need to make sure that none of its forecasts would have produced enough wind exposure to trigger).

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from io import BytesIO

import ocha_stratus as stratus
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rioxarray.exceptions import NoDataInBounds
from shapely.ops import unary_union
from tqdm.auto import tqdm

from src.datasources import fms, ibtracs, worldpop, codab
from src.constants import *
from src.blob import PROJECT_PREFIX, upload_geoparquet_to_blob
```

```python
blob_names = stratus.list_container_blobs(
    name_starts_with=f"{PROJECT_PREFIX}/raw/fms/TC Data"
)
```

```python
fcast_blob_names = [x for x in blob_names if "Forecast_Track" in x]
```

```python
obsv_blob_names = [x for x in blob_names if "Best_Track" in x]
```

```python
speeds = [34, 50, 64]
```

```python
quads = ["ne", "nw", "se", "sw"]
cols = ["valid_time", "Latitude", "Longitude", "Category", "MeanWind"] + [
    f"quadrant_radius_{speed}_{x}" for speed in speeds for x in quads
]
```

```python
cols
```

```python
n_points = 360
```

```python
def convert_nm_to_m(value_nm: float):
    return np.nan_to_num(value_nm * NM_TO_M, nan=0)
```

```python
speed2word = {34: "Gale", 50: "Storm", 64: "Hurricane"}
```

```python
speed2word.items()
```

```python
speed2word = {34: "Gale", 50: "Storm", 64: "Hurricane"}
quads = ["ne", "se", "sw", "nw"]
quad_col_rename = {}
for speed, speedword in speed2word.items():
    quad_col_rename.update(
        {
            f"{x.upper()}{speedword}Radius": f"quadrant_radius_{speed}_{x}"
            for x in quads
        }
    )
```

```python
{
    f"{x.upper()}{speedword}Radius": f"quadrant_radius_{speed}_{x}"
    for speed, speedword in speed2word.items()
    for x in quads
}
```

```python
quad_col_rename
```

```python
def calculate_fms_buffers(gdf, best_track: bool = False):
    gdf["Longitude"] = gdf["Longitude"].apply(lambda x: (x + 360) % 360)
    name_season = gdf.iloc[0]["Name Season"]
    if not best_track:
        issued_time = gdf.iloc[0]["base_time"]
    for speed in speeds:
        speedword = speed2word[speed]
        gdf = gdf.rename(
            columns={
                f"{x.upper()}{speedword}Radius": f"quadrant_radius_{speed}_{x}"
                for x in quads
            }
        )

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
    for speed in speeds:
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
        merged_gs = gpd.GeoSeries([merged], crs=gdf.crs)
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
```

```python
dicts = []
geoms = []
dfs = []
for blob_name in tqdm(fcast_blob_names):
    data = stratus.load_blob_data(blob_name)
    gdf = fms.parse_fms_forecast(BytesIO(data))
    gdf = gdf.rename(columns={"forecast_time": "valid_time"})
    geoms_in, dicts_in = calculate_fms_buffers(gdf)
    geoms.extend(geoms_in)
    dicts.extend(dicts_in)
    dfs.append(gdf.drop(columns="geometry"))

gdf_buffers = gpd.GeoDataFrame(data=dicts, geometry=geoms, crs=3832)
gdf_buffers = gdf_buffers.to_crs(FJI_CRS)
df_tracks = pd.concat(dfs)
```

```python
gdf_buffers
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/forecast_tracks_buffers.parquet"
upload_geoparquet_to_blob(gdf_buffers, blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/forecast_tracks.parquet"
stratus.upload_parquet_to_blob(df_tracks, blob_name)
```

```python
dicts = []
geoms = []
dfs = []
for blob_name in tqdm(obsv_blob_names):
    data = stratus.load_blob_data(blob_name)
    gdf = fms.parse_fms_forecast(BytesIO(data), best_track=True)
    geoms_in, dicts_in = calculate_fms_buffers(gdf, best_track=True)
    geoms.extend(geoms_in)
    dicts.extend(dicts_in)
    dfs.append(gdf.drop(columns="geometry"))

gdf_buffers_best = gpd.GeoDataFrame(data=dicts, geometry=geoms, crs=3832)
gdf_buffers_best = gdf_buffers_best.to_crs(FJI_CRS)
df_tracks_best = pd.concat(dfs)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/best_tracks.parquet"
stratus.upload_parquet_to_blob(df_tracks_best, blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/best_tracks.parquet"
df_tracks_best = stratus.load_parquet_from_blob(blob_name)
```

```python
da_wp = worldpop.load_worldpop_from_blob()
```

```python
da_wp
```

```python
da_wp_reproject = da_wp.rio.reproject(3832)
```

```python
# da_wp = da_wp.assign_coords({"x": (((da_wp.x + 360) % 360))}).sortby("x")
```

```python
adm0 = codab.load_codab_from_blob(admin_level=0)
```

```python
da_wp_clip = da_wp.rio.clip(adm0.geometry)
```

```python
da_wp_clip_new = da_wp.rio.clip(adm0.geometry)
```

```python
dicts = []
for _, row in gdf_buffers.iterrows():
    if not row.geometry:
        pop_exposed = 0
    else:
        try:
            da_wp_clip_buffer = da_wp.rio.clip([row.geometry])
            pop_exposed = int(da_wp_clip_buffer.sum())
        except NoDataInBounds as e:
            pop_exposed = 0
    dicts.append(
        {
            "issued_time": row["issued_time"],
            "name_season": row["name_season"],
            "buffer_speed": row["buffer_speed"],
            "pop_exposed": pop_exposed,
        }
    )
df_exp_new = pd.DataFrame(dicts)
```

```python
df_exp_new[df_exp_new["buffer_speed"] == 64]
```

```python
df_exp[df_exp["buffer_speed"] == 64]
```

```python
df_exp
```

```python
dicts = []
for _, row in gdf_buffers.iterrows():
    if not row.geometry:
        pop_exposed = 0
    else:
        try:
            da_wp_clip_buffer = da_wp_clip.rio.clip([row.geometry])
            pop_exposed = int(da_wp_clip_buffer.sum())
        except NoDataInBounds as e:
            pop_exposed = 0
    dicts.append(
        {
            "issued_time": row["issued_time"],
            "name_season": row["name_season"],
            "buffer_speed": row["buffer_speed"],
            "pop_exposed": pop_exposed,
        }
    )

df_exp = pd.DataFrame(dicts)
```

```python
df_exp.groupby(["name_season", "buffer_speed"])[
    "pop_exposed"
].max().reset_index()
```

```python
dicts = []
for name_season, row in gdf_buffers_best.set_index("name_season").iterrows():
    if not row.geometry:
        pop_exposed = 0
    else:
        try:
            da_wp_clip_buffer = da_wp_clip.rio.clip([row.geometry])
            pop_exposed = int(da_wp_clip_buffer.sum())
        except NoDataInBounds as e:
            pop_exposed = 0
    dicts.append(
        {
            "name_season": name_season,
            "pop_exposed": pop_exposed,
            "buffer_speed": row["buffer_speed"],
        }
    )
```

```python
df_exp_best = pd.DataFrame(dicts)
```

```python
df_exp_best["sid"] = df_exp_best["name_season"].replace(NAMESEASON2SID)
```

```python
df_exp_best
```

```python
df_exp_best
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/fms_besttrack_exp.parquet"
stratus.upload_parquet_to_blob(df_exp_best, blob_name)
```

```python
fig, ax = plt.subplots()
gdf_buffer_best.boundary.plot(ax=ax)
adm0.boundary.plot(ax=ax)
```

```python

```
