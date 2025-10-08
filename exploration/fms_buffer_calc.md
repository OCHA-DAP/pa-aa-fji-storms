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

# FMS buffer estimation
<!-- markdownlint-disable MD013 -->

Estimating the FMS wind buffers, using the model that was fitted on the historical USA values.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import io

import pandas as pd
import numpy as np
import ocha_lens as lens
import ocha_stratus as stratus
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from tqdm.auto import tqdm

from src.datasources import ibtracs, codab
from src.datasources.ibtracs import interpolate_track, expand_quad_col
from src.constants import *
from src.blob import PROJECT_PREFIX
```

## Load data

```python
adm0 = codab.load_codab_from_blob(admin_level=0).to_crs(FJI_CRS)
```

```python
query = """
SELECT *
FROM storms.ibtracs_storms
"""
with stratus.get_engine(stage="dev").connect() as conn:
    df_storms = pd.read_sql(query, conn)
```

```python
query = """
SELECT *
FROM storms.ibtracs_tracks_geo
WHERE basin = 'SP'
"""
with stratus.get_engine(stage="dev").connect() as conn:
    gdf_tracks = gpd.read_postgis(query, conn, geom_col="geometry")
```

```python
gdf_tracks = gdf_tracks.merge(df_storms)
```

```python
gdf_tracks = gdf_tracks[~gdf_tracks["provisional"]]
```

```python
gdf_tracks["provider"].value_counts()
```

```python
gdf_tracks["longitude"] = gdf_tracks.geometry.x
gdf_tracks["latitude"] = gdf_tracks.geometry.y
```

```python
gdf_tracks.columns
```

```python
cols = ["sid", "valid_time", "wind_speed", "longitude", "latitude"]
df_tracks = gdf_tracks[cols].dropna()
```

```python
df_tracks["longitude"] = df_tracks["longitude"].apply(
    lambda x: (x + 360) % 360
)
```

```python
df_tracks["lat_abs"] = df_tracks["latitude"].apply(abs)
```

```python
df_tracks = df_tracks[df_tracks["wind_speed"] > 0]
```

```python
df_tracks["wind_speed"].min()
```

```python
log_cols = ["wind_speed", "lat_abs"]
for col in log_cols:
    df_tracks[f"{col}_log"] = np.log(df_tracks[col])
```

```python
buffer_speeds = [34, 50, 64]
```

```python
for speed in buffer_speeds:
    params = WIND_RADIUS_PARAMS[speed]
    df_tracks[f"r{speed}_log"] = (
        params["const"]
        + df_tracks["wind_speed_log"] * params["wind_speed_log"]
        + df_tracks["lat_abs_log"] * params["lat_abs_log"]
    )
    df_tracks[f"r{speed}"] = np.exp(df_tracks[f"r{speed}_log"])
    df_tracks[f"r{speed}"] = df_tracks.apply(
        lambda row: 0 if row["wind_speed"] < speed else row[f"r{speed}"],
        axis=1,
    )
    df_tracks[f"r{speed}_m"] = df_tracks[f"r{speed}"] * NM_TO_M
```

```python

```

```python
df_tracks_interp = (
    df_tracks.groupby("sid")
    .apply(interpolate_track, include_groups=False)
    .reset_index()
    .drop(columns=["level_1"])
)
```

```python
df_tracks_interp
```

```python
gdf_tracks = gpd.GeoDataFrame(
    data=df_tracks,
    geometry=gpd.points_from_xy(df_tracks["longitude"], df_tracks["latitude"]),
    crs=FJI_CRS,
)
```

```python
gdf_tracks_interp = gpd.GeoDataFrame(
    data=df_tracks_interp,
    geometry=gpd.points_from_xy(
        df_tracks_interp["longitude"], df_tracks_interp["latitude"]
    ),
    crs=FJI_CRS,
)
```

```python
gdf_tracks_interp
```

```python

```

```python
dicts = []
geoms = []

for sid, group in tqdm(gdf_tracks_interp.to_crs(3832).groupby("sid")):
    for speed in buffer_speeds:
        polys = [
            row.geometry.buffer(row[f"r{speed}_m"])
            for _, row in group.iterrows()
        ]
        # for _, row in group.iterrows():
        #     polys.append(row.geometry.buffer(row[f"r{speed}_m"]))
        gdf = gpd.GeoDataFrame(geometry=polys, crs=3832)
        merged = unary_union(gdf.geometry.values)
        merged_gs = gpd.GeoSeries([merged], crs=3832)
        dicts.append({"sid": sid, "buffer_speed": speed})
        geoms.append(merged)
```

```python
gdf_buffers = gpd.GeoDataFrame(data=dicts, geometry=geoms, crs=3832)
```

```python
def plot_buffers(sid):
    fig, ax = plt.subplots(dpi=300)
    adm0.to_crs(3832).boundary.plot(ax=ax, color="k", linewidth=0.5)
    gdf_tracks.to_crs(3832).set_index("sid").loc[sid].plot(
        ax=ax, markersize=5, marker=".", edgecolor="none", color="red"
    )
    gdf_tracks_interp.to_crs(3832).set_index("sid").loc[sid].plot(
        ax=ax, markersize=1, marker=".", edgecolor="none"
    )
    gdf_buffers.set_index("sid").loc[sid].plot(
        ax=ax, linewidth=0.5, column="buffer_speed", alpha=0.2
    )
    ax.axis("off")
```

```python
plot_buffers(WINSTON_SID)
```

```python
plot_buffers(YASA_SID)
```

```python
plot_buffers(HAROLD_SID)
```

```python
plot_buffers(MAL_SID)
```

```python
buf = io.BytesIO()
gdf_buffers.to_parquet(buf, index=False)  # writes GeoParquet metadata
buf.seek(0)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/fms_wind_buffers.parquet"
stratus.upload_blob_data(data=buf.getvalue(), blob_name=blob_name)
```
