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

# Forecast uncertainty exposure

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
adm0 = codab.load_codab_from_blob(admin_level=0).to_crs(FJI_CRS)
adm3 = codab.load_codab_from_blob(admin_level=3).to_crs(FJI_CRS)
```

```python
da_wp = worldpop.load_worldpop_from_blob(iso3="fji")
da_wp = da_wp.assign_coords({"x": (((da_wp.x + 360) % 360))}).sortby("x")
```

```python
da_wp_clip = da_wp.rio.clip(adm3.geometry)
```

```python
da_wp_clip_adm0.plot()
```

```python
da_wp_clip_adm0.sum().values
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
speeds = [34, 50, 64]
quads = ["ne", "se", "sw", "nw"]
cols = ["valid_time", "Latitude", "Longitude", "Category", "MeanWind"] + [
    f"quadrant_radius_{speed}_{x}" for speed in speeds for x in quads
]
```

```python
gdfs = []

for blob_name in tqdm(fcast_blob_names):
    data = stratus.load_blob_data(blob_name)
    gdf = fms.parse_fms_forecast(BytesIO(data))
    gdf = gdf.rename(columns={"forecast_time": "valid_time"})
    gdfs.append(gdf)

gdf_tracks = pd.concat(gdfs)
```

```python
gdf_tracks = gdf_tracks.rename(columns=fms.QUAD_COL_RENAME)
```

```python
gdf_tracks
```

```python
gdf_tracks["sid"] = gdf_tracks["Name Season"].replace(NAMESEASON2SID)
```

```python
gdf_tracks[df_tracks["sid"].isnull()]
```

```python
gdf_tracks["sid"].unique()
```

```python
gdf_tracks = gdf_tracks.to_crs(3832)
```

```python
gdf_tracks.crs.axis_info[0].unit_name
```

```python
group.columns
```

```python
gdf_tracks["uncertainty_m"] = gdf_tracks["Uncertainty"] * NM_TO_M
```

```python
gdf_tracks.groupby(["Name Season", "base_time"]).first()
```

```python
_gdf_buffers.to_crs(FJI_CRS).iloc[0]
```

```python
_gdf_buffers.to_crs(FJI_CRS).iloc[0].geometry.bounds[0]
```

```python
# gdfs_buffers = []
gdfs_shifts = []
gdfs_shifts_buffers = []
dicts = []

for sid, sid_group in tqdm(gdf_tracks.groupby("sid")):
    # if sid != YASA_SID:
    #     continue
    for issued_time, group in tqdm(sid_group.groupby("base_time")):
        # _gdf_buffers = ibtracs.calculate_wind_buffers_gdf(group)
        # _gdf_buffers["issued_time"] = issued_time
        # _gdf_buffers["sid"] = sid
        # gdfs_buffers.append(_gdf_buffers)

        for shift_deg in tqdm(range(0, 360, 10), disable=True):
            _gdf_shift = fms.shift_gdf_points(group, shift_deg)
            _gdf_shift["issued_time"] = issued_time
            _gdf_shift["sid"] = sid
            _gdf_shift_buffers = ibtracs.calculate_wind_buffers_gdf(_gdf_shift)
            _gdf_shift_buffers = _gdf_shift_buffers.to_crs(FJI_CRS)
            _gdf_shift_buffers["shift_deg"] = shift_deg
            _gdf_shift_buffers["issued_time"] = issued_time
            _gdf_shift_buffers["sid"] = sid

            gdfs_shifts.append(_gdf_shift)
            gdfs_shifts_buffers.append(_gdf_shift_buffers)
            for _, row in _gdf_shift_buffers.iterrows():
                if not row.geometry:
                    pop_exposed = 0
                else:
                    try:
                        _da_clip = da_wp_clip.rio.clip([row.geometry])
                        pop_exposed = int(_da_clip.sum())
                    except NoDataInBounds as e:
                        pop_exposed = 0
                dicts.append(
                    {
                        "sid": sid,
                        "issued_time": issued_time,
                        "speed": row["speed"],
                        "shift_deg": shift_deg,
                        "pop_exposed": pop_exposed,
                    }
                )
        # break
```

```python
gdf_shift_tracks = pd.concat(gdfs_shifts)
gdf_shift_buffers = pd.concat(gdfs_shifts_buffers)
df_exp_shift_raw = pd.DataFrame(dicts)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/fms_shift_tracks.parquet"
stratus.upload_parquet_to_blob(
    gdf_shift_tracks.drop(columns="geometry"), blob_name
)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/fms_shift_buffers.parquet"
upload_geoparquet_to_blob(gdf_shift_buffers, blob_name)
```

```python
df_exp_shift = df_exp_shift_raw.pivot(
    columns="speed",
    index=["sid", "issued_time", "shift_deg"],
    values="pop_exposed",
)
df_exp_shift.columns = [f"exp_{x}" for x in df_exp_shift.columns]
df_exp_shift = df_exp_shift.reset_index()
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/fms_shift_exp.parquet"
stratus.upload_parquet_to_blob(df_exp_shift, blob_name)
```

```python
rows = []
for (sid, issued_time), group in df_exp_shift.groupby(["sid", "issued_time"]):
    group = group.copy()
    group = group.sort_values(
        [f"exp_{x}" for x in [64, 50, 34]], ascending=False
    )
    worst_row = group.iloc[0].copy()
    worst_row["level"] = "worst"
    best_row = group.iloc[-1].copy()
    best_row["level"] = "best"
    rows.extend([worst_row, best_row])
```

```python
df_exp_range = pd.DataFrame(rows)
```

```python
df_exp_range
```

```python
for sid, group in df_exp_range.groupby("sid"):
    group.pivot(index="issued_time", values="exp_64", columns="level").plot()
```

```python
group.pivot(index="issued_time", values="exp_34", columns="level").plot()
```

```python
gdf_shifted_buffers_all[gdf_shifted_buffers_all["speed"] == 34]
```

```python
gdf_shifted_buffers_all[gdf_shifted_buffers_all["speed"] == 34].plot(alpha=0.1)
```

```python
gdf_shifted_all = pd.concat(gdfs)
```

```python
gdf_shifted_all.plot()
```

```python
ax = gdf_shifted_all[gdf_shifted_all["shift_deg"] == 0].plot()
gdf_shifted_all[gdf_shifted_all["shift_deg"] == 90].plot(ax=ax)
```

```python
df_exp_shifted_raw = pd.DataFrame(dicts)
```

```python
df_exp_shifted = df_exp_shifted_raw.pivot(
    columns="speed", index=["issued_time", "shift_deg"], values="pop_exposed"
)
df_exp_shifted.columns = [f"exp_{x}" for x in df_exp_shifted.columns]
df_exp_shifted = df_exp_shifted.reset_index()
```

```python
df_exp_shifted = df_exp_shifted.sort_values(
    [f"exp_{x}" for x in [64, 50, 34]], ascending=False
)
```

```python
worst_row = df_exp_shifted.iloc[0]
best_row = df_exp_shifted.iloc[-1]
```

```python
fig, ax = plt.subplots()
adm3.plot(ax=ax)
group.to_crs(FJI_CRS).plot(ax=ax, color="k")
gdf_shifted_all[gdf_shifted_all["shift_deg"] == worst_row["shift_deg"]].plot(
    ax=ax, color="crimson"
)
gdf_shifted_all[gdf_shifted_all["shift_deg"] == best_row["shift_deg"]].plot(
    ax=ax, color="green"
)
ax.axis("off")
```

```python
fig, ax = plt.subplots()
group.plot(ax=ax)
gdf_out.plot(ax=ax)
```

```python
test.iloc[0]
```

```python
fig, ax = plt.subplots()
group.plot(ax=ax, alpha=0.1)
test[test["speed"] == 34].plot(ax=ax, alpha=0.1)
test[test["speed"] == 50].plot(ax=ax, alpha=0.1)
test[test["speed"] == 64].plot(ax=ax, alpha=0.1)
```

```python
gpd.GeoDataFrame(geometry=[test]).plot(alpha=0.1)
```

```python

```
