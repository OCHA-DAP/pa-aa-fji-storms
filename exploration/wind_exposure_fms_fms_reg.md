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

# Wind exposure - FMS buffers

Calculating wind exposure using estimated FMS buffers

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import io

import matplotlib.pyplot as plt
import ocha_stratus as stratus
import geopandas as gpd
import pandas as pd
from rioxarray.exceptions import NoDataInBounds
from tqdm.auto import tqdm

from src.datasources import worldpop, ibtracs, codab
from src.constants import *
from src.blob import PROJECT_PREFIX
```

## Load data

```python
adm0 = codab.load_codab_from_blob(admin_level=0).to_crs(FJI_CRS)
adm3 = codab.load_codab_from_blob(admin_level=3).to_crs(FJI_CRS)
```

```python
da_wp = worldpop.load_worldpop_from_blob(iso3="fji")
```

```python
da_wp.sum().values
```

```python
# da_wp = da_wp.assign_coords({"x": (((da_wp.x + 360) % 360))}).sortby("x")
```

```python
da_wp_clip = da_wp.rio.clip(adm3.geometry)
```

```python
da_wp_clip.sum().values
```

```python
fig, ax = plt.subplots(dpi=200)
da_wp_clip.plot(ax=ax, vmax=100)
adm0.boundary.plot(ax=ax, linewidth=0.5)
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/ibtracs/fms_wind_buffers_fms_reg.parquet"
)
gdf_buffers = gpd.read_parquet(io.BytesIO(stratus.load_blob_data(blob_name)))
```

```python
gdf_buffers = gdf_buffers.to_crs(FJI_CRS)
```

```python
fig, ax = plt.subplots(dpi=200)
da_wp_clip.plot(ax=ax, vmax=100)
gdf_buffers[gdf_buffers["sid"] == WINSTON_SID].plot(ax=ax, alpha=0.1)
adm0.boundary.plot(ax=ax, linewidth=0.5)
```

## Calculate exposure

### At adm0 level

```python
dicts = []
for _, row in tqdm(gdf_buffers.iterrows(), total=len(gdf_buffers)):
    if not row.geometry:
        pop_exposed = 0
    else:
        try:
            da_clip_sid = da_wp_clip.rio.clip([row.geometry])
            pop_exposed = int(da_clip_sid.sum())
        except NoDataInBounds as e:
            pop_exposed = 0
    dicts.append(
        {
            "sid": row["sid"],
            "buffer_speed": row["buffer_speed"],
            "pop_exposed": pop_exposed,
        }
    )
```

```python
df_exp = pd.DataFrame(dicts)
```

```python
df_exp[df_exp["sid"] == WINSTON_SID]
```

```python
df_exp[
    (df_exp["buffer_speed"] == 34) & (df_exp["pop_exposed"] > 0)
].sort_values("pop_exposed", ascending=False).iloc[:20]
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/fms_wind_buffers_exposure_fms_reg.parquet"
stratus.upload_parquet_to_blob(df_exp, blob_name)
```

### At adm3 level

```python
adm3 = adm3.to_crs(da_wp.rio.crs)
```

```python
da_wp_clip_all = da_wp.rio.clip(adm3.geometry, all_touched=True)
```

```python
da_wp.rio.crs
```

```python
gdf_buffers_nowrap = gdf_buffers.to_crs(da_wp.rio.crs)
```

```python
dicts = []
for pcode, adm_row in tqdm(
    adm3.set_index("ADM3_PCODE").iterrows(), total=len(adm3)
):
    da_clip_adm = da_wp_clip_all.rio.clip([adm_row.geometry])
    for _, row in gdf_buffers_nowrap.iterrows():
        if not row.geometry:
            pop_exposed = 0
        else:
            try:
                da_clip_sid = da_clip_adm.rio.clip([row.geometry])
                pop_exposed = int(da_clip_sid.sum())
            except NoDataInBounds as e:
                pop_exposed = 0
        dicts.append(
            {
                "sid": row["sid"],
                "ADM3_PCODE": pcode,
                "buffer_speed": row["buffer_speed"],
                "pop_exposed": pop_exposed,
            }
        )
```

```python
df_exp_adm3 = pd.DataFrame(dicts)
```

```python
df_exp_adm3[
    (df_exp_adm3["ADM3_PCODE"] == WAINIKELI3)
    & (df_exp_adm3["sid"] == WINSTON_SID)
]
```

```python
# blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/fms_wind_buffers_exposure_adm3.parquet"
# stratus.upload_parquet_to_blob(df_exp_adm3, blob_name)
```
