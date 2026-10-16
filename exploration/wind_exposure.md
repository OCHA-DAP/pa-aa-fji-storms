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

# Wind exposure

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
from src.constants import FJI_CRS
from src.blob import PROJECT_PREFIX
```

```python
# codab.download_codab_to_blob()
```

```python
adm0 = codab.load_codab_from_blob(admin_level=0).to_crs(FJI_CRS)
adm3 = codab.load_codab_from_blob(admin_level=3).to_crs(FJI_CRS)
```

```python
# worldpop.download_worldpop_to_blob(iso3="fji")
```

```python
da_wp = worldpop.load_worldpop_from_blob(iso3="fji")
```

```python
da_wp.rio.crs
```

```python
da_wp
```

```python
da_wp = da_wp.assign_coords({"x": (((da_wp.x + 360) % 360))}).sortby("x")
```

```python
da_wp_clip = da_wp.rio.clip(adm0.geometry, all_touched=True)
```

```python
fig, ax = plt.subplots(dpi=200)
da_wp_clip.plot(ax=ax, vmax=100)
adm0.boundary.plot(ax=ax, linewidth=0.5)
```

```python
CAUKODROVE3 = "FJ10301"
```

```python
da_wp_clip.rio.clip(adm3[adm3["ADM3_PCODE"] == CAUKODROVE3].geometry).plot()
```

```python
# blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/wind_buffers.shp"
# gdf_buffers = stratus.load_shp_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/wind_buffers.parquet"
gdf_buffers = gpd.read_parquet(io.BytesIO(stratus.load_blob_data(blob_name)))
```

```python
gdf_buffers = gdf_buffers.to_crs(FJI_CRS)
```

```python
WINSTON_SID = "2016041S14170"
MICK_SID = "2009346S10172"
```

```python
fig, ax = plt.subplots(dpi=200)
da_wp_clip.plot(ax=ax, vmax=100)
gdf_buffers[gdf_buffers["sid"] == MICK_SID].plot(ax=ax, alpha=0.1)
adm0.boundary.plot(ax=ax, linewidth=0.5)
```

```python
gdf_buffers
```

```python
test_row = gdf_buffers[gdf_buffers["sid"] == WINSTON_SID].iloc[2]
```

```python
ax = adm0.boundary.plot()
da_wp.rio.clip([test_row.geometry]).plot(ax=ax)
```

```python
dicts = []
for _, row in tqdm(gdf_buffers.iterrows(), total=len(gdf_buffers)):
    if not row.geometry:
        continue
    try:
        da_clip_sid = da_wp_clip.rio.clip([row.geometry])
    except NoDataInBounds as e:
        continue
    dicts.append(
        {
            "sid": row["sid"],
            "buffer_speed": row["buffer_speed"],
            "pop_exposed": int(da_clip_sid.sum()),
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
int(da_wp_clip.sum())
```

```python
df_exp[df_exp["buffer_speed"] == 64].sort_values(
    "pop_exposed", ascending=False
)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/wind_buffers_exposure.parquet"
stratus.upload_parquet_to_blob(df_exp, blob_name)
```
