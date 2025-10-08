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

# Plot testing

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import math
import io

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import ocha_stratus as stratus
from shapely.geometry import box
from rioxarray.exceptions import NoDataInBounds
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


from src.datasources import codab, worldpop
from src.constants import *
from src.blob import PROJECT_PREFIX
from src.plotting import (
    build_circle_template,
    plot_bullseye_exposures,
    plot_template_circles,
    lighten,
)
```

```python
query = """
SELECT *
FROM storms.ibtracs_storms
WHERE genesis_basin = 'SP'
"""
with stratus.get_engine(stage="dev").connect() as con:
    df_storms = pd.read_sql(query, con)
```

```python
adm3 = codab.load_codab_from_blob(admin_level=3)
```

```python
da_wp = worldpop.load_worldpop_from_blob()
```

```python
da_wp = da_wp.assign_coords({"x": (((da_wp.x + 360) % 360))}).sortby("x")
```

```python
da_wp_clip = da_wp.rio.clip(adm3.geometry)
```

```python
dicts = []
for pcode, row in adm3.set_index("ADM3_PCODE").iterrows():
    try:
        da_clip = da_wp_clip.rio.clip([row.geometry])
    except NoDataInBounds as e:
        continue
    dicts.append({"pcode": pcode, "pop": int(da_clip.sum())})
```

```python
df_pop = pd.DataFrame(dicts)
```

```python
gdf_admin = adm3.merge(df_pop.rename(columns={"pcode": "ADM3_PCODE"})).rename(
    columns={"pop": "pop_total"}
)
```

```python
gdf_admin
```

```python
gdf_admin_no_rotuma = gdf_admin[gdf_admin["ADM2_PCODE"] != ROTUMA1]
```

```python
template_df = build_circle_template(
    gdf_admin_no_rotuma, crs_equal_area="EPSG:3832", area_per_person=80000
)
template_df = template_df.merge(gdf_admin[["ADM3_PCODE", "ADM3_EN"]])
plot_template_circles(template_df, label_col="ADM3_EN")
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/fms_wind_buffers_exposure_adm3.parquet"
df_exp = stratus.load_parquet_from_blob(blob_name)
```

```python
def plot_storm_exp(sid):
    row = df_storms.set_index("sid").loc[sid]
    name_season = f"{row['name'].capitalize()} {row['season']}"
    fig, ax = plot_bullseye_exposures(
        template_df,
        df_exp.set_index("sid").loc[sid],
        label_col="ADM3_EN",
    )
    ax.set_title(f"{name_season}: wind speed population exposure by Tikina")
```

```python
plot_storm_exp(WINSTON_SID)
```

```python
plot_storm_exp(YASA_SID)
```

```python
plot_storm_exp(HAROLD_SID)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/fms_wind_buffers.parquet"
gdf_buffers = gpd.read_parquet(io.BytesIO(stratus.load_blob_data(blob_name)))
gdf_buffers = gdf_buffers.to_crs(FJI_CRS)
```

```python
def plot_wind_buffers(sid):
    row = df_storms.set_index("sid").loc[sid]
    name_season = f"{row['name'].capitalize()} {row['season']}"
    fig, ax = plt.subplots(dpi=200, figsize=(10, 8))
    colors = {34: "gold", 50: "crimson", 64: "indigo"}
    colors_pale = colors_pale = {s: lighten(colors[s]) for s in colors}

    gdf_admin_no_rotuma.to_crs(FJI_CRS).boundary.plot(
        ax=ax, color="black", linewidth=0.5
    )
    xlims, ylims = ax.get_xlim(), ax.get_ylim()

    ax.axis("off")
    gdf_buffers_sid = gdf_buffers[gdf_buffers["sid"] == sid]
    for speed, color in colors_pale.items():
        gdf_buffers_sid[gdf_buffers_sid["buffer_speed"] == speed].plot(
            ax=ax, color=color
        )

    legend_patches = [
        mpatches.Patch(facecolor=colors_pale[34], label="34 kt"),
        mpatches.Patch(facecolor=colors_pale[50], label="50 kt"),
        mpatches.Patch(facecolor=colors_pale[64], label="64 kt"),
    ]
    ax.legend(
        handles=legend_patches,
        title="Wind speed",
        frameon=True,
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
    )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_title(f"{name_season}: wind speed buffers")
```

```python
plot_wind_buffers(WINSTON_SID)
```

```python
plot_wind_buffers(YASA_SID)
```

```python
plot_wind_buffers(HAROLD_SID)
```

```python

```
