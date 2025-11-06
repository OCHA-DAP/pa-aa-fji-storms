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
import matplotlib.patches as mpatches
import pandas as pd
import ocha_stratus as stratus
from shapely.geometry import box
from rioxarray.exceptions import NoDataInBounds
from matplotlib.patches import Circle
from tqdm.auto import tqdm

from src.datasources import codab, worldpop
from src.constants import *
from src.blob import PROJECT_PREFIX
from src.plotting import (
    build_circle_template,
    plot_bullseye_exposures,
    plot_template_circles,
    lighten,
    wrap_text,
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
# adm3 = adm3.to_crs(FJI_CRS)
```

```python
adm3 = adm3.to_crs(4326)
```

```python
adm3
```

```python
da_wp = worldpop.load_worldpop_from_blob()
```

```python
da_wp.assign_coords({"x": (((da_wp.x + 360) % 360))}).sortby("x").rio.clip(
    adm3.to_crs(FJI_CRS).geometry
).plot()
```

```python
da_wp.sum().values
```

```python
# da_wp = da_wp.assign_coords({"x": (((da_wp.x + 360) % 360))}).sortby("x")
```

```python
# da_wp = da_wp.assign_coords({"x": (((da_wp.x + 180) % 360) - 180)}).sortby("x")
```

```python
da_wp_clip = da_wp.rio.clip(adm3.geometry, all_touched=True)
```

```python
da_wp_clip.sum().values
```

```python
dicts = []
empty_adms = []
for pcode, row in tqdm(
    adm3.set_index("ADM3_PCODE").iterrows(), total=len(adm3)
):
    try:
        da_clip = da_wp_clip.rio.clip([row.geometry])
    except NoDataInBounds as e:
        empty_adms.append(pcode)
        print(f"no pop found for {pcode}")
        print(row.geometry.bounds)
        continue
    dicts.append({"pcode": pcode, "pop": int(da_clip.sum())})
```

```python
df_pop = pd.DataFrame(dicts)
```

```python
df_pop
```

```python
row.geometry.bounds
```

```python
da_wp_clip.rio.bounds()
```

```python
gdf_admin = adm3.merge(df_pop.rename(columns={"pcode": "ADM3_PCODE"})).rename(
    columns={"pop": "pop_total"}
)
```

```python
gdf_admin_simple = gdf_admin[~gdf_admin["ADM2_PCODE"].isin([ROTUMA2, LAU2])]
```

```python
gdf_admin["adm_label"] = gdf_admin["ADM3_EN"].apply(
    wrap_text, max_len=9, break_anywhere=True
)
```

```python
template_df = build_circle_template(
    gdf_admin, crs_equal_area="EPSG:3832", area_per_person=80000
)
```

```python
template_df_simple = build_circle_template(
    gdf_admin_simple, crs_equal_area="EPSG:3832", area_per_person=80000
)
```

```python
plot_template_circles(
    template_df.merge(gdf_admin[["ADM3_PCODE", "adm_label"]]),
    label_col="adm_label",
    min_font=4,
    max_font=20,
)
```

```python
plot_template_circles(
    template_df_simple.merge(gdf_admin[["ADM3_PCODE", "adm_label"]]),
    label_col="adm_label",
    min_font=4,
    max_font=20,
)
```

```python
gdf_admin2 = (
    gdf_admin[["ADM2_PCODE", "ADM2_EN", "pop_total", "geometry"]]
    .dissolve(["ADM2_PCODE", "ADM2_EN"], aggfunc="sum")
    .reset_index()
)
```

```python
gdf_admin2["adm_label"] = (
    gdf_admin2["ADM2_EN"]
    .str.replace("_", " ")
    .apply(wrap_text, max_len=9, break_anywhere=True)
)
```

```python
template_df_adm2 = build_circle_template(
    gdf_admin2,
    crs_equal_area="EPSG:3832",
    area_per_person=80_000,
    id_col="ADM2_PCODE",
)
```

```python
template_df_adm2.loc[template_df_adm2["ADM2_PCODE"] == ROTUMA2, "y"] = -1.75e6
```

```python
plot_template_circles(
    template_df_adm2.merge(gdf_admin2[["ADM2_PCODE", "adm_label"]]),
    label_col="adm_label",
    min_font=6,
    max_font=30,
)
```

```python
gdf_admin1 = (
    gdf_admin[["ADM1_PCODE", "ADM1_EN", "pop_total", "geometry"]]
    .dissolve(["ADM1_PCODE", "ADM1_EN"], aggfunc="sum")
    .reset_index()
)
```

```python
gdf_admin1["adm_label"] = (
    gdf_admin1["ADM1_EN"].str.replace("  ", " ").str.replace(" ", "\n")
)
```

```python
template_df_adm1 = build_circle_template(
    gdf_admin1,
    crs_equal_area="EPSG:3832",
    area_per_person=80_000,
    id_col="ADM1_PCODE",
)
template_df_adm1 = template_df_adm1.merge(
    gdf_admin1[["ADM1_PCODE", "adm_label"]]
)
template_df_adm1.loc[template_df_adm1["ADM1_PCODE"] == EASTERN1, "x"] = 3.32e6
```

```python
plot_template_circles(
    template_df_adm1,
    label_col="adm_label",
    min_font=12,
    max_font=30,
)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/fms_wind_buffers_exposure_adm3.parquet"
df_exp = stratus.load_parquet_from_blob(blob_name)
```

```python
df_exp[(df_exp["ADM3_PCODE"] == WAINIKELI3) & (df_exp["sid"] == WINSTON_SID)]
```

```python
def plot_storm_exp(sid):
    row = df_storms.set_index("sid").loc[sid]
    name_season = f"{row['name'].capitalize()} {row['season']}"
    fig, ax = plot_bullseye_exposures(
        template_df_simple.merge(gdf_admin[["ADM3_PCODE", "adm_label"]]),
        df_exp[df_exp["sid"] == sid],
        label_col="adm_label",
        min_font=4,
        max_font=20,
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
