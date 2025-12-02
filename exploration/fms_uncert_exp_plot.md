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

# FMS uncertainty exposure plotting

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import math
import io
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import ocha_stratus as stratus
from svgutils.compose import Figure, SVG
from shapely.geometry import box
from rioxarray.exceptions import NoDataInBounds
from matplotlib.patches import Circle
from tqdm.auto import tqdm

from src.datasources import codab, worldpop
from src.constants import *
from src.blob import PROJECT_PREFIX, load_geoparquet_from_blob
from src.plotting import (
    build_circle_template,
    plot_bullseye_exposures,
    plot_template_circles,
    lighten,
    wrap_text,
    plot_wind_buffers,
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
adm3["adm_label"] = adm3["ADM3_EN"].apply(
    wrap_text, max_len=9, break_anywhere=True
)
```

```python
adm3 = adm3.to_crs(FJI_CRS)
```

```python
adm3_no_rotuma_lau = adm3[~(adm3["ADM2_PCODE"].isin([ROTUMA2, LAU2]))]
```

```python
da_wp = worldpop.load_worldpop_from_blob(iso3="fji")
da_wp = da_wp.assign_coords({"x": (((da_wp.x + 360) % 360))}).sortby("x")
```

```python
da_wp_clip = da_wp.rio.clip(adm3.geometry)
```

```python
da_wp_clip.plot()
```

```python

```

```python
blob_name = f"{PROJECT_PREFIX}/processed/plotting/adm3_simple_template.parquet"
adm3_simple_template = stratus.load_parquet_from_blob(blob_name)
```

```python
plot_template_circles(
    adm3_simple_template.merge(adm3[["ADM3_PCODE", "adm_label"]]),
    label_col="adm_label",
    min_font=4,
    max_font=20,
)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/fms_shift_buffers.parquet"
gdf_buffers = load_geoparquet_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/forecast_tracks_buffers.parquet"
gdf_buffers_single = load_geoparquet_from_blob(blob_name)
```

```python
gdf_buffers_single
```

```python
gdf_buffers = gdf_buffers.rename(columns={"speed": "buffer_speed"})
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/fms_shift_tracks.parquet"
df_shift_tracks = stratus.load_parquet_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/fms_shift_exp.parquet"
df_shift_exp = stratus.load_parquet_from_blob(blob_name)
```

```python
sid = YASA_SID
```

```python
df_shift_exp_sid = df_shift_exp[df_shift_exp["sid"] == sid]
```

```python
df_shift_exp_sid
```

```python
df_shift_exp_sid["issued_time"].unique()
```

```python
issued_time = df_shift_exp_sid["issued_time"].unique()[1]
```

```python
issued_time
```

```python
df_shift_exp_sid_issued = df_shift_exp_sid[
    df_shift_exp_sid["issued_time"] == issued_time
]
```

```python
df_shift_exp_sid_issued = df_shift_exp_sid_issued.sort_values(
    [f"exp_{x}" for x in [64, 50, 34]], ascending=False
)
```

```python
gdf_buffers_sid_isused = gdf_buffers[
    (gdf_buffers["sid"] == sid) & (gdf_buffers["issued_time"] == issued_time)
]
```

```python
middle_buffers = gdf_buffers_single[
    gdf_buffers_single["issued_time"] == issued_time
]
```

```python
gdf_buffers_sid_issued_single
```

```python
worst_row = df_shift_exp_sid_issued.iloc[0]
best_row = df_shift_exp_sid_issued.iloc[-1]
```

```python
best_row
```

```python
worst_deg = worst_row["shift_deg"]
best_deg = best_row["shift_deg"]
```

```python
worst_buffers = gdf_buffers_sid_isused[
    gdf_buffers_sid_isused["shift_deg"] == worst_deg
]
best_buffers = gdf_buffers_sid_isused[
    gdf_buffers_sid_isused["shift_deg"] == best_deg
]
```

```python
df_shift_tracks_sid_issued = df_shift_tracks[
    (df_shift_tracks["sid"] == sid)
    & (df_shift_tracks["issued_time"] == issued_time)
]
```

```python
worst_track = df_shift_tracks_sid_issued[
    df_shift_tracks_sid_issued["shift_deg"] == worst_deg
]
best_track = df_shift_tracks_sid_issued[
    df_shift_tracks_sid_issued["shift_deg"] == best_deg
]
```

```python
fig, ax = plt.subplots()
adm3.boundary.plot(ax=ax)
worst_buffers.plot(ax=ax, alpha=0.3)
best_buffers.plot(ax=ax, alpha=0.3)
```

```python
da_wp_clip
```

```python
gdf_buffers_sid_issued_single
```

```python
dicts = []

for buffers, limit in [
    (gdf_buffers_sid_issued_single, "middle"),
    (worst_buffers, "worst"),
    (best_buffers, "best"),
]:
    for _, buffer_row in tqdm(buffers.iterrows(), total=len(buffers)):
        da_clip_buffer = da_wp_clip.rio.clip([buffer_row.geometry])
        for _, adm_row in adm3.iterrows():
            try:
                da_clip_buffer_adm = da_clip_buffer.rio.clip(
                    [adm_row.geometry], all_touched=True
                )
                pop_exposed = int(da_clip_buffer_adm.sum())
            except NoDataInBounds as e:
                pop_exposed = 0
            dicts.append(
                {
                    "ADM3_PCODE": adm_row["ADM3_PCODE"],
                    "limit": limit,
                    "buffer_speed": buffer_row["buffer_speed"],
                    "pop_exposed": pop_exposed,
                }
            )
```

```python
df_exp_adm3 = pd.DataFrame(dicts)
```

```python
df_exp_adm3
```

```python
def to_fji_time(dt):
    dt_utc = dt.replace(tzinfo=ZoneInfo("UTC"))
    dt_fiji = dt_utc.astimezone(ZoneInfo("Pacific/Fiji"))
    return dt_fiji
```

```python
def fji_time_str(dt):
    dt_fiji = to_fji_time(dt)
    return f"{dt_fiji:%Y-%m-%d %H:%M} (Fiji time)"
```

```python
fji_time_str(issued_time)
```

```python
issued_time_fji = to_fji_time(issued_time)
```

```python
issued_time_fji
```

```python
storm_issued_str = f"Yasa: forecast issued {fji_time_str(issued_time)}"
```

```python
storm_issued_str
```

```python
for row, stuff in enumerate(
    [
        ("middle", "Most likely"),
        ("worst", "Highest reasonable"),
        ("best", "Lowest reasonable"),
    ]
):
    print(row, stuff)
```

```python
axes[0]
```

```python
title
```

```python
y_center
```

```python
issued_time
```

```python
f"{issued_time:%Y%m%dT%H%MZ}"
```

```python
storm_name = "Yasa"
```

```python
fig, axes = plt.subplots(3, 2, figsize=(12, 18))

row_specs = [
    ("middle", "Most likely track", middle_buffers),
    (
        "worst",
        "Highest reasonable exposure\n(worst case scenario)",
        worst_buffers,
    ),
    ("best", "Lowest reasonable exposure\n(best case scenario)", best_buffers),
]

# ---- Plotting Loop ----
for row, (limit, title_str, gdf_buffers) in enumerate(row_specs):
    # ---- Left column: wind swaths ----
    left_ax = axes[row][0]
    plot_wind_buffers(adm3_no_rotuma_lau, gdf_buffers, ax=left_ax)

    # Row label on the left axis
    left_ax.text(
        -0.05,  # x-position (axes coords; <0 means “left of axis”)
        0.5,  # y-position (centered vertically)
        title_str,
        fontsize=16,
        va="center",
        ha="center",
        transform=left_ax.transAxes,
        rotation=90,
    )

    # ---- Right column: population exposure ----
    plot_bullseye_exposures(
        adm3_simple_template.merge(adm3[["ADM3_PCODE", "adm_label"]]),
        df_exp_adm3[df_exp_adm3["limit"] == limit],
        label_col="adm_label",
        min_font=4,
        max_font=20,
        ax=axes[row][1],
    )

# ---- Column Titles ----
axes[0][0].set_title("Wind swaths", fontsize=18, pad=15)
axes[0][1].set_title("Population exposure", fontsize=18, pad=15)

# ---- Overall Figure Title ----
fig.suptitle(
    f"{storm_name}: forecast issued {fji_time_str(issued_time)}",
    fontsize=22,
    fontweight="bold",
    y=0.97,
)

# Adjust layout so nothing gets chopped off
fig.tight_layout(rect=[0, 0, 1, 0.98])

fig.savefig(
    f"temp/{storm_name}_fcast_{issued_time:%Y%m%dT%H%MZ}.pdf",
    format="pdf",
    bbox_inches="tight",
)
```
