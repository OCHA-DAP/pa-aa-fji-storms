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
from io import BytesIO
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

from src.datasources import codab, worldpop, fms
from src.constants import *
from src.blob import PROJECT_PREFIX, load_geoparquet_from_blob
from src.plotting import (
    build_circle_template,
    plot_bullseye_exposures,
    plot_template_circles,
    lighten,
    wrap_text,
    plot_wind_buffers,
    plot_thermometer,
    fig_to_base64,
    plot_bubbles_and_swaths,
)
from src.exposure_calc import calculate_multi_adm_exposure
from src.email.content import render_template
from src import listmonk
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
gdf_buffers_sid_issued_single = gdf_buffers_single[
    (gdf_buffers_single["name_season"] == "Yasa 2021")
    & (gdf_buffers_single["issued_time"] == issued_time)
]
gdf_buffers_sid_issued_single
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
da_wp_clip.plot()
```

```python
gdf_buffers_sid_issued_single
```

```python
TEST_FORECAST_BLOB_NAME = "pa-aa-fji-storms/raw/fms/TC Data/TC Cody/20220110T000000Z_GP_Forecast_Track_2122_03F_CODY_(CO_DEE).csv"
```

```python
decoded_csv = BytesIO(stratus.load_blob_data(TEST_FORECAST_BLOB_NAME))
gdf_forecast = fms.parse_fms_forecast(decoded_csv)
gdf_forecast = gdf_forecast.rename(columns={"forecast_time": "valid_time"})
```

```python
gdf_readiness = gdf_forecast[gdf_forecast["leadtime"] <= 120].copy()
gdf_buffers_readiness = fms.calculate_fms_buffers_gdf(gdf_readiness)
```

```python
(
    df_exp_shift,
    gdf_shift_buffers,
    gdf_shift_tracks,
) = fms.calculate_shifted_exposures(
    gdf_readiness, da_wp_clip, disable_tqdm=False
)
df_exp_shift = df_exp_shift.sort_values(
    [f"exp_{x}" for x in [64, 50, 34]], ascending=False
)
worst_row = df_exp_shift.iloc[0].copy()
worst_row["level"] = "worst"
best_row = df_exp_shift.iloc[-1].copy()
best_row["level"] = "best"
```

```python
worst_buffers = gdf_shift_buffers[
    gdf_shift_buffers["shift_deg"] == worst_row["shift_deg"]
]
best_buffers = gdf_shift_buffers[
    gdf_shift_buffers["shift_deg"] == best_row["shift_deg"]
]
```

```python
dfs = []

for buffers, limit in [
    (gdf_buffers_readiness, "middle"),
    (worst_buffers, "worst"),
    (best_buffers, "best"),
]:
    df_in = calculate_multi_adm_exposure(
        buffers, da_wp_clip, adm3, disable_tqdm=False
    )
    df_in["limit"] = limit
    dfs.append(df_in)

df_exp_adm3 = pd.concat(dfs, ignore_index=True)
```

```python
df_adm3_out = df_exp_adm3[df_exp_adm3["limit"] == "middle"].pivot(
    columns="buffer_speed", index="ADM3_PCODE", values="pop_exposed"
)
df_adm3_out = df_adm3_out.rename(
    columns={x: f"exp_{x}_knot" for x in df_adm3_out.columns}
)
df_adm3_out = df_adm3_out.reset_index()
df_adm3_out.columns.name = None
cols = [
    "ADM1_PCODE",
    "ADM1_EN",
    "ADM2_PCODE",
    "ADM2_EN",
    "ADM3_PCODE",
    "ADM3_EN",
]
df_adm3_out = adm3[cols].merge(df_adm3_out)
df_adm3_out = df_adm3_out.sort_values("exp_64_knot", ascending=False)
df_adm3_out
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
issued_time
```

```python
f"{issued_time:%Y%m%dT%H%MZ}"
```

```python
storm_name = "Yasa"
```

```python
fig, ax = plt.subplots()
adm3.plot(ax=ax)
middle_buffers.plot(alpha=0.3, ax=ax)
```

```python
fig, axs, bubbles_plot_filepath = plot_bubbles_and_swaths(
    gdf_mostlikely_buffers=gdf_buffers_readiness,
    gdf_worst_buffers=worst_buffers,
    gdf_best_buffers=best_buffers,
    gdf_adm3_swath_plot=adm3_no_rotuma_lau,
    df_adm3_template=adm3_simple_template,
    gdf_adm3=adm3,
    df_exp_adm3=df_exp_adm3,
    cyclone_name=cyclone_name,
    forecast_display_str=forecast_display_str,
    forecast_id="TEST_ID",
    save_local=True,
)
```

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

row_specs = [
    ("middle", "Most likely track", middle_buffers),
    (
        "worst",
        "Upper bound exposure\n(worst case scenario)",
        worst_buffers,
    ),
    ("best", "Lower bound exposure\n(best case scenario)", best_buffers),
]

# ---- Plotting loop ----
for col, (limit, title_str, gdf_buffers) in enumerate(row_specs):
    top_ax = axes[0, col]
    bottom_ax = axes[1, col]

    # Wind swaths
    plot_wind_buffers(adm3_no_rotuma_lau, gdf_buffers, ax=top_ax)

    # Column titles
    if col == 0:
        title_color = "black"
        title_weight = "bold"
    elif col == 1:
        title_color = "red"
        title_weight = "normal"
    else:
        title_color = "green"
        title_weight = "normal"

    top_ax.set_title(
        title_str,
        fontsize=20,
        fontweight=title_weight,
        color=title_color,
        pad=6,
    )

    # Population exposure
    plot_bullseye_exposures(
        adm3_simple_template.merge(adm3[["ADM3_PCODE", "adm_label"]]),
        df_exp_adm3[df_exp_adm3["limit"] == limit],
        label_col="adm_label",
        min_font=4,
        max_font=20,
        ax=bottom_ax,
    )

# ---- Row labels (left side) ----
for row_idx, row_label in enumerate(["Wind swaths", "Population exposure"]):
    axes[row_idx, 0].text(
        -0.02,
        0.5,
        row_label,
        fontsize=18,
        va="center",
        ha="right",
        rotation=90,
        transform=axes[row_idx, 0].transAxes,
    )

# ---- Main title ----
fig.suptitle(
    f"{storm_name}: forecast issued {fji_time_str(issued_time)}",
    fontsize=22,
    fontweight="bold",
    y=1,
)

# ---- Layout ----
fig.tight_layout(rect=[0, 0, 1, 1])

# ---- Save ----
fig.savefig(
    f"temp/{storm_name}_fcast_{issued_time:%Y%m%dT%H%MZ}.pdf",
    format="pdf",
    bbox_inches="tight",
)
```

```python
filename = f"temp/{storm_name}_fcast_{issued_time:%Y%m%dT%H%MZ}.pdf"
```

```python
filename
```

```python
with open(filename, "rb") as f:
    files = {"file": f.read()}
```

```python
listmonk.upload_file(filename)
```

## Thermometer

```python
df_stats = fms.load_historical_stats()
```

```python
df_stats
```

```python
worst_row
```

```python
best_row
```

```python
impact_thresh = 10_000
```

```python
df_stats_major = df_stats[df_stats["exp_64"] > 5000]
```

```python
int(da_wp_clip.sum())
```

```python
df_stats_major
```

```python
cyclone_name = "Yasa"
```

```python
forecast_display_str = "2020-12-16 12:00 (Fiji time)"
```

```python
main_value = 48389
max_value = 150_000
trigger_threshold = EXP_THRESHOLD_64_KNOTS
low_bound = 1856
high_bound = 679947
```

```python
fig, ax = plot_thermometer(
    main_value=main_value,
    low_bound=low_bound,
    high_bound=high_bound,
    df_stats=df_stats,
    cyclone_name=cyclone_name,
    forecast_display_str=forecast_display_str,
)
```

```python
img_base64 = fig_to_base64(fig)
```

```python
html_str = render_template(
    template_name="informational.html",
    variables={"thermometer_plot": img_base64},
)
```

```python
with open("temp/test.html", "w", encoding="utf-8") as f:
    f.write(html_str)
```

```python
fig.savefig(
    f"temp/test.png",
    format="png",
    bbox_inches="tight",
)
```
