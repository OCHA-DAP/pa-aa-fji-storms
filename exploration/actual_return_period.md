---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: pa-aa-fji-storms
    language: python
    name: pa-aa-fji-storms
---

# Actual return period

Return period based on actual tracks from FMS

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

import utils
```

```python
load_dotenv()

MB_TOKEN = os.environ["MAPBOX_TOKEN"]
```

```python
fms = utils.load_cyclonetracks()
fms = utils.interpolate_cyclonetracks(fms)
adm0 = utils.load_codab(level=0)
```

```python
# if needed, filter to only include those with ecmwf hindcasts
# ecmwf = utils.load_ecmwf_besttrack_hindcasts()
# ecnameyears = ecmwf["nameyear"].unique()
# fms = fms[fms["nameyear"].isin(ecnameyears)]
```

```python
fms["distance"] = (
    fms.to_crs(3832).geometry.distance(adm0.to_crs(3832).iloc[0].geometry)
    / 1000
)
```

```python
fms = fms.sort_values("datetime", ascending=False)
```

```python
# find triggers based on various composite triggers

close_distances = [-1, 0, 25, 50, 100]
far_distances = [-1, 100, 150, 200, 250]

triggers = pd.DataFrame()

for close_distance in close_distances:
    for far_distance in far_distances:
        thresholds = [
            {"name": "close", "distance": close_distance, "category": 3},
            {"name": "far", "distance": far_distance, "category": 4},
        ]
        dff = pd.DataFrame()
        for threshold in thresholds:
            df_add = fms[
                (fms["distance"] <= threshold.get("distance"))
                & (fms["Category numeric"] >= threshold.get("category"))
            ]
            dff = pd.concat([dff, df_add])
        dff = dff.sort_values("datetime", ascending=False)
        nameseasons = dff["Name Season"].unique()
        df_add = pd.DataFrame(
            {
                "close": close_distance,
                "far": far_distance,
                "count": len(nameseasons),
                "nameseasons": "<br>".join(nameseasons),
            },
            index=[0],
        )
        triggers = pd.concat([triggers, df_add], ignore_index=True)

triggers["return"] = len(fms["datetime"].dt.year.unique()) / triggers["count"]

df_freq = triggers.pivot(
    index="close",
    columns="far",
    values="return",
)
df_freq = df_freq.sort_values("close", ascending=True)
df_freq.columns = df_freq.columns.astype(str)
df_freq.index = df_freq.index.astype(str)
df_freq = df_freq.astype(float).round(2)

df_records = triggers.pivot(
    index="close",
    columns="far",
    values="nameseasons",
)
df_records = df_records.sort_values("close", ascending=True)
df_records.columns = df_records.columns.astype(str)
df_records.index = df_records.index.astype(str)

fig = px.imshow(
    df_freq,
    text_auto=True,
    range_color=[2, 4],
    color_continuous_scale="Reds",
)
fig.update(
    data=[
        {
            "customdata": df_records,
            "hovertemplate": "Cyclones triggered:<br>%{customdata}",
        }
    ]
)

fig.update_traces(name="")
fig.update_layout(
    coloraxis_colorbar_title="Return<br>period<br>(years)",
)
fig.update_xaxes(side="top", title_text="Distance for Cat 4+")
fig.update_yaxes(title_text="Distance for Cat 3+")

fig.show()
```

```python
display(pd.DataFrame(df_records.loc["0", "250"].split("<br>")))
```

```python
# save specific trigger

close_distance = 0
far_distance = 200

threshold_name = f"d{close_distance}c3_d{far_distance}c4"
thresholds = [
    {"name": "close", "distance": close_distance, "category": 3},
    {"name": "far", "distance": far_distance, "category": 4},
]

fms_f = pd.DataFrame()

for threshold in thresholds:
    df_add = fms[
        (fms["distance"] <= threshold.get("distance"))
        & (fms["Category numeric"] >= threshold.get("category"))
    ]
    fms_f = pd.concat([fms_f, df_add])
fms_f = fms_f.drop_duplicates()

triggers = pd.DataFrame()

for nameyear in fms_f["nameyear"].unique():
    dff = fms[fms["nameyear"] == nameyear]

    # find actual landfall
    df_actual_landfall = dff[dff["distance"] == 0]
    if df_actual_landfall.empty:
        landfall_date = None
        # find closest pass instead
        min_distance = dff["distance"].min()
        min_distance_date = dff[dff["distance"] == min_distance][
            "datetime"
        ].min()

    else:
        landfall_date = df_actual_landfall["datetime"].min()
        min_distance_date = landfall_date

    df_add = pd.DataFrame(
        {
            "nameyear": nameyear,
            "landfall_date": landfall_date,
            "min_distance_date": min_distance_date,
        },
        index=[0],
    )

    triggers = pd.concat([triggers, df_add], ignore_index=True)

triggers.to_csv(
    utils.EXP_DIR / f"fms_triggers_{threshold_name}.csv", index=False
)
```

```python

```
