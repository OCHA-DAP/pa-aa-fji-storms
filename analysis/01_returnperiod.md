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

# Return period

Calculate return period of tropical cyclones \(TCs)
by category and distance to Fiji.
Can also calculate composite triggers
\(i.e. multiple distance / strength combinations).

In the end, the selected AA framework trigger is a TC that is forecasted to be:

- Category 4 or 5 while within 250 km of Fiji, _or_
- Category 3, 4, 5 while making landfall in Fiji

## Data sources

- TCs best tracks sent by Fiji Met Services \(FMS)
  - Category is using Australian scale
- CODAB adm0 from HDX

## Notes

This notebook only looks at the _actual_ path of TCs, whereas of course in
reality, we will be triggering off the forecast. Thus, we assume that the best
track forecasts are sufficiently accurate and unbiased so that the return
period based on the actual tracks is close to the return period of a trigger
based on forecasts. Nevertheless, the return period will always be lower
\(_i.e._, storms will appear to be more frequent) with forecasts, since a
forecast could indicate a storm will pass within the trigger zone, but in
reality it veers away \(_i.e._, a false positive). By design, there are no
false negatives since as soon as a TC enters the trigger zone, it will also
be forecast to enter the trigger zone \(forecasting 0 hours into the future).

CRS:

- `EPSG:3832` is projected CRS for Fiji region \(in metres)
- `EPSG:4326` generally does **not** work for plots because Fiji spans the antimeridian
- the CRS `utils.FJI_CRS` should be used instead for plots using degrees
  - for reference, it is `"+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84
  +no_defs"`

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import utils
```

## Load data

```python
fms = utils.load_cyclonetracks()
adm0 = utils.load_codab(level=0)
```

## Interpolate and calculate distance

```python
# interpolate FMS tracks to hourly
fms = utils.interpolate_cyclonetracks(fms)
```

```python
# calculate distance between each point and adm0 (slow-ish)
fms["Distance (km)"] = (
    fms.to_crs(epsg=3832).apply(
        lambda point: point.geometry.distance(adm0.geometry),
        axis=1,
    )
    / 1000
)
```

## Calculate simple trigger return periods

Simple trigger is just whether path of TC meets distance and strength
criteria, _e.g._:

- Category 4 or 5 while within 250 km of Fiji

```python
distances = [0, 100, 250, 370, 500]
categories = [2, 3, 4, 5]

triggers = pd.DataFrame()

for distance in distances:
    for category in categories:
        thresholds = [
            {
                "distance": distance,
                "category": category,
            },
        ]
        dff = pd.DataFrame()
        for threshold in thresholds:
            # cycle through composite thresholds
            df_add = fms[
                (fms["Distance (km)"] <= threshold.get("distance"))
                & (fms["Category numeric"] >= threshold.get("category"))
            ]
            dff = pd.concat([dff, df_add])
        dff = dff.sort_values("datetime", ascending=False)
        nameseasons = dff["Name Season"].unique()
        df_add = pd.DataFrame(
            {
                "distance": distance,
                "category": category,
                "count": len(nameseasons),
                "nameseasons": "<br>".join(nameseasons),
            },
            index=[0],
        )
        triggers = pd.concat([triggers, df_add], ignore_index=True)

triggers["return"] = len(fms["datetime"].dt.year.unique()) / triggers["count"]

# reshape return period
df_freq = triggers.pivot(
    index="category",
    columns="distance",
    values="return",
)
df_freq = df_freq.sort_values("category", ascending=False)
df_freq.columns = df_freq.columns.astype(str)
df_freq.index = df_freq.index.astype(str)
df_freq = df_freq.astype(float).round(2)

# reshape lists of cyclones triggered
df_records = triggers.pivot(
    index="category",
    columns="distance",
    values="nameseasons",
)
df_records = df_records.sort_values("category", ascending=False)
df_records.columns = df_records.columns.astype(str)
df_records.index = df_records.index.astype(str)

fig = px.imshow(
    df_freq,
    text_auto=True,
    range_color=[1, 5],
    color_continuous_scale="Reds",
)
# add lists of cyclones triggered as customdata for hover
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
fig.update_xaxes(side="top", title_text=f"Distance (km)")
fig.update_yaxes(title_text=f"Category")

# note: can change renderer to show in browser instead

# if plot doesn't initially show up, switch to renderer="svg"
# then back to "notebook"
fig.show(renderer="notebook")
```

## Calculate composite trigger return periods

Composite trigger consists of multiple distance / strength
combinations, e.g.:

- Category 4 or 5 while within 250 km of Fiji, _or_
- Category 3, 4, 5 while making landfall in Fiji

Here we compare how adjusting the threshold distances for
Category 4+ and Category 3+ affect the return period.

```python
# find triggers based on various composite triggers

# -1 included in distance list to cancel out that part of the trigger
close_distances = [-1, 0, 25, 50, 100]
far_distances = [-1, 100, 150, 200, 250, 300]

close_category = 3
far_category = 4

triggers = pd.DataFrame()

for close_distance in close_distances:
    for far_distance in far_distances:
        thresholds = [
            {
                "name": "close",
                "distance": close_distance,
                "category": close_category,
            },
            {
                "name": "far",
                "distance": far_distance,
                "category": far_category,
            },
        ]
        dff = pd.DataFrame()
        for threshold in thresholds:
            # cycle through composite thresholds
            df_add = fms[
                (fms["Distance (km)"] <= threshold.get("distance"))
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

# reshape return period
df_freq = triggers.pivot(
    index="close",
    columns="far",
    values="return",
)
df_freq = df_freq.sort_values("close", ascending=True)
df_freq.columns = df_freq.columns.astype(str)
df_freq.index = df_freq.index.astype(str)
df_freq = df_freq.astype(float).round(2)

# reshape lists of cyclones triggered
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
# add lists of cyclones triggered as customdata for hover
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
fig.update_xaxes(side="top", title_text=f"Distance for Cat {far_category}+")
fig.update_yaxes(title_text=f"Distance for Cat {close_category}+")

# note: can change renderer to show in browser instead
fig.show()
```

```python
# save specific trigger

close_distance = 0
far_distance = 250

threshold_name = f"d{close_distance}c3_d{far_distance}c4"
thresholds = [
    {"name": "close", "distance": close_distance, "category": 3},
    {"name": "far", "distance": far_distance, "category": 4},
]

fms_f = pd.DataFrame()

for threshold in thresholds:
    df_add = fms[
        (fms["Distance (km)"] <= threshold.get("distance"))
        & (fms["Category numeric"] >= threshold.get("category"))
    ]
    fms_f = pd.concat([fms_f, df_add])
fms_f = fms_f.drop_duplicates()

triggers = pd.DataFrame()

for nameyear in fms_f["nameyear"].unique():
    dff = fms[fms["nameyear"] == nameyear]

    # find actual landfall
    df_actual_landfall = dff[dff["Distance (km)"] == 0]
    if df_actual_landfall.empty:
        landfall_date = None
        # find closest pass instead
        min_distance = dff["Distance (km)"].min()
        min_distance_date = dff[dff["Distance (km)"] == min_distance][
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

# NaT in landfall_date implies storm didn't actually make landfall
# (based on 1-hr interpolations of storm track)
display(triggers)
triggers.to_csv(
    utils.EXP_DIR / f"fms_triggers_{threshold_name}.csv", index=False
)
```

```python

```
