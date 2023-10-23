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

# Hindcasts

From Fiji Met Services

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
import geopandas as gpd
from pyproj import pyproj
from shapely.geometry import Point, LineString
from shapely.validation import make_valid, explain_validity
from shapely.ops import transform
import fiona
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import ModelDesc, Term, LookupFactor
import plotly.figure_factory as ff
import chart_studio
import chart_studio.plotly as py
import plotly.io as pio

from src import utils
from src.constants import FJI_CRS
```

```python
load_dotenv()

AA_DATA_DIR = Path(os.environ["AA_DATA_DIR"])
EXP_DIR = AA_DATA_DIR / "public/exploration/fji"
FCAST_DIR = EXP_DIR / "rsmc/forecasts"

MB_TOKEN = os.environ["MAPBOX_TOKEN"]
CS_KEY = os.environ["CHARTSTUDIO_APIKEY"]
chart_studio.tools.set_credentials_file(
    username="tristandowning", api_key=CS_KEY
)
```

## Load data

```python
gdf_adm0 = gpd.read_file(
    utils.ADM0_PATH, layer="fji_polbnda_adm0_country"
).set_crs(3832)
```

```python
forecasts = utils.load_hindcasts()
actual = utils.load_cyclonetracks()
forecast_nameseasons = forecasts["Name Season"].unique()
actual = actual[actual["Name Season"].isin(forecast_nameseasons)]
```

```python
actual_interp = utils.interpolate_cyclonetracks(actual)
```

```python
df = forecasts.merge(
    actual_interp,
    left_on=["Name Season", "forecast_time"],
    right_on=["Name Season", "datetime"],
    suffixes=["_forecast", "_actual"],
)

df["leadtime"] = df["forecast_time"] - df["base_time"]
df["leadtime (days)"] = (
    df["leadtime"].dt.days + df["leadtime"].dt.seconds / 3600 / 24
)

# calculate forecast error in deg
df["error (deg)"] = df.apply(
    lambda row: row["geometry_actual"].distance(row["geometry_forecast"]),
    axis=1,
)

# calculate forecast error in km
# I'm sure there is a better way to do this
distances = (
    gpd.GeoDataFrame(geometry=df["geometry_actual"])
    .to_crs(3832)
    .distance(gpd.GeoDataFrame(geometry=df["geometry_forecast"]).to_crs(3832))
    / 1000
)
df["error (km)"] = distances

df["category error"] = df["Category"] - df["Category numeric"]
df["category abs error"] = df["category error"].abs()
```

## Plot errors

```python
df_plot = (
    df.groupby(["Category", "leadtime"]).mean(numeric_only=True).reset_index()
)

# distance
fig = px.line(
    df_plot,
    x="leadtime (days)",
    y="error (km)",
    color="Category",
)
fig.update_layout(title="Forecast distance error", template="simple_white")
fig.update_xaxes(rangemode="tozero", title="Forecast leadtime (days)")
fig.update_yaxes(
    rangemode="tozero",
    title="Forecast error (km)",
)
fig.show()

# category
fig = px.line(
    df_plot,
    x="leadtime (days)",
    y="category error",
    color="Category",
)
fig.update_layout(title="Forecast category error", template="simple_white")
fig.update_xaxes(rangemode="tozero", title="Forecast leadtime (days)")
fig.update_yaxes(
    rangemode="tozero",
    title="Forecast error (category)",
)
fig.show()
```

## Check historical triggers

```python
# load buffer (default 250 km)

buffer_distance = 250
# utils.process_buffer(buffer_distance)
trigger_zone = utils.load_buffer(buffer_distance)
trigger_zone = trigger_zone.to_crs(src.constants.FJI_CRS)
```

```python
# calculate distance to buffer of forecast
df[f"forecast_to_{buffer_distance}_buffer"] = (
    gpd.GeoDataFrame(geometry=df["geometry_forecast"])
    .to_crs(3832)
    .geometry.distance(trigger_zone.to_crs(3832).iloc[0].geometry)
    / 1000
)
# calculate distance to Fiji of forecast
df[f"forecast_to_adm0"] = (
    gpd.GeoDataFrame(geometry=df["geometry_forecast"])
    .to_crs(3832)
    .geometry.distance(gdf_adm0.iloc[0].geometry)
    / 1000
)
# calculate distance to Fiji of actual
actual_interp["actual_to_adm0"] = (
    actual_interp.to_crs(3832).geometry.distance(gdf_adm0.iloc[0].geometry)
    / 1000
)
```

```python
# find historical triggers

df_triggers = pd.DataFrame()

for name_season in df["Name Season"].unique():
    print(name_season)
    dff = df[df["Name Season"] == name_season]
    actual_interp_f = actual_interp[
        actual_interp["Name Season"] == name_season
    ]

    # find first forecast trigger
    df_triggered = dff[
        (dff[f"forecast_to_{buffer_distance}_buffer"] == 0)
        & (dff["Category"] > 3)
    ]
    first_trigger_date = df_triggered["base_time"].min()
    print(f"First trigger: {first_trigger_date}")

    # find actual trigger
    df_actual_triggered = actual_interp_f[
        (actual_interp_f["actual_to_adm0"] < buffer_distance)
        & (actual_interp_f["Category numeric"] > 3)
    ]
    first_actual_trigger = df_actual_triggered["datetime"].min()
    print(f"First actual trigger: {first_actual_trigger}")

    # find actual landfall
    df_actual_landfall = actual_interp_f[
        actual_interp_f["actual_to_adm0"] == 0
    ]
    if df_actual_landfall.empty:
        landfall_date = None
        # find closest pass instead
        min_distance = actual_interp_f["actual_to_adm0"].min()
        min_distance_date = actual_interp_f[
            actual_interp_f["actual_to_adm0"] == min_distance
        ]["datetime"].min()
        print(f"No landfall; min distance at: {min_distance_date}")
    else:
        landfall_date = df_actual_landfall["datetime"].min()
        min_distance_date = landfall_date
        print(f"Actual landfall date: {landfall_date}")
    print("")

    df_add = pd.DataFrame(
        {
            "name_season": name_season,
            "first_trigger_date": first_trigger_date,
            "first_actual_trigger": first_actual_trigger,
            "landfall_date": landfall_date,
            "min_distance_date": min_distance_date,
        },
        index=[0],
    )

    df_triggers = pd.concat([df_triggers, df_add], ignore_index=True)

df_triggers["leadtime"] = (
    df_triggers["min_distance_date"] - df_triggers["first_trigger_date"]
)
```

```python
df_triggers
```

```python
mean_fms_leadtime = df_triggers["leadtime"].mean()
print(mean_fms_leadtime.days + mean_fms_leadtime.seconds / 3600 / 24)
```

## Plot tracks

```python
pio.renderers.default = "notebook"
pio.renderers.default = "browser"
# utils.process_buffer(200)
trigger_zone = utils.load_buffer(250)
trigger_zone = trigger_zone.to_crs(src.constants.FJI_CRS)

for name_season in df["Name Season"].unique():
    df_plot = df[df["Name Season"] == name_season].sort_values("base_time")
    actual_f = actual[actual["Name Season"] == name_season]

    fig = px.choropleth_mapbox(
        trigger_zone,
        geojson=trigger_zone.geometry,
        locations=trigger_zone.index,
        mapbox_style="open-street-map",
    )
    fig.update_traces(
        marker_opacity=0.3,
        name="Area within 250km of Fiji",
    )

    fig.add_trace(
        go.Scattermapbox(
            lon=actual_f["Longitude"],
            lat=actual_f["Latitude"],
            mode="lines+markers",
            line=dict(color="black"),
            name="Actual",
            customdata=actual_f[["Category numeric", "datetime"]],
            hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
        )
    )

    for base_time in df_plot["base_time"].unique():
        dff = df_plot[df_plot["base_time"] == base_time]
        dff = dff.sort_values("forecast_time")
        fig.add_trace(
            go.Scattermapbox(
                lon=dff["Longitude_forecast"],
                lat=dff["Latitude_forecast"],
                mode="lines+markers",
                line=dict(width=1),
                marker=dict(size=3),
                name=base_time.strftime("%b %d, %H:%M"),
                customdata=dff[["Category", "forecast_time"]],
                hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
            )
        )

    fig.update_layout(
        mapbox_style="mapbox://styles/tristandownin/clk2sq75x01lc01pc9xxn7ov9",
        mapbox_accesstoken=MB_TOKEN,
        mapbox_zoom=4.5,
        mapbox_center_lat=-17,
        mapbox_center_lon=179,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title=f"{name_season}<br><sup>FMS 72hr forecasts</sup>",
    )

    fig.show()
```

## Plot timeline

```python jupyter={"outputs_hidden": true}
for name_season in df["Name Season"].unique():
    df_plot = df[df["Name Season"] == name_season]

    fig = go.Figure()
    for base_time in df_plot["base_time"].unique():
        dff = df_plot[df_plot["base_time"] == base_time]
        fig.add_trace(
            go.Scatter(
                x=dff["forecast_time"],
                y=dff["forecast_to_adm0"],
                name=str(base_time),
            )
        )

    fig.update_layout(template="simple_white")
    fig.show()
```

```python

```
