---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: venv
    language: python
    name: venv
---

# Storm Tracks

```python
%load_ext jupyter_black
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
from shapely.geometry import Point
from shapely.validation import make_valid, explain_validity
from shapely.ops import transform
import fiona
import matplotlib.pyplot as plt

pyo.init_notebook_mode()
```

```python
load_dotenv()
EXP_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/exploration/fji"
CYCLONETRACKS_PATH = (
    EXP_DIR / "rsmc/RSMC TC Tracks Historical 1969_70 to 2022_23 Seasons.csv"
)
RAW_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/raw/fji"
CODAB_PATH = RAW_DIR / "cod_ab/fji_polbnda_adm0_country"
# CODAB_PATH = RAW_DIR / "cod_ab/fji_polbnda_adm1_district"
# CODAB_PATH = RAW_DIR / "cod_ab/fji_polbnda_adm.gdb"
```

```python
# Load and process cyclone tracks

df = pd.read_csv(CYCLONETRACKS_PATH)
df["Date"] = df["Date"].apply(lambda x: x.strip())
df["datetime"] = df["Date"] + " " + df["Time"]
df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %HZ")
df = df.drop(["Date", "Time"], axis=1)
cyclone_names = df["Cyclone Name"].unique()
seasons = df["Season"].unique()
df["Category numeric"] = pd.to_numeric(df["Category"], errors="coerce")

gdf_tracks = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"])
)
gdf_tracks.crs = "EPSG:4326"

# Read CODAB

gdf_adm0 = gpd.read_file(CODAB_PATH, layer="fji_polbnda_adm0_country").set_crs(
    "EPSG:3832"
)

# calculate distance from Fiji
gdf_tracks["Distance (km)"] = (
    gdf_tracks.to_crs(epsg=3832).apply(
        lambda point: point.geometry.distance(gdf_adm0.geometry),
        axis=1,
    )
    / 1000
)

df_agg = (
    gdf_tracks.groupby(["Cyclone Name", "Season"])
    .agg(
        {
            "Category numeric": "max",
            "Pressure": "min",
            "Wind (Knots)": "max",
            "Distance (km)": "min",
        }
    )
    .reset_index()
)

print(df_agg)
```

```python
# Plot tracks and CODAB

gdf_adm0_4326 = gdf_adm0.to_crs("EPSG:4326")
fig = px.choropleth(
    gdf_adm0_4326,
    geojson=gdf_adm0_4326.geometry,
    locations=gdf_adm0_4326.index,
)
fig.update_layout(
    template="simple_white",
    geo=dict(
        lataxis=dict(range=[-25, -9]),
        lonaxis=dict(range=[164, -166]),
        visible=False,
    ),
)

PLOT_NAMES = ["WINSTON", "HAROLD 2020"]

dff = gdf_tracks[gdf_tracks["Cyclone Name"].isin(PLOT_NAMES)]

for name in dff["Cyclone Name"].unique():
    dfff = dff[dff["Cyclone Name"] == name]
    fig.add_trace(
        go.Scattergeo(
            lat=dfff["Latitude"],
            lon=dfff["Longitude"],
            name=name,
            customdata=dfff[
                ["Category", "Wind (Knots)", "datetime", "Distance (km)"]
            ],
            marker_size=dfff["Wind (Knots)"].fillna(0) / 5,
            mode="lines+markers",
            line_width=0.5,
            hovertemplate=(
                "Category: %{customdata[0]}<br>"
                "Wind speed: %{customdata[1]} knots<br>"
                "Datetime: %{customdata[2]}<br>"
                "Distance: %{customdata[3]:,.0f} km"
            ),
        )
    )

pyo.iplot(fig)
```

```python
# Calculate recurrences

distances = [0, 50, 100, 200, 300, 400, 500]

df_recur = pd.DataFrame()

for distance in distances:
    dff_agg = df_agg[df_agg["Distance (km)"] <= distance]
    df_count = (
        dff_agg.groupby("Category numeric")
        .size()
        .reset_index()
        .rename(columns={0: "Total count"})
    )
    df_count["Distance cutoff (km)"] = distance
    df_recur = pd.concat([df_recur, df_count], ignore_index=True)

df_recur["Recurrence"] = len(seasons) / df_recur["Total count"]
df_recur = df_recur.rename(columns={"Category numeric": "Category"})
df_recur["Category"] = df_recur["Category"].astype(int)

df_recur = df_recur.pivot(
    index="Category",
    columns="Distance cutoff (km)",
    values="Recurrence",
)
df_recur = df_recur.sort_values("Category", ascending=False)
df_recur.columns = df_recur.columns.astype(str)
df_recur.index = df_recur.index.astype(str)
df_recur = df_recur.round(1)

print(df_recur)

fig = px.imshow(df_recur, text_auto=True, range_color=[1, 5])
fig.update_layout(coloraxis_colorbar_title="Recurrence (years)")
fig.update_xaxes(side="top", title_text="Minimum distance to Fiji (km)")

pyo.iplot(fig)
```

```python
# Plot recurrence by Category

df_cat_count = (
    df_max.groupby("Category numeric")
    .size()
    .reset_index()
    .rename(columns={0: "Total count"})
)
df_cat_count["Count per year"] = df_cat_count["Total count"] / len(seasons)
df_cat_count["Recurrence (years)"] = 1 / df_cat_count["Count per year"]
print(df_cat_count)

fig = go.Figure()
fig.update_layout(
    template="simple_white", title_text="Average count by Category"
)
fig.update_xaxes(title_text="Category")
fig.update_yaxes(title_text="Average count per year")
fig.add_trace(
    go.Bar(
        x=df_cat_count["Category numeric"], y=df_cat_count["Count per year"]
    )
)
pyo.iplot(fig)
```

```python
# Plot recurrence by max wind speed

bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
df_speed_count = (
    df_max.groupby(pd.cut(df_max["Wind (Knots)"] - 0.01, bins))
    .size()
    .reset_index()
    .rename(columns={0: "Total count"})
)

df_speed_count["Count per year"] = df_speed_count["Total count"] / len(seasons)
df_speed_count["Recurrence (years)"] = 1 / df_speed_count["Count per year"]
print(df_speed_count)

fig = go.Figure()
fig.update_layout(
    template="simple_white", title_text="Total count by max wind speed"
)
fig.update_xaxes(title_text="Max wind speed (knots)")
fig.update_yaxes(title_text="Total count")
fig.add_trace(go.Histogram(x=df_max["Wind (Knots)"] - 0.01, nbinsx=7))
pyo.iplot(fig)
```

```python
# Plot recurrence by min pressure

print([df_max["Pressure"].max(), df_max["Pressure"].min()])
start = 980
stop = 1015
step = 5
bins = np.arange(start, stop, step)
print(bins)
df_speed_count = (
    df_max.groupby(pd.cut(df_max["Pressure"], bins, include_lowest=True))
    .size()
    .reset_index()
    .rename(columns={0: "Total count"})
)

df_speed_count["Count per year"] = df_speed_count["Total count"] / len(seasons)
df_speed_count["Recurrence (years)"] = 1 / df_speed_count["Count per year"]
print(df_speed_count)

fig = go.Figure()
fig.update_layout(
    template="simple_white", title_text="Total count by min pressure"
)
fig.update_xaxes(title_text="Min pressure")
fig.update_yaxes(title_text="Total count")
fig.add_trace(
    go.Histogram(
        x=df_max["Pressure"], xbins=dict(start=start, size=step, end=stop)
    )
)
pyo.iplot(fig)
```

```python

```
