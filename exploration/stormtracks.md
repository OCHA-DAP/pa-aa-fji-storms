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
df["Cyclone Name"] = df["Cyclone Name"].apply(
    lambda name: "".join([x for x in name if not x.isdigit()]).strip()
)
df["Name Season"] = df["Cyclone Name"] + " " + df["Season"]
cyclone_names = df["Name Season"].unique()
seasons = df["Season"].unique()
df["Category numeric"] = pd.to_numeric(df["Category"], errors="coerce").fillna(
    0
)


cyclone_names = df["Name Season"].unique()

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
```

```python
df_agg = pd.DataFrame()

df_agg = (
    gdf_tracks.groupby("Name Season")
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

for name_season in df["Name Season"].unique():
    dff = gdf_tracks[gdf_tracks["Name Season"] == name_season]
    min_distance = dff["Distance (km)"].min()
    closest_datetime = dff[dff["Distance (km)"] == min_distance][
        "datetime"
    ].iloc[0]
    first_datetime = dff["datetime"].iloc[0]
    distance_leadtime = closest_datetime - first_datetime

    max_category = dff["Category numeric"].max()
    strongest_datetime = dff[dff["Category numeric"] == max_category][
        "datetime"
    ].iloc[0]
    strength_leadtime = strongest_datetime - first_datetime

    df_agg.loc[
        df_agg["Name Season"] == name_season,
        ["Leadtime (distance)", "Leadtime (strength)"],
    ] = [distance_leadtime, strength_leadtime]

for l in ["strength", "distance"]:
    df_agg[f"Leadtime ({l})"] = df_agg[f"Leadtime ({l})"].apply(
        lambda x: x.days + x.seconds / 3600 / 24
    )

print(df_agg)
```

```python
# drop ridiculously long cyclone
df_agg = df_agg[~(df_agg["Name Season"] == "UNNAMED-SP 1971/1972")]
df_agg = df_agg[
    (df_agg["Leadtime (distance)"] > 0) & (df_agg["Leadtime (distance)"] > 0)
]

bins = range(10)
for cat in df_agg["Category numeric"].unique():
    fig, ax = plt.subplots()
    df_agg[df_agg["Category numeric"] == cat]["Leadtime (distance)"].hist(
        ax=ax, bins=bins
    )
    ax.set_title(f"Category: {cat}")
    ax.set_xlabel("Days to closest pass")

distances = range(100, 601, 100)

for dist in distances:
    fig, ax = plt.subplots()
    df_agg[df_agg["Distance (km)"] < dist]["Leadtime (distance)"].hist(
        ax=ax, bins=bins
    )
    ax.set_title(f"Minimum distance: {dist}")
    ax.set_xlabel("Days to closest pass")

fig, ax = plt.subplots()
df_agg[(df_agg["Distance (km)"] < 200) & (df_agg["Category numeric"] == 5)][
    "Leadtime (distance)"
].hist(ax=ax, bins=bins)
ax.set_title("Distance < 200 and Category 5")
ax.set_xlabel("Days to closest pass")

fig, ax = plt.subplots()
df_agg[(df_agg["Distance (km)"] < 300) & (df_agg["Category numeric"] >= 4)][
    "Leadtime (distance)"
].hist(ax=ax, bins=bins)
ax.set_title("Distance < 300 and Category 4")
ax.set_xlabel("Days to closest pass")

fig, ax = plt.subplots()
df_agg[(df_agg["Distance (km)"] < 300) & (df_agg["Category numeric"] >= 4)][
    "Leadtime (strength)"
].hist(ax=ax, bins=bins)
ax.set_title("Distance < 300 and Category 4")
ax.set_xlabel("Days to highest strength")
```

```python
df_agg = df_agg.sort_values(by=["Category numeric"], ascending=False)

fig = px.histogram(
    df_agg,
    x="Leadtime (distance)",
    color="Category numeric",
    color_discrete_sequence=px.colors.sequential.thermal_r,
)

pyo.iplot(fig)

fig = px.histogram(
    df_agg,
    x="Leadtime (strength)",
    color="Category numeric",
    color_discrete_sequence=px.colors.sequential.thermal_r,
)

pyo.iplot(fig)
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

PLOT_NAMES = ["WINSTON 2015/2016", "HAROLD 2019/2020", "YASA 2020/2021"]

dff = gdf_tracks[gdf_tracks["Name Season"].isin(PLOT_NAMES)]

for name in dff["Name Season"].unique():
    dfff = dff[dff["Name Season"] == name]
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

distances = np.round(np.arange(100, 500 + 0.01, 100), 0)

df_recur = pd.DataFrame()

range_color = [1, 5]

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

fig = px.imshow(df_recur, text_auto=True, range_color=range_color)
fig.update_layout(
    coloraxis_colorbar_title="Recurrence (years)",
    coloraxis_colorbar_tickvals=range_color,
)
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
