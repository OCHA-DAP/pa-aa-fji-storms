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

pyo.init_notebook_mode()
```

```python
load_dotenv()
EXP_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/exploration/fji"
CYCLONETRACKS_PATH = (
    EXP_DIR / "rsmc/RSMC TC Tracks Historical 1969_70 to 2022_23 Seasons.csv"
)
RAW_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/raw/fji"
CODAB_PATH = (
    RAW_DIR / "cod_ab/fji_polbnda_adm0_country/fji_polbnda_adm0_country.shp"
)
```

```python
# Load and process cyclone tracks

df = pd.read_csv(CYCLONETRACKS_PATH)
df["Date"] = df["Date"].apply(lambda x: x.strip())
df["datetime"] = df["Date"] + " " + df["Time"]
df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %HZ")
df = df.drop(["Date", "Time"], axis=1)
print(df["Longitude"].max())
print(df)
cyclone_names = df["Cyclone Name"].unique()
seasons = df["Season"].unique()
print(len(seasons))
df["Category numeric"] = pd.to_numeric(df["Category"], errors="coerce")

gdf_tracks = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"])
)
gdf_tracks.crs = "EPSG:4326"

numeric_columns = ["Category numeric", "Pressure", "Wind (Knots)"]
df_max = (
    df.groupby(["Cyclone Name", "Season"])[numeric_columns].max().reset_index()
)
print(df_max)
print(df_max.groupby("Category numeric").size())
```

```python
# Load and process CODAB

gdf_admin0 = gpd.read_file(CODAB_PATH)
gdf_admin0 = gdf_admin0.to_crs(epsg=4326)
```

```python
# Calculate distance from each point to Admin0 (slow)

gdf_admin0_3857 = gdf_admin0.to_crs(3857)
gdf_tracks["distance"] = gdf_tracks.to_crs(epsg=3857).apply(
    lambda point: point.geometry.distance(gdf_admin0_3857.geometry),
    axis=1,
)
gdf_admin0_3857.plot()
```

```python
# Plot CODAB

fig = px.choropleth(
    gdf_plot,
    geojson=gdf_plot.geometry,
    locations=gdf_plot.index,
)
fig.update_geos(fitbounds="locations", visible=False)
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
# Plot all cyclone tracks

fig = px.choropleth(
    gdf_admin0, geojson=gdf_admin0.geometry, locations=gdf_admin0.index
)
fig.update_layout(
    template="simple_white",
    geo=dict(
        lataxis=dict(range=[-25, -9]),
        lonaxis=dict(range=[164, -166]),
        visible=False,
    ),
)


for name in df_max.loc[df_max["Category numeric"] > 3.0]["Cyclone Name"]:
    gdff = gdf_tracks.loc[gdf_tracks["Cyclone Name"] == name]
    fig.add_trace(
        go.Scattergeo(
            lat=gdff["Latitude"],
            lon=gdff["Longitude"],
            name=name,
            customdata=gdff[
                ["Category", "Wind (Knots)", "datetime", "distance"]
            ],
            marker_size=gdff["distance"].fillna(0) / 10e4,
            mode="lines+markers",
            line_width=0.5,
            hovertemplate=(
                "Category: %{customdata[0]}<br>"
                "Wind (Knots): %{customdata[1]}<br>"
                "Datetime: %{customdata[2]}<br>"
                "Distance: %{customdata[3]}"
            ),
        )
    )

# fig.add_trace(go.Choroplethmapbox(geojson=shp.geometry, locations=shp.index))

pyo.iplot(fig)
```

```python

```
