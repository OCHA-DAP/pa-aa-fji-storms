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

# Impact data

Process sub-national impact data from NDMO, and plots it with cyclone tracks.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
import json

from dotenv import load_dotenv
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from shapely import LineString
import shapely

from src import utils
```

```python
load_dotenv()
```

```python
utils.download_codab(clobber=True)
```

```python
# if needed, process impact data
utils.process_housing_impact()
```

```python
# load impact data
cod1 = utils.load_codab(level=1)
cod2 = utils.load_codab(level=2)
cod3 = utils.load_codab(level=3)
impact = utils.load_geo_impact()
des = utils.load_desinventar()
housing = utils.load_housing_impact()
```

```python
housing
```

```python
# load tracks
ecmwf = utils.load_ecmwf_besttrack_hindcasts()
fms = utils.load_cyclonetracks()
fmsnameyears = fms["nameyear"].unique()
ecmwf = ecmwf[ecmwf["nameyear"].isin(fmsnameyears)]
ecmwf["time"] = pd.to_datetime(ecmwf["time"])
ecmwf["forecast_time"] = pd.to_datetime(ecmwf["forecast_time"])
ecnameyears = ecmwf["nameyear"].unique()
fms = fms[fms["nameyear"].isin(ecnameyears)]
fms = fms.merge(
    ecmwf[["time", "speed_knots"]], left_on="datetime", right_on="time"
)
zone = utils.load_buffer()
hindcasts = utils.load_hindcasts()
hindcasts = hindcasts.merge(
    fms[["Name Season", "nameyear"]], on="Name Season", how="left"
)
```

```python
# process geoimpact
dff = impact.groupby(["Event", utils.ADM3]).size().reset_index()
dff = dff.rename(columns={0: "count"})

# plot geo impact by event
event = "Winston 2015/2016"
gdf = cod3.merge(dff[dff["Event"] == event], on=utils.ADM3, how="left")
gdf["count"] = gdf["count"].fillna(0)
gdf.plot(column="count")
```

```python
impact["Event"].value_counts()
```

```python
housing["nameseason"].value_counts()
```

```python
utils.process_fms_cyclonetracks()
```

```python
# housing plot with forecasts and bands

# nameseason = "Tino 2019/2020"
nameseason = "Winston 2015/2016"
nameseason = "Yasa 2020/2021"
# nameseason = "Evan 2012/2013"
dff = housing[housing["nameseason"] == nameseason]
if dff["ADM2_PCODE"].isnull().values.any():
    codn = cod1.copy()
    admn = 1
else:
    codn = cod2.copy()
    admn = 2
dff = codn.merge(dff, on=f"ADM{admn}_PCODE", how="left")
cols = ["Destroyed", "Major Damage"]
dff[cols] = dff[cols].fillna(0)
dff.geometry = dff.geometry.simplify(100)
dff = dff.to_crs(utils.FJI_CRS)
dff = dff.set_index(f"ADM{admn}_PCODE")
fig = px.choropleth(
    dff,
    geojson=dff.geometry,
    locations=dff.index,
    hover_name=f"ADM{admn}_NAME_x",
    color="Major Damage",
    hover_data=["Destroyed", "Major Damage"],
    color_continuous_scale="Blues",
)
fig.update_coloraxes(showscale=False)
fig.update_traces(marker_line_width=0.5)
fig.update_geos(fitbounds="locations", visible=False)

fig.show()

trigger_zone = utils.load_buffer(250)
trigger_zone = trigger_zone.to_crs(utils.FJI_CRS)
distances = [50, 100, 200]
colors = ["Reds", "Oranges", ""]


def gdf_buffers(gdf, distances):
    ls = LineString(gdf.geometry.to_crs(3832))
    polys = []
    distances = [50, 100, 200]
    prev_poly = None
    for d in distances:
        poly = ls.buffer(d * 1000)
        for prev_poly in polys:
            poly = shapely.difference(poly, prev_poly)
        polys.append(poly)

    buffers = gpd.GeoDataFrame(
        data=distances, geometry=polys, crs=3832
    ).to_crs(utils.FJI_CRS)
    buffers = buffers.rename(columns={0: "distance"})
    return buffers


nameyear_sel = [
    #     "yasa2020",
    #     "harold2020",
    #     "winston2016",
    "evan2012",
]

# just produce one at a time

nameyear = nameyear_sel[0]

ec_f = ecmwf[ecmwf["nameyear"] == nameyear].sort_values("forecast_time").copy()
fm_f = (
    hindcasts[hindcasts["nameyear"] == nameyear]
    .sort_values("base_time")
    .copy()
)
ac_f = fms[fms["nameyear"] == nameyear].copy()
ac_f["simple_date"] = ac_f["datetime"].apply(
    lambda x: x.strftime("%b %d, %H:%M")
)

name_season = ac_f["Name Season"].iloc[0]

# x, y = trigger_zone.geometry[0].boundary.xy
# fig = px.line_mapbox(lat=y, lon=x, mapbox_style="open-street-map")
# fig.update_traces(name="Area within 250km of Fiji", line_width=1)

# plot actual (FMS)
fig.add_trace(
    go.Scattergeo(
        lon=ac_f["Longitude"],
        lat=ac_f["Latitude"],
        mode="lines+text",
        text=ac_f["Category"],
        textfont=dict(size=20, color="black"),
        line=dict(color="black", width=2),
        marker=dict(size=5),
        name="Actual",
        customdata=ac_f[["Category numeric", "simple_date"]],
        hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
        legendgroup="actual",
        legendgrouptitle_text="",
    )
)

# plot actual buffer
buffers = gdf_buffers(ac_f, distances)
for distance in distances:
    buffer = buffers[buffers["distance"] == distance]
fig.add_trace(
    go.Choropleth(
        geojson=json.loads(buffers.geometry.to_json()),
        locations=buffers.index,
        z=buffers["distance"],
        marker_opacity=0.2,
        marker_line_width=0,
        colorscale="YlOrRd_r",
        name="50/100/200km buffer",
        legendgroup="actual",
        showlegend=False,
        showscale=False,
        zmin=50,
        zmid=200,
        zmax=250,
        hoverinfo="skip",
    )
)

# FMS forecasts
for base_time in fm_f["base_time"].unique():
    date_str = base_time.strftime("%b %d, %H:%M")
    fms_historical = True
    dff = fm_f[fm_f["base_time"] == base_time]
    dff = dff.sort_values("forecast_time")
    fig.add_trace(
        go.Scattergeo(
            lon=dff["Longitude"],
            lat=dff["Latitude"],
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=5),
            name=date_str,
            customdata=dff[["Category", "forecast_time"]],
            hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
            legendgroup=date_str,
            legendgrouptitle_text="",
        )
    )

    buffers = gdf_buffers(dff, distances)
    fig.add_trace(
        go.Choropleth(
            geojson=json.loads(buffers.geometry.to_json()),
            locations=buffers.index,
            z=buffers["distance"],
            marker_opacity=0.2,
            marker_line_width=0,
            colorscale="YlOrRd_r",
            legendgroup=date_str,
            name="50/100/200km buffer",
            showlegend=False,
            showscale=False,
            zmin=50,
            zmid=200,
            zmax=250,
            hoverinfo="skip",
        )
    )

fig.update_layout(
    #     mapbox_style="mapbox://styles/tristandownin/clk2sq75x01lc01pc9xxn7ov9",
    #     mapbox_accesstoken=MB_TOKEN,
    #     mapbox_zoom=4.5,
    #     mapbox_center_lat=-17,
    #     mapbox_center_lon=179,
    #     margin={"r": 0, "t": 50, "l": 0, "b": 0},
    title=f"{name_season}<br><sup>FMS 72hr forecasts with ECMWF 120hr forecasts</sup>",
)

fig.show(renderer="browser")
```

```python
housing
```
