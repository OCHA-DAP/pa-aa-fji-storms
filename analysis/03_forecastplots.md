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

# Forecast plots

Create plots with historical forecasts to show TC forecast evolution.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import json

import chart_studio
import chart_studio.plotly as py
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from shapely.geometry import LineString
import shapely

from src import utils, constants
```

## Load data

```python
ecmwf = utils.load_ecmwf_besttrack_hindcasts()
hindcasts = utils.load_hindcasts()
fms = utils.load_cyclonetracks()
hindcasts = hindcasts.merge(fms[["nameyear", "Name Season"]], on="Name Season")
trigger_zone = utils.load_buffer(250)

trigger_zone = trigger_zone.to_crs(constants.FJI_CRS)
cod1 = utils.load_codab(level=1)
cod2 = utils.load_codab(level=2)
cod3 = utils.load_codab(level=3)
```

```python
housing = utils.load_housing_impact()
```

```python
triggers = utils.load_historical_triggers()
triggers = triggers.set_index("nameyear")
```

```python
housing["nameyear"].unique()
```

## Process data

```python
ecmwf["forecast_time"] = pd.to_datetime(ecmwf["forecast_time"])
ecmwf["fms_speed"] = ecmwf["speed_knots"] * 0.940729 + 14.9982
ecmwf["fms_cat"] = ecmwf["fms_speed"].apply(utils.knots2cat)
```

## Plot forecasts

```python
distances = [50, 100, 200]


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
    ).to_crs(constants.FJI_CRS)
    buffers = buffers.rename(columns={0: "distance"})
    return buffers


for nameyear in housing["nameyear"].unique():
    trigger = triggers.loc[nameyear]

    ec_f = (
        ecmwf[ecmwf["nameyear"] == nameyear]
        .sort_values("forecast_time")
        .copy()
    )
    fm_f = (
        hindcasts[hindcasts["nameyear"] == nameyear]
        .sort_values("base_time")
        .copy()
    )
    fm_f["cat_str"] = fm_f["Category"].apply(lambda x: str(x).split(".")[0])
    ac_f = fms[fms["nameyear"] == nameyear].copy()
    ac_f["simple_date"] = ac_f["datetime"].apply(
        lambda x: x.strftime("%b %d, %H:%M")
    )

    name_season = ac_f["Name Season"].iloc[0]

    x, y = trigger_zone.geometry[0].boundary.xy
    # fig = px.line_mapbox(lat=y, lon=x)

    dff = housing[housing["nameyear"] == nameyear]
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
    dff = dff.to_crs(constants.FJI_CRS)
    dff = dff.set_index(f"ADM{admn}_PCODE")

    # plot housing impact
    fig = px.choropleth_mapbox(
        dff,
        geojson=dff.geometry,
        locations=dff.index,
        hover_name=f"ADM{admn}_NAME_x",
        color="Major Damage",
        hover_data=["Destroyed", "Major Damage"],
        color_continuous_scale="Blues",
        opacity=0.7,
    )
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker_line_width=0.5)
    fig.update_geos(fitbounds="locations", visible=False)

    # plot buffer zone
    fig.add_trace(
        go.Scattermapbox(
            lat=np.array(y),
            lon=np.array(x),
            mode="lines",
            name="Area within 250km of Fiji",
            line=dict(width=1, color="dodgerblue"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # plot actual (FMS)
    fig.add_trace(
        go.Scattermapbox(
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
            visible="legendonly",
        )
    )

    # plot actual buffer
    buffers = gdf_buffers(ac_f, distances)
    for distance in distances:
        buffer = buffers[buffers["distance"] == distance]
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=json.loads(buffers.geometry.to_json()),
            locations=buffers.index,
            z=buffers["distance"],
            marker_opacity=0.3,
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
            visible="legendonly",
        )
    )

    # FMS forecasts
    for base_time in fm_f["base_time"].unique():
        if base_time == trigger["fms_fcast_date"]:
            act = "A: "
        else:
            act = ""
        date_str = base_time.strftime("%b %d, %H:%M")
        fms_historical = True
        dff = fm_f[fm_f["base_time"] == base_time]
        dff = dff.sort_values("forecast_time")
        fig.add_trace(
            go.Scattermapbox(
                lon=dff["Longitude"],
                lat=dff["Latitude"],
                mode="text+lines",
                text=dff["cat_str"],
                textfont=dict(size=20, color="black"),
                line=dict(width=2),
                marker=dict(size=10),
                name=act + date_str,
                customdata=dff[["Category", "forecast_time"]],
                hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
                legendgroup=date_str,
                legendgrouptitle_text="",
                visible="legendonly",
            )
        )

        buffers = gdf_buffers(dff, distances)
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=json.loads(buffers.geometry.to_json()),
                locations=buffers.index,
                z=buffers["distance"],
                marker_opacity=0.3,
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
                visible="legendonly",
            )
        )

    # EC forecasts
    if fm_f.empty:
        line = dict(width=2)
        colorscale = "YlOrRd_r"
    else:
        line = dict(width=2, color="grey")
        colorscale = "Greys_r"
    for base_time in ec_f["forecast_time"].unique():
        if base_time == trigger["ec_5day_date"]:
            red = "R: "
        else:
            red = ""
        if base_time == trigger["ec_3day_date"] and fm_f.empty:
            act = "A: "
        else:
            act = ""
        date_str = base_time.strftime("%b %d, %H:%M")
        dff = ec_f[ec_f["forecast_time"] == base_time]
        dff = dff.sort_values("time")
        fig.add_trace(
            go.Scattermapbox(
                lon=dff["lon"],
                lat=dff["lat"],
                mode="text+lines",
                text=dff["fms_cat"].astype(str),
                textfont=dict(size=20, color="black"),
                line=line,
                marker=dict(size=5),
                name=f"{red}{act} EC {date_str}",
                customdata=dff[["fms_cat", "time"]],
                hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
                legendgroup=f"EC {date_str}",
                legendgrouptitle_text="",
                visible="legendonly",
            )
        )

        buffers = gdf_buffers(dff, distances)
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=json.loads(buffers.geometry.to_json()),
                locations=buffers.index,
                z=buffers["distance"],
                marker_opacity=0.3,
                marker_line_width=0,
                colorscale=colorscale,
                legendgroup=f"EC {date_str}",
                name="50/100/200km buffer",
                showlegend=False,
                showscale=False,
                zmin=50,
                zmid=200,
                zmax=250,
                hoverinfo="skip",
                visible="legendonly",
            )
        )

    if fm_f.empty:
        subtitle = (
            "ECMWF 120hr forecasts in colour; R: readiness, A: activation"
        )
    else:
        subtitle = (
            "Fiji Met Services official 72hr forecasts in colour, "
            "ECMWF 120hr forecasts in grey; R: readiness, A: activation"
        )
    fig.update_layout(
        mapbox_style="basic",
        mapbox_accesstoken=utils.MB_TOKEN,
        mapbox_zoom=5.5,
        mapbox_center_lat=-17,
        mapbox_center_lon=179,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title=f"{name_season}<br>" f"<sup>{subtitle}</sup>",
    )

    fig.show()

    # save as html
    filename = f"{nameyear}_forecasts.html"

    f = open(utils.MAPS_DIR / filename, "w")
    f.close()
    with open(utils.MAPS_DIR / filename, "a") as f:
        f.write(
            fig.to_html(
                full_html=True, include_plotlyjs="cdn", auto_play=False
            )
        )
    f.close()
```

```python

```
