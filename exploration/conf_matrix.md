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

# Confusion matrix

Produce confusion matrixes based on trigger thresholds

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from dotenv import load_dotenv
import shapely
from shapely.geometry import Point, LineString

import utils
```

```python
load_dotenv()

MB_TOKEN = os.environ["MAPBOX_TOKEN"]
```

```python
# if needed, process best tracks
# utils.process_ecmwf_besttrack_hindcasts()
```

```python
ecmwf = utils.load_ecmwf_besttrack_hindcasts()
fms = utils.load_cyclonetracks()
fmsnameyears = fms["nameyear"].unique()
ecmwf = ecmwf[ecmwf["nameyear"].isin(fmsnameyears)]
ecmwf["time"] = pd.to_datetime(ecmwf["time"])
ecmwf["forecast_time"] = pd.to_datetime(ecmwf["forecast_time"])
ecnameyears = ecmwf["nameyear"].unique()
fms = fms[fms["nameyear"].isin(ecnameyears)]
fms = fms.merge(
    ecmwf_a[["time", "speed_knots"]], left_on="datetime", right_on="time"
)
zone = utils.load_buffer()
hindcasts = utils.load_hindcasts()
hindcasts = hindcasts.merge(
    fms[["Name Season", "nameyear"]], on="Name Season", how="left"
)
```

```python
ecmwf["nameyear"].unique()
```

```python
px.scatter(
    fms[fms["speed_knots"] > 0],
    y="Wind (Knots)",
    x="speed_knots",
    trendline="ols",
)
```

```python
# interpolate ecmwf wind speed to fms wind speed
# based on trendline above
ecmwf["fms_speed"] = ecmwf["speed_knots"] * 0.940729 + 14.9982
ecmwf["fms_cat"] = ecmwf["fms_speed"].apply(utils.knots2cat)
```

```python
# interpolate ECMWF forecasts

dfs = []
for nameyear in tqdm(ecmwf["nameyear"].unique()):
    dff = ecmwf[ecmwf["nameyear"] == nameyear]
    for forecast_time in dff["forecast_time"].unique():
        dfff = dff[dff["forecast_time"] == forecast_time]
        cols = [
            "speed_knots",
            "lat",
            "lon",
            "category_numeric",
            "lead_time",
            "fms_speed",
            "fms_cat",
        ]
        dfff = dfff.groupby("time").first()[cols]
        dfff = dfff.resample("H").interpolate().reset_index()
        dfff["nameyear"] = nameyear
        dfff["forecast_time"] = forecast_time
        dfs.append(dfff)

ecmwf_i = pd.concat(dfs, ignore_index=True)
```

```python
ecmwf_i
```

```python
ecmwf_i = gpd.GeoDataFrame(
    data=ecmwf_i,
    geometry=gpd.points_from_xy(ecmwf_i["lon"], ecmwf_i["lat"], crs=4326),
)
```

```python
ecmwf_i[ecmwf_i["nameyear"] == "winston2016"].plot()
```

```python
ecmwf_i["distance"] = (
    ecmwf_i.to_crs(3832).geometry.distance(adm0.iloc[0].geometry) / 1000
)
```

```python
close_distance = 0
far_distance = 250

thresholds = [
    {"name": "close", "distance": close_distance, "category": 3},
    {"name": "far", "distance": far_distance, "category": 4},
]

total_years = len(ecmwf_i["time"].dt.year.unique()) - 1

ecmwf_f = pd.DataFrame()

for threshold in thresholds:
    df_add = ecmwf_i[
        (ecmwf_i["distance"] <= threshold.get("distance"))
        & (ecmwf_i["fms_cat"] >= threshold.get("category"))
    ]
    ecmwf_f = pd.concat([ecmwf_f, df_add])
emcwf_f = ecmwf_f.drop_duplicates()
```

```python
ecmwf_f["nameyear"].unique()
```

```python
threshold_name = f"d{close_distance}c3_d{250}c4"
actual_triggers = pd.read_csv(
    utils.EXP_DIR / f"fms_triggers_{threshold_name}.csv"
)
actual_triggers["min_distance_date"] = pd.to_datetime(
    actual_triggers["min_distance_date"]
)
```

```python
actual_triggers.sort_values("min_distance_date", ascending=False)
```

```python
leadtimes = pd.DataFrame()

for max_leadtime in [2, 4]:
    print(max_leadtime + 1)
    dff = ecmwf_f[ecmwf_f["lead_time"] <= (max_leadtime + 1) * 24]
    if dff.empty:
        continue
    first_trigger = (
        dff.groupby("nameyear")["forecast_time"].min().reset_index()
    )

    first_trigger = first_trigger.merge(
        actual_triggers, on="nameyear", how="outer"
    )
    first_trigger["leadtime"] = (
        first_trigger["min_distance_date"] - first_trigger["forecast_time"]
    )
    display(first_trigger)
    mean_lt = first_trigger["leadtime"].mean()
    print(mean_lt.days + mean_lt.seconds / 3600 / 24)
```

```python
# calculate simple conf

fms_complete = utils.load_cyclonetracks()
conf = fms_complete.groupby("nameyear")[["Season", "Name Season"]].first()
display(conf)
conf = conf.join(ecmwf.groupby("nameyear")["name"].first(), how="left")
conf["in_ec"] = conf["name"].apply(lambda x: isinstance(x, str))
conf = conf.drop(columns="name")
display(conf)
for max_leadtime in [2, 4]:
    print(max_leadtime + 1)
    dff = ecmwf_f[ecmwf_f["lead_time"] <= (max_leadtime + 1) * 24]
    conf = conf.join(dff.groupby("nameyear")["forecast_time"].min())
    conf = conf.rename(
        columns={"forecast_time": f"ec_{max_leadtime + 1}day_date"}
    )
    conf[f"ec_{max_leadtime + 1}day_trig"] = conf[
        f"ec_{max_leadtime + 1}day_date"
    ].apply(lambda x: not pd.isna(x))

threshold_name = f"d{0}c3_d{250}c4"
actual_triggers = pd.read_csv(
    utils.EXP_DIR / f"fms_triggers_{threshold_name}.csv"
)
actual_triggers["min_distance_date"] = pd.to_datetime(
    actual_triggers["min_distance_date"]
)
conf = conf.join(actual_triggers.set_index("nameyear")["min_distance_date"])
conf = conf.rename(columns={"min_distance_date": "fms_actual_date"})
conf["fms_actual_trig"] = conf["fms_actual_date"].apply(
    lambda x: not pd.isnull(x)
)
display(conf)
conf[(conf["ec_5day_trig"] | conf["fms_actual_trig"]) & conf["in_ec"]]
```

```python
fms
```

```python
# create plots for pptx and demo
import src.constants
import src.check_trigger

pio.renderers.default = "browser"
# pio.renderers.default = "notebook"
# utils.process_buffer(200)
trigger_zone = utils.load_buffer(250)
trigger_zone = trigger_zone.to_crs(src.constants.FJI_CRS)

nameyear_sel = [
    #     "yasa2020",
    #     "harold2020",
    #     "winston2016",
    #     "keni2018",
    #     "pola2019"
    "evan2012",
]

for nameyear in nameyear_sel:
    df_plot = ecmwf[ecmwf["nameyear"] == nameyear].sort_values("forecast_time")
    actual_f = fms[fms["nameyear"] == nameyear]
    hindcasts_f = hindcasts[hindcasts["nameyear"] == nameyear].sort_values(
        "base_time"
    )

    # plot trigger zone
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

    # plot actual (FMS)
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

    fms_historical = False
    # plot FMS historical forecasts
    for base_time in hindcasts_f["base_time"].unique():
        fms_historical = True
        dff = hindcasts_f[hindcasts_f["base_time"] == base_time]
        dff = dff.sort_values("forecast_time")
        fig.add_trace(
            go.Scattermapbox(
                lon=dff["Longitude"],
                lat=dff["Latitude"],
                mode="lines",
                line=dict(width=3),
                marker=dict(size=3),
                name=base_time.strftime("%b %d, %H:%M"),
                customdata=dff[["Category", "forecast_time"]],
                hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
            )
        )

    line = dict(width=3, color="grey") if fms_historical else dict(width=3)

    # plot ECMWF
    for base_time in df_plot["forecast_time"].unique():
        dff = df_plot[df_plot["forecast_time"] == base_time]
        dff = dff.sort_values("time")
        fig.add_trace(
            go.Scattermapbox(
                lon=dff["lon"],
                lat=dff["lat"],
                mode="lines",
                line=line,
                showlegend=True,
                name=base_time.strftime("EC %b %d, %H:%M"),
                customdata=dff[["fms_cat", "time"]],
                hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
            )
        )

    name_season = actual_f["Name Season"].iloc[0]

    fig.update_layout(
        mapbox_style="mapbox://styles/tristandownin/clk2sq75x01lc01pc9xxn7ov9",
        mapbox_accesstoken=MB_TOKEN,
        mapbox_zoom=4.5,
        mapbox_center_lat=-17,
        mapbox_center_lon=179,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title=f"{name_season}<br><sup>FMS 72hr forecasts with ECMWF 120hr forecasts</sup>",
    )

    fig.show()
```

```python
# create plots for simulation (with buffer)
import src.constants
import src.check_trigger

pio.renderers.default = "browser"
# pio.renderers.default = "notebook"
# utils.process_buffer(200)
trigger_zone = utils.load_buffer(250)
trigger_zone = trigger_zone.to_crs(src.constants.FJI_CRS)
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
    ).to_crs(src.constants.FJI_CRS)
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
fm_f["cat_str"] = fm_f["Category"].apply(lambda x: str(x).split(".")[0])
ac_f = fms[fms["nameyear"] == nameyear].copy()
ac_f["simple_date"] = ac_f["datetime"].apply(
    lambda x: x.strftime("%b %d, %H:%M")
)

name_season = ac_f["Name Season"].iloc[0]

x, y = trigger_zone.geometry[0].boundary.xy
# fig = px.line_mapbox(lat=y, lon=x)

fig = go.Figure()
fig.add_trace(
    go.Scattermapbox(
        lat=np.array(y),
        lon=np.array(x),
        mode="lines",
        name="Area within 250km of Fiji",
        line_width=1,
        hoverinfo="skip",
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
    )
)

# FMS forecasts
for base_time in fm_f["base_time"].unique():
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
            name=date_str,
            customdata=dff[["Category", "forecast_time"]],
            hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
            legendgroup=date_str,
            legendgrouptitle_text="",
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
        )
    )

# EC forecasts
for base_time in ec_f["forecast_time"].unique():
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
            line=dict(width=2, color="grey"),
            marker=dict(size=5),
            name=f"EC {date_str}",
            customdata=dff[["fms_cat", "forecast_time"]],
            hovertemplate="Category: %{customdata[0]}<br>Datetime: %{customdata[1]}",
            legendgroup=f"EC {date_str}",
            legendgrouptitle_text="",
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
            colorscale="Greys_r",
            legendgroup=f"EC {date_str}",
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
    mapbox_style="mapbox://styles/tristandownin/clk2sq75x01lc01pc9xxn7ov9",
    #     mapbox_style="white-bg",
    mapbox_accesstoken=MB_TOKEN,
    mapbox_zoom=4.5,
    mapbox_center_lat=-17,
    mapbox_center_lon=179,
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    title=f'Cyclone "Daniel" 2023',
)

fig.show()
```

```python
fm_f.columns
```

```python
type(np.array(x))
```

```python
for distance, buffer in buffers.set_index("distance").iterrows():
    print(distance)
    display(buffer.geometry)
    buffer.geometry
```
