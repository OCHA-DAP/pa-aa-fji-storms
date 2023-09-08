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

# Historical triggers

Calculating the leadtime of triggers by comparing:

1. the time they would have triggered from the forecasts
\(either ECMWF or FMS)
2. the time they passed closest to Fiji

Also used to determine "false alarms" where trigger would have gone off
but TC's actual track did not meet trigger criteria.

## Data sources

- Forecasts
  - Fiji Met Service \(FMS) historical official forecasts \(72hrs)
    - currently only available for Yasa, Harold, and Evan
  - ECMWF historical forecasts
    \(available [here](https://rda.ucar.edu/datasets/ds330.3/), 120hrs)
- Actual tracks: FMS

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm

from src import utils
```

## Load data

```python
# if needed, process best tracks
# utils.process_ecmwf_besttrack_hindcasts()
ecmwf = utils.load_ecmwf_besttrack_hindcasts()
fms = utils.load_cyclonetracks()
zone = utils.load_buffer()
adm0 = utils.load_codab(level=0)
# load FMS hindcasts (strictly speaking, historical forecasts not hindcasts)
hindcasts = utils.load_hindcasts()

# keep only EMCWF forecasts for which we have official FMS tracks
fmsnameyears = fms["nameyear"].unique()
ecmwf = ecmwf[ecmwf["nameyear"].isin(fmsnameyears)]
ecmwf["time"] = pd.to_datetime(ecmwf["time"])
ecmwf["forecast_time"] = pd.to_datetime(ecmwf["forecast_time"])
ecnameyears = ecmwf["nameyear"].unique()
fms = fms[fms["nameyear"].isin(ecnameyears)]

# calculate "actual" ECMWF points - with 0hrs leadtime
ecmwf_a = ecmwf[ecmwf["lead_time"] == 0]
fms = fms.merge(
    ecmwf_a[["time", "speed_knots"]], left_on="datetime", right_on="time"
)

hindcasts = hindcasts.merge(
    fms[["Name Season", "nameyear"]], on="Name Season", how="left"
).drop_duplicates()
```

## Process data

```python
# plot comparison between FMS official tracks and EMCWF leadtime=0
px.scatter(
    fms[fms["speed_knots"] > 0],
    y="Wind (Knots)",
    x="speed_knots",
    trendline="ols",
).show()
```

```python
# based on trendline above, change ECMWF speeds to FMS speeds
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

ecmwf_i = gpd.GeoDataFrame(
    data=ecmwf_i,
    geometry=gpd.points_from_xy(ecmwf_i["lon"], ecmwf_i["lat"], crs=4326),
)
```

```python
# calculate ECMWF distances

ecmwf_i["distance"] = (
    ecmwf_i.to_crs(3832).geometry.distance(adm0.iloc[0].geometry) / 1000
)
```

```python
# interpolate FMS forecasts

dfs = []
for nameyear in tqdm(hindcasts["nameyear"].unique()):
    dff = hindcasts[hindcasts["nameyear"] == nameyear]
    for base_time in dff["base_time"].unique():
        dfff = dff[dff["base_time"] == base_time]
        cols = [
            "Latitude",
            "Longitude",
            "Category",
            "leadtime",
        ]
        dfff = dfff.groupby("forecast_time").first()[cols]
        dfff = dfff.resample("H").interpolate().reset_index()
        dfff["nameyear"] = nameyear
        dfff["base_time"] = base_time
        dfs.append(dfff)

hindcasts_i = pd.concat(dfs, ignore_index=True)

hindcasts_i = gpd.GeoDataFrame(
    data=hindcasts_i,
    geometry=gpd.points_from_xy(
        hindcasts_i["Longitude"], hindcasts_i["Latitude"], crs=4326
    ),
)

hindcasts_i = hindcasts_i.rename(
    columns={"forecast_time": "time", "base_time": "forecast_time"}
)
```

```python
# calculate FMS forecast distances

hindcasts_i["distance"] = (
    hindcasts_i.to_crs(3832).geometry.distance(adm0.iloc[0].geometry) / 1000
)
```

## Compare forecast triggers with actual

### Combine into one DataFrame

Get triggers for forecasts and actual tracks

```python
# specify threshold
close_distance = 0
far_distance = 250

close_category = 3
far_category = 4

thresholds = [
    {"name": "close", "distance": close_distance, "category": close_category},
    {"name": "far", "distance": far_distance, "category": far_category},
]
threshold_name = (
    f"d{close_distance}c{close_category}_d{far_distance}c{far_category}"
)

# load FMS actual tracks as complete list of cyclones
fms_complete = utils.load_cyclonetracks()
conf = fms_complete.groupby("nameyear")[["Season", "Name Season"]].first()

# check which TCs are included in ecmwf hindcasts
conf = conf.join(ecmwf.groupby("nameyear")["name"].first(), how="left")
conf["in_ec"] = conf["name"].apply(lambda x: isinstance(x, str))
conf = conf.drop(columns="name")

# check which TCs are included in FMS historical forecasts
conf = conf.join(
    hindcasts.groupby("nameyear")["cyclone_name"].first(), how="left"
)
conf["in_fms_fcast"] = conf["cyclone_name"].apply(lambda x: isinstance(x, str))
conf = conf.drop(columns="cyclone_name")

# get ECMWF triggers
ecmwf_f = pd.DataFrame()

for threshold in thresholds:
    df_add = ecmwf_i[
        (ecmwf_i["distance"] <= threshold.get("distance"))
        & (ecmwf_i["fms_cat"] >= threshold.get("category"))
    ]
    ecmwf_f = pd.concat([ecmwf_f, df_add])

ecmwf_f = ecmwf_f.drop_duplicates()
for max_leadtime in [2, 4]:
    dff = ecmwf_f[ecmwf_f["lead_time"] <= (max_leadtime + 1) * 24]
    conf = conf.join(dff.groupby("nameyear")["forecast_time"].min())
    conf = conf.rename(
        columns={"forecast_time": f"ec_{max_leadtime + 1}day_date"}
    )
    conf[f"ec_{max_leadtime + 1}day_trig"] = conf[
        f"ec_{max_leadtime + 1}day_date"
    ].apply(lambda x: not pd.isna(x))

# get FMS forecast triggers
hindcasts_f = pd.DataFrame()
for threshold in thresholds:
    df_add = hindcasts_i[
        (hindcasts_i["distance"] <= threshold.get("distance"))
        & (hindcasts_i["Category"] >= threshold.get("category"))
    ]
    hindcasts_f = pd.concat([hindcasts_f, df_add])
conf = conf.join(hindcasts_f.groupby("nameyear")["forecast_time"].min())
conf = conf.rename(columns={"forecast_time": "fms_fcast_date"})
conf[f"fms_fcast_trig"] = conf[f"fms_fcast_date"].apply(
    lambda x: not pd.isna(x)
)

# get actual triggers
threshold_name = (
    f"d{close_distance}c{close_category}_d{far_distance}c{far_category}"
)
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

conf.reset_index().to_csv(
    utils.PROC_PATH / "historical_triggers.csv", index=False
)
```

### ECMWF metrics

```python
# 5-day (i.e. readiness trigger)
dff = conf[conf["in_ec"]]
conf_ec5 = pd.crosstab(dff["ec_5day_trig"], dff["fms_actual_trig"])
display(conf_ec5)
TP = conf_ec5.loc[True, True]
TN = conf_ec5.loc[False, False]
FP = conf_ec5.loc[True, False]
FN = conf_ec5.loc[False, True]
print(f"Detection rate = {TP / (TP + FN)}")
print(f"False alarm rate = {FP / (TP + FN)}")
dfff = dff[dff["ec_5day_trig"] & dff["fms_actual_trig"]].copy()
dfff["leadtime"] = dfff["fms_actual_date"] - dfff["ec_5day_date"]
mean_lt = dfff["leadtime"].mean()
mean_lt_days = mean_lt.days + mean_lt.seconds / 3600 / 24
print(
    f"Mean leadtime = {mean_lt_days:.2f} days ({mean_lt_days * 24:.0f} hours)"
)
print("Triggered cyclones:")
display(dff[dff["ec_5day_trig"]])
print("Missed cyclones:")
display(dff[~dff["ec_5day_trig"] & dff["fms_actual_trig"]])
print("False alarm cyclones:")
display(dff[dff["ec_5day_trig"] & ~dff["fms_actual_trig"]])
```

### FMS forecast metrics

```python
# activation trigger
dff = conf[conf["in_fms_fcast"]]
conf_ec5 = pd.crosstab(dff["fms_fcast_trig"], dff["fms_actual_trig"])
display(conf_ec5)
# since only contains true positives, detection rate is 1 and false alarm 0
dfff = dff[dff["fms_fcast_trig"] & dff["fms_actual_trig"]].copy()
dfff["leadtime"] = dfff["fms_actual_date"] - dfff["fms_fcast_date"]
mean_lt = dfff["leadtime"].mean()
mean_lt_days = mean_lt.days + mean_lt.seconds / 3600 / 24
print(
    f"Mean leadtime = {mean_lt_days:.2f} days ({mean_lt_days * 24:.0f} hours)"
)
print("Triggered cyclones:")
display(dff[dff["fms_fcast_trig"]])
```

```python

```
