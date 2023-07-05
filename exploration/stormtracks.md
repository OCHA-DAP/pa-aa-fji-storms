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
import statsmodels.formula.api as smf
from patsy import ModelDesc, Term, LookupFactor

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
IMPACT_PATH = EXP_DIR / "rsmc/FIJI_ DesInventar data 20230626.xlsx"
```

```python
# Load and process impact data

df = pd.read_excel(IMPACT_PATH, skiprows=[0], index_col=None)
df = df.dropna(how="all")
df = df.reset_index(drop=True)
df_clean = df.copy()

# re-assemble file
# iterate over rows
last_valid_index = None
valid_indices = []
for index, row in df.loc[2:].iterrows():
    # check if row starts with serial number
    if isinstance(row.iloc[0], int):
        last_valid_index = index
        valid_indices.append(index)
        last_valid_col_i = np.argwhere(~row.isnull()).max()
        last_valid_col = df.columns[last_valid_col_i]
        if last_valid_col_i + 1 < len(df.columns):
            first_empty_col = df.columns[last_valid_col_i + 1]
    else:
        # if row after col A is empty, append to last valid cell of last valid row
        df_clean.loc[last_valid_index, last_valid_col] = df_clean.loc[
            last_valid_index, last_valid_col
        ] + str(row[0])
        # if row has other content, fill in rest of last valid row
        if not row[1:].dropna().empty:
            cells_available = len(
                df_clean.loc[last_valid_index, first_empty_col:]
            )
            df_clean.loc[last_valid_index, first_empty_col:] = row.values[
                1 : cells_available + 1
            ]

df_clean = df_clean.loc[valid_indices]
df_clean = df_clean[df_clean["Event"] == "TC - Tropical Cyclone"]
metrics = [
    "Deaths",
    "Injured",
    "Missing",
    "Houses Destroyed",
    "Houses Damaged",
    "Directly affected",
    "Indirectly Affected",
    "Relocated",
    "Evacuated",
    "Losses $USD",
    "Losses $Local",
    "Education centers",
    "Hospitals",
]
df_clean = df_clean.dropna(subset=metrics)
df_clean[metrics] = df_clean[metrics].astype(int)


def clean_date(raw_date):
    if isinstance(raw_date, str):
        date_list = [int(x) for x in raw_date.split("/")]
        if date_list[1] == 0:
            date_list[1] = 1
        if date_list[2] == 0:
            date_list[2] = 1
        return datetime(date_list[0], date_list[1], date_list[2])
    return raw_date


df_clean["datetime"] = df_clean["Date (YMD)"].apply(clean_date)


def datetime_to_season(date):
    eff_date = date - pd.Timedelta(days=200)
    return f"{eff_date.year}/{eff_date.year + 1}"


df_clean["Season"] = df_clean["datetime"].apply(datetime_to_season)


def clean_name(raw_name):
    return (
        raw_name.removesuffix("")
        .removeprefix("Tropical Cyclone")
        .removeprefix(" - ")
        .removesuffix("(possibly)")
        .strip()
    )


df_clean["Cyclone Name"] = df_clean["Description of Cause"].apply(clean_name)
df_clean["Name Season"] = df_clean["Cyclone Name"] + " " + df_clean["Season"]

# clean up errors
df_clean.loc[df_clean["Name Season"] == "Daman 2007/2008", "Deaths"] = 0

# keep only relevant columns
df_clean = df_clean[["Name Season"] + metrics]

df_clean.to_csv(EXP_DIR / "impact_data.csv", index=False)
```

```python
# metrics histograms

for metric in metrics:
    fig, ax = plt.subplots()
    df_clean[metric][df_clean[metric] > 0].hist(ax=ax, bins=20)
    ax.set_title(metric)
```

```python
# Load and process cyclone tracks

df = pd.read_csv(CYCLONETRACKS_PATH)
df["Date"] = df["Date"].apply(lambda x: x.strip())
df["datetime"] = df["Date"] + " " + df["Time"]
df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %HZ")
df = df.drop(["Date", "Time"], axis=1)
df["Cyclone Name"] = df["Cyclone Name"].apply(
    lambda name: "".join([x for x in name if not x.isdigit()])
    .strip()
    .capitalize()
)
df["Name Season"] = df["Cyclone Name"] + " " + df["Season"]
cyclone_names = df["Name Season"].unique()
seasons = df["Season"].unique()
df["Category numeric"] = pd.to_numeric(df["Category"], errors="coerce").fillna(
    0
)

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
# aggregate cyclone tracks and attach impact data

df_agg = pd.DataFrame()

df_agg = (
    gdf_tracks.groupby(["Name Season", "Season"])
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


def categorynumeric_to_category(cat_num):
    if cat_num == 0:
        return "L"
    else:
        return int(cat_num)


df_agg["Category"] = df_agg["Category numeric"].apply(
    categorynumeric_to_category
)

df_agg["Season numeric"] = df_agg["Season"].apply(
    lambda x: int(x.split("/")[0])
)

for name_season in df_agg["Name Season"].unique():
    dff = gdf_tracks[gdf_tracks["Name Season"] == name_season]
    min_distance = dff["Distance (km)"].min()

    # distance leadtime
    closest_datetime = dff[dff["Distance (km)"] == min_distance][
        "datetime"
    ].iloc[0]
    first_datetime = dff["datetime"].iloc[0]
    distance_leadtime = closest_datetime - first_datetime

    # strength/category leadtime
    max_category = dff["Category numeric"].max()
    strongest_datetime = dff[dff["Category numeric"] == max_category][
        "datetime"
    ].iloc[0]
    strength_leadtime = strongest_datetime - first_datetime

    # props at min distance
    landfall_row = dff[dff["datetime"] == closest_datetime].iloc[0]

    # props within distance
    distances = [100, 200, 300, 400, 500]
    for distance in distances:
        dfff = dff[dff["Distance (km)"] < distance]
        if dfff.empty:
            continue
        time_spent = dfff["datetime"].iloc[-1] - dfff["datetime"].iloc[0]
        time_spent = time_spent.days + time_spent.seconds / 3600 / 24
        df_agg.loc[
            df_agg["Name Season"] == name_season,
            [
                f"Max speed < {distance} km",
                f"Max cat. < {distance} km",
                f"Min pressure < {distance} km",
                f"Days spent < {distance} km",
            ],
        ] = [
            dfff["Wind (Knots)"].max(),
            dfff["Category numeric"].max(),
            dfff["Pressure"].min(),
            time_spent,
        ]

    df_agg.loc[
        df_agg["Name Season"] == name_season,
        [
            "Leadtime (distance)",
            "Leadtime (strength)",
            "Cat. at min distance",
            "Speed at min distance",
        ],
    ] = [
        distance_leadtime,
        strength_leadtime,
        landfall_row["Category numeric"],
        landfall_row["Wind (Knots)"],
    ]

for l in ["strength", "distance"]:
    df_agg[f"Leadtime ({l})"] = df_agg[f"Leadtime ({l})"].apply(
        lambda x: x.days + x.seconds / 3600 / 24
    )

# merge track and impact data
df_agg = pd.merge(df_agg, df_clean, on="Name Season", how="outer")
```

```python
# plot prop metric relationships

plot_x = ["Max speed < 100 km", "Season numeric"]
plot_y = ["Deaths", "Houses Destroyed", "Losses $Local"]
df_plot = df_agg[df_agg["Category numeric"] > 0]
df_plot = df_plot.sort_values(by=["Category numeric"], ascending=False)

for x in plot_x:
    for y in plot_y:
        fig = px.scatter(
            df_plot,
            x=x,
            y=y,
            hover_name="Name Season",
            trendline="ols",
            color="Category",
            trendline_color_override="grey",
            trendline_scope="overall",
            color_discrete_sequence=None,
        )

        fig.update_layout(template="simple_white")
        pyo.iplot(fig)
```

```python
# run regressions and plot

distances = [100]
indep_combos = [
    *[[f"Max speed < {x} km", f"Days spent < {x} km"] for x in distances],
    *[
        [f"Max speed < {x} km", f"Days spent < {x} km", "Season numeric"]
        for x in distances
    ],
    #     *[[f"Min pressure < {x} km"] for x in distances],
    *[[f"Min pressure < {x} km", f"Days spent < {x} km"] for x in distances],
    *[[f"Max cat. < {x} km", f"Days spent < {x} km"] for x in distances],
]
deps = ["Deaths", "Houses Destroyed", "Losses $Local", "Losses $USD"]

df_reg = pd.DataFrame(columns=["Indeps", "Dep", "R2adj"])


def ols_results(dep, indeps):
    response_terms = [Term([LookupFactor(dep)])]
    model_terms = [Term([])]
    model_terms += [Term([LookupFactor(c)]) for c in indeps]
    model_desc = ModelDesc(response_terms, model_terms)
    return smf.ols(model_desc, df_agg).fit()


for dep in deps:
    for indeps in indep_combos:
        res = ols_results(dep, indeps)
        df_reg.loc[len(df_reg)] = {
            "Indeps": ", ".join(indeps),
            "Dep": dep,
            "R2adj": round(res.rsquared_adj, 2),
        }

fig = px.imshow(
    df_reg.pivot(index="Indeps", columns="Dep", values="R2adj"),
    text_auto=True,
)
fig.update_layout(
    coloraxis_colorbar_title="R2 adjusted",
)
pyo.iplot(fig)

print(
    ols_results(
        "Losses $USD",
        ["Max speed < 100 km", "Days spent < 100 km"],
    ).summary()
)
print(
    ols_results(
        "Deaths",
        ["Max speed < 100 km", "Days spent < 100 km"],
    ).summary()
)
```

```python
# Calculate and plot recurrences by category

distances = range(100, 501, 100)
categories = range(3, 6)

df_recur = pd.DataFrame()

range_color = [0.5, 8]

for distance in distances:
    for category in categories:
        dff_agg = df_agg[
            (df_agg["Distance (km)"] <= distance)
            & (df_agg["Category numeric"] >= category)
        ].sort_values("Season numeric", ascending=False)
        count = len(dff_agg)
        if count < 21:
            names = "<br>".join([x for x in dff_agg["Name Season"]])
        else:
            names = str(count)
        dff_agg = dff_agg[metrics].mean()
        dff_agg["Cyclones"] = names
        dff_agg["Count"] = count
        dff_agg["Category"] = category
        dff_agg["Distance (km)"] = distance
        df_recur = pd.concat(
            [df_recur, dff_agg.to_frame().T], ignore_index=True
        )


df_recur["Recurrence"] = (
    len(df_agg["Season numeric"].dropna().unique()) / df_recur["Count"]
)

df_freq = df_recur.pivot(
    index="Category",
    columns="Distance (km)",
    values="Recurrence",
)
df_freq = df_freq.sort_values("Category", ascending=False)
df_freq.columns = df_freq.columns.astype(str)
df_freq.index = df_freq.index.astype(str)
df_freq = df_freq.astype(float).round(2)

df_records = df_recur.pivot(
    index="Category",
    columns="Distance (km)",
    values="Cyclones",
)
df_records = df_records.sort_values("Category", ascending=False)
df_records.columns = df_records.columns.astype(str)
df_records.index = df_records.index.astype(str)

fig = px.imshow(
    df_freq,
    text_auto=True,
    range_color=range_color,
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
    coloraxis_colorbar_title="Recurrence (years)",
)
fig.update_xaxes(side="top", title_text="Minimum distance to Fiji (km)")

pyo.iplot(fig)
```

```python
# Calculate and plot recurrences by wind speed
# should combine this with cell above...

speeds = range(100, 501, 100)
categories = range(80, 140, 10)

df_recur = pd.DataFrame()

range_color = [0.5, 8]

for distance in distances:
    for category in categories:
        dff_agg = df_agg[
            (df_agg["Distance (km)"] <= distance)
            & (df_agg["Wind (Knots)"] >= category)
        ].sort_values("Season numeric", ascending=False)
        count = len(dff_agg)
        if count < 16:
            names = "<br>".join([x for x in dff_agg["Name Season"]])
        else:
            names = str(count)
        dff_agg = dff_agg[metrics].mean()
        dff_agg["Cyclones"] = names
        dff_agg["Count"] = count
        dff_agg["Category"] = category
        dff_agg["Distance (km)"] = distance
        df_recur = pd.concat(
            [df_recur, dff_agg.to_frame().T], ignore_index=True
        )


df_recur["Recurrence"] = (
    len(df_agg["Season numeric"].dropna().unique()) / df_recur["Count"]
)

df_freq = df_recur.pivot(
    index="Category",
    columns="Distance (km)",
    values="Recurrence",
)
df_freq = df_freq.sort_values("Category", ascending=False)
df_freq.columns = df_freq.columns.astype(str)
df_freq.index = df_freq.index.astype(str)
df_freq = df_freq.astype(float).round(2)

df_records = df_recur.pivot(
    index="Category",
    columns="Distance (km)",
    values="Cyclones",
)
df_records = df_records.sort_values("Category", ascending=False)
df_records.columns = df_records.columns.astype(str)
df_records.index = df_records.index.astype(str)

fig = px.imshow(
    df_freq,
    text_auto=True,
    range_color=range_color,
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
    coloraxis_colorbar_title="Recurrence (years)",
)
fig.update_xaxes(side="top", title_text="Minimum distance to Fiji (km)")
fig.update_yaxes(title_text="Max. wind speed (knots)")

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
fig.update_traces(showlegend=False)
fig.update_layout(
    template="simple_white",
    geo=dict(
        lataxis=dict(range=[-25, -9]),
        lonaxis=dict(range=[164, -166]),
        visible=False,
    ),
)

PLOT_NAMES = [
    "Meli 1978/1979",
    "Winston 2015/2016",
    "Harold 2019/2020",
    "Yasa 2020/2021",
]

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
# plot leadstimes by prop

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
# plot leadtimes

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
