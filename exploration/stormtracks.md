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

pyo.init_notebook_mode()
load_dotenv()
```

```python
CS_KEY = os.environ["CHARTSTUDIO_APIKEY"]
chart_studio.tools.set_credentials_file(
    username="tristandowning", api_key=CS_KEY
)
MB_TOKEN = os.environ["MAPBOX_TOKEN"]
```

```python
EXP_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/exploration/fji"
CYCLONETRACKS_PATH = (
    EXP_DIR / "rsmc/RSMC TC Tracks Historical 1969_70 to 2022_23 Seasons.csv"
)
RAW_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/raw/fji"
CODAB_PATH = RAW_DIR / "cod_ab/fji_polbnda_adm0_country"
CODAB3_PATH = RAW_DIR / "cod_ab/fji_polbnda_adm3_tikina"
IMPACT_PATH = EXP_DIR / "rsmc/FIJI_ DesInventar data 20230626.xlsx"
PROC_PATH = Path(os.environ["AA_DATA_DIR"]) / "public/processed/fji"
SAVE_DIR = Path("/Users/tdowning/OCHA/data/fji")
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

# drop duplicates - Odette
df_clean = df_clean.drop_duplicates(subset="Name Season", keep="first")

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
df["Birth"] = df.groupby("Name Season")["datetime"].transform(min)
df["Age (days)"] = df["datetime"] - df["Birth"]
df["Age (days)"] = df["Age (days)"].apply(
    lambda x: x.days + x.seconds / 24 / 3600
)


gdf_tracks = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"])
)
gdf_tracks.crs = "EPSG:4326"

# Read CODAB

gdf_adm0 = gpd.read_file(CODAB_PATH, layer="fji_polbnda_adm0_country").set_crs(
    "EPSG:3832"
)
```

```python
# interpolate cyclone tracks

dfs = []

for name in gdf_tracks["Name Season"].unique():
    dff = gdf_tracks[gdf_tracks["Name Season"] == name][
        [
            "Latitude",
            "Longitude",
            "Category numeric",
            "Wind (Knots)",
            "Pressure",
            "datetime",
            "Age (days)",
        ]
    ]
    dff = dff.groupby("datetime").first()
    dff = dff.resample("H").interpolate().reset_index()
    dff["Name Season"] = name
    dfs.append(dff)

df_interp = pd.concat(dfs)

gdf_interp = gpd.GeoDataFrame(
    df_interp,
    geometry=gpd.points_from_xy(df_interp["Longitude"], df_interp["Latitude"]),
)
gdf_interp.crs = "EPSG:4326"
```

```python
# calculate distance for admin0 (no interp)
# note - for admin0, it doesn't make much of a difference whether using
# interpolated storm tracks or not
gdf_tracks["Distance (km)"] = (
    gdf_tracks.to_crs(epsg=3832).apply(
        lambda point: point.geometry.distance(gdf_adm0.geometry),
        axis=1,
    )
    / 1000
)
```

```python
# calculate distance for admin0 (with interp)
# note - for admin0, it doesn't make much of a difference whether using
# interpolated storm tracks or not
gdf_interp["Distance (km)"] = (
    gdf_interp.to_crs(epsg=3832).apply(
        lambda point: point.geometry.distance(gdf_adm0.geometry),
        axis=1,
    )
    / 1000
)
```

```python
# agg with interp - not necesary for adm0

df_agg = pd.DataFrame()

df_agg = (
    gdf_interp.groupby(["Name Season"])
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

df_agg["Season numeric"] = df_agg["Name Season"].apply(
    lambda x: int(x.split("/")[0][-4:])
)
```

```python
# calculate adm3 distances with interp
# filter interp to reduce time to calculate adm3 distances

gdf_adm3 = gpd.read_file(CODAB3_PATH, layer="fji_polbnda_adm3_tikina").set_crs(
    "EPSG:3832"
)

dff = gdf_interp.groupby("Name Season").agg(
    {"Distance (km)": "min", "Category numeric": "max"}
)
names = dff[(dff["Category numeric"] > 2) & (dff["Distance (km)"] < 300)].index

df_distances = gdf_interp[gdf_interp["Name Season"].isin(names)]
df_distances = df_distances.to_crs(epsg=3832)

gdf_adm3 = gdf_adm3.set_index("ADM3_PCODE")

for pcode in gdf_adm3.index:
    df_distances[pcode] = (
        df_distances.apply(
            lambda row: row.geometry.distance(gdf_adm3.loc[pcode].geometry),
            axis=1,
        )
        / 1000
    )
```

```python
df_d_melt = df_distances.melt(
    id_vars=[
        "datetime",
        "Category numeric",
        "Name Season",
        "Wind (Knots)",
        "Pressure",
        "Age (days)",
    ],
    value_vars=gdf_adm3.index,
    var_name="ADM3_PCODE",
    value_name="Adm3 distance (km)",
)
df_d_melt
```

```python
cutoff_cat = 3
cutoff_distance = 200

dff = df_d_melt.copy()
dff = dff[
    (dff["Adm3 distance (km)"] <= cutoff_distance)
    & (dff["Category numeric"] >= cutoff_cat)
]
dff = dff.sort_values("datetime", ascending=False)

for pcode in gdf_adm3.index:
    dfff = dff[dff["ADM3_PCODE"] == pcode]
    names = dfff["Name Season"].unique()
    names_text = "<br>".join(names)
    names_count = len(names)
    days_spent = len(dfff) / 24
    time_per_storm = days_spent / names_count if names_count > 0 else np.nan

    min_ds = dfff.groupby("Name Season")["Adm3 distance (km)"].min()
    leadtimes = dfff[dfff["Adm3 distance (km)"].isin(min_ds)]["Age (days)"]

    gdf_adm3.loc[
        pcode,
        [
            "days",
            "count",
            "days per storm",
            "storm names",
            "median leadtime",
            "mean leadtime",
        ],
    ] = [
        days_spent,
        names_count,
        time_per_storm,
        names_text,
        leadtimes.median(),
        leadtimes.mean(),
    ]

# plot adm3 count
# terrible terrible bodge to deal with shp spanning -180 deg
gdf_t = gdf_adm3.to_crs("EPSG:3832")
gdf_t.geometry = gdf_t.geometry.translate(-1000000, 0)
gdf_t = gdf_t.to_crs("EPSG:4326")
# realign by trial and error . . .
gdf_t.geometry = gdf_t.geometry.translate(8.984, 0)


fig = px.choropleth_mapbox(
    gdf_t,
    geojson=gdf_t.geometry,
    locations=gdf_t.index,
    color="count",
)

fig.update_traces(
    marker_line_width=0.01,
    customdata=gdf_t[["ADM3_NAME", "storm names", "count"]],
    hovertemplate="<b>%{customdata[0]}</b><br>"
    "Total: %{customdata[2]}<br><br>%{customdata[1]}",
)

fig.update_layout(
    mapbox_style="white-bg",
    mapbox_accesstoken=MB_TOKEN,
    mapbox_zoom=6.5,
    mapbox_center_lat=-17.8,
    mapbox_center_lon=179.5,
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    title=dict(
        text=f"Count of Category ≥{cutoff_cat} cyclones within "
        "{cutoff_distance} km of each Tikina "
        "since 1969<br>"
        f"<sup>Cyclone counted if its Category is ≥{cutoff_cat} while "
        "it is within {cutoff_distance} km "
        "of the Tikina",
        x=0.01,
    ),
    coloraxis_colorbar_title="Total number<br>of cyclones",
)

fig.show()

filename = "fji_tikina_count.html"

f = open(SAVE_DIR / filename, "w")
f.close()
with open(SAVE_DIR / filename, "a") as f:
    f.write(
        fig.to_html(full_html=True, include_plotlyjs="cdn", auto_play=False)
    )
f.close()

# plot adm3 leadtime
# terrible terrible bodge to deal with shp spanning -180 deg
gdf_t = gdf_adm3.to_crs("EPSG:3832")
gdf_t.geometry = gdf_t.geometry.translate(-1000000, 0)
gdf_t = gdf_t.to_crs("EPSG:4326")
# realign by trial and error . . .
gdf_t.geometry = gdf_t.geometry.translate(8.984, 0)


fig = px.choropleth_mapbox(
    gdf_t,
    geojson=gdf_t.geometry,
    locations=gdf_t.index,
    color="median leadtime",
)

fig.update_traces(
    marker_line_width=0.01,
    customdata=gdf_t[["ADM3_NAME", "storm names", "median leadtime"]],
    hovertemplate="<b>%{customdata[0]}</b><br>"
    "Median leadtime: %{customdata[2]:.1f} days<br><br>%{customdata[1]}",
)

fig.update_layout(
    mapbox_style="white-bg",
    mapbox_accesstoken=MB_TOKEN,
    mapbox_zoom=6.5,
    mapbox_center_lat=-17.8,
    mapbox_center_lon=179.5,
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    title=dict(
        text=f"Median leadtime of Category ≥{cutoff_cat} cyclones within"
        " {cutoff_distance} km of each Tikina "
        "since 1969<br>"
        "<sup>Leadtime calculated as number of days between first "
        "tracking storm, and storm passing closest to "
        "the Tikina",
        x=0.01,
    ),
    coloraxis_colorbar_title="Median leadtime<br>(days)",
)


fig.show()

filename = "fji_tikina_leadtimes.html"

f = open(SAVE_DIR / filename, "w")
f.close()
with open(SAVE_DIR / filename, "a") as f:
    f.write(
        fig.to_html(full_html=True, include_plotlyjs="cdn", auto_play=False)
    )
f.close()
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
    dff = gdf_tracks[gdf_tracks["Name Season"] == name_season].sort_values(
        "datetime"
    )
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

    # 4-5 leadtime
    if not dff[dff["Category numeric"] == 5].empty:
        low_datetime = dff[dff["Category numeric"] == 4]["datetime"].iloc[0]
        high_datetime = dff[dff["Category numeric"] == 5]["datetime"].iloc[0]
        l = high_datetime - low_datetime
        fourfive_leadtime = l.days + l.seconds / 3600 / 24
    else:
        fourfive_leadtime = None

    # 3-4 leadtime
    if not dff[dff["Category numeric"] == 4].empty:
        low_datetime = dff[dff["Category numeric"] == 3]["datetime"].iloc[0]
        high_datetime = dff[dff["Category numeric"] == 4]["datetime"].iloc[0]
        l = high_datetime - low_datetime
        threefour_leadtime = l.days + l.seconds / 3600 / 24
    else:
        threefour_leadtime = None

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
            "Leadtime (4-5)",
            "Leadtime (3-4)",
            "Cat. at min distance",
            "Speed at min distance",
        ],
    ] = [
        distance_leadtime,
        strength_leadtime,
        fourfive_leadtime,
        threefour_leadtime,
        landfall_row["Category numeric"],
        landfall_row["Wind (Knots)"],
    ]

for l in ["strength", "distance"]:
    df_agg[f"Leadtime ({l})"] = df_agg[f"Leadtime ({l})"].apply(
        lambda x: x.days + x.seconds / 3600 / 24
    )

# drop ridiculously long cyclone
df_agg = df_agg[~(df_agg["Name Season"] == "Unnamed-sp 1971/1972")]


df_agg.to_csv(EXP_DIR / "aggregated_cyclone_tracks.csv", index=False)

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
    *[[f"Max speed < {x} km"] for x in distances],
    *[[f"Max speed < {x} km", f"Days spent < {x} km"] for x in distances],
    *[
        [f"Max speed < {x} km", f"Days spent < {x} km", "Season numeric"]
        for x in distances
    ],
    #     *[[f"Min pressure < {x} km"] for x in distances],
    *[[f"Min pressure < {x} km", f"Days spent < {x} km"] for x in distances],
    *[[f"Max cat. < {x} km", f"Days spent < {x} km"] for x in distances],
    ["Category"],
]
deps = ["Deaths", "Houses Destroyed", "Losses $Local", "Losses $USD"]

df_reg = pd.DataFrame(columns=["Indeps", "Dep", "R2adj"])


def ols_results(dep, indeps, df):
    response_terms = [Term([LookupFactor(dep)])]
    model_terms = [Term([])]
    model_terms += [Term([LookupFactor(c)]) for c in indeps]
    model_desc = ModelDesc(response_terms, model_terms)
    return smf.ols(model_desc, df).fit()


for dep in deps:
    for indeps in indep_combos:
        res = ols_results(dep, indeps, df_agg)
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
```

```python
print(ols_results("Losses $Local", ["Max cat. < 100 km"], df_agg).summary())
print(
    ols_results(
        "Losses $Local", ["Max speed < 100 km", "Days spent < 100 km"], df_agg
    ).summary()
)
print(
    ols_results(
        "Deaths",
        ["Max speed < 100 km", "Days spent < 100 km", "Season numeric"],
        df_agg,
    ).summary()
)
```

```python
# plot category and strength by year

categories = range(2, 6, 1)
speeds = range(60, 181, 40)
speed_labels = [f"{speeds[j]}-{speeds[j+1]}" for j in range(len(speeds) - 1)]

bins = range(1970, 2031, 10)
labels = [f"{x}'s" for x in bins[:-1]]
labels[-1] = labels[-1] + " (projected)"
valid_years = [min(10, 2022 - year) for year in bins[:-1]]

cutoff_distance = 200

# category

fig = go.Figure()
fig.update_layout(template="simple_white")

for category in categories:
    dff = df_agg[
        (df_agg["Category numeric"] == category)
        & (df_agg["Distance (km)"] < cutoff_distance)
    ]
    dff = (
        dff.groupby(
            pd.cut(
                dff["Season numeric"], bins=bins, right=False, labels=labels
            )
        )
        .agg(
            names=("Name Season", list),
            count=("Name Season", "count"),
        )
        .reset_index()
    )
    dff["count"] = dff["count"] / valid_years * 10
    fig.add_trace(
        go.Bar(
            x=dff["Season numeric"],
            y=dff["count"],
            name=category,
        )
    )

pyo.iplot(fig)

# speed

fig = go.Figure()
fig.update_layout(
    template="simple_white",
    title="Cyclones within 200 km of Fiji by wind speed",
)

for j in range(len(speeds) - 1):
    dff = df_agg[
        (df_agg["Max speed < 200 km"] >= speeds[j])
        & (df_agg["Max speed < 200 km"] < speeds[j + 1])
        & (df_agg["Distance (km)"] < cutoff_distance)
    ]
    dff = (
        dff.groupby(
            pd.cut(
                dff["Season numeric"], bins=bins, right=False, labels=labels
            )
        )
        .agg(
            names=("Name Season", list),
            count=("Name Season", "count"),
        )
        .reset_index()
    )
    dff["count"] = dff["count"] / valid_years * 10
    fig.add_trace(
        go.Scatter(
            x=dff["Season numeric"],
            y=dff["count"],
            name=speed_labels[j],
            stackgroup="one",
        )
    )

fig.update_xaxes(title="Decade")
fig.update_yaxes(title="Number of cyclones per decade")
fig.update_legends(title="Max. speed within<br>200 km of Fiji<br>(knots)")

pyo.iplot(fig)

col = "Max speed < 100 km"
df_plot = df_agg.groupby("Season numeric")[col].mean().reset_index()
px.scatter(df_plot, x="Season numeric", y=col).show()
```

```python
col = "Max speed < 200 km"

colors = ["blue", "green", "red"]

hist_data = []

years = range(1970, 2031, 20)
valid_years = [min(20, 2023 - year) for year in years[:-1]]
year_labels = [f"{years[j]}-{years[j+1]-1}" for j in range(len(years) - 2)]
year_labels.append(f"{years[-2]}-")
medians = []

for j in range(len(years) - 1):
    dff = df_agg[
        (df_agg["Season numeric"] >= years[j])
        & (df_agg["Season numeric"] < years[j + 1])
    ][col]
    dff = dff.dropna()
    medians.append(dff.median())
    hist_data.append(dff)

fig = ff.create_distplot(
    hist_data,
    group_labels=year_labels,
    bin_size=20,
    show_rug=False,
    colors=colors,
)

y_pos = [0.033, 0.030, 0.027]
for median, year, color, y_p in zip(medians, years[:-1], colors, y_pos):
    fig.add_trace(
        go.Scatter(
            y=[0, y_p, 0.04],
            x=[median, median, median],
            showlegend=False,
            line_dash="dash",
            line_color=color,
            mode="lines+text",
            text=["", f" median<br> ={median}", ""],
            textposition="bottom right",
            textfont_color=color,
        )
    )

fig.update_traces(marker_line_width=0, marker_opacity=0.1)
fig.update_layout(
    template="simple_white",
    title="Historical distribution of wind speeds",
    height=600,
)
fig.update_xaxes(title="Max. speed within 200 km of Fiji (knots)")
fig.update_yaxes(range=[0, 0.034], title="Fraction of cyclones in period")
fig.update_legends(
    title="Season start year",
    y=0.99,
    x=0.99,
    xanchor="right",
)
fig.show()
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
        #         dff_agg = dff_agg[dff_agg["Season numeric"] > 1999]
        count = len(dff_agg)
        if count < 21:
            names = "<br>".join([x for x in dff_agg["Name Season"]])
        else:
            names = str(count)
        #         dff_agg = dff_agg[metrics].mean()
        dff_agg["Cyclones"] = names
        dff_agg["Count"] = count
        dff_agg["Category"] = category
        dff_agg["Distance (km)"] = distance
        #         df_recur = pd.concat(
        #             [df_recur, dff_agg.to_frame().T], ignore_index=True
        #         )
        df_count = pd.DataFrame(
            {
                "Cyclones": names,
                "Count": count,
                "Category": category,
                "Distance (km)": distance,
            },
            index=[1],
        )
        df_recur = pd.concat([df_recur, df_count])


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
    coloraxis_colorbar_title="Return period (years)",
)
fig.update_xaxes(side="top", title_text="Minimum distance to Fiji (km)")

pyo.iplot(fig)
```

```python
# Calculate and plot recurrences by wind speed
# should combine this with cell above...

speeds = range(100, 501, 100)
distances = range(100, 501, 100)
categories = range(70, 140, 10)

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
# Calculate and plot recurrences by wind speed < 100km and days spent < 100 km
# should combine this with cell above...

distances = range(0, 1, 1)
categories = range(60, 140, 10)

df_recur = pd.DataFrame()
df_trig = df_agg.copy()
df_trig = df_trig.dropna(subset=["Deaths", "Pressure"])

range_color = [0.5, 8]

cutoff_year = 1000

num_years = min(
    2022 - cutoff_year, len(df_agg["Season numeric"].dropna().unique())
)

for distance in distances:
    for category in categories:
        df_trig[trigger_wording] = False
        dff_agg = df_agg[
            (df_agg["Days spent < 100 km"] >= distance)
            & (df_agg["Max speed < 100 km"] >= category)
        ].sort_values("Season numeric", ascending=False)
        dff_agg = dff_agg[dff_agg["Season numeric"] > cutoff_year]
        df_trig.loc[
            df_trig["Name Season"].isin(dff_agg["Name Season"]),
            trigger_wording,
        ] = True
        count = len(dff_agg)
        if count < 20:
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
        trigger_wording = f"Max speed < 100 km = {category}"


df_recur["Recurrence"] = num_years / df_recur["Count"]

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
fig.update_xaxes(side="top", title_text="Minimum days within 100 km")
fig.update_yaxes(title_text="Max speed within 100 km")

pyo.iplot(fig)
```

```python
plot_metrics = ["Deaths", "Losses $Local", "Houses Destroyed"]
trigger_wordings = ["Max speed < 100 km = 90", "Max speed < 100 km = 80"]

for metric in plot_metrics:
    for trigger_wording in trigger_wordings:
        fig = go.Figure()
        fig.update_layout(template="simple_white")
        df_plot = (
            df_trig[df_trig[metric] > 0]
            .sort_values(metric, ascending=False)
            .iloc[:20]
        )

        fig.add_trace(
            go.Bar(
                x=df_plot["Name Season"],
                y=df_plot[metric],
            )
        )

        fig.add_trace(
            go.Bar(
                x=df_plot[df_plot[trigger_wording]]["Name Season"],
                y=df_plot[df_plot[trigger_wording]][metric],
            )
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
fig.update_traces(showlegend=False)
fig.update_layout(
    template="simple_white",
    geo=dict(
        lataxis=dict(range=[-24, -10]),
        lonaxis=dict(range=[165, -167]),
        visible=True,
    ),
)

PLOT_NAMES = [
    "Meli 1978/1979",
    "Winston 2015/2016",
    "Harold 2019/2020",
    "Yasa 2020/2021",
    "Odette 1984/1985",
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

if True:
    filename = "fji_stormtracks.html"

    f = open(SAVE_DIR / filename, "w")
    f.close()
    with open(SAVE_DIR / filename, "a") as f:
        f.write(fig.to_html(full_html=True, include_plotlyjs="cdn"))
    f.close()
```

```python



def points_to_linetring(points):
    return LineString(points.tolist())


print(gdf_tracks)
gdf_agg = gdf_tracks.groupby("Name Season").agg(
    {"geometry": points_to_linetring, "Wind (Knots)": list}
)

print(gdf_agg)
```

```python
# plot heatmap

fig = px.density_mapbox(
    df_interp,
    lat="Latitude",
    lon="Longitude",
    z="Wind (Knots)",
    radius=10,
)

fig.update_layout(
    mapbox_style="light",
    mapbox_accesstoken=MB_TOKEN,
    mapbox_zoom=5,
    mapbox_center_lat=-17,
    mapbox_center_lon=178,
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

fig.show()
```

```python
# plot mapbox traces

initial_plot_names = [
    "Winston 2015/2016",
    "Harold 2019/2020",
    "Yasa 2020/2021",
]

cutoff_cat = 3
cutoff_distance = 300

plot_names = df_agg[
    (df_agg[f"Max cat. < {cutoff_distance} km"] >= cutoff_cat)
]["Name Season"]

df_plot = gdf_tracks.copy()
df_plot = df_plot[df_plot["Name Season"].isin(plot_names)]

df_plot = df_plot.sort_values("datetime", ascending=False)

cat2color = {
    5: "darkred",
    4: "red",
    3: "orange",
    2: "gold",
    1: "dimgray",
    0: "darkgray",
}

fig = go.Figure()

for name in df_plot["Name Season"].unique():
    dff = df_plot[df_plot["Name Season"] == name].sort_values(
        "datetime", ascending=True
    )
    dff["Category count"] = (
        dff["Category numeric"].ne(dff["Category numeric"].shift()).cumsum()
    )
    max_wind = dff["Wind (Knots)"].max()
    cat_count_max = dff[dff["Wind (Knots)"] == max_wind].iloc[0][
        "Category count"
    ]

    for cat_count in dff["Category count"].unique():
        dfff = dff[dff["Category count"] == cat_count]
        last_date = dfff["datetime"].iloc[-1]
        next_date = dff.loc[dff["datetime"].shift().eq(last_date)]
        dfff = pd.concat([dfff, next_date])
        category = dfff["Category numeric"].iloc[0]

        fig.add_trace(
            go.Scattermapbox(
                lat=dfff["Latitude"],
                lon=dfff["Longitude"],
                name=name,
                line_color=cat2color.get(category),
                opacity=0.5,
                customdata=dfff[
                    [
                        "Category",
                        "Wind (Knots)",
                        "datetime",
                        "Distance (km)",
                        "Age (days)",
                    ]
                ],
                mode="lines",
                line_width=category**2 / 2 + 2,
                hovertemplate=(
                    "Category: %{customdata[0]}<br>"
                    "Wind speed: %{customdata[1]} knots<br>"
                    "Datetime: %{customdata[2]}<br>"
                    "Distance: %{customdata[3]:,.0f} km<br>"
                    "Age: %{customdata[4]:.1f} days"
                ),
                showlegend=bool(cat_count == cat_count_max),
                legendgroup=name,
            )
        )

color_text = " ".join(
    [
        f"<span style='color:{color}'>{cat}</span>"
        for cat, color in cat2color.items()
    ]
)
fig.update_layout(
    mapbox_style="mapbox://styles/tristandownin/clk2sq75x01lc01pc9xxn7ov9",
    mapbox_accesstoken=MB_TOKEN,
    mapbox_zoom=5,
    mapbox_center_lat=-17,
    mapbox_center_lon=178,
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    title=dict(
        text=f"Category ≥ {cutoff_cat} within "
        "{cutoff_distance} km of Fiji since 1969<br>"
        f"<sup>Category indicated by line width and colour: "
        "<b>{color_text}</b></sup>",
        x=0.01,
    ),
)
fig.update_legends(
    title="Double-click to<br>isolate cyclone:",
    tracegroupgap=2,
)

fig.show(config={"displayModeBar": False})

# py.plot(fig, filename="Fiji Storm Tracks", auto_open=False)
```

```python
py.plot(
    fig,
    filename="Fiji Storm Tracks",
    auto_open=False,
    config={"displayModeBar": False},
)
```

```python

```
