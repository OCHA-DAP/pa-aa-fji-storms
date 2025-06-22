---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: pa-aa-fji-storms
    language: python
    name: pa-aa-fji-storms
---

# Observational optimization
<!-- markdownlint-disable MD013 -->

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import duckdb
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dask.diagnostics import ProgressBar

from src.datasources import ibtracs
from src import utils
```

```python
query = """
SELECT *
FROM public.imerg
WHERE pcode = 'FJ'
ORDER BY valid_date ASC
"""
df_imerg = pd.read_sql(query, stratus.get_engine("prod"))
```

```python
df_imerg["roll2_mean"] = df_imerg["mean"].rolling(2).sum()
```

```python
gdf_buffer = utils.load_buffer()
```

```python
from src.datasources import ibtracs_backup
```

```python
# ibtracs_backup.download_ibtracs(temp_dir="temp")
```

```python
ds = ibtracs_backup.load_ibtracs("temp/IBTrACS.SP.v04r01.nc")
```

```python
df_best = ibtracs_backup.get_best_tracks(ds)
```

```python
df_prov = ibtracs_backup.get_provisional_tracks(ds)
```

```python
df_storms = ibtracs_backup.get_storms(ds)
```

```python
df_ibtracs = pd.concat([df_best, df_prov], ignore_index=True)
```

```python
blob_name = "pa-aa-fji-storms/processed/ibtracs/ibtracs_sp.parquet"
stratus.upload_parquet_to_blob(df_ibtracs, blob_name)
```

```python
# get blob URL - note that this contains the SAS so be careful
blob_name = "emdat/processed/emdat_all.parquet"
url = (
    stratus.get_container_client(container_name="global")
    .get_blob_client(blob_name)
    .url
)

# load using duckdb (query is for flooding in Cameroon)
con = duckdb.connect()
df_emdat = con.execute(
    f"""
    SELECT *
    FROM read_parquet('{url}')
    WHERE ISO = 'FJI' AND "Disaster Subtype" = 'Tropical cyclone' AND Historic = 'No'
"""
).df()
```

```python
df_emdat
```

```python
HAROLD = "2020092S09155"
YASA = "2020346S13168"
WINSTON = "2016041S14170"
CERF_SIDS = [HAROLD, YASA, WINSTON]
```

```python
sids = [
    ["2003011S09182",   "Ami 2003"],
    ["2004098S15173",  "Gale 2004"],
    "2006025S18147",  # Jim 2006
    "2007091S14175",  # Cliff 2007
    "2007337S12186",  # Daman 2007
    "2008026S12179",  # Gene 2008
    "2009346S10172",  # Mick 2009
    "2010069S12188",  # Tomas 2010
    "2012346S14180",  # Evan 2012
    WINSTON,  # Winston 2016
    "2016095S13162",  # Zena 2016
    "2018089S18172",  # Josie 2018
    "2018096S15172",  # Keni 2018
    "2019359S08175",  # Sarai 2019
    "2020015S12170",  # Tino 2020
    HAROLD,  # Harold 2020
    YASA,  # Yasa 2020
    "2021029S16171",  # Ana 2021
    "2022008S17173",  # Cody 2022
    "2023316S09167",  # Mal 2023
]
```

```python
df_emdat["sid"] = sids
```

```python
df_emdat["cerf"] = df_emdat["sid"].apply(lambda x: x in CERF_SIDS)
```

```python
gdf_ibtracs = gpd.GeoDataFrame(
    data=df_ibtracs,
    geometry=gpd.points_from_xy(
        df_ibtracs["longitude"], df_ibtracs["latitude"]
    ),
    crs=4326,
)
```

```python
gdf_ibtracs = gdf_ibtracs.to_crs(utils.FJI_CRS)
```

```python
gdf_ibtracs.plot()
```

```python
gdf_ibtracs["in_buffer"] = gdf_ibtracs.within(
    gdf_buffer.to_crs(utils.FJI_CRS).iloc[0].geometry
)
```

```python
gdf_ibtracs
```

```python
gdf_ibtracs_buf = gdf_ibtracs[gdf_ibtracs["in_buffer"]]
```

```python
gdf_ibtracs_buf
```

```python
gdf_ibtracs_buf.plot()
```

```python
gdf_ibtracs_buf_recent = gdf_ibtracs_buf[
    gdf_ibtracs_buf["valid_time"] > "2000-06-01"
]
```

```python
group
```

```python
dicts = []
for sid, group in gdf_ibtracs_buf_recent.groupby("sid"):
    start_date = group["valid_time"].min().date()
    end_date = group["valid_time"].max().date() + pd.Timedelta(days=1)
    df_imerg_f = df_imerg[
        (df_imerg["valid_date"] >= start_date)
        & (df_imerg["valid_date"] <= end_date)
    ]
    dicts.append(
        {
            "sid": sid,
            "wind": group["wind_speed"].max(),
            "roll2_mean": df_imerg_f["roll2_mean"].max(),
        }
    )
```

```python
df_stats = pd.DataFrame(dicts)
```

```python
df_storms
```

```python
df_stats = df_stats.merge(
    df_emdat[["sid", "Total Affected", "cerf"]], how="outer"
).merge(df_storms[["sid", "name", "season"]])
```

```python
df_stats["season"] = df_stats["season"].astype(int)
df_stats["cat"] = df_stats["wind"].apply(utils.knots2cat)
```

```python
df_stats
```

```python
(25 + 1) / 6
```

```python
# CERF RP
(2024 - 2006 + 1 + 1) / 3
```

```python
rain_threshs = {}
for cat in [1, 2, 3, 4]:
    df_stats_cat = df_stats[df_stats["cat"] >= cat].sort_values(
        "roll2_mean", ascending=False
    )
    rain_thresh = None
    for check_thresh in df_stats_cat["roll2_mean"]:
        dff = df_stats_cat[df_stats_cat["roll2_mean"] >= check_thresh]
        if dff["season"].nunique() > 6:
            break
        rain_thresh = check_thresh

    rain_thresh = (rain_thresh + check_thresh) / 2

    rain_threshs.update({cat: rain_thresh})
```

```python
rain_threshs
```

```python
df_stats
```

```python
import inspect

print(inspect.getsource(utils.knots2cat))
```

```python
cat2knots = {5: 108, 4: 86, 3: 64, 2: 48, 1: 34}
```

```python
def plot_threshs(rain_thresh, wind_cat_thresh):
    wind_thresh = cat2knots[wind_cat_thresh]
    ymax = df_stats["roll2_mean"].max() * 1.1
    xmax = df_stats["wind"].max() * 1.1

    fig, ax = plt.subplots(dpi=200, figsize=(7, 7))

    bubble_sizes = df_stats["Total Affected"].fillna(0)
    # Optional: scale for visual clarity
    bubble_sizes_scaled = (
        bubble_sizes / bubble_sizes.max() * 5000
    )  # Adjust 300 as needed

    # Plot bubbles
    ax.scatter(
        df_stats["wind"],
        df_stats["roll2_mean"],
        s=bubble_sizes_scaled,
        alpha=0.3,
        color="crimson",
        edgecolor="none",
        zorder=1,
    )

    for _, row in df_stats.iterrows():
        triggered = (row["roll2_mean"] >= rain_thresh) & (
            row["wind"] >= wind_thresh
        )
        ax.annotate(
            row["name"].capitalize() + "\n" + str(row["season"]),
            (row["wind"], row["roll2_mean"]),
            ha="center",
            va="center",
            fontsize=6,
            color="crimson" if row["cerf"] == True else "k",
            zorder=10 if row["cerf"] else 9,
            alpha=0.8,
            fontstyle="italic" if triggered else "normal",
            fontweight="bold" if triggered else "normal",
        )

    trig_color = "orange"
    ax.axvline(
        wind_thresh,
        color=trig_color,
        linewidth=0.5,
        zorder=0,
    )
    ax.annotate(
        f"  Cat. {wind_cat_thresh} \n({wind_thresh} kn) ",
        (wind_thresh, 0),
        va="top",
        ha="center",
        rotation=90,
        color=trig_color,
        fontsize=8,
        fontstyle="italic",
    )
    ax.axhline(
        rain_thresh,
        color=trig_color,
        linewidth=0.5,
        zorder=0,
    )
    ax.annotate(
        f"{rain_thresh:.0f} mm",
        (0, rain_thresh),
        va="center",
        ha="right",
        color=trig_color,
        fontsize=8,
        fontstyle="italic",
    )
    ax.add_patch(
        mpatches.Rectangle(
            (wind_thresh, rain_thresh),  # bottom left
            xmax - wind_thresh,  # width
            ymax - rain_thresh,  # height
            facecolor=trig_color,
            alpha=0.07,
            zorder=0,
        )
    )

    ax.annotate(
        "\n"
        "    Size of bubble corresponds to\n"
        "    total number of people affected [EM-DAT]\n\n"
        "    Red text indicates CERF allocation",
        (0, ymax),
        va="top",
        fontsize=6,
        fontstyle="italic",
        color="grey",
    )
    ax.annotate(
        "\nTriggered    \nstorms    ",
        (xmax, ymax),
        ha="right",
        va="top",
        color=trig_color,
        fontstyle="italic",
    )

    ax.set_ylabel(
        "Total 2-day rainfall, average over whole country (mm) [IMERG]"
    )
    ax.set_xlabel("\nMax. wind speed while in 250 km buffer (knots) [FMS]")
    ax.set_title(
        "Fiji triggered storms (since 2000)\n"
        f"Cat. {cat} and {rain_thresh:.0f} mm rainfall trigger"
    )

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
```

```python
cat = 2
plot_threshs(rain_threshs[cat], cat)
```

```python
cat = 3
plot_threshs(rain_threshs[cat], cat)
```

```python
cat = 4
plot_threshs(rain_threshs[cat], cat)
```

```python

```
