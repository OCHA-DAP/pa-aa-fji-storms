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

# Exposure trigger

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import geopandas as gpd
from matplotlib.ticker import StrMethodFormatter
from tqdm.auto import tqdm
from src.constants import *

from src.blob import PROJECT_PREFIX
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/fms_besttrack_exp.parquet"
df_exp_besttrack = stratus.load_parquet_from_blob(blob_name)
```

```python
df_exp_besttrack = df_exp_besttrack.rename(
    columns={"pop_exposed": "pop_exposed_besttrack"}
)
```

```python
df_exp_besttrack
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/fms_wind_buffers_exposure_fms_reg.parquet"
df_exp_raw = stratus.load_parquet_from_blob(blob_name)
```

```python
df_exp_raw
```

```python
df_exp = df_exp_raw.merge(
    df_exp_besttrack[["sid", "pop_exposed_besttrack", "buffer_speed"]],
    how="left",
)

df_exp["pop_exposed"] = (
    df_exp["pop_exposed_besttrack"].fillna(df_exp["pop_exposed"]).astype(int)
)

df_exp = df_exp.pivot(
    index="sid", columns="buffer_speed", values="pop_exposed"
)
df_exp = df_exp.rename(columns={x: f"exp_{x}" for x in df_exp}).reset_index()
df_exp.columns.name = None
```

```python
df_exp
```

```python
df_exp = df_exp.fillna(0)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/storm_stats_buffer250.parquet"
df_stats_raw = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats_all = df_stats_raw.merge(df_exp, how="outer")
```

```python
# take only from 2001 since this is the only season with full EM-DAT data
min_season = 2001
max_season = 2024
num_seasons = max_season - min_season + 1
df_stats = df_stats_all[
    (df_stats_all["season"] >= min_season)
    & (df_stats_all["season"] <= max_season)
]
```

```python
df_stats["name_season"] = (
    df_stats["name"].str.capitalize()
    + " "
    + df_stats["season"].astype(int).astype(str)
)
```

```python
target_rp = 3
```

```python
# take floor of target years
target_years = int((num_seasons + 1) / target_rp)
actual_rp = (num_seasons + 1) / target_years
```

```python
actual_rp
```

```python
target_years
```

```python
# just check what historical CERF RP is
(max_season - 2007 + 1 + 1) / 3
```

```python
for x in df_stats["Total Affected"].sort_values().to_list():
    n_trig_seasons = df_stats[df_stats["Total Affected"] >= x][
        "season"
    ].nunique()
    if n_trig_seasons == target_years:
        break

impact_thresh = x
```

```python
impact_thresh
```

```python
df_stats["target"] = df_stats["Total Affected"] >= impact_thresh
```

```python
df_stats.sort_values("Total Affected", ascending=False)
```

```python
df_stats["cerf_num"] = df_stats["cerf"].apply(
    lambda x: int(x) if isinstance(x, bool) else None
)
```

```python
df_stats[~df_stats["cerf_num"].isnull()].corr(numeric_only=True)[
    "cerf_num"
].plot.bar()
```

```python
df_stats.corr(numeric_only=True)["Total Affected"].plot.bar()
```

```python
df_stats.plot.scatter(x="wind", y="Total Affected")
```

```python
df_reg = df_stats.dropna().copy()
```

```python
def calc_reg(target_col, var_cols):
    X = df_reg[var_cols]
    y = df_reg[target_col]

    # Add a constant term (intercept)
    X = sm.add_constant(X)

    # Fit the regression
    model = sm.OLS(y, X).fit()

    # Get results
    print(model.summary())
    return model
```

```python
var_cols = ["exp_34", "roll2_mean"]
target_col = "target"
calc_reg(target_col, var_cols)
```

```python
var_cols = [
    "exp_64",
    "exp_50",
    "exp_34",
    # "exp_64_50_34_weighted",
    "wind",
    "roll2_mean",
]
dicts = []
for var_col in var_cols:
    for var_thresh in df_stats[var_col].sort_values():
        dff = df_stats[df_stats[var_col] >= var_thresh]
        if var_col == "exp_64":
            print(var_thresh)
        if dff["season"].nunique() == target_years:
            df_stats[f"{var_col}_trig"] = df_stats[var_col] >= var_thresh
            dicts.append({"var_col": var_col, "var_thresh": var_thresh})
            break

df_threshs = pd.DataFrame(dicts)

# df_stats["exp_64_trig"] = df_stats["exp_64"] > 0
```

```python
df_threshs
```

```python
dicts = []
for var_col in var_cols:
    p = df_stats["target"]
    pp = df_stats[f"{var_col}_trig"]
    tp = p & pp
    fp = ~p & pp
    tn = ~p & ~pp
    fn = p & ~pp
    tpr = tp.sum() / p.sum()
    ppv = tp.sum() / pp.sum()
    f1 = 2 * tpr * ppv / (ppv + tpr)
    corr = df_stats[["Total Affected", var_col]].corr().iloc[0, -1]
    dicts.append({"var_col": var_col, "f1": f1, "corr": corr})

df_metrics = pd.DataFrame(dicts)
```

```python
disp_names = {
    "exp_64": "Exp. to\n64 kt.",
    "exp_50": "Exp. to\n50 kt.",
    "exp_34": "Exp. to\n34 kt.",
    # "exp_64_50_34_weighted": "Weighted\nexp.",
    "wind": "Max. wind\nwithin 250km",
    "roll2_mean": "Two-day\nrain",
}

xcol, ycol = "corr", "f1"

fig, ax = plt.subplots(dpi=200, figsize=(7, 7))

df_metrics.plot.scatter(x=xcol, y=ycol, ax=ax)

for var_col, row in df_metrics.set_index("var_col").iterrows():
    ax.annotate(
        disp_names[var_col],
        row[[xcol, ycol]],
        va="top",
        ha="left",
        fontsize=8,
        fontstyle="italic",
    )

ax.set_xlabel("Correlation")
ax.set_ylabel("F1 score")
ax.set_title(
    "Accuracy metrics of indicators\ncompared to total impact [EM-DAT]"
)

lims = (0, 1)
ax.set_xlim(lims)
ax.set_ylim(lims)

[ax.spines[x].set_visible(False) for x in ["top", "right"]]
```

```python
def highlight_true(val):
    if isinstance(val, bool) and val is True:
        return "background-color: crimson"
    return ""
```

```python
cols = ["name_season", "Total Affected", "cerf", "target", "cat"] + [
    x for x in df_stats.columns if "trig" in x
]
```

```python
df_stats[cols].sort_values("Total Affected", ascending=False).style.format(
    {"Total Affected": "{:,.0f}", "cat": "{:.0f}"}
).map(highlight_true)
```

```python
df_threshs
```

```python
indicator_col = "exp_34"
cols = [
    "name_season",
    "Total Affected",
    "target",
    "cerf",
    indicator_col,
    f"{indicator_col}_trig",
]
df_stats[cols].sort_values(indicator_col, ascending=False).style.format(
    {"Total Affected": "{:,.0f}", "exp_34": "{:,.0f}"}
).map(highlight_true)
```

```python
indicator_col = "roll2_mean"
cols = [
    "name_season",
    "Total Affected",
    "target",
    "cerf",
    indicator_col,
    f"{indicator_col}_trig",
]
df_stats[cols].sort_values(indicator_col, ascending=False).style.format(
    {"Total Affected": "{:,.0f}", "exp_34": "{:,.0f}"}
).map(highlight_true)
```

```python
fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
df_stats.plot.scatter(x="exp_64", y="exp_34", ax=ax, color="crimson")
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

lims = (0, 850_000)
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel('Population exposed to 64-knot ("hurricane-force") wind speed')
ax.set_ylabel('Population exposed to 34-knot ("gale-force") wind speed')

[ax.spines[x].set_visible(False) for x in ["top", "right"]]
```

```python
def plot_indicators(
    xcol,
    ycol,
    show_bubbles: bool = True,
    xthresh: float = None,
    ythresh: float = None,
    cond: str = None,
):
    ymax = df_stats[ycol].max() * 1.1
    xmax = df_stats[xcol].max() * 1.1

    fig, ax = plt.subplots(dpi=200, figsize=(7, 7))

    if show_bubbles:
        bubble_sizes = df_stats["Total Affected"].fillna(0)
        # Optional: scale for visual clarity
        bubble_sizes_scaled = (
            bubble_sizes / bubble_sizes.max() * 5000
        )  # Adjust 300 as needed

    # Plot bubbles
    ax.scatter(
        df_stats[xcol],
        df_stats[ycol],
        s=bubble_sizes_scaled if show_bubbles else 0,
        alpha=0.3,
        color="crimson",
        edgecolor="none",
        zorder=1,
    )

    for _, row in df_stats.iterrows():
        fontweight = "normal"
        if xthresh is not None and ythresh is not None:
            if cond == "or":
                if row[xcol] >= xthresh or row[ycol] >= ythresh:
                    fontweight = "bold"
            elif cond == "and":
                if row[xcol] >= xthresh and row[ycol] >= ythresh:
                    fontweight = "bold"
            else:
                raise ValueError("incorrect cond")
        ax.annotate(
            f'{row["name"].capitalize()}\n{row["season"]:.0f}',
            (row[xcol], row[ycol]),
            ha="center",
            va="center",
            fontsize=6,
            color="crimson" if row["cerf"] == True else "k",
            zorder=10 if row["cerf"] else 9,
            alpha=0.8,
            fontweight=fontweight,
        )

    if xthresh is not None and ythresh is not None:
        ax.axvline(xthresh, color="darkorange", linewidth=0.5)
        ax.axhline(ythresh, color="darkorange", linewidth=0.5)

    legend_text = "\n    Red text indicates CERF allocation\n\n"
    if show_bubbles:
        legend_text += (
            "    Size of bubble corresponds to\n"
            "    total number of people affected [EM-DAT]"
        )
    ax.annotate(
        legend_text,
        (0, ymax),
        va="top",
        fontsize=6,
        fontstyle="italic",
        color="grey",
    )

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    return fig, ax
```

```python
def disp_threshs(
    xcol,
    ycol,
    xthresh: float = None,
    ythresh: float = None,
    cond: str = "or",
    cols=None,
):
    df_disp = df_stats.copy()
    df_disp[f"{xcol}_trig"] = df_stats[xcol] >= xthresh
    df_disp[f"{ycol}_trig"] = df_stats[ycol] >= ythresh
    if cond == "or":
        df_disp["trig"] = df_disp[f"{xcol}_trig"] | df_disp[f"{ycol}_trig"]
    elif cond == "and":
        df_disp["trig"] = df_disp[f"{xcol}_trig"] & df_disp[f"{ycol}_trig"]
    else:
        raise ValueError("wrong cond")
    if cols is None:
        cols = [
            "name_season",
            # "target",
            "cerf",
            xcol,
            f"{xcol}_trig",
            ycol,
            f"{ycol}_trig",
            "trig",
            "exp_34",
            "Total Affected",
        ]
    return (
        df_disp[cols]
        .sort_values(["Total Affected", "trig", xcol, ycol], ascending=False)
        .rename(
            columns={
                "exp_34": "Pop. exposed<br>to 34 knots",
                "exp_64": "Pop. exposed<br>to 64 knots",
                "exp_64_trig": "Pop. exp. trig.",
                "roll2_mean": "Two-day rainfall",
                "roll2_mean_trig": "Rain trig.",
                "cerf": "CERF",
                "name_season": "Name",
                "trig": "Either trig.",
            }
        )
        .iloc[:20]
        .style.format(
            {
                "Total Affected": "{:,.0f}",
                "Pop. exposed<br>to 64 knots": "{:,.0f}",
                "Pop. exposed<br>to 34 knots": "{:,.0f}",
                "Two-day rainfall": "{:,.0f}",
            }
        )
        .map(highlight_true)
        .bar(
            subset="Total Affected",
            vmin=0,
            color="darkorange",
        )
        .set_table_styles(
            {
                "Total Affected": [
                    {"selector": "th", "props": [("text-align", "left")]},
                    {
                        "selector": "td",
                        "props": [("text-align", "left"), ("width", "300px")],
                    },
                ]
            }
        )
    )
```

```python
dicts = []
for rain_thresh in tqdm(df_stats["roll2_mean"].sort_values()):
    for wind_col in ["wind"] + [
        x for x in df_stats.columns if "exp" in x and "trig" not in x
    ]:
        for wind_thresh in df_stats[wind_col].sort_values():
            rain_cond = df_stats["roll2_mean"] >= rain_thresh
            wind_cond = df_stats[wind_col] >= wind_thresh
            dff_and = df_stats[rain_cond & wind_cond]
            dff_or = df_stats[rain_cond | wind_cond]

            if dff_and["season"].nunique() == target_years:
                # check if all CERF storms triggered with fcast
                cerf_fcast_count = dff_and["cerf"].sum()
                dicts.append(
                    {
                        "cond": "and",
                        "rain_thresh": rain_thresh,
                        "wind_col": wind_col,
                        "wind_thresh": wind_thresh,
                        "sum_total_affected": dff_and["Total Affected"].sum(),
                        "cerf_fcast_count": cerf_fcast_count,
                        "fcast_frac": 1,
                    }
                )
            if dff_or["season"].nunique() == target_years:
                # check if all CERF storms triggered with fcast
                cerf_fcast_count = df_stats[wind_cond]["cerf"].sum()
                fcast_count = len(df_stats[wind_cond])
                fcast_frac = fcast_count / len(dff_or)
                dicts.append(
                    {
                        "cond": "or",
                        "rain_thresh": rain_thresh,
                        "wind_col": wind_col,
                        "wind_thresh": wind_thresh,
                        "sum_total_affected": dff_or["Total Affected"].sum(),
                        "cerf_fcast_count": cerf_fcast_count,
                        "fcast_frac": fcast_frac,
                    }
                )
```

```python
df_results = pd.DataFrame(dicts)
```

```python
df_results
```

```python
df_results = df_results.sort_values("sum_total_affected", ascending=False)
```

```python
# df_results_simple = df_results[~df_results["wind_col"].str.contains("_only")]
# df_results_simple = df_results[df_results["wind_col"] == "exp_34"]
df_results_simple = df_results
```

```python
df_results_simple_or = df_results_simple[df_results_simple["cond"] == "or"]
```

```python
df_results_simple_or_cerf = df_results_simple_or[
    df_results_simple_or["cerf_fcast_count"] == 3
]
```

```python
for impact in df_results_simple_or_cerf["sum_total_affected"].unique():
    print(impact)
    display(
        df_results_simple_or_cerf[
            df_results_simple_or_cerf["sum_total_affected"] == impact
        ]
    )
```

```python
xcol, ycol = "exp_64", "roll2_mean"
xthresh, ythresh = 5000, 190

knots = xcol.removeprefix("exp_")

fig, ax = plot_indicators(
    xcol, ycol, xthresh=xthresh, ythresh=ythresh, cond="or"
)
ax.set_xlabel(f"Population exposed to ≥ {knots} knot wind [FMS, WorldPop]")
ax.set_ylabel("Total 2-day rainfall, average over whole country (mm) [IMERG]")
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

plt.show()
```

```python
disp_threshs(xcol, ycol, xthresh=xthresh, ythresh=ythresh, cond="or")
```

```python
df_disp = df_stats.copy()
xcol, ycol = "exp_64", "roll2_mean"
xthresh, ythresh = 5000, 190
df_disp["Total Affected"] = df_disp["Total Affected"].fillna(0)
df_disp[f"{xcol}_trig"] = df_stats[xcol] >= xthresh
df_disp[f"{ycol}_trig"] = df_stats[ycol] >= ythresh
df_disp["trig"] = df_disp[f"{xcol}_trig"] | df_disp[f"{ycol}_trig"]
cols = ["name_season", "target", "cerf", "trig", "wind_trig", "Total Affected"]
(
    df_disp[cols]
    .sort_values(["Total Affected", "trig", "wind_trig"], ascending=False)
    .iloc[:20]
    .rename(
        columns={
            "target": "Target",
            "cerf": "CERF",
            "trig": "New trig.",
            "wind_trig": "Old trig.",
            "name_season": "Name",
        }
    )
    .style.format({"Total Affected": "{:,.0f}"})
    .map(highlight_true)
    .bar(
        subset="Total Affected",
        vmin=0,
        color="darkorange",
    )
    .set_table_styles(
        {
            "Total Affected": [
                {"selector": "th", "props": [("text-align", "left")]},
                {
                    "selector": "td",
                    "props": [("text-align", "left"), ("width", "300px")],
                },
            ]
        }
    )
)
```

```python
len(df_stats)
```

```python
(num_seasons + 1) / 8
```

```python
(num_seasons + 1) / 9
```

```python
(num_seasons + 1) / 10
```

```python
8 / (num_seasons + 1)
```

```python
df_disp = df_stats.copy()
xcol, ycol = "exp_64", "roll2_mean"
xthresh, ythresh = 5000, 190
df_disp["Total Affected"] = df_disp["Total Affected"].fillna(0)
df_disp[f"{xcol}_trig"] = df_stats[xcol] >= xthresh
df_disp[f"{ycol}_trig"] = df_stats[ycol] >= ythresh
df_disp["trig"] = df_disp[f"{xcol}_trig"] | df_disp[f"{ycol}_trig"]
cols = [
    "name_season",
    # "target",
    "cerf",
    # "exp_64_trig",
    # "roll2_mean_trig",
    "trig",
    "wind_trig",
    "Total Affected",
]
(
    df_disp[cols]
    .sort_values(["Total Affected", "trig", "wind_trig"], ascending=False)
    .iloc[:20]
    .rename(
        columns={
            "target": "Target",
            "cerf": "CERF",
            "exp_64_trig": "Wind exp. trig.",
            "roll2_mean_trig": "Rain trig.",
            "trig": "New trig.",
            "wind_trig": "Old trig.",
            "name_season": "Name",
        }
    )
    .style.format({"Total Affected": "{:,.0f}"})
    .map(highlight_true)
    .bar(
        subset="Total Affected",
        vmin=0,
        color="darkorange",
    )
    .set_table_styles(
        {
            "Total Affected": [
                {"selector": "th", "props": [("text-align", "left")]},
                {
                    "selector": "td",
                    "props": [("text-align", "left"), ("width", "300px")],
                },
            ]
        }
    )
)
```

```python
xcol, ycol = "exp_50", "roll2_mean"
xthresh, ythresh = 33993.0, 188.198907

knots = xcol.removeprefix("exp_")

fig, ax = plot_indicators(
    xcol, ycol, xthresh=xthresh, ythresh=ythresh, cond="or"
)
ax.set_xlabel(f"Population exposed to ≥ {knots} knot wind [FMS, WorldPop]")
ax.set_ylabel("Total 2-day rainfall, average over whole country (mm) [IMERG]")
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

plt.show()

disp_threshs(xcol, ycol, xthresh=xthresh, ythresh=ythresh, cond="or")
```

```python
df_stats.sort_values("roll2_mean", ascending=False)
```

```python
xcol, ycol = "exp_64", "roll2_mean"
xthresh, ythresh = 32472.0, 121.61126

fig, ax = plot_indicators(
    xcol, ycol, xthresh=xthresh, ythresh=ythresh, cond="or"
)
ax.set_xlabel("Population exposed to ≥ 34 knot wind [FMS, WorldPop]")
ax.set_ylabel("Total 2-day rainfall, average over whole country (mm) [IMERG]")
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

plt.show()

disp_threshs(xcol, ycol, xthresh=xthresh, ythresh=ythresh, cond="or")
```

```python
fig, ax = plot_indicators(
    "exp_50", "roll2_mean", xthresh=619821.0, ythresh=121.61126, cond="or"
)
ax.set_xlabel("Population exposed to ≥ 34 knot wind [FMS, WorldPop]")
ax.set_ylabel("Total 2-day rainfall, average over whole country (mm) [IMERG]")
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
```

```python
fig, ax = plot_indicators(
    "exp_34", "roll2_mean", xthresh=760508.0, ythresh=121.61126, cond="or"
)
ax.set_xlabel("Population exposed to ≥ 34 knot wind [FMS, WorldPop]")
ax.set_ylabel("Total 2-day rainfall, average over whole country (mm) [IMERG]")
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
```

```python
df_stats
```

```python
# fig, ax = plot_indicators(
#     "exp_34", "roll2_mean", xthresh=697414.0, ythresh=194.592564, cond="or"
# )
fig, ax = plot_indicators(
    "exp_34", "roll2_mean", xthresh=250_000, ythresh=190, cond="or"
)
ax.set_xlabel("Population exposed to ≥ 34 knot wind [FMS, WorldPop]")
ax.set_ylabel("Total 2-day rainfall, average over whole country (mm) [IMERG]")
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
```

```python
df_stats["sel_trig"] = (df_stats["roll2_mean"] >= 121.61126) | (
    df_stats["exp_34"] >= 700802.0
)
```

```python
df_stats["Total Affected"] = df_stats["Total Affected"].fillna(0)
```

```python
df_stats.sort_values("Total Affected")
```

```python
target_years
```

```python
cols = [
    "name_season",
    "target",
    "cerf",
    "exp_34_trig",
    "sel_trig",
    "wind_trig",
    "Total Affected",
]
df_stats[cols].sort_values(
    ["Total Affected", "sel_trig", "wind_trig"], ascending=False
).style.format({"Total Affected": "{:,.0f}", "exp_34": "{:,.0f}"}).map(
    highlight_true
).bar(
    subset="Total Affected",
    vmin=0,
    color="darkorange",
).set_table_styles(
    {
        "Total Affected": [
            {"selector": "th", "props": [("text-align", "left")]},
            {
                "selector": "td",
                "props": [("text-align", "left"), ("width", "500px")],
            },
        ]
    }
)
```

```python
fig, ax = plot_indicators("exp_34", "roll2_mean")
ax.set_xlabel("Population exposed to ≥ 34 knot wind [FMS, WorldPop]")
ax.set_ylabel("Total 2-day rainfall, average over whole country (mm) [IMERG]")
ax.axhline(126.13058)
ax.axvline(700802.0)
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
```

```python
for impact in df_results["sum_total_affected"].unique():
    print(impact)
    display(df_results[df_results["sum_total_affected"] == impact])
```

```python
df_results.iloc[:20]
```

```python
df_results_best = df_results[
    df_results["sum_total_affected"] == df_results["sum_total_affected"].max()
]
```

```python
df_results_best["rain_thresh"].unique()
```

```python
df_results_best
```

```python

```
