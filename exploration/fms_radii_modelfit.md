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

# FMS tracks radius model fit
<!-- markdownlint-disable MD013 -->
Fitting model for wind radii to FMS tracks

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_lens as lens
import ocha_stratus as stratus
import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.constants import *
from src.blob import PROJECT_PREFIX
from src.datasources.ibtracs import expand_quad_col
```

## Load and process data

```python
blob_name = f"{PROJECT_PREFIX}/processed/fms/best_tracks.parquet"
df_tracks = stratus.load_parquet_from_blob(blob_name)
```

```python
df_tracks.count()
```

```python
quads = ["ne", "nw", "se", "sw"]
```

```python
speeds = [34, 50, 64]
```

```python
speed2word = {34: "Gale", 50: "Storm", 64: "Hurricane"}
```

```python
for speed in speeds:
    speedword = speed2word[speed]
    df_tracks = df_tracks.rename(
        columns={
            f"{x.upper()}{speedword}Radius": f"quadrant_radius_{speed}_{x}"
            for x in quads
        }
    )
    cols = [f"quadrant_radius_{speed}_{quad}" for quad in quads]
    df_tracks[f"quadrant_radius_{speed}_mean"] = df_tracks[cols].mean(axis=1)
```

```python
all_quad_cols = [x for x in df_tracks.columns if "quadrant" in x]
```

```python
df_tracks[all_quad_cols] = df_tracks[all_quad_cols].fillna(0)
```

Let's see how things look for each buffer speed.

```python
for speed in speeds:
    df_tracks.plot.scatter(
        x="MeanWind",
        y=f"quadrant_radius_{speed}_mean",
        alpha=0.1,
    )
```

## Regression

### Compare various methods

Here we can compare:

- Linear vs. log-log (`log` in the results table)
- Whether we should also include latitude (or only wind speed) (`dof` in the results table)

First we set up the columns.

```python
all_quad_cols
```

```python
df_tracks["lat_abs"] = df_tracks["Latitude"].abs()
```

```python
for col in ["MeanWind", "lat_abs"] + all_quad_cols:
    df_tracks[f"{col}_log"] = np.log(df_tracks[col])
```

Then we iterate over the possible setups.

```python
dicts = []
for col in all_quad_cols:
    print(col)
    for log_str in ["", "_log"]:
        target_col = f"{col}{log_str}"
        for var_cols in [
            [f"MeanWind{log_str}"],
            [f"MeanWind{log_str}", f"lat_abs{log_str}"],
        ]:
            df_reg = df_tracks[[target_col] + var_cols].dropna()
            # exclude zero values (which would show up as -np.inf for the log)
            df_reg = df_reg[df_reg[target_col] > 0]
            X = df_reg[var_cols]
            y = df_reg[target_col]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            print(var_cols)
            display(model.summary())
            dicts.append(
                {
                    "col": col,
                    "dof": len(var_cols),
                    "log": log_str,
                    "r2adj": model.rsquared_adj,
                }
            )
```

```python
df_results = pd.DataFrame(dicts)
```

```python
df_results
```

From the results (column `r2adj` is the adjusted $R^2$ value) above it looks like:

1. Log-log performs better than linear for 34, but other way around for 50 and 64 (but isn't a huge difference for those)
2. Including latitude (`dof=2`) is better than only wind
3. Also it seems like we generally get better performance for the `mean` prediction instead of by quadrant.

So, we'll go with that.

### Make predictions

```python
# quad_cols = [f"quadrant_radius_34_{x}" for x in quads]
```

```python
mean_quad_cols = [f"quadrant_radius_{x}_mean" for x in speeds]
```

```python
dicts_params = []
for col in mean_quad_cols:
    df_reg = df_tracks[
        [
            "Name Season",
            "valid_time",
            "MeanWind_log",
            "lat_abs_log",
            f"{col}_log",
        ]
    ].dropna()
    df_reg = df_reg[df_reg[f"{col}_log"] > 0]
    X = df_reg[["MeanWind_log", "lat_abs_log"]]
    y = df_reg[f"{col}_log"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    dicts_params.append({"col": col, "params": model.params})
    X_pred = df_tracks[["MeanWind_log", "lat_abs_log"]]
    X_pred = sm.add_constant(X_pred)
    df_tracks[f"{col}_log_pred"] = model.predict(X_pred)
```

Here are the parameters which we will save in `src.constants` so we can apply them to new forecasts:

```python
dicts_params
```

We also just want to set anywhere with a maximum wind speed lower than the buffer speed to a radius of 0.

```python
for speed in speeds:
    df_tracks[f"quadrant_radius_{speed}_mean_pred"] = np.exp(
        df_tracks[f"quadrant_radius_{speed}_mean_log_pred"]
    )
    df_tracks[f"quadrant_radius_{speed}_mean_pred"] = df_tracks.apply(
        lambda row: 0
        if row["MeanWind"] < speed
        else row[f"quadrant_radius_{speed}_mean_pred"],
        axis=1,
    )
    df_tracks[f"quadrant_radius_{speed}_mean_pred"] = df_tracks[
        f"quadrant_radius_{speed}_mean_pred"
    ].fillna(0)
```

### Plot results

Here we plot the results - looks ok, although we often predict a non-zero radius when the radius. But this could be from the weird points above where it's possible that JTWC didn't calculate a radius.

```python
for speed in speeds:
    df_tracks.plot.scatter(
        x=f"quadrant_radius_{speed}_mean",
        y=f"quadrant_radius_{speed}_mean_pred",
        alpha=0.05,
    )
```

We can also calculate the $R^2$ and correlation (including the zeros, so pessimistic estimate).

```python
def calc_r2(y_pred, y_true, k):
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return r2, r2_adj
```

```python
for speed in speeds:
    print(speed)
    print("corr:")
    print(
        df_tracks[
            [
                f"quadrant_radius_{speed}_mean",
                f"quadrant_radius_{speed}_mean_pred",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )
    print("r2, r2adj:")
    r2, r2_adj = calc_r2(
        df_tracks[f"quadrant_radius_{speed}_mean_pred"],
        df_tracks[f"quadrant_radius_{speed}_mean"],
        2,
    )
    print(r2, r2_adj)
    print()
```

### Plot examples

```python
df_tracks["Name Season"].unique()
```

```python
for name_season in df_tracks["Name Season"].unique():
    for speed in speeds:
        df_tracks.set_index("Name Season").loc[name_season].plot(
            x="valid_time",
            y=[
                f"quadrant_radius_{speed}_mean",
                f"quadrant_radius_{speed}_mean_pred",
            ],
        )
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/usa_only_wind_radii.parquet"
df_usa = stratus.load_parquet_from_blob(blob_name)
```

```python
for speed in speeds:
    df_usa = expand_quad_col(df_usa, f"usa_r{speed}")
```

```python
for speed in speeds:
    cols = [f"usa_r{speed}_{quad}" for quad in quads]
    df_usa[f"usa_r{speed}_mean"] = df_usa[cols].mean(axis=1)
```

```python
df_usa
```

```python
df_tracks["sid"] = df_tracks["Name Season"].replace(NAMESEASON2SID)
```

```python
df_tracks[df_tracks["sid"].isnull()]
```

```python
df_compare = df_tracks.merge(df_usa)
```

```python
for speed in speeds:
    for quad in quads + ["mean"]:
        df_compare.plot.scatter(
            x=f"quadrant_radius_{speed}_{quad}", y=f"usa_r{speed}_{quad}"
        )
```

```python
for speed in speeds:
    df_reg = df_compare[
        [
            "Name Season",
            "valid_time",
            f"quadrant_radius_{speed}_mean",
            f"usa_r{speed}_mean",
        ]
    ].dropna()
    X = df_reg[[f"usa_r{speed}_mean"]]
    y = df_reg[f"quadrant_radius_{speed}_mean"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    X_pred = df_compare[[f"usa_r{speed}_mean"]]
    X_pred = sm.add_constant(X_pred)
    df_compare[f"quadrant_radius_{speed}_mean_usa_pred"] = model.predict(
        X_pred
    )
    df_compare[f"quadrant_radius_{speed}_mean_usa_pred"] = df_compare[
        f"quadrant_radius_{speed}_mean_usa_pred"
    ].fillna(0)
```

```python
for speed in speeds:
    mean_abs_error_usa = (
        (
            df_compare[f"quadrant_radius_{speed}_mean_usa_pred"]
            - df_compare[f"quadrant_radius_{speed}_mean"]
        )
        .abs()
        .mean()
    )
    mean_abs_error_fms = (
        (
            df_compare[f"quadrant_radius_{speed}_mean_pred"]
            - df_compare[f"quadrant_radius_{speed}_mean"]
        )
        .abs()
        .mean()
    )
    print(mean_abs_error_usa, mean_abs_error_fms)
```

```python
for speed in speeds:
    print(speed)
    print("corr:")
    print(
        df_compare[
            [
                f"quadrant_radius_{speed}_mean",
                f"quadrant_radius_{speed}_mean_usa_pred",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )
    print("r2, r2adj:")
    r2, r2_adj = calc_r2(
        df_compare[f"quadrant_radius_{speed}_mean_usa_pred"],
        df_compare[f"quadrant_radius_{speed}_mean"],
        2,
    )
    print(r2, r2_adj)
    print()
```

```python
for name_season in df_compare["Name Season"].unique():
    for speed in speeds:
        df_compare.set_index("Name Season").loc[name_season].plot(
            x="valid_time",
            y=[
                f"quadrant_radius_{speed}_mean",
                f"quadrant_radius_{speed}_mean_usa_pred",
            ],
        )
```

```python

```
