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

# USA tracks radius model fit
<!-- markdownlint-disable MD013 -->
Fitting model for wind radii to USA tracks

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

## Processing data

Just extracting USA values from IBTrACS - can all be skipped if file is already saved to blob

```python
# ds = lens.ibtracs.load_ibtracs(dataset="SP")
```

```python
# df_storm = lens.ibtracs.get_storms(ds)
```

```python
# gdf_tracks = lens.ibtracs.get_tracks(ds)
```

```python
# speeds = [34, 50, 64]
```

```python
quad_cols = [f"usa_r{x}" for x in speeds]
```

```python
ds
```

```python
other_cols = ["sid", "usa_lat", "usa_lon", "usa_wind"]
```

```python
var_cols = other_cols + quad_cols
```

```python
var_cols
```

```python
ds_subset = ds[var_cols]
df_select = ds_subset.to_dataframe().reset_index()
```

```python
df_ = lens.ibtracs.normalize_radii(df_select, radii_cols=quad_cols)
df_["valid_time"] = df_["time"].dt.round("min")
df_ = lens.ibtracs._convert_string_columns(df_, ["sid"])
```

```python
df_ = df_[var_cols + ["valid_time"]]
df_ = df_[df_.valid_time.notna()]
```

```python
df_
```

```python
df_usa = df_.copy()
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/usa_only_wind_radii.parquet"
```

```python
stratus.upload_parquet_to_blob(df_usa, blob_name)
```

```python
df_usa = stratus.load_parquet_from_blob(blob_name)
```

```python
query = """
SELECT *
FROM storms.ibtracs_storms
"""
with stratus.get_engine(stage="dev").connect() as conn:
    df_storms = pd.read_sql(query, conn)
```

```python
df_usa = df_usa.merge(df_storms)
```

```python
df_usa
```

```python
buffer_speeds = [34, 50, 64]
for buffer_speed in buffer_speeds:
    df_usa = expand_quad_col(df_usa, f"usa_r{buffer_speed}")
```

```python
quads = ["ne", "nw", "se", "sw"]
```

```python
df_usa[[f"usa_r34_{x}" for x in quads]].mean()
```

```python
df_usa.groupby("season")["usa_r34_nw"].count().plot()
```

```python
df_usa[df_usa["season"] >= 1980].groupby("season")["usa_r34_nw"].count()
```

```python
min_season = 2003
```

```python
df_usa_recent = df_usa[df_usa["season"] >= min_season]
df_usa_recent = df_usa_recent.dropna(subset="usa_wind")
```

```python
radius_cols = [
    f"usa_r{speed}_{quad}" for speed in buffer_speeds for quad in quads
]
```

```python
df_usa_recent[radius_cols] = df_usa_recent[radius_cols].fillna(0)
```

```python
for speed in buffer_speeds:
    cols = [f"usa_r{speed}_{quad}" for quad in quads]
    df_usa_recent[f"usa_r{speed}_mean"] = df_usa_recent[cols].mean(axis=1)
```

```python
df_usa_recent
```

```python
for speed in buffer_speeds:
    df_usa_recent.plot.scatter(
        x="usa_wind",
        y=f"usa_r{speed}_mean",
        alpha=0.1,
    )
```

```python
mean_quad_cols = [f"usa_r{speed}_mean" for speed in buffer_speeds]
```

```python
mean_quad_cols
```

```python
df_usa_recent["usa_lat_abs"] = df_usa_recent["usa_lat"].abs()
```

```python
for col in ["usa_wind", "usa_lat_abs"] + mean_quad_cols:
    df_usa_recent[f"{col}_log"] = np.log(df_usa_recent[col])
```

```python
dicts = []
for col in mean_quad_cols:
    print(col)
    for log_str in ["", "_log"]:
        target_col = f"{col}{log_str}"
        for var_cols in [
            [f"usa_wind{log_str}"],
            [f"usa_wind{log_str}", f"usa_lat_abs{log_str}"],
        ]:
            df_reg = df_usa_recent[[target_col] + var_cols].dropna()
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

```python
dicts_params = []
for col in mean_quad_cols:
    df_reg = df_usa_recent[
        ["sid", "valid_time", "usa_wind_log", "usa_lat_abs_log", f"{col}_log"]
    ].dropna()
    df_reg = df_reg[df_reg[f"{col}_log"] > 0]
    X = df_reg[["usa_wind_log", "usa_lat_abs_log"]]
    y = df_reg[f"{col}_log"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    dicts_params.append({"col": col, "params": model.params})
    X_pred = df_usa_recent[["usa_wind_log", "usa_lat_abs_log"]]
    X_pred = sm.add_constant(X_pred)
    df_usa_recent[f"{col}_log_pred"] = model.predict(X_pred)
```

```python
dicts_params
```

```python
log_pred_cols = [f"usa_r{speed}_mean_log_pred" for speed in buffer_speeds]
```

```python
log_pred_cols
```

```python
for speed in buffer_speeds:
    df_usa_recent[f"usa_r{speed}_mean_pred"] = np.exp(
        df_usa_recent[f"usa_r{speed}_mean_log_pred"]
    )
    df_usa_recent[f"usa_r{speed}_mean_pred"] = df_usa_recent.apply(
        lambda row: 0
        if row["usa_wind"] < speed
        else row[f"usa_r{speed}_mean_pred"],
        axis=1,
    )
```

```python
df_usa_recent
```

```python
for col in mean_quad_cols:
    df_usa_recent.plot.scatter(
        x=f"{col}",
        y=f"{col}_pred",
        alpha=0.05,
    )
```

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
for speed in buffer_speeds:
    print(speed)
    print("corr:")
    print(
        df_usa_recent[[f"usa_r{speed}_mean", f"usa_r{speed}_mean_pred"]]
        .corr()
        .iloc[0, 1]
    )
    print("r2, r2adj:")
    r2, r2_adj = calc_r2(
        df_usa_recent[f"usa_r{speed}_mean_pred"],
        df_usa_recent[f"usa_r{speed}_mean"],
        2,
    )
    print(r2, r2_adj)
    print()
```

```python
for sid in [WINSTON_SID, YASA_SID, HAROLD_SID]:
    for speed in buffer_speeds:
        df_usa_recent.set_index("sid").loc[sid].plot(
            x="valid_time", y=[f"usa_r{speed}_mean", f"usa_r{speed}_mean_pred"]
        )
```

```python

```

```python

```
