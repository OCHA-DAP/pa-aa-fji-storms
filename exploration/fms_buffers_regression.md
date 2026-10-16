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

# Wind buffers - FMS approximation
<!-- markdownlint-disable MD013 -->
Estimating wind buffers of FMS tracks, using the JTWC wind radii, since FMS doesn't provide wind radii.

Trying first using regression.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.constants import *
from src.blob import PROJECT_PREFIX
from src.datasources.ibtracs import expand_quad_col
```

```python
query = """
SELECT *
FROM storms.ibtracs_storms
WHERE genesis_basin = 'SP'
"""
with stratus.get_engine(stage="dev").connect() as conn:
    df_storms = pd.read_sql(query, conn)
```

```python
query = """
SELECT *
FROM storms.ibtracs_tracks_geo
WHERE basin = 'SP'
"""
with stratus.get_engine(stage="dev").connect() as conn:
    gdf_tracks = gpd.read_postgis(query, conn, geom_col="geometry")
```

```python
gdf_tracks = gdf_tracks.merge(df_storms)
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/ibtracs/ibtracs_with_usa_radii.parquet"
)
df_tracks_usa = stratus.load_parquet_from_blob(blob_name)
```

```python
df_tracks_usa
```

```python
buffer_speeds = [34, 50, 64]
for buffer_speed in buffer_speeds:
    df_tracks_usa = expand_quad_col(
        df_tracks_usa, f"quadrant_radius_{buffer_speed}"
    )
```

```python
gdf_tracks_best = gdf_tracks[~gdf_tracks["provisional"]]
```

```python
gdf_tracks_best
```

```python
sorted(gdf_tracks_best["season"].unique())
```

```python
RADIUS_COL = "quadrant_radius_{speed}_{quad}"
```

```python
quads = ["ne", "nw", "se", "sw"]
```

```python
df_tracks_merge = gdf_tracks_best.merge(
    df_tracks_usa, on=["sid", "valid_time"], suffixes=("_wmo", "_usa")
)
```

```python
df_tracks_merge.columns
```

```python
for speed in buffer_speeds:
    cols = [RADIUS_COL.format(speed=speed, quad=x) for x in quads]
    df_tracks_merge[
        RADIUS_COL.format(speed=speed, quad="mean")
    ] = df_tracks_merge[cols].mean(axis=1)
```

```python
df_tracks_merge
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
col_sets = (
    ("wind", ["wind_speed_wmo"]),
    ("wind_pres", ["wind_speed_wmo", "pressure_wmo"]),
)
dicts = []
df_reg = df_tracks_merge.dropna(subset=["wind_speed_wmo", "pressure_wmo"])
df_reg = df_reg[df_reg["wind_speed_wmo"] >= 0]
conversion_factor = 1
for speed in buffer_speeds:
    cutoff_speed = speed * conversion_factor
    for quad in quads + ["mean"]:
        target_col = RADIUS_COL.format(speed=speed, quad=quad)
        df_reg[target_col] = df_reg[target_col].fillna(0)
        for cutoff in [True, False]:
            if cutoff:
                df_model = df_reg[df_reg["wind_speed_wmo"] >= cutoff_speed]
            else:
                df_model = df_reg
            for col_set in col_sets:
                var_cols = col_set[1]
                X = df_model[var_cols]
                y = df_model[target_col]
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()

                X_pred = df_reg[var_cols]
                X_pred = sm.add_constant(X_pred)
                df_reg["pred"] = model.predict(X_pred)
                if cutoff:
                    df_reg["pred"] = df_reg.apply(
                        lambda row: 0
                        if row["wind_speed_wmo"] <= cutoff
                        else row["pred"],
                        axis=1,
                    )
                df_reg["pred"] = df_reg["pred"].apply(
                    lambda x: 0 if x < 0 else x
                )
                r2, r2_adj = calc_r2(
                    df_reg["pred"], df_reg[target_col], k=len(var_cols)
                )
                dicts.append(
                    {
                        "col_set": col_set[0],
                        "cutoff": cutoff,
                        "speed": speed,
                        "quad": quad,
                        "r2": r2,
                        "r2_adj": r2_adj,
                        "summary": model.summary(),
                        "params": model.params,
                    }
                )
        # for plot_col in ["wind_speed_wmo", "pressure_wmo"]:
        #     df_reg.plot.scatter(
        #         x=plot_col,
        #         y=target_col,
        #         alpha=0.1,
        #     )
```

```python
df_reg[["wind_speed_wmo", "pressure_wmo"]].corr()
```

```python
df_r2 = pd.DataFrame(dicts)
```

```python
df_r2[df_r2["quad"] == "mean"]
```

```python
df_r2[
    (df_r2["col_set"] == "wind_pres")
    & df_r2["cutoff"]
    & (df_r2["quad"] == "mean")
]["summary"].apply(display)
```

```python
df_r2.iloc[3]["summary"]
```

```python
r2
```

```python
WINSTON_SID
```

```python

```

```python
df_tracks_merge["season"].unique()
```

```python
df_tracks_merge["wind_diff"] = (
    df_tracks_merge["wind_speed_wmo"] - df_tracks_merge["wind_speed_usa"]
)
```

```python
df_tracks_merge["wind_diff"].hist()
```

```python
df_tracks_merge[df_tracks_merge["wind_diff"].abs() > 10][
    ["sid", "valid_time", "wind_speed_wmo", "wind_speed_usa", "wind_diff"]
]
```

```python
df_tracks_merge.set_index("sid").loc[WINSTON_SID][
    ["wind_speed_wmo", "wind_speed_usa"]
]
```

```python

```
