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

# Distances

Calculating the distance of the forecasts and actual path to admin levels.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src import utils
```

## Load data

```python
triggers = utils.load_historical_triggers()
# only take actual triggered storms with ECMWF forecasts
triggers = triggers[triggers[["fms_actual_trig", "ec_5day_trig"]].all(axis=1)]
triggers = triggers.set_index("nameyear")
fms = utils.load_cyclonetracks()
fms = fms[fms["nameyear"].isin(triggers.index)]
ecmwf = utils.load_ecmwf_besttrack_hindcasts()
ecmwf = ecmwf[ecmwf["nameyear"].isin(triggers.index)]

cod2 = utils.load_codab(level=2).to_crs(3832)
```

```python
cod2
```

## Calculate distance

```python
# calculate distance at readiness, activation, actual
cols = ["ADM2_PCODE", "ADM2_NAME", "ADM1_PCODE", "ADM1_NAME", "geometry"]
distances = cod2[cols].copy()

cols = ["ec_5day_date", "ec_3day_date", "fms_actual_date"]
labels = ["readiness", "activation", "actual"]
for nameyear, row in triggers.iterrows():
    for col, label in zip(cols, labels):
        if label == "actual":
            track = fms[fms["nameyear"] == nameyear]
            track = track.sort_values("datetime")
        else:
            track = ecmwf[
                (ecmwf["nameyear"] == nameyear)
                & (ecmwf["forecast_time"] == row[col])
            ]
            track = track.sort_values("time")
        track = track.to_crs(3832)
        track = LineString([(p.x, p.y) for p in track.geometry])
        distances[f"{nameyear}_{label}_distance"] = np.round(
            track.distance(cod2.geometry) / 1000
        ).astype(int)
distances = distances.drop(columns="geometry")
distances.to_csv(utils.PROC_PATH / "trigger_adm2_distances.csv", index=False)
```

```python
display(distances)
```

```python
# calculate "hits" i.e. districts within 50km
cutoffs = [50, 100]

for cutoff in cutoffs:
    hits = distances.copy()
    hits = hits.applymap(lambda x: x <= cutoff if isinstance(x, int) else x)

    # make nicer spreadsheet output

    cols = ["ADM2_PCODE", "ADM1_PCODE"]
    hits_out = hits.drop(columns=cols)
    hits_out = hits[["ADM1_NAME", *hits_out.columns.drop("ADM1_NAME")]]
    cols = ["ADM1_NAME", "ADM2_NAME"]
    hits_out = hits_out.sort_values(cols)
    hits_out = hits_out.set_index(cols)
    display(hits_out)

    with pd.ExcelWriter(
        utils.PROC_PATH / f"trigger_adm2_within_{cutoff}km.xlsx"
    ) as writer:
        for nameyear in triggers.index:
            dff = pd.DataFrame()
            dff[[label for label in labels]] = hits_out[
                [f"{nameyear}_{label}_distance" for label in labels]
            ]
            dff = dff.reset_index()
            dff.to_excel(writer, sheet_name=nameyear, index=False)
```

```python

```
