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

# ECMWF tracks

Loading ECMWF tracks to get longer leadtime

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import re

import requests
from dateutil import rrule
from dotenv import load_dotenv
import getpass
import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px
from sklearn.linear_model import LogisticRegression

import utils
```

```python
# if needed, download hindcasts
# utils.download_ecmwf_hindcasts()
```

```python
# if needed, process hindcasts
# utils.process_ecmwf_hindcasts()
```

```python
# if needed, process best tracks
# utils.process_ecmwf_besttrack_hindcasts()
```

```python
utils.process_ecmwf_besttrack_hindcasts()
```

```python
# plot by name
name = "winston"
df = pd.read_csv(ECMWF_PROCESSED / f"csv/{name}_all.csv")

# plot forecast
px.line(
    df[df["mtype"] == "forecast"],
    x="lon",
    y="lat",
    color="forecast_time",
    hover_data="lead_time",
).show()

# plot ensembles
forecast_time = df["forecast_time"].unique()[0]
px.line(
    df[
        (df["mtype"] == "ensembleforecast")
        & (df["forecast_time"] == forecast_time)
    ],
    x="lon",
    y="lat",
    color="ensemble",
    hover_data="lead_time",
)
```

```python
# check historical triggers
buffer = utils.load_buffer()
buffer.plot()
```

```python
df = fms.dropna()
X = df[["Wind (Knots)", "Pressure"]].values
y = df["Category numeric"].values
model = LogisticRegression(random_state=0).fit(X, y)
```

```python
df["pred"] = model.predict(df[["Wind (Knots)", "Pressure"]]).astype(str)
```

```python
px.scatter(df, x="Wind (Knots)", y="Pressure", color="pred")
```

```python
fms["Wind_prev"] = fms["Wind (Knots)"].shift()
px.scatter(fms, x="Wind (Knots)", y="Pressure", color="Category")
```

```python
fms
```
