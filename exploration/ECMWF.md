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
import numpy as np
from tqdm.auto import tqdm
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import geopandas as gpd
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = "notebook"

from src import utils
from src.constants import FJI_CRS
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
utils.process_ecmwf_besttrack_hindcasts()
```

```python
fms = utils.load_cyclonetracks()
nameyears = fms["nameyear"].unique()
forecast = utils.load_ecmwf_besttrack_hindcasts()
forecast.plot()
forecast = forecast[forecast["nameyear"].isin(nameyears)]
forecast.plot()
```

```python
forecast.plot()
```

```python
forecast
```

```python
trigger_zone = utils.load_buffer()
trigger_zone = trigger_zone.to_crs(FJI_CRS)
```

```python
forecast.plot()
```

```python
# plot by name
name = "winston"
df = forecast[forecast["name"] == name]
df = df.sort_values(["forecast_time", "time"])
display(df)
# plot all forecasts
fig = px.line(
    df,
    x="lon",
    y="lat",
    color="forecast_time",
    hover_data="lead_time",
)
fig.show()

# plot only actual track

fig = px.line(
    df[df["lead_time"] == 0],
    x="lon",
    y="lat",
    hover_data="lead_time",
)
fms_f = fms[fms["nameyear"] == "winston2016"]
fig.add_trace(go.Scatter(x=fms_f["Longitude"], y=fms_f["Latitude"]))
fig.show()
```

```python
forecast["speed"].max()
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
