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

# Impact data

Process sub-national impact data from NDMO

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from dotenv import load_dotenv
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

import utils
```

```python
load_dotenv()
```

```python
impact = utils.load_geo_impact()
```

```python
impact["Event"].value_counts()
```

```python
cod.join(impact.groupby(ADM_ID).size().to_frame("events")).plot(
    column="events"
)
```

```python
utils.load_housing_impact()
```
