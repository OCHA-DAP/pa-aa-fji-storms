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

# CODAB

Getting bounding box for CODAB for info email plotting.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from src import utils, constants
```

```python
codab = utils.load_codab(level=0)
```

```python
codab["centroid"] = codab.centroid
codab = codab.to_crs(constants.FJI_CRS)
codab.plot()
```

```python
codab["centroid"].to_crs(constants.FJI_CRS)
```
