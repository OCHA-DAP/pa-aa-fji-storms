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

# Ray check

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point


from src import utils, constants
```

```python
adm = utils.load_codab()
```

```python
adm_dissolve = adm.dissolve()
```

```python
adm_dissolve.plot()
```

```python
ls = LineString(
    [[360 - 179.7, -19.0], [360 - 179.9, -21.4], [360 - 179.2, -23.7]]
)
```

```python
# madeup to test trigger

ls = LineString([[178, -17.0], [360 - 179.9, -21.4], [360 - 179.2, -23.7]])
```

```python
ls
```

```python
ls.intersects(adm_dissolve.to_crs(constants.FJI_CRS).geometry)
```

```python
ls.distance(adm_dissolve.to_crs(constants.FJI_CRS).geometry)
```

```python
fig, ax = plt.subplots(dpi=300)

adm_dissolve.to_crs(constants.FJI_CRS).plot(ax=ax)
gdf = gpd.GeoDataFrame(geometry=[ls], crs=constants.FJI_CRS)

# If the ax is already defined, you can plot the GeoDataFrame on it
gdf.plot(ax=ax, color="blue", linewidth=2)
```

```python
gdf.to_crs(3832).distance(adm_dissolve.to_crs(3832)) / 1000
```

```python

```
