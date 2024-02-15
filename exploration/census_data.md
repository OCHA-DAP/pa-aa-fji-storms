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

# Fiji census data

Data from [ArcGIS hub](https://hub.arcgis.com/datasets/eaglegis::fiji-enumerationzones-1/explore?location=-17.799991%2C178.018270%2C9.88).

This seems to be from the [Fiji Census](https://fiji.popgis.spc.int/#c=home)

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from geocube.api.core import make_geocube
from shapely.geometry import box
from tqdm.notebook import tqdm

from src import utils
```

## Load census data

```python
census = utils.load_raw_census()
census = census.to_crs(3832)
census["area_km2"] = census.geometry.area / 1000 / 1000
census["pop_density_km2"] = census["TotalPop"] / census["area_km2"]
# calculate centroids using projected geometry (EPSG:3832)
census_centroids = census.copy()
census_centroids.geometry = census.geometry.centroid
census = census.to_crs(utils.FJI_CRS)
```

```python
# plot with maximum at 100 people / km2 to easily see urban areas
census.plot(column="pop_density_km2", legend=True, figsize=(10, 10), vmax=100)
```

## Method 1: `gpd.sjoin()`

```python
# create 0.1 deg grid

# note: the "grid" gdf can be arbitrary, I'm just creating a 0.1 deg grid
# to nicely match the grid we're already using, for demonstration purposes


def create_grid(xmin, ymin, xmax, ymax, width=0.1, height=0.1):
    rows = int(np.ceil((ymax - ymin) / height))
    cols = int(np.ceil((xmax - xmin) / width))
    grid = []
    for i in range(cols):
        for j in range(rows):
            grid.append(
                box(
                    xmin + i * width,
                    ymin + j * height,
                    xmin + (i + 1) * width,
                    ymin + (j + 1) * height,
                )
            )
    return gpd.GeoDataFrame(grid, columns=["geometry"])


xmin, ymin, xmax, ymax = census.total_bounds
grid = create_grid(xmin, ymin, xmax, ymax)
grid = grid.set_crs(utils.FJI_CRS)
grid = grid.to_crs(3832)
grid.plot()
```

```python
# join using centroids and "contains"

# if you join using the actual census shapes and with "overlaps", you end up
# double counting some census tracts
joined = gpd.sjoin(grid, census_centroids, how="left", predicate="contains")
```

```python
population_sum = (
    joined.groupby(joined.index).agg({"TotalPop": "sum"}).reset_index()
)

# Add the geometry back to the summarized DataFrame
population_grid = grid.merge(population_sum, left_index=True, right_on="index")
population_grid = population_grid.to_crs(utils.FJI_CRS)

population_grid[population_grid["TotalPop"] > 0].plot(
    column="TotalPop", legend=True
)
```

```python
# check that total populations match
# they do, no one got lost
print(population_grid["TotalPop"].sum())
print(census["TotalPop"].sum())
```

## Method 2: raster with `geocube`

Note: I don't think this will work.
Geocube does quick rasterization of `gdf` but
I'm not sure if can easily sum the values in the
polygons like with `gpd.sjoin()`

```python
resolution = (
    -0.001,
    0.001,
)  # Negative value for longitude to indicate westward direction

# Use geocube to rasterize the population data
cube = make_geocube(
    vector_data=census, measurements=["pop_density_km2"], resolution=resolution
)
```

```python
# find smallest census tract
xmin, ymin, xmax, ymax = census[
    census["area_km2"] == census["area_km2"].min()
].geometry.total_bounds
```

```python
# check that resolution is fine enough to capture smallest census tract
# looks like it doesn't, so would have to go to higher resolution
# this would take too long, not worth pursuing
fig, ax = plt.subplots(figsize=(15, 15))
census.boundary.plot(ax=ax, linewidth=0.05, color="k")
cube["pop_density_km2"].plot(ax=ax, vmax=200)

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
```

```python

```
