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

# IMERG

IMERG plotting

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from dask.diagnostics import ProgressBar

from src.datasources import codab
from src.constants import *
```

```python
adm3 = codab.load_codab_from_blob(admin_level=3)
```

```python
adm3 = adm3.to_crs(FJI_CRS)
```

```python
adm3_no_rotuma_lau = adm3[~(adm3["ADM2_PCODE"].isin([ROTUMA2, LAU2]))]
```

```python
adm3_no_rotuma_lau.plot()
```

```python
query = """
SELECT *
FROM public.imerg
WHERE pcode = 'FJ'
ORDER BY valid_date ASC
"""
df_imerg = pd.read_sql(query, stratus.get_engine("prod"))
```

```python
df_imerg["valid_date"] = pd.to_datetime(df_imerg["valid_date"])
```

```python
dates = pd.date_range("2020-12-10", "2020-12-30")
```

```python
df_imerg_storm = df_imerg[df_imerg["valid_date"].isin(dates)]
```

```python
df_imerg_storm
```

```python
raster_dates = pd.date_range("2020-12-16", "2020-12-17")
```

```python
IMERG_BLOB_NAME = (
    "imerg/daily/late/v7/processed/imerg-daily-late-{valid_date_str}.tif"
)
```

```python
das = []
for d in raster_dates:
    blob_name = IMERG_BLOB_NAME.format(valid_date_str=d.date())
    da_in = stratus.open_blob_cog(
        blob_name, stage="prod", container_name="raster"
    )
    da_in["valid_date"] = d
    das.append(da_in)
```

```python
da_imerg = xr.concat(das, dim="valid_date")
```

```python
da_imerg = da_imerg.assign_coords({"x": (((da_imerg.x + 360) % 360))}).sortby(
    "x"
)
```

```python
da_imerg
```

```python
da_imerg_clip = da_imerg.rio.clip(adm3.geometry, all_touched=True)
```

```python
with ProgressBar():
    da_imerg_clip_computed = da_imerg_clip.compute()
```

```python
da_imerg_sum = da_imerg_clip_computed.sum(dim=["valid_date"]).squeeze(
    drop=True
)
```

```python
da_imerg_sum.where(da_imerg_sum >= 0).plot()
```

```python
def upsample_dataarray(
    da: xr.DataArray,
    resolution: float = 0.1,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.DataArray:
    new_lat = np.arange(
        da[lat_dim].min() - 1, da[lat_dim].max() + 1, resolution
    )
    new_lon = np.arange(
        da[lon_dim].min() - 1, da[lon_dim].max() + 1, resolution
    )
    return da.interp(
        coords={
            lat_dim: new_lat,
            lon_dim: new_lon,
        },
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )
```

```python
da_imerg_sum_up = upsample_dataarray(
    da_imerg_sum, lat_dim="y", lon_dim="x", resolution=0.01
)
```

```python
da_imerg_sum_up_clip = da_imerg_sum_up.rio.clip(adm3.geometry)
```

```python
levels = [25, 50, 100, 150, 200, 300, 400, 500, 750]
colors = [
    "lawngreen",
    "limegreen",
    "yellow",
    "gold",
    "darkorange",
    "red",
    "firebrick",
    "magenta",
    "darkmagenta",
]
cbar_kwargs = {
    "label": "Precipitation (mm)",  # Set label for the colorbar
    "shrink": 0.8,  # Shrink the colorbar to 80% of its default size
}
```

```python
fig, ax = plt.subplots(dpi=200, figsize=(10, 5))

adm3.boundary.plot(ax=ax, linewidth=0.5, color="k")
da_imerg_sum_up_clip.where(da_imerg_sum_up_clip >= 0).plot(
    ax=ax, levels=levels, colors=colors, extend="max", cbar_kwargs=cbar_kwargs
)
ax.axis("off")
ax.set_ylim(-19.5, -16)
ax.set_title(
    f"Cyclone Yasa: 2-day rainfall\nAverage over whole country: {198.548186:.0f} mm"
)

fig.savefig("temp/fji_obvs_test.png", dpi=200, bbox_inches="tight")
```

```python

```
