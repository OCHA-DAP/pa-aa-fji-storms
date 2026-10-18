"""Synthetic IMERG-like daily rainfall for the exercise storms.

Rain falls in a blob around the storm centre (plus a smooth random texture so
the map doesn't look drawn with a compass). The country average follows the
observational-trigger method in exploration/rain_plot.md: 2-day sum on the
0.1-degree grid, upsampled to 0.01 degrees, clipped to the admin boundaries,
then averaged.
"""

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from src.constants import FJI_CRS
from src.datasources.worldpop import upsample_dataarray

GRID_X = np.round(np.arange(175.0, 184.0, 0.1), 2)
GRID_Y = np.round(np.arange(-22.5, -13.5, 0.1), 2)


def _empty() -> xr.DataArray:
    da = xr.DataArray(
        np.zeros((len(GRID_Y), len(GRID_X))),
        coords={"y": GRID_Y, "x": GRID_X},
        dims=("y", "x"),
    )
    return da.rio.write_crs(FJI_CRS)


def daily_rain(centres, peak_mm: float, sigma_km: float = 130, seed=0):
    """Rain for one day: centres = storm positions (lat, lon) during the day."""
    xx, yy = np.meshgrid(GRID_X, GRID_Y)
    field = np.zeros_like(xx)
    for lat, lon in centres:
        dx = (xx - lon) * 111.32 * np.cos(np.deg2rad(lat))
        dy = (yy - lat) * 110.57
        field += np.exp(-(dx**2 + dy**2) / (2 * sigma_km**2))
    field /= len(centres)
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.normal(size=field.shape), 3)
    texture = 1 + 0.9 * texture / np.abs(texture).max()
    da = _empty()
    da.values = np.clip(peak_mm * field * texture, 0, None)
    return da


def national_mean(da: xr.DataArray, adm) -> float:
    up = upsample_dataarray(da, resolution=0.01)
    up = up.rio.write_crs(FJI_CRS)
    return float(up.rio.clip(adm.geometry).mean())


def land_only(da: xr.DataArray, adm) -> xr.DataArray:
    """For the map: keep pixels touching land, like the email plots."""
    return da.rio.clip(adm.geometry, all_touched=True, drop=False)


def scale_to(da: xr.DataArray, adm, target_mm: float) -> xr.DataArray:
    return da * (target_mm / national_mean(da, adm))
