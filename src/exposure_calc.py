import geopandas as gpd
import pandas as pd
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

from src.constants import FJI_CRS


def calculate_single_adm_exposure(
    gdf_buffers: gpd.GeoDataFrame, da_wp: xr.DataArray
) -> pd.DataFrame:
    # ensure correct CRS
    gdf_buffers = gdf_buffers.to_crs(FJI_CRS)
    da_wp = da_wp.assign_coords({"x": ((da_wp.x + 360) % 360)}).sortby("x")

    records = []
    for _, row in gdf_buffers.iterrows():
        row_data = row.drop(labels="geometry").to_dict()

        if not row.geometry or row.geometry.is_empty:
            pop_exposed = 0
        else:
            try:
                da_wp_clip_buffer = da_wp.rio.clip([row.geometry])
                pop_exposed = int(da_wp_clip_buffer.sum())
            except NoDataInBounds:
                pop_exposed = 0

        row_data["pop_exposed"] = pop_exposed
        records.append(row_data)

    return pd.DataFrame(records)
