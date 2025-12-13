import geopandas as gpd
import pandas as pd
import xarray as xr
from rioxarray.exceptions import NoDataInBounds

from src.constants import FJI_CRS


def calculate_single_adm_exposure(
    gdf_buffers: gpd.GeoDataFrame, da_wp: xr.DataArray
) -> pd.DataFrame:
    # ensure correct crs
    gdf_buffers = gdf_buffers.to_crs(FJI_CRS)
    da_wp = da_wp.assign_coords({"x": ((da_wp.x + 360) % 360)}).sortby("x")
    dicts = []
    for _, row in gdf_buffers.iterrows():
        if not row.geometry:
            pop_exposed = 0
        else:
            try:
                da_wp_clip_buffer = da_wp.rio.clip([row.geometry])
                pop_exposed = int(da_wp_clip_buffer.sum())
            except NoDataInBounds:
                pop_exposed = 0
        dicts.append(
            {
                "issued_time": row["issued_time"],
                "name_season": row["name_season"],
                "buffer_speed": row["buffer_speed"],
                "pop_exposed": pop_exposed,
            }
        )
    df_exp = pd.DataFrame(dicts)
    return df_exp
