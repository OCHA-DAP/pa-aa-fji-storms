import numpy as np
import ocha_stratus as stratus
import requests

from src.blob import PROJECT_PREFIX
from src.constants import ISO3

WORLDPOP_BASE_URL = (
    "https://data.worldpop.org/GIS/Population/"
    "Global_2000_2020_1km_UNadj/2020/{iso3_upper}/"
    "{iso3}_ppp_2020_1km_Aggregated_UNadj.tif"
)


def get_blob_name(iso3: str):
    iso3 = iso3.lower()
    return (
        f"{PROJECT_PREFIX}/raw/worldpop/"
        f"{iso3}_ppp_2020_1km_Aggregated_UNadj.tif"
    )


def download_worldpop_to_blob(iso3: str = ISO3, clobber: bool = False):
    iso3 = iso3.lower()
    blob_name = get_blob_name(iso3)
    if not clobber and blob_name in stratus.list_container_blobs(
        name_starts_with=f"{PROJECT_PREFIX}/raw/worldpop/", stage="dev"
    ):
        print(f"{blob_name} already exists in blob storage")
        return
    url = WORLDPOP_BASE_URL.format(iso3_upper=iso3.upper(), iso3=iso3)
    response = requests.get(url)
    response.raise_for_status()
    stratus.upload_blob_data(response.content, blob_name, stage="dev")


def load_worldpop_from_blob(iso3: str = ISO3):
    iso3 = iso3.lower()
    blob_name = get_blob_name(iso3)
    da = stratus.open_blob_cog(blob_name, stage="dev")
    da = da.where(da != da.attrs["_FillValue"]).squeeze(drop=True)
    da.attrs["_FillValue"] = np.nan
    return da
