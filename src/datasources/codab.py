import ocha_stratus as stratus
import requests

from src.blob import PROJECT_PREFIX
from src.constants import ISO3

FIELDMAPS_BASE_URL = "https://data.fieldmaps.io/cod/originals/{iso3}.shp.zip"


def get_blob_name(iso3: str = ISO3):
    iso3 = iso3.lower()
    return f"{PROJECT_PREFIX}/raw/codab/{iso3}.shp.zip"


def download_codab_to_blob(clobber: bool = False, iso3: str = ISO3):
    blob_name = get_blob_name(iso3=iso3)
    if not clobber and blob_name in stratus.list_container_blobs(
        name_starts_with=f"{PROJECT_PREFIX}/raw/codab/"
    ):
        print(f"{blob_name} already exists in blob storage")
        return
    url = FIELDMAPS_BASE_URL.format(iso3=iso3)
    response = requests.get(url)
    response.raise_for_status()
    stratus.upload_blob_data(response.content, blob_name, stage="dev")


def load_codab_from_blob(admin_level: int = 0, iso3: str = ISO3):
    shapefile = f"{iso3}_adm{admin_level}.shp"
    gdf = stratus.load_shp_from_blob(
        blob_name=get_blob_name(iso3),
        shapefile=shapefile,
        stage="dev",
    )
    return gdf


def load_buffer():
    blob_name = f"{PROJECT_PREFIX}/processed/buffer/fji_250km_buffer.zip"
    return stratus.load_shp_from_blob(blob_name=blob_name)
