import argparse
from io import BytesIO

import geopandas as gpd
import ocha_stratus as stratus
from pipeline_utils import get_logger, load_boolean_env

from src.blob import PROJECT_PREFIX
from src.datasources.fms import (
    calculate_fms_buffers,
    decode_b64_string,
    parse_fms_forecast,
)
from src.listmonk import create_campaign, send_campaign

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=str, default="")
    return parser.parse_args()


TEST_EMAIL = load_boolean_env("TEST_EMAIL", True)
SIMULATE_TRIGGER = load_boolean_env("FORCE_TRIGGER", False)
DRY_RUN = load_boolean_env("DRY_RUN", True)

YASA_TEST_BLOB_NAME = f"{PROJECT_PREFIX}/raw/fms/TC Data/TC Yasa/20201216T000000Z_Official_Forecast_Track_2021_02F_YASA.csv"

if __name__ == "__main__":
    # Init
    logger.info("Monitor forecast trigger pipeline started.")
    logger.info(
        f"Env vars set to: {TEST_EMAIL=}, {SIMULATE_TRIGGER=}, {DRY_RUN=}"
    )
    args = parse_args()
    csv_str = args.csv
    if csv_str:
        decoded_csv = decode_b64_string(csv_str)
    elif SIMULATE_TRIGGER:
        decoded_csv = BytesIO(stratus.load_blob_data(YASA_TEST_BLOB_NAME))
    else:
        logger.info(
            "No CSV input provided and FORCE_TRIGGER is False. Exiting."
        )
        exit(0)
    gdf_forecast = parse_fms_forecast(decoded_csv)
    gdf_forecast = gdf_forecast.rename(columns={"forecast_time": "valid_time"})
    geoms_in, dicts_in = calculate_fms_buffers(gdf_forecast)
    gdf_buffers = gpd.GeoDataFrame(data=dicts_in, geometry=geoms_in, crs=3832)
    print(gdf_buffers.head())

    # Calculate single track buffers

    # Calculate population exposure

    # Send trigger email
    if not DRY_RUN:
        campaign_id = create_campaign()
        logger.info(f"Created campaign with ID: {campaign_id}")
        send_campaign(campaign_id)

    # Calculate uncertainty buffers

    logger.info("Monitor forecast trigger pipeline completed.")
