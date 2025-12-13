import argparse
import os
from io import BytesIO

import ocha_stratus as stratus
from pipeline_utils import get_logger, load_boolean_env

from src.blob import PROJECT_PREFIX
from src.constants import EXP_THRESHOLD_64_KNOTS, FJI_CRS
from src.datasources import codab, worldpop
from src.datasources.fms import (
    calculate_fms_buffers_gdf,
    decode_b64_string,
    get_forecast_display_str,
    get_forecast_id,
    parse_fms_forecast,
)
from src.exposure_calc import calculate_single_adm_exposure
from src.listmonk import TRISTAN_ONLY_LIST_ID, create_and_send_campaign

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=str, default="")
    return parser.parse_args()


TEST_EMAIL = load_boolean_env("TEST_EMAIL", True)
SIMULATE_TRIGGER = load_boolean_env("SIMULATE_TRIGGER", False)
DRY_RUN = load_boolean_env("DRY_RUN", True)

TEST_FORECAST_BLOB_NAME = os.getenv("TEST_FORECAST_BLOB_NAME", "")

YASA_TEST_BLOB_NAME = f"{PROJECT_PREFIX}/raw/fms/TC Data/TC Yasa/20201216T000000Z_Official_Forecast_Track_2021_02F_YASA.csv"

LIST_IDS = [TRISTAN_ONLY_LIST_ID] if TEST_EMAIL else [TRISTAN_ONLY_LIST_ID]

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
    elif TEST_FORECAST_BLOB_NAME:
        decoded_csv = BytesIO(stratus.load_blob_data(TEST_FORECAST_BLOB_NAME))
    else:
        logger.info("No forecast CSV provided and no test blob set; exiting.")
        exit(0)

    gdf_forecast = parse_fms_forecast(decoded_csv)
    gdf_forecast = gdf_forecast.rename(columns={"forecast_time": "valid_time"})

    forecast_id = get_forecast_id(gdf_forecast)
    logger.info(f"Forecast ID: {forecast_id=}")

    forecast_display_str = get_forecast_display_str(gdf_forecast)
    logger.info(f"Forecast Display String: {forecast_display_str=}")

    gdf_readiness = gdf_forecast[gdf_forecast["leadtime"] <= 120].copy()
    gdf_action = gdf_forecast[gdf_forecast["leadtime"] <= 72].copy()

    # Calculate single track buffers
    logger.info("Calculating forecast buffers.")
    gdf_buffers_readiness = calculate_fms_buffers_gdf(gdf_readiness)
    gdf_buffers_action = calculate_fms_buffers_gdf(gdf_action)

    # Calculate population exposure
    logger.info("Calculating population exposure.")
    adm3 = codab.load_codab_from_blob(admin_level=0).to_crs(FJI_CRS)
    da_wp = worldpop.load_worldpop_from_blob()
    da_wp = da_wp.assign_coords({"x": ((da_wp.x + 360) % 360)}).sortby("x")
    da_wp_clip = da_wp.rio.clip(adm3.geometry)

    df_exp_readiness = calculate_single_adm_exposure(
        gdf_buffers_readiness, da_wp_clip
    )
    df_exp_action = calculate_single_adm_exposure(
        gdf_buffers_action, da_wp_clip
    )

    # Check trigger conditions
    readiness_exp = df_exp_readiness[
        df_exp_readiness["buffer_speed"] == 64
    ].iloc[0]["pop_exposed"]
    action_exp = df_exp_action[df_exp_action["buffer_speed"] == 64].iloc[0][
        "pop_exposed"
    ]
    logger.info(
        f"Readiness exposure: {readiness_exp}, Action exposure: {action_exp}"
    )
    trigger_readiness = readiness_exp >= EXP_THRESHOLD_64_KNOTS
    trigger_action = action_exp >= EXP_THRESHOLD_64_KNOTS

    email_base_name = "[TEST]" if TEST_EMAIL else ""
    email_base_name = email_base_name + forecast_id

    # Send trigger emails
    if not DRY_RUN:
        if trigger_readiness:
            logger.info("Readiness trigger condition met; sending email.")
            subject_readiness = (
                f"Fiji AA: Readiness ACTIVATED by {forecast_display_str}"
            )
            create_and_send_campaign(
                subject=subject_readiness,
                name=f"{email_base_name} Readiness Trigger",
                list_ids=LIST_IDS,
            )
        else:
            logger.info("Readiness trigger condition not met; no email sent.")

    # Calculate uncertainty buffers

    logger.info("Monitor forecast trigger pipeline completed.")
