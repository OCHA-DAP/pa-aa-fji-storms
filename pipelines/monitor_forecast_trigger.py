import argparse
import os
from io import BytesIO

import ocha_stratus as stratus
import pandas as pd
from dotenv import load_dotenv

from src.blob import PROJECT_PREFIX
from src.constants import EXP_THRESHOLD_64_KNOTS, FJI_CRS
from src.datasources import codab, worldpop
from src.datasources.fms import (
    calculate_fms_buffers_gdf,
    calculate_shifted_exposures,
    decode_b64_string,
    fji_time_str,
    load_historical_stats,
    parse_fms_forecast,
)
from src.email.content import render_template
from src.exposure_calc import calculate_single_adm_exposure
from src.listmonk import TRISTAN_ONLY_LIST_ID, create_and_send_campaign
from src.pipeline_utils import get_logger, load_boolean_env

load_dotenv()

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=str, default="")
    return parser.parse_known_args()[0]


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

    row = gdf_forecast.iloc[0]
    issued_time = row["base_time"]
    cyclone_name = row["cyclone_name"]
    season = row["season"]
    logger.info(f"Issue time: {issued_time}, Cyclone name: {cyclone_name}")
    forecast_display_str = fji_time_str(issued_time)
    forecast_id = f"{cyclone_name.lower().replace(' ', '_')}_{season}_{issued_time:%Y%m%dT%H%MZ}"

    logger.info(f"Forecast ID: {forecast_id=}")

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
        "Readiness exposure:\n%s", df_exp_readiness.to_string(index=False)
    )
    logger.info("Action exposure:\n%s", df_exp_action.to_string(index=False))

    trigger_readiness = readiness_exp >= EXP_THRESHOLD_64_KNOTS
    trigger_action = action_exp >= EXP_THRESHOLD_64_KNOTS

    email_base_name = "[TEST]_" if TEST_EMAIL else ""
    email_base_name = email_base_name + forecast_id

    # Send trigger emails
    if not DRY_RUN or False:
        if trigger_readiness:
            logger.info("Readiness trigger condition met; sending email.")
            subject = f"Anticipatory action Fiji: Cyclone {cyclone_name} readiness trigger ACTIVATED"
            body = render_template(
                "readiness.html",
                {
                    "cyclone_name": cyclone_name,
                    "forecast_display_str": forecast_display_str,
                },
            )
            create_and_send_campaign(
                subject=subject,
                name=f"{email_base_name}_readiness",
                list_ids=LIST_IDS,
                body=body,
            )
        if trigger_action:
            logger.info("Action trigger condition met; sending email.")
            subject = f"Anticipatory action Fiji: Cyclone {cyclone_name} action trigger ACTIVATED"
            body = render_template(
                "action.html",
                {
                    "cyclone_name": cyclone_name,
                    "forecast_display_str": forecast_display_str,
                },
            )
            create_and_send_campaign(
                subject=subject,
                name=f"{email_base_name}_action",
                list_ids=LIST_IDS,
                body=body,
            )
        else:
            logger.info("Trigger conditions not met; no emails sent.")

    # Calculate uncertainty exposure
    (
        df_exp_shift,
        gdf_shift_buffers,
        gdf_shift_tracks,
    ) = calculate_shifted_exposures(
        gdf_readiness, da_wp_clip, disable_tqdm=False
    )
    df_exp_shift = df_exp_shift.sort_values(
        [f"exp_{x}" for x in [64, 50, 34]], ascending=False
    )
    worst_row = df_exp_shift.iloc[0].copy()
    worst_row["level"] = "worst"
    best_row = df_exp_shift.iloc[-1].copy()
    best_row["level"] = "best"
    df_exp_shift_summary = pd.DataFrame([worst_row, best_row])
    logger.info(
        "Exposure under uncertainty:\n%s",
        df_exp_shift_summary.to_string(index=False),
    )
    df_stats = load_historical_stats()

    # Send info email
    if trigger_action:
        activation_subject = "(ACTION TRIGGER ACTIVATED)"
    elif trigger_readiness:
        activation_subject = "(READINESS TRIGGER ACTIVATED)"
    else:
        activation_subject = "(NOT ACTIVATED)"
    if not DRY_RUN:
        logger.info("Sending info email.")
        subject = f"Anticipatory action Fiji: Cyclone {cyclone_name} forecast information {activation_subject}"
        body = render_template(
            "informational.html",
            {
                "cyclone_name": cyclone_name,
                "forecast_display_str": forecast_display_str,
                "readiness_str": "ACTIVATED"
                if trigger_readiness
                else "NOT ACTIVATED",
                "action_str": "ACTIVATED"
                if trigger_action
                else "NOT ACTIVATED",
                "readiness_exp": f"{readiness_exp:,.0f}",
                "action_exp": f"{action_exp:,.0f}",
            },
        )
        create_and_send_campaign(
            subject=subject,
            name=f"{email_base_name}_forecast_info",
            list_ids=LIST_IDS,
            body=body,
        )

    logger.info("Monitor forecast trigger pipeline completed.")
