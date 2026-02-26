import argparse
import os
from io import BytesIO

import ocha_stratus as stratus
import pandas as pd
from dotenv import load_dotenv

from src.blob import PROJECT_PREFIX
from src.constants import EXP_THRESHOLD_64_KNOTS, FJI_CRS, LAU2, ROTUMA2
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
from src.exposure_calc import (
    calculate_multi_adm_exposure,
    calculate_single_adm_exposure,
)
from src.listmonk import (
    PROD_INFO_LIST_IDS,
    PROD_TRIGGER_LIST_IDS,
    TEST_LIST_IDS,
    create_and_send_campaign,
    upload_file,
)
from src.pipeline_utils import get_logger, load_boolean_env
from src.plotting import (
    fig_to_base64,
    plot_bubbles_and_swaths,
    plot_thermometer,
)

load_dotenv()

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=str, default="")
    return parser.parse_known_args()[0]


# load standard email switches
TEST_EMAIL = load_boolean_env("TEST_EMAIL", True)
SIMULATE_TRIGGER = load_boolean_env("SIMULATE_TRIGGER", False)
DRY_RUN = load_boolean_env("DRY_RUN", True)

TEST_FORECAST_BLOB_NAME = os.getenv("TEST_FORECAST_BLOB_NAME", "")

YASA_TEST_BLOB_NAME = f"{PROJECT_PREFIX}/raw/fms/TC Data/TC Yasa/20201216T000000Z_Official_Forecast_Track_2021_02F_YASA.csv"

INFO_LIST_IDS = TEST_LIST_IDS if TEST_EMAIL else PROD_INFO_LIST_IDS
TRIGGER_LIST_IDS = TEST_LIST_IDS if TEST_EMAIL else PROD_TRIGGER_LIST_IDS

# enable/disable tqdm progress bars for running locally
DISABLE_TQDM = load_boolean_env("DISABLE_TQDM", True)

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

    # Load forecast
    gdf_forecast = parse_fms_forecast(decoded_csv)
    gdf_forecast = gdf_forecast.rename(columns={"forecast_time": "valid_time"})

    row = gdf_forecast.iloc[0]
    issued_time = row["base_time"]
    cyclone_name = row["cyclone_name"]
    season = row["season"]
    logger.info(f"Issue time: {issued_time}, Cyclone name: {cyclone_name}")
    forecast_display_str = fji_time_str(issued_time)
    forecast_id = (
        f"{cyclone_name.lower().replace(' ', '_')}_{season}_fcast_"
        f"{issued_time:%Y%m%dT%H%MZ}"
    )

    logger.info(f"Forecast ID: {forecast_id=}")

    gdf_readiness = gdf_forecast[gdf_forecast["leadtime"] <= 120].copy()
    gdf_action = gdf_forecast[gdf_forecast["leadtime"] <= 72].copy()

    # Calculate single track buffers
    logger.info("Calculating forecast buffers.")
    gdf_buffers_readiness = calculate_fms_buffers_gdf(gdf_readiness)
    gdf_buffers_action = calculate_fms_buffers_gdf(gdf_action)

    # Calculate population exposure
    logger.info("Calculating population exposure.")
    adm3 = codab.load_codab_from_blob(admin_level=3).to_crs(FJI_CRS)
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

    email_base_name = "[TEST]_" if TEST_EMAIL or SIMULATE_TRIGGER else ""
    email_base_name = email_base_name + forecast_id

    # Send trigger emails
    if not DRY_RUN:
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
                list_ids=TRIGGER_LIST_IDS,
                body=body,
            )
        else:
            logger.info(
                "Readiness trigger conditions not met; not sending readiness email."
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
                list_ids=TRIGGER_LIST_IDS,
                body=body,
            )
        else:
            logger.info(
                "Action trigger conditions not met; not sending action email."
            )

    # Calculate perturbed tracks exposure at adm0 level
    (
        df_exp_shift,
        gdf_shift_buffers,
        gdf_shift_tracks,
    ) = calculate_shifted_exposures(
        gdf_readiness, da_wp_clip, disable_tqdm=DISABLE_TQDM
    )
    df_exp_shift = df_exp_shift.sort_values(
        [f"exp_{x}" for x in [64, 50, 34]], ascending=False
    )
    worst_row = df_exp_shift.iloc[0].copy()
    worst_row["level"] = "worst"
    best_row = df_exp_shift.iloc[-1].copy()
    best_row["level"] = "best"
    logger.info(
        "Best case exposure:\n%s",
        best_row.to_frame().T.to_string(index=False),
    )
    logger.info(
        "Worst case exposure:\n%s",
        worst_row.to_frame().T.to_string(index=False),
    )
    df_stats = load_historical_stats()

    any_64_knot_exposure = worst_row["exp_64"] > 0
    # Produce thermometer plot
    if any_64_knot_exposure:
        fig_thermometer, ax = plot_thermometer(
            main_value=readiness_exp,
            low_bound=best_row["exp_64"],
            high_bound=worst_row["exp_64"],
            df_stats=df_stats,
            cyclone_name=cyclone_name,
            forecast_display_str=forecast_display_str,
        )
        img_base64_thermometer = fig_to_base64(fig_thermometer)
    else:
        img_base64_thermometer = None

    # Calculate exposure at adm3 level for most likely
    logger.info("Calculating ADM3 level exposure for most likely track.")
    df_exp_adm3_mostlikely = calculate_multi_adm_exposure(
        gdf_buffers_readiness, da_wp_clip, adm3, disable_tqdm=DISABLE_TQDM
    )

    # Save adm3 exposure CSV file for attachment to email
    df_adm3_out = df_exp_adm3_mostlikely.pivot(
        columns="buffer_speed", index="ADM3_PCODE", values="pop_exposed"
    )
    df_adm3_out = df_adm3_out.rename(
        columns={x: f"exp_{x}_knot" for x in df_adm3_out.columns}
    )
    df_adm3_out = df_adm3_out.reset_index()
    df_adm3_out.columns.name = None
    cols = [
        "ADM1_PCODE",
        "ADM1_EN",
        "ADM2_PCODE",
        "ADM2_EN",
        "ADM3_PCODE",
        "ADM3_EN",
    ]
    df_adm3_out = adm3[cols].merge(df_adm3_out)
    df_adm3_out = df_adm3_out.sort_values(
        [f"exp_{x}_knot" for x in [64, 50, 34]], ascending=False
    )
    adm3_exp_filename = f"temp/{forecast_id}_adm3_exposure.csv"
    if not os.path.exists("temp/"):
        os.makedirs("temp/")
    df_adm3_out.to_csv(adm3_exp_filename, index=False)
    if not DRY_RUN:
        adm3_exp_id = upload_file(adm3_exp_filename)["id"]
        logger.info(
            f"ADM3 exposure file uploaded with media ID: {adm3_exp_id}"
        )
    else:
        adm3_exp_id = "DRY_RUN_MEDIA_ID"
        logger.info("DRY RUN: ADM3 exposure file not uploaded.")

    # Calculate best and worst case adm3 exposures
    worst_buffers = gdf_shift_buffers[
        gdf_shift_buffers["shift_deg"] == worst_row["shift_deg"]
    ]
    best_buffers = gdf_shift_buffers[
        gdf_shift_buffers["shift_deg"] == best_row["shift_deg"]
    ]
    logger.info("Calculating ADM3 level exposure for worst case tracks.")
    df_exp_adm3_worst = calculate_multi_adm_exposure(
        worst_buffers, da_wp_clip, adm3, disable_tqdm=DISABLE_TQDM
    )
    logger.info("Calculating ADM3 level exposure for best case tracks.")
    df_exp_adm3_best = calculate_multi_adm_exposure(
        best_buffers, da_wp_clip, adm3, disable_tqdm=DISABLE_TQDM
    )

    # Create bubble plot
    blob_name = (
        f"{PROJECT_PREFIX}/processed/plotting/adm3_simple_template.parquet"
    )
    adm3_simple_template = stratus.load_parquet_from_blob(blob_name)
    df_exp_adm3_best["limit"] = "best"
    df_exp_adm3_worst["limit"] = "worst"
    df_exp_adm3_mostlikely["limit"] = "middle"
    df_exp_adm3 = pd.concat(
        [
            df_exp_adm3_best,
            df_exp_adm3_mostlikely,
            df_exp_adm3_worst,
        ],
        ignore_index=True,
    )
    adm3_no_rotuma_lau = adm3[~(adm3["ADM2_PCODE"].isin([ROTUMA2, LAU2]))]
    fig, axs, bubbles_plot_filepath = plot_bubbles_and_swaths(
        gdf_mostlikely_buffers=gdf_buffers_readiness,
        gdf_worst_buffers=worst_buffers,
        gdf_best_buffers=best_buffers,
        gdf_adm3_swath_plot=adm3_no_rotuma_lau,
        df_adm3_template=adm3_simple_template,
        gdf_adm3=adm3,
        df_exp_adm3=df_exp_adm3,
        cyclone_name=cyclone_name,
        forecast_display_str=forecast_display_str,
        forecast_id=forecast_id,
        save_local=True,
    )
    if not DRY_RUN:
        bubbles_plot_id = upload_file(bubbles_plot_filepath)["id"]
        logger.info(f"Bubbles plot uploaded with media ID: {bubbles_plot_id}")
    else:
        bubbles_plot_id = "DRY_RUN_MEDIA_ID"
        logger.info("DRY RUN: Bubbles plot not uploaded.")

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
                "img_base64_thermometer": img_base64_thermometer,
                "forecast_id": forecast_id,
                "any_64_knot_exposure": any_64_knot_exposure,
            },
        )
        create_and_send_campaign(
            subject=subject,
            name=f"{email_base_name}_forecast_info",
            list_ids=INFO_LIST_IDS,
            body=body,
            media=[adm3_exp_id, bubbles_plot_id],
        )

    logger.info("Monitor forecast trigger pipeline completed.")
