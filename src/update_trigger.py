import argparse
import base64
import os
import smtplib
import ssl
import zipfile
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from io import StringIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from ochanticipy.utils.hdx_api import load_resource_from_hdx
from shapely.geometry import LineString

FJI_CRS = "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"


def load_fms_forecast(path: Path | StringIO) -> pd.DataFrame:
    """
    Loads FMS raw forecast
    Parameters
    ----------
    path: Path | StringIO
        Path to raw forecast CSV. Path can be a StringIO
        (so CSV can be passed as an encoded string from Power Automate)


    Returns
    -------
    DataFrame of processed forecast
    """
    df_date = pd.read_csv(path, header=None, nrows=3)
    date_str = df_date.iloc[0, 1].removeprefix("baseTime=")
    base_time = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    cyclone_name = (
        df_date.iloc[2, 0].removeprefix("# CycloneName=").capitalize()
    )
    if isinstance(path, StringIO):
        path.seek(0)
    df_data = pd.read_csv(
        path,
        skiprows=range(6),
    )
    df_data = df_data.drop([0])
    df_data = df_data.rename(
        columns={"Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']": "forecast_time"}
    )
    df_data["forecast_time"] = pd.to_datetime(
        df_data["forecast_time"]
    ).dt.tz_localize(None)
    df_data["cyclone_name"] = cyclone_name
    df_data["base_time"] = base_time
    df_data["season"] = datetime_to_season(base_time)
    df_data["Name Season"] = df_data["cyclone_name"] + " " + df_data["season"]
    df_data["leadtime"] = df_data["forecast_time"] - df_data["base_time"]
    df_data["leadtime"] = (
        df_data["leadtime"].dt.days * 24
        + df_data["leadtime"].dt.seconds / 3600
    ).astype(int)
    return df_data


def datetime_to_season(date):
    # July 1 (182nd day of the year) is technically the start of the season
    eff_date = date - pd.Timedelta(days=182)
    return f"{eff_date.year}/{eff_date.year + 1}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    return parser.parse_args()


def load_adm0():
    resource_name = "fji_polbnda_adm0_country.zip"
    zip_path = Path(resource_name)
    filename = zip_path.stem
    load_resource_from_hdx("cod-ab-fji", resource_name, zip_path)
    extract_path = resource_name.removesuffix(".zip")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    gdf = gpd.read_file(Path(extract_path), layer=filename).set_crs(3832)
    return gdf


def check_trigger(csv: str) -> dict:
    """
    Checks trigger, from GitHub Action

    Parameters
    ----------
    csv: str
        Encoded Base64 string outputted from Power Automate of
        raw CSV file of forecast

    Returns
    -------

    """
    print("Loading forecast...")
    bytes = csv.encode("ascii") + b"=="
    converted_bytes = base64.b64decode(bytes)
    csv_str = converted_bytes.decode("ascii")
    filepath = StringIO(csv_str)
    fcast = load_fms_forecast(filepath)
    print("Loading adm0...")
    adm0 = load_adm0()
    print("Processing buffer...")
    buffer = adm0.simplify(10 * 1000).buffer(250 * 1000)
    print("Checking trigger...")
    thresholds = [
        {"distance": 250, "category": 4},
        {"distance": 0, "category": 3},
    ]
    readiness, activation = False, False
    cyclone = fcast.iloc[0]["Name Season"]
    base_time = str(fcast.iloc[0]["base_time"])
    fcast = fcast.set_index("leadtime")
    fcast[["prev_category", "prev_lat", "prev_lon"]] = fcast.shift()[
        ["Category", "Latitude", "Longitude"]
    ]
    for threshold in thresholds:
        cat = threshold.get("category")
        dist = threshold.get("distance")
        if dist == 0:
            zone = adm0.to_crs(FJI_CRS)
        else:
            zone = buffer.to_crs(FJI_CRS)
        for leadtime in fcast.index[:-1]:
            row = fcast.loc[leadtime]
            if row["Category"] >= cat and row["prev_category"] >= cat:
                ls = LineString(
                    [
                        row[["Longitude", "Latitude"]],
                        row[["prev_lon", "prev_lat"]],
                    ]
                )
                if ls.intersects(zone.geometry)[0]:
                    readiness = True
                    if leadtime <= 72:
                        activation = True
    report = {
        "cyclone": cyclone,
        "publication_time": base_time,
        "readiness": readiness,
        "activation": activation,
    }
    return report


def send_trigger_email(report: dict):
    triggers = []
    if report.get("readiness"):
        triggers.append("readiness")
    if report.get("activation"):
        triggers.append("action")

    PORT = 465  # For SSL
    PASSWORD = os.getenv("G_P_APP_PWD")
    USERNAME = os.getenv("G_P_ACCOUNT")
    sender_email = formataddr(("OCHA Centre for Humanitarian Data", USERNAME))
    SERVER = os.getenv("G_P_SERVER")
    mailing_list = ["tristan.downing@un.org"]

    environment = Environment(loader=FileSystemLoader("src/email/"))

    for trigger_type in triggers:
        template = environment.get_template(f"{trigger_type}.html")

        message = MIMEMultipart("alternative")
        message["Subject"] = (
            "Anticipatory action Fiji – "
            f"{trigger_type.capitalize()} trigger reached"
        )
        message["From"] = sender_email
        message["To"] = ", ".join(mailing_list)

        html = template.render(name=report.get("cyclone"))
        html_part = MIMEText(html, "html")
        message.attach(html_part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SERVER, PORT, context=context) as server:
            server.login(USERNAME, PASSWORD)
            server.sendmail(USERNAME, mailing_list, message.as_string())


if __name__ == "__main__":
    args = parse_args()
    report = check_trigger(csv=args.csv)
    print(report)
    send_trigger_email(report)
