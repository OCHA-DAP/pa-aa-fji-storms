import argparse
import base64
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, make_msgid
from io import StringIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from ochanticipy.utils.hdx_api import load_resource_from_hdx
from shapely.geometry import LineString

load_dotenv()

FJI_CRS = "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
EMAIL_HOST = os.getenv("CHD_DS_HOST")
EMAIL_PORT = int(os.getenv("CHD_DS_PORT"))
EMAIL_PASSWORD = os.getenv("CHD_DS_EMAIL_PASSWORD")
EMAIL_USERNAME = os.getenv("CHD_DS_EMAIL_USERNAME")
EMAIL_ADDRESS = os.getenv("CHD_DS_EMAIL_ADDRESS")


def decode_forecast_csv(csv: str) -> StringIO:
    bytes_str = csv.encode("ascii") + b"=="
    converted_bytes = base64.b64decode(bytes_str)
    csv_str = converted_bytes.decode("ascii")
    filepath = StringIO(csv_str)
    return filepath


def process_fms_forecast(
    path: Path | StringIO, save: bool = True
) -> pd.DataFrame:
    """
    Loads FMS raw forecast
    Parameters
    ----------
    path: Path | StringIO
        Path to raw forecast CSV. Path can be a StringIO
        (so CSV can be passed as an encoded string from Power Automate)
    save: bool = True
        If True, saves forecast as CSV

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
    base_time_file_str = (
        base_time.replace(microsecond=0).isoformat().replace(":", "")
    )
    if save:
        df_data.to_csv(f"data/forecast_{base_time_file_str}.csv", index=False)
    return df_data


def datetime_to_season(date):
    # July 1 (182nd day of the year) is technically the start of the season
    eff_date = date - pd.Timedelta(days=182)
    return f"{eff_date.year}/{eff_date.year + 1}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # if no CSV supplied, set to Yasa (readiness and action activation)
    yasa = os.getenv("YASA_MOD")
    parser.add_argument("csv", nargs="?", type=str, default=yasa)
    parser.add_argument("--suppress-send", action="store_true")
    return parser.parse_args()


def load_adm(level: int = 0) -> gpd.GeoDataFrame:
    """
    Loads adm from repo file structure
    Returns
    -------
    GeoDF of adm0
    """
    adm_name = ""
    if level == 0:
        adm_name = "country"
    elif level == 1:
        adm_name = "district"
    elif level == 2:
        adm_name = "province"
    elif level == 3:
        adm_name = "tikina"
    resource_name = f"fji_polbnda_adm{level}_{adm_name}.zip"
    zip_path = "data" / Path(resource_name)
    if zip_path.exists():
        print(f"adm{level} already exists")
    else:
        print(f"adm{level} does not exist, downloading now")
        load_resource_from_hdx("cod-ab-fji", resource_name, zip_path)
    gdf = gpd.read_file(
        f"zip://{zip_path.as_posix()}", layer=zip_path.stem
    ).set_crs(3832)
    return gdf


def load_buffer() -> gpd.GeoDataFrame:
    """
    Loads buffer from repo file structure
    Returns
    -------
    GeoDataFrame of buffer
    """
    buffer_name = "fji_250km_buffer"
    buffer_dir = "data" / Path(buffer_name)
    buffer_path = buffer_dir / f"{buffer_name}.shp"
    if buffer_path.exists():
        print("buffer already exists")
        buffer = gpd.read_file(buffer_path)
    else:
        print("processing buffer")
        adm0 = load_adm(level=0)
        buffer = adm0.simplify(10 * 1000).buffer(250 * 1000)
        if not buffer_dir.exists():
            os.mkdir(buffer_dir)
        buffer.to_file(buffer_path)
    return buffer


def check_trigger(forecast: pd.DataFrame) -> dict:
    """
    Checks trigger, from GitHub Action

    Parameters
    ----------
    forecast: pd.DataFrame
        df of processed forecast

    Returns
    -------

    """
    if not Path("data").exists():
        os.mkdir("data")
    print("Loading adm...")
    adm0 = load_adm(level=0)
    print("Processing buffer...")
    buffer = load_buffer()
    print("Checking trigger...")
    thresholds = [
        {"distance": 250, "category": 4},
        {"distance": 0, "category": 3},
    ]
    readiness, action = False, False
    cyclone = forecast.iloc[0]["Name Season"]
    base_time = forecast.iloc[0]["base_time"]
    base_time_str = base_time.replace(microsecond=0).isoformat()
    forecast = forecast.set_index("leadtime")
    forecast[["prev_category", "prev_lat", "prev_lon"]] = forecast.shift()[
        ["Category", "Latitude", "Longitude"]
    ]
    for threshold in thresholds:
        cat = threshold.get("category")
        dist = threshold.get("distance")
        if dist == 0:
            zone = adm0.to_crs(FJI_CRS)
        else:
            zone = buffer.to_crs(FJI_CRS)
        for leadtime in forecast.index[:-1]:
            row = forecast.loc[leadtime]
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
                        action = True
    report = {
        "cyclone": cyclone,
        "publication_time": base_time_str,
        "readiness": readiness,
        "action": action,
    }
    pub_time_file_str = base_time_str.replace(":", "")
    with open(f"data/report_{pub_time_file_str}.json", "w") as outfile:
        json.dump(report, outfile)
    return report


def calculate_distances(forecast: pd.DataFrame, codab: gpd.GeoDataFrame):
    # fcast = pd.read_csv("data/forecast.csv")

    pass


def send_trigger_email(
    report: dict, suppress_send: bool = False, save: bool = True
):
    triggers = []
    if report.get("readiness"):
        triggers.append("readiness")
    if report.get("action"):
        triggers.append("action")

    sender_email = formataddr(
        ("OCHA Centre for Humanitarian Data", EMAIL_ADDRESS)
    )

    mailing_list = ["tristan.downing@un.org"]

    environment = Environment(loader=FileSystemLoader("src/email/"))

    for trigger in triggers:
        template = environment.get_template(f"{trigger}.html")
        message = MIMEMultipart("alternative")
        message["Subject"] = (
            "Anticipatory action Fiji – "
            f"{trigger.capitalize()} trigger reached"
        )
        message["From"] = sender_email
        message["To"] = ", ".join(mailing_list)
        pub_time_split = report.get("publication_time").split("T")
        html = template.render(
            name=report.get("cyclone").split(" ")[0],
            pub_time=pub_time_split[1],
            pub_date=pub_time_split[0],
        )
        html_part = MIMEText(html, "html")
        message.attach(html_part)

        context = ssl.create_default_context()
        if not suppress_send:
            with smtplib.SMTP_SSL(
                EMAIL_HOST, EMAIL_PORT, context=context
            ) as server:
                server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
                server.sendmail(
                    EMAIL_ADDRESS, mailing_list, message.as_string()
                )
        if save:
            pub_time_file_str = report.get("publication_time").replace(":", "")
            with open(
                f"data/{trigger}_activation_email_{pub_time_file_str}.html",
                "w",
            ) as outfile:
                outfile.write(message.as_string())


def plot_forecast(
    report: dict, forecast: pd.DataFrame, save_html: bool = False
) -> go.Figure:
    colors = (
        (5, "rebeccapurple"),
        (4, "crimson"),
        (3, "orange"),
        (2, "limegreen"),
        (1, "skyblue"),
    )
    pub_time_split = report.get("publication_time").split("T")
    trigger_zone = load_buffer().to_crs(FJI_CRS)
    fig = go.Figure()
    x, y = trigger_zone.geometry[0].boundary.xy
    fig.add_trace(
        go.Scattermapbox(
            lat=np.array(y),
            lon=np.array(x),
            mode="lines",
            name="Area within 250km of Fiji",
            line=dict(width=1),
            hoverinfo="skip",
        )
    )
    official = forecast[forecast["leadtime"] <= 72]
    fig.add_trace(
        go.Scattermapbox(
            lat=official["Latitude"],
            lon=official["Longitude"],
            mode="lines",
            line=dict(width=2, color="black"),
            name="Official 72-hour forecast",
            customdata=official[["Category", "forecast_time"]],
            hovertemplate="Category: %{customdata[0]}<br>"
            "Datetime: %{customdata[1]}",
        )
    )
    for color in colors:
        dff = official[official["Category"] == color[0]]
        fig.add_trace(
            go.Scattermapbox(
                lat=dff["Latitude"],
                lon=dff["Longitude"],
                mode="markers",
                line=dict(width=2, color=color[1]),
                marker=dict(size=10),
                name="Official 72-hour forecast",
                hoverinfo="skip",
            )
        )
    print(forecast.columns)
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=4.5,
        mapbox_center_lat=-17,
        mapbox_center_lon=179,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title=f"RSMC Nadi forecast for {report.get('cyclone')}<br>"
        f"<sup>Produced at {pub_time_split[1]} (UTC) on {pub_time_split[0]}",
    )
    pub_time_file_str = report.get("publication_time").replace(":", "")
    filepath_stem = f"data/forecast_plot_{pub_time_file_str}"
    fig.write_image(f"{filepath_stem}.png", scale=3)
    if save_html:
        fig.write_html(f"{filepath_stem}.html")
    return fig


def send_informational_email(
    report: dict,
    forecast: pd.DataFrame,
    suppress_send: bool = False,
    save: bool = True,
):
    pub_time_file_str = report.get("publication_time").replace(":", "")
    # adm2 = load_adm(level=2)
    plot_forecast(report, forecast, save_html=True)
    if save:
        pass

    environment = Environment(loader=FileSystemLoader("src/email/"))
    template = environment.get_template("informational.html")

    sender_email = formataddr(
        ("OCHA Centre for Humanitarian Data", EMAIL_ADDRESS)
    )
    mailing_list = ["tristan.downing@un.org"]

    message = MIMEMultipart("alternative")
    message["Subject"] = "Anticipatory action Fiji – Forecast information"
    message["From"] = sender_email
    message["To"] = ", ".join(mailing_list)

    pub_time_split = report.get("publication_time").split("T")

    plot_cid = make_msgid(domain="humdata.org")
    print(plot_cid)
    html = template.render(
        name=report.get("cyclone").split(" ")[0],
        pub_date=pub_time_split[0],
        pub_time=pub_time_split[1],
        readiness="ACTIVATED" if report.get("readiness") else "NOT ACTIVATED",
        action="ACTIVATED" if report.get("action") else "NOT ACTIVATED",
        plot_cid=plot_cid.replace("<", "").replace(">", ""),
    )
    html_part = MIMEText(html, "html")
    message.attach(html_part)

    image = MIMEImage(
        open(f"data/forecast_plot_{pub_time_file_str}.png", "rb").read()
    )
    image.add_header("Content-ID", plot_cid)
    message.attach(image)

    context = ssl.create_default_context()
    if not suppress_send:
        with smtplib.SMTP_SSL(
            EMAIL_HOST, EMAIL_PORT, context=context
        ) as server:
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, mailing_list, message.as_string())
    if save:
        with open(
            f"data/informational_email_{pub_time_file_str}.html",
            "w",
        ) as outfile:
            outfile.write(message.as_string())
    pass


if __name__ == "__main__":
    args = parse_args()
    filepath = decode_forecast_csv(args.csv)
    forecast = process_fms_forecast(path=filepath, save=True)
    report = check_trigger(forecast)
    print(report)
    send_trigger_email(report, suppress_send=args.suppress_send)
    send_informational_email(
        report, forecast, suppress_send=args.suppress_send
    )
