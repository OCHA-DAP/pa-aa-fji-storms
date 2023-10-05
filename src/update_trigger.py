import argparse
import base64
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import make_msgid
from io import StringIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from html2text import html2text
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
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
TRIGGER_TO = os.getenv("TRIGGER_TO")
TRIGGER_CC = os.getenv("TRIGGER_CC")
INFO_TO = os.getenv("INFO_TO")
INFO_CC = os.getenv("INFO_CC")
CAT2COLOR = (
    (5, "rebeccapurple"),
    (4, "crimson"),
    (3, "orange"),
    (2, "limegreen"),
    (1, "skyblue"),
)


def decode_forecast_csv(csv: str) -> StringIO:
    bytes_str = csv.encode("ascii") + b"=="
    converted_bytes = base64.b64decode(bytes_str)
    csv_str = converted_bytes.decode("ascii")
    filepath = StringIO(csv_str)
    return filepath


def process_fms_forecast(
    path: Path | StringIO, save: bool = True
) -> gpd.GeoDataFrame:
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
    df_data["Category"] = df_data["Category"].astype(int)
    base_time_file_str = base_time.isoformat(timespec="minutes").replace(
        ":", ""
    )
    if save:
        df_data.to_csv(
            OUTPUT_DIR / f"forecast_{base_time_file_str}.csv", index=False
        )
    gdf = gpd.GeoDataFrame(
        df_data,
        geometry=gpd.points_from_xy(df_data["Longitude"], df_data["Latitude"]),
    )
    gdf = gdf.set_crs(FJI_CRS)
    return gdf


def datetime_to_season(date):
    # July 1 (182nd day of the year) is technically the start of the season
    eff_date = date - pd.Timedelta(days=182)
    return f"{eff_date.year}/{eff_date.year + 1}"


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
    zip_path = INPUT_DIR / resource_name
    if zip_path.exists():
        print(f"adm{level} already exists")
    else:
        print(f"adm{level} does not exist, downloading now")
        load_resource_from_hdx("cod-ab-fji", resource_name, zip_path)
    gdf = gpd.read_file(
        f"zip://{zip_path.as_posix()}", layer=zip_path.stem
    ).set_crs(3832)
    if level >= 2:
        gdf["ADM1_NAME"] = gdf["ADM1_NAME"].apply(
            lambda x: x.replace("  ", " ")
        )
    return gdf


def load_buffer() -> gpd.GeoDataFrame:
    """
    Loads buffer from repo file structure
    Returns
    -------
    GeoDataFrame of buffer
    """
    buffer_name = "fji_250km_buffer"
    buffer_dir = INPUT_DIR / buffer_name
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


def check_trigger(forecast: gpd.GeoDataFrame) -> dict:
    """
    Checks trigger, from GitHub Action

    Parameters
    ----------
    forecast: pd.DataFrame
        df of processed forecast

    Returns
    -------

    """
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
    base_time_str = base_time.isoformat(timespec="minutes")
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
    with open(OUTPUT_DIR / f"report_{pub_time_file_str}.json", "w") as outfile:
        json.dump(report, outfile)
    return report


def plot_forecast(
    report: dict, forecast: gpd.GeoDataFrame, save_html: bool = False
) -> go.Figure:
    pub_time_split = report.get("publication_time").split("T")
    trigger_zone = load_buffer().to_crs(FJI_CRS)
    forecast = forecast.to_crs(3832)
    official = forecast[forecast["leadtime"] <= 72]
    unofficial = forecast[forecast["leadtime"] >= 72]
    # produce uncertainty cone
    circles = []
    for _, row in official.iterrows():
        circles.append(row["geometry"].buffer(row["Uncertainty"] * 1000))
    o_zone = (
        gpd.GeoDataFrame(geometry=circles)
        .dissolve()
        .set_crs(3832)
        .to_crs(FJI_CRS)
    )
    circles = []
    for _, row in forecast.iterrows():
        circles.append(row["geometry"].buffer(row["Uncertainty"] * 1000))
    u_zone = (
        gpd.GeoDataFrame(geometry=circles)
        .dissolve()
        .set_crs(3832)
        .to_crs(FJI_CRS)
    )
    fig = go.Figure()
    # trigger zone
    x_b, y_b = trigger_zone.geometry[0].boundary.xy
    fig.add_trace(
        go.Scattermapbox(
            lat=np.array(y_b),
            lon=np.array(x_b),
            mode="lines",
            name="Area within 250 km of Fiji",
            line=dict(width=1),
            hoverinfo="skip",
        )
    )
    # official forecast
    fig.add_trace(
        go.Scattermapbox(
            lat=official["Latitude"],
            lon=official["Longitude"],
            mode="lines",
            line=dict(width=2, color="black"),
            name="Best Track",
            customdata=official[["Category", "forecast_time"]],
            hovertemplate="Category: %{customdata[0]}<br>"
            "Datetime: %{customdata[1]}",
            legendgroup="official",
            legendgrouptitle_text="Official 72-hour forecast",
        )
    )
    # unofficial forecast
    fig.add_trace(
        go.Scattermapbox(
            lat=unofficial["Latitude"],
            lon=unofficial["Longitude"],
            mode="lines",
            line=dict(width=2, color="white"),
            name="Best Track",
            customdata=unofficial[["Category", "forecast_time"]],
            hovertemplate="Category: %{customdata[0]}<br>"
            "Datetime: %{customdata[1]}",
            legendgroup="unofficial",
            legendgrouptitle_text="Unofficial 120-hour forecast",
        )
    )
    # by category
    for color in CAT2COLOR:
        dff = forecast[forecast["Category"] == color[0]]
        fig.add_trace(
            go.Scattermapbox(
                lat=dff["Latitude"],
                lon=dff["Longitude"],
                mode="markers",
                line=dict(width=2, color=color[1]),
                marker=dict(size=10),
                name=f"Category {color[0]}",
                hoverinfo="skip",
            )
        )
    # uncertainty
    x_u, y_u = u_zone.geometry[0].boundary.xy
    fig.add_trace(
        go.Scattermapbox(
            lat=np.array(y_u),
            lon=np.array(x_u),
            mode="lines",
            name="Uncertainty",
            line=dict(width=1, color="white"),
            hoverinfo="skip",
            legendgroup="unofficial",
        )
    )
    x_o, y_o = o_zone.geometry[0].boundary.xy
    fig.add_trace(
        go.Scattermapbox(
            lat=np.array(y_o),
            lon=np.array(x_o),
            mode="lines",
            name="Uncertainty",
            line=dict(width=1, color="black"),
            hoverinfo="skip",
            legendgroup="official",
        )
    )
    # set map bounds based on uncertainty cone of unofficial forecast
    lat_max = max(y_u)
    lat_min = min(y_u)
    lon_max = max(x_u)
    lon_min = min(x_u)
    max_bound = max(lon_max - lon_min, lat_max - lat_min) * 111
    zoom = 12.7 - np.log(max_bound)
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=zoom,
        mapbox_center_lat=(lat_max + lat_min) / 2,
        mapbox_center_lon=(lon_max + lon_min) / 2,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title=f"RSMC Nadi forecast for {report.get('cyclone')}<br>"
        f"<sup>Produced at {pub_time_split[1]} (UTC) "
        f"on {pub_time_split[0]}",
        legend=dict(xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.3)"),
        height=800,
        width=800,
    )
    pub_time_file_str = report.get("publication_time").replace(":", "")
    filepath_stem = OUTPUT_DIR / f"forecast_plot_{pub_time_file_str}"
    fig.write_image(f"{filepath_stem}.png", scale=4)
    if save_html:
        fig.write_html(f"{filepath_stem}.html")
    return fig


def calculate_distances(
    report: dict, forecast: gpd.GeoDataFrame, save: bool = True
) -> pd.DataFrame:
    pub_time_file_str = report.get("publication_time").replace(":", "")
    adm2 = load_adm(level=2)
    cols = ["ADM1_PCODE", "ADM1_NAME", "ADM2_PCODE", "ADM2_NAME", "geometry"]
    distances = adm2[cols].copy()
    forecast = forecast.to_crs(3832)
    track = LineString([(p.x, p.y) for p in forecast.geometry])
    distances["distance (km)"] = np.round(
        track.distance(adm2.geometry) / 1000
    ).astype(int)
    distances["uncertainty (km)"] = None
    distances["category"] = None
    # find closest point to use for uncertainty
    for i, row in distances.iterrows():
        forecast["distance"] = row.geometry.distance(forecast.geometry)
        i_min = forecast["distance"].idxmin()
        distances.loc[i, "uncertainty (km)"] = np.round(
            forecast.loc[i_min, "Uncertainty"]
        ).astype(int)
        distances.loc[i, "category"] = forecast.loc[i_min, "Category"]
    distances = distances.drop(columns="geometry")
    distances = distances.sort_values("distance (km)")
    if save:
        distances.to_csv(
            OUTPUT_DIR / f"distances_{pub_time_file_str}.csv", index=False
        )
    return distances


def plot_distances(report: dict, distances: pd.DataFrame) -> go.Figure():
    pub_time_split = report.get("publication_time").split("T")
    fig = go.Figure()
    distances = distances.sort_values("distance (km)", ascending=False)
    distances["adm2_adm1"] = distances.apply(
        lambda row: f"{row['ADM2_NAME']} "
        f"({row['ADM1_NAME'].removesuffix(' Division')})",
        axis=1,
    )
    adm_order = distances["adm2_adm1"]
    for color in CAT2COLOR:
        dff = distances[distances["category"] == color[0]]
        fig.add_trace(
            go.Scatter(
                y=dff["adm2_adm1"],
                x=dff["distance (km)"],
                mode="markers",
                error_x=dict(
                    type="data", array=dff["uncertainty (km)"], visible=True
                ),
                marker=dict(size=8, color=color[1]),
                name=color[0],
            )
        )
    fig.update_yaxes(categoryorder="array", categoryarray=adm_order)
    fig.update_xaxes(
        rangemode="nonnegative",
        title="Minimum distance from best track forecast to Province (km)",
    )
    title = (
        f"{report.get('cyclone').split(' ')[0]} "
        f"{pub_time_split[1]} {pub_time_split[0]} "
        "forecast distances<br>"
        "<sup>Error bars estimated based on uncertainty cone of forecast</sup>"
    )
    fig.update_layout(
        template="simple_white",
        title_text=title,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        legend=dict(
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.3)",
            title="Category at<br>closest pass",
        ),
    )
    pub_time_file_str = report.get("publication_time").replace(":", "")
    filepath_stem = OUTPUT_DIR / f"distances_plot_{pub_time_file_str}"
    fig.write_image(f"{filepath_stem}.png", scale=4)
    return fig


def send_trigger_email(
    report: dict,
    suppress_send: bool = False,
    save: bool = True,
    test_email: bool = False,
):
    test_subject = "[TEST] " if test_email else ""
    triggers = []
    if report.get("readiness"):
        triggers.append("readiness")
    if report.get("action"):
        triggers.append("action")
    pub_time_split = report.get("publication_time").split("T")

    to_list = [x.strip() for x in TRIGGER_TO.split(",") if x]
    cc_list = [x.strip() for x in TRIGGER_CC.split(",") if x]

    environment = Environment(loader=FileSystemLoader("src/email/"))

    for trigger in triggers:
        template = environment.get_template(f"{trigger}.html")
        msg = EmailMessage()
        msg["Subject"] = (
            f"{test_subject}Anticipatory action Fiji – "
            f"{trigger.capitalize()} trigger reached"
        )
        msg["From"] = Address(
            "OCHA Centre for Humanitarian Data",
            EMAIL_ADDRESS.split("@")[0],
            EMAIL_ADDRESS.split("@")[1],
        )
        for mail_list, name in zip([to_list, cc_list], ["To", "Cc"]):
            msg[name] = [
                Address(x.split("@")[0], x.split("@")[0], x.split("@")[1])
                for x in mail_list
            ]

        html_str = template.render(
            name=report.get("cyclone").split(" ")[0],
            pub_time=pub_time_split[1],
            pub_date=pub_time_split[0],
            test_email=test_email,
        )
        text_str = html2text(html_str)
        msg.set_content(text_str)
        msg.add_alternative(html_str, subtype="html")

        context = ssl.create_default_context()
        if not suppress_send:
            with smtplib.SMTP_SSL(
                EMAIL_HOST, EMAIL_PORT, context=context
            ) as server:
                server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
                server.sendmail(
                    EMAIL_ADDRESS, to_list + cc_list, msg.as_string()
                )
        if save:
            pub_time_file_str = report.get("publication_time").replace(":", "")
            name_stem = f"{trigger}_activation_email_{pub_time_file_str}"
            with open(OUTPUT_DIR / f"{name_stem}.txt", "w") as f:
                f.write(text_str)
            with open(OUTPUT_DIR / f"{name_stem}.html", "w") as f:
                f.write(html_str)
            with open(OUTPUT_DIR / f"{name_stem}.msg", "wb") as f:
                f.write(bytes(msg))


def send_info_email(
    report: dict,
    suppress_send: bool = False,
    save: bool = True,
    test_email: bool = False,
):
    test_subject = "[TEST] " if test_email else ""
    pub_time_split = report.get("publication_time").split("T")
    pub_time_file_str = report.get("publication_time").replace(":", "")

    environment = Environment(loader=FileSystemLoader("src/email/"))
    template = environment.get_template("informational.html")

    to_list = [x.strip() for x in INFO_TO.split(",") if x]
    cc_list = [x.strip() for x in INFO_CC.split(",") if x]

    msg = EmailMessage()
    msg[
        "Subject"
    ] = f"{test_subject}Anticipatory action Fiji – Forecast information"
    msg["From"] = Address(
        "OCHA Centre for Humanitarian Data",
        EMAIL_ADDRESS.split("@")[0],
        EMAIL_ADDRESS.split("@")[1],
    )
    for mail_list, name in zip([to_list, cc_list], ["To", "Cc"]):
        msg[name] = [
            Address(x.split("@")[0], x.split("@")[0], x.split("@")[1])
            for x in mail_list
        ]

    map_cid = make_msgid(domain="humdata.org")
    distances_cid = make_msgid(domain="humdata.org")
    html_str = template.render(
        name=report.get("cyclone").split(" ")[0],
        pub_date=pub_time_split[0],
        pub_time=pub_time_split[1],
        readiness="ACTIVATED" if report.get("readiness") else "NOT ACTIVATED",
        action="ACTIVATED" if report.get("action") else "NOT ACTIVATED",
        map_cid=map_cid[1:-1],
        distances_cid=distances_cid[1:-1],
        test_email=test_email,
    )
    text_str = html2text(html_str)
    msg.set_content(text_str)
    msg.add_alternative(html_str, subtype="html")

    for plot, cid in zip(["forecast", "distances"], [map_cid, distances_cid]):
        img_path = OUTPUT_DIR / f"{plot}_plot_{pub_time_file_str}.png"
        with open(img_path, "rb") as img:
            msg.get_payload()[1].add_related(
                img.read(), "image", "png", cid=cid
            )

    csv_name = f"distances_{pub_time_file_str}.csv"
    with open(OUTPUT_DIR / csv_name, "rb") as f:
        f_data = f.read()
    msg.add_attachment(
        f_data, maintype="text", subtype="csv", filename=csv_name
    )

    context = ssl.create_default_context()
    if not suppress_send:
        with smtplib.SMTP_SSL(
            EMAIL_HOST, EMAIL_PORT, context=context
        ) as server:
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_list + cc_list, msg.as_string())
    if save:
        file_stem = f"informational_email_{pub_time_file_str}"
        with open(OUTPUT_DIR / f"{file_stem}.txt", "w") as f:
            f.write(text_str)
        with open(OUTPUT_DIR / f"{file_stem}.html", "w") as f:
            f.write(html_str)
        with open(OUTPUT_DIR / f"{file_stem}.msg", "wb") as f:
            f.write(bytes(msg))
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # if no CSV supplied, set to Yasa (readiness and action activation)
    yasa = os.getenv("YASA_MOD")
    parser.add_argument("csv", nargs="?", type=str, default=yasa)
    parser.add_argument("--suppress-send", action="store_true")
    parser.add_argument("--test-email", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not OUTPUT_DIR.exists():
        os.mkdir(OUTPUT_DIR)
    if not INPUT_DIR.exists():
        os.mkdir(INPUT_DIR)
    filepath = decode_forecast_csv(args.csv)
    forecast = process_fms_forecast(path=filepath, save=True)
    report = check_trigger(forecast)
    print(report)
    send_trigger_email(
        report, suppress_send=args.suppress_send, test_email=args.test_email
    )
    plot_forecast(report, forecast, save_html=True)
    distances = calculate_distances(report, forecast)
    plot_distances(report, distances)
    send_info_email(
        report,
        suppress_send=args.suppress_send,
        test_email=args.test_email,
    )
