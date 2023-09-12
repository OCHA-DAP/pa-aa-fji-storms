import argparse
import base64
import zipfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
from ochanticipy.utils.hdx_api import load_resource_from_hdx


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


def check_trigger(csv: str):
    """
    Checks trigger, from GitHub Action

    Parameters
    ----------
    csv: str
        Encoded string outputted from Power Automate of
        raw CSV file of forecast

    Returns
    -------

    """
    bytes = csv.encode("ascii") + b"=="
    converted_bytes = base64.b64decode(bytes)
    # print(converted_bytes)
    # csv_json = converted_bytes.decode("utf8")
    # print(csv_json)
    csv_str = converted_bytes.decode("ascii")
    filepath = StringIO(csv_str)
    df = load_fms_forecast(filepath)
    gdf = load_adm0()
    print(gdf)
    print(df)


if __name__ == "__main__":
    args = parse_args()
    check_trigger(csv=args.csv)
