import getpass
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from dateutil import rrule
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

AA_DATA_DIR = Path(os.environ["AA_DATA_DIR"])
EXP_DIR = AA_DATA_DIR / "public/exploration/fji"
FCAST_DIR = EXP_DIR / "rsmc/forecasts"
CYCLONETRACKS_PATH = (
    EXP_DIR / "rsmc/RSMC TC Tracks Historical 1969_70 to 2022_23 Seasons.csv"
)
IMPACT_PATH = EXP_DIR / "rsmc/FIJI_ DesInventar data 20230626.xlsx"
RAW_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/raw/fji"
ADM0_PATH = RAW_DIR / "cod_ab/fji_polbnda_adm0_country"
PROC_PATH = Path(os.environ["AA_DATA_DIR"]) / "public/processed/fji"
FJI_CRS = "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
CODAB_PATH = RAW_DIR / "cod_ab"
NDMO_DIR = AA_DATA_DIR / "private/raw/fji/ndmo"
ADM3 = "ADM3_PCODE"
ADM2 = "ADM2_PCODE"
ADM1 = "ADM1_PCODE"
ECMWF_RAW = AA_DATA_DIR / "public/exploration/glb/ecmwf/cyclone_hindcasts"
ECMWF_PROCESSED = (
    AA_DATA_DIR / "public/exploration/fji/ecmwf/cyclone_hindcasts"
)
KNOTS_PER_MS = 1.94384


def load_hindcasts() -> gpd.GeoDataFrame:
    """
    Loads RSMC / FMS hindcasts
    Returns
    -------
    gdf of hindcasts
    """
    filenames = os.listdir(FCAST_DIR)

    df = pd.DataFrame()
    for filename in filenames:
        if filename.startswith("."):
            continue
        df_date = pd.read_csv(FCAST_DIR / filename, header=None, nrows=3)
        date_str = df_date.iloc[0, 1].removeprefix("baseTime=")
        base_time = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        cyclone_name = (
            df_date.iloc[2, 0].removeprefix("# CycloneName=").capitalize()
        )
        df_data = pd.read_csv(
            FCAST_DIR / filename,
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
        df_data["Name Season"] = (
            df_data["cyclone_name"] + " " + df_data["season"]
        )

        df = pd.concat([df, df_data], ignore_index=True)

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"])
    )
    gdf.crs = "EPSG:4326"

    return gdf


def load_cyclonetracks() -> gpd.GeoDataFrame:
    df = pd.read_csv(CYCLONETRACKS_PATH)
    df["Date"] = df["Date"].apply(lambda x: x.strip())
    df["datetime"] = df["Date"] + " " + df["Time"]
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %HZ")
    df = df.drop(["Date", "Time"], axis=1)
    df["Cyclone Name"] = df["Cyclone Name"].apply(
        lambda name: "".join([x for x in name if not x.isdigit()])
        .strip()
        .capitalize()
    )
    df["Name Season"] = df["Cyclone Name"] + " " + df["Season"]
    df["Category numeric"] = pd.to_numeric(
        df["Category"], errors="coerce"
    ).fillna(0)
    df["Birth"] = df.groupby("Name Season")["datetime"].transform(min)
    df["Age (days)"] = df["datetime"] - df["Birth"]
    df["Age (days)"] = df["Age (days)"].apply(
        lambda x: x.days + x.seconds / 24 / 3600
    )
    df["nameyear"] = df["Cyclone Name"].apply(
        lambda x: x.lower()
    ) + df.groupby("Name Season")["Birth"].transform(
        lambda x: x.dt.year
    ).astype(
        str
    )

    gdf_tracks = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"])
    )
    gdf_tracks.crs = "EPSG:4326"
    return gdf_tracks


def interpolate_cyclonetracks(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Interpolate cyclone tracks hourly

    Parameters
    ----------
    gdf: GeoDataFrame
        cyclone tracks

    Returns
    -------
    Interpolated GeoDataFrame
    """
    dfs = []

    for name in gdf["Name Season"].unique():
        dff = gdf[gdf["Name Season"] == name][
            [
                "Latitude",
                "Longitude",
                "Category numeric",
                "Wind (Knots)",
                "Pressure",
                "datetime",
                "Age (days)",
                "nameyear",
            ]
        ]
        dff = dff.groupby("datetime").first()
        nameyear = dff.iloc[0]["nameyear"]
        dff = dff.resample("H").interpolate().reset_index()
        dff["Name Season"] = name
        dff["nameyear"] = nameyear
        dfs.append(dff)

    df_interp = pd.concat(dfs)

    gdf_interp = gpd.GeoDataFrame(
        df_interp,
        geometry=gpd.points_from_xy(
            df_interp["Longitude"], df_interp["Latitude"]
        ),
    )
    gdf_interp.crs = "EPSG:4326"

    return gdf_interp


def load_codab(level: int = 3) -> gpd.GeoDataFrame:
    """
    Load Fiji codab
    Parameters
    ----------
    level: int = 3
        admin level

    Returns
    -------
    gdf: gpd.GeoDataFrame
        includes setting CRS EPSG:3832
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
    filename = f"fji_polbnda_adm{level}_{adm_name}"
    gdf = gpd.read_file(CODAB_PATH / filename, layer=filename).set_crs(3832)
    if level > 0:
        gdf["ADM1_NAME"] = gdf["ADM1_NAME"].replace(
            "Northern  Division", "Northern Division"
        )
    return gdf


def load_geo_impact() -> gpd.GeoDataFrame:
    """
    Load processed geo impact (from NDMO)
    Returns
    -------
    gdf: gpd.GeoDataFrame
    """
    df = pd.read_csv(
        AA_DATA_DIR / "private/processed/fji/ndmo/processed_geo_impact.csv"
    )
    gdf = gpd.GeoDataFrame(
        data=df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=FJI_CRS
    )
    return gdf


def process_geo_impact():
    """
    Process geo-located impact (mostly infrastructure) from NDMO.
    Combines both files we received from them (xls and shp),
    removes duplicates.
    Finds adm3 (tikina) for each event.
    Saves in private/processed.
    """
    cod = load_codab()
    cod = cod.set_index(ADM3)
    cod = cod.to_crs(FJI_CRS)

    # read shp
    gdf = gpd.read_file(NDMO_DIR / "TC_Merged_cleaned/TC_Merged_Cleaned.shp")
    impact_crs = gdf.crs
    gdf = gdf.drop(columns=["FID_", "OID_", "geometry"])

    # read xls
    df = pd.read_excel(NDMO_DIR / "Flood Events.xls")

    concat = pd.concat([df, gdf])
    str_cols = ["Name", "Hazard", "Infrastruc", "Event", "Location"]
    concat[str_cols] = concat[str_cols].astype(str)
    num_cols = ["Year", "X_COORD", "Y_COORD"]
    concat[num_cols] = concat[num_cols].replace([0, ""], np.nan)
    concat = concat.dropna(subset=num_cols)
    concat[num_cols] = concat[num_cols].astype(int)
    concat["Name"] = concat["Name"].apply(lambda x: x.title())
    unique_cols = ["Name", "Event", "X_COORD", "Y_COORD"]
    concat = concat.drop_duplicates(subset=unique_cols)

    impact = gpd.GeoDataFrame(
        data=concat,
        geometry=gpd.points_from_xy(
            concat["X_COORD"], concat["Y_COORD"], crs=impact_crs
        ),
    )
    impact = impact.to_crs(FJI_CRS)

    id_cols = impact.columns
    # find the adm3 for each event
    for adm_id, adm in cod.iterrows():
        impact[adm_id] = impact.within(adm.geometry)

    # find events which don't have an adm3
    missing_events = impact[impact[cod.index].sum(axis=1) == 0]
    impact = impact[impact[cod.index].sum(axis=1) > 0]

    # calculate distance to adm3s instead
    cod = cod.to_crs(3832)
    missing_events = missing_events.to_crs(3832)
    for adm_id, adm in cod.iterrows():
        missing_events[f"{adm_id}"] = missing_events.distance(adm.geometry)
    missing_events[ADM3] = missing_events[cod.index].idxmin(axis=1)
    missing_events = missing_events.drop(columns=cod.index)
    missing_events = missing_events.to_crs(FJI_CRS)

    impact = impact.melt(id_vars=id_cols, var_name=ADM3)
    impact = impact[impact["value"]].drop(columns=["value"])
    impact = pd.concat([impact, missing_events], ignore_index=True)

    replace = {
        "TC Winston": "Winston 2015/2016",
        "March Flood (TD17F)": "TD17F 2011/2012",
        # note: Ana is "Ana 2020/2021"
        "TC_YASA_ANA": "Yasa 2020/2021",
        "TC Cody": "Cody 2021/2022",
        "January Flood (TD07F)": "TD07F 2011/2012",
        "TC ZENA": "Zena 2015/2016",
        "TC Sarai": "Sarai 2019/2020",
        # note: includes TD04F, TD05F, and TC Hettie
        "TD_Jan_Feb": "TD 2008/2009",
        "TC Tino": "Tino 2019/2020",
        "TC Josie": "Josie 2017/2018",
        "TD-17F": "TD17F 2015/2016",
        "TC Tomas": "Tomas 2009/2010",
        "TC_BINA": "Bina 2020/2021",
    }
    impact["Event"] = impact["Event"].replace(replace)

    impact["lon"] = impact.geometry.x
    impact["lat"] = impact.geometry.y
    impact = impact.drop(columns=["X_COORD", "Y_COORD", "geometry"])
    impact.to_csv(
        AA_DATA_DIR / "private/processed/fji/ndmo/processed_geo_impact.csv",
        index=False,
    )


def knots2cat(knots):
    category = 0
    if knots > 107:
        category = 5
    elif knots > 85:
        category = 4
    elif knots > 63:
        category = 3
    elif knots > 47:
        category = 2
    elif knots > 33:
        category = 1
    return category


def load_ecmwf_besttrack_hindcasts():
    df = pd.read_csv(ECMWF_PROCESSED / "besttrack_forecasts.csv")
    gdf = gpd.GeoDataFrame(
        data=df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"),
    )
    gdf = gdf.to_crs(FJI_CRS)
    return gdf


def process_ecmwf_besttrack_hindcasts():
    """
    Take best track forecasts from ECMWF CSVs.
    Also set correct starting year for each cyclone, including ones with
    duplicated years across years, and ones spanning one calendar year to the
    next.
    """
    ecmwf_filelist = os.listdir(ECMWF_PROCESSED / "csv")
    ecmwf_filelist = [x for x in ecmwf_filelist if not x.startswith(".")]

    # take only best-track forecasts
    forecast = pd.DataFrame()
    for filename in ecmwf_filelist:
        df = pd.read_csv(ECMWF_PROCESSED / "csv" / filename)
        df = df[df["mtype"] == "forecast"]
        drop_cols = ["mtype", "product", "ensemble"]
        df = df.drop(columns=drop_cols)
        df["name"] = filename.split("_")[0]
        forecast = pd.concat([forecast, df], ignore_index=True)

    forecast["time"] = pd.to_datetime(forecast["time"])
    forecast["forecast_time"] = pd.to_datetime(forecast["forecast_time"])
    # deal with "double negatives" (i.e. negative degrees West)
    forecast[["lat", "lon"]] = forecast[["lat", "lon"]].applymap(
        lambda x: str(x).replace("--", "")
    )
    forecast["lon"] = pd.to_numeric(forecast["lon"])
    forecast["lon"] = forecast["lon"].apply(lambda x: x + 360 if x < 0 else x)

    forecast["speed_knots"] = forecast["speed"] * KNOTS_PER_MS
    forecast["category_numeric"] = forecast["speed_knots"].apply(knots2cat)

    # set correct start years for each cylone
    # including duplicate cyclone names
    forecast["year"] = forecast["forecast_time"].dt.year
    forecast = forecast.sort_values("forecast_time")
    forecast["nameyear"] = ""
    for name in forecast["name"].unique():
        dff = forecast[forecast["name"] == name].groupby("year").first()
        if len(dff) == 1:
            forecast.loc[forecast["name"] == name, "nameyear"] = name + str(
                dff.index[0]
            )
            continue
        dff = dff.reset_index()
        j = 0
        while j < len(dff):
            year0 = dff.iloc[j]["year"]
            if j == len(dff) - 1:
                year1 = 0
                forecast.loc[
                    (forecast["name"] == name) & (forecast["year"] == year0),
                    "nameyear",
                ] = name + str(dff.index[0])
            else:
                year1 = dff.iloc[j + 1]["year"]
            if year1 == year0 + 1:
                j += 1
            else:
                year1 = 0
            forecast.loc[
                (forecast["name"] == name)
                & ((forecast["year"] == year0) | (forecast["year"] == year1)),
                "nameyear",
            ] = name + str(year0)
            j += 1

    forecast = forecast.drop(columns="year")
    forecast.to_csv(ECMWF_PROCESSED / "besttrack_forecasts.csv", index=False)


def process_ecmwf_hindcasts(dry_run: bool = False):
    """
    Produce CSVs from ECMWF hindcast XMLs
    """
    gdf = load_cyclonetracks()
    df_typhoons = pd.DataFrame()
    df_typhoons["international"] = (
        gdf["Cyclone Name"].apply(lambda x: x.lower()).unique()
    )

    def xml2csv(filename):
        try:
            tree = ET.parse(filename)
        except ET.ParseError:
            print("Error with file, skipping")
            return
        root = tree.getroot()

        prod_center = root.find("header/productionCenter").text
        baseTime = root.find("header/baseTime").text

        # Create one dictonary for each time point, and append it to a list
        for members in root.findall("data"):
            mtype = members.get("type")
            if mtype not in ["forecast", "ensembleForecast"]:
                continue
            for members2 in members.findall("disturbance"):
                cyclone_name = [
                    name.text.lower().strip()
                    for name in members2.findall("cycloneName")
                ]
                if not cyclone_name:
                    continue
                cyclone_name = cyclone_name[0].lower()
                if cyclone_name not in list(df_typhoons["international"]):
                    continue
                # print(f"Found typhoon {cyclone_name}")
                for members3 in members2.findall("fix"):
                    tem_dic = {}
                    tem_dic["mtype"] = [mtype]
                    tem_dic["product"] = [
                        re.sub("\\s+", " ", prod_center).strip().lower()
                    ]
                    tem_dic["cyc_number"] = [
                        name.text for name in members2.findall("cycloneNumber")
                    ]
                    tem_dic["ensemble"] = [members.get("member")]
                    tem_dic["speed"] = [
                        name.text
                        for name in members3.findall(
                            "cycloneData/maximumWind/speed"
                        )
                    ]
                    tem_dic["pressure"] = [
                        name.text
                        for name in members3.findall(
                            "cycloneData/minimumPressure/pressure"
                        )
                    ]
                    time = [
                        name.text for name in members3.findall("validTime")
                    ]
                    tem_dic["time"] = [
                        "/".join(time[0].split("T")[0].split("-"))
                        + ", "
                        + time[0].split("T")[1][:-1]
                    ]
                    # set sign of lat/lon based on N/S/E/W
                    tem_dic["lat"] = [
                        "-" + name.text
                        if name.attrib.get("units") == "deg S"
                        else name.text
                        for name in members3.findall("latitude")
                    ]
                    tem_dic["lon"] = [
                        "-" + name.text
                        if name.attrib.get("units") == "deg W"
                        else name.text
                        for name in members3.findall("longitude")
                    ]
                    tem_dic["lead_time"] = [members3.get("hour")]
                    tem_dic["forecast_time"] = [
                        "/".join(baseTime.split("T")[0].split("-"))
                        + ", "
                        + baseTime.split("T")[1][:-1]
                    ]
                    tem_dic1 = dict(
                        [
                            (k, "".join(str(e).lower().strip() for e in v))
                            for k, v in tem_dic.items()
                        ]
                    )
                    # Save to CSV
                    if not dry_run:
                        outfile = (
                            ECMWF_PROCESSED / f"csv/{cyclone_name}_all.csv"
                        )
                        pd.DataFrame(tem_dic1, index=[0]).to_csv(
                            outfile,
                            mode="a",
                            header=not os.path.exists(outfile),
                            index=False,
                        )

    filename_list = sorted(list(Path(ECMWF_RAW / "xml").glob("*.xml")))

    completed_xmls = pd.read_csv(ECMWF_PROCESSED / "completed_xmls.csv")[
        "0"
    ].to_list()
    for filename in tqdm(filename_list):
        if str(filename) not in completed_xmls or dry_run:
            xml2csv(filename)
            if not dry_run:
                completed_xmls.append(filename)
    pd.DataFrame(completed_xmls).to_csv(
        ECMWF_PROCESSED / "completed_xmls.csv", index=False
    )


def download_ecmwf_hindcasts(
    start_date: datetime = datetime(2022, 8, 22, 0, 0, 0)
):
    """
    Downloads ECMWF cyclone hindcasts to
        public/exploration/glb/ecmwf/cyclone_hindcasts
    """
    save_dir = AA_DATA_DIR / "public/exploration/glb/ecmwf/cyclone_hindcasts"
    email = input("email: ")
    pswd = getpass.getpass("password: ")

    values = {"email": email, "passwd": pswd, "action": "login"}
    login_url = "https://rda.ucar.edu/cgi-bin/login"

    ret = requests.post(login_url, data=values)
    if ret.status_code != 200:
        print("Bad Authentication")
        print(ret.text)
        exit(1)

    # note this url changed recently, this is the correct one as of Aug 2023
    dspath = "https://data.rda.ucar.edu/ds330.3/"
    date_list = rrule.rrule(
        rrule.HOURLY,
        dtstart=start_date,
        until=datetime.utcnow().date(),
        interval=12,
    )
    verbose = True

    for date in date_list:
        ymd = date.strftime("%Y%m%d")
        ymdhms = date.strftime("%Y%m%d%H%M%S")
        server = "test" if date < datetime(2008, 8, 1) else "prod"
        file = (
            f"ecmf/{date.year}/{ymd}/z_tigge_c_ecmf_{ymdhms}_"
            f"ifs_glob_{server}_all_glo.xml"
        )
        filename = dspath + file
        outfile = save_dir / "xml" / os.path.basename(filename)
        # Don't download if exists already
        if outfile.exists():
            if verbose:
                print(f"{file} already exists")
            continue
        req = requests.get(filename, cookies=ret.cookies, allow_redirects=True)
        if req.status_code != 200:
            if verbose:
                print(f"{file} invalid URL")
            continue
        if verbose:
            print(f"{file} downloading")
        open(outfile, "wb").write(req.content)


def load_housing_impact() -> pd.DataFrame:
    """
    Load processed housing impact from NDMO
    """
    return pd.read_csv(
        AA_DATA_DIR / "private/processed/fji/ndmo/processed_house_impact.csv"
    )


def process_housing_impact():
    """
    Process housing impact (by adm 1 or 2) from NDMO
    """
    cod = load_codab(level=2)

    cyclones = ["Winston", "Sarai", "Tino", "Harold", "Yasa", "Ana", "Evan"]

    dfs = []
    name2season = {
        "Winston": "2015/2016",
        "Sarai": "2019/2020",
        "Tino": "2019/2020",
        "Harold": "2019/2020",
        "Yasa": "2020/2021",
        "Ana": "2020/2021",
        "Evan": "2012/2013",
    }
    for cyclone in cyclones:
        evan = "_Evan" if cyclone == "Evan" else ""
        df_in = pd.read_excel(
            NDMO_DIR / f"Disaster Damaged Housing{evan}.xlsx",
            sheet_name=f"TC_{cyclone}",
        )
        df_in = df_in.rename(
            columns={
                "Completely Damaged": "Destroyed",
                "Partially Damaged": "Major Damage",
            }
        )
        df_in = df_in.dropna()
        df_in["nameseason"] = f"{cyclone} {name2season.get(cyclone)}"
        dfs.append(df_in)

    df = pd.concat(dfs, ignore_index=True)
    df["ADM1_NAME"] = df["Division"] + " Division"
    df["Province"] = df["Province"].replace("Nadroga/Navosa", "Nadroga_Navosa")
    df = df.merge(
        cod[["ADM1_PCODE", "ADM1_NAME"]].drop_duplicates(),
        on="ADM1_NAME",
        how="left",
    )
    df = df.merge(
        cod[["ADM2_PCODE", "ADM2_NAME"]],
        right_on="ADM2_NAME",
        left_on="Province",
        how="left",
    )
    df.to_csv(
        AA_DATA_DIR / "private/processed/fji/ndmo/processed_house_impact.csv",
        index=False,
    )


def load_desinventar() -> pd.DataFrame:
    """
    Load processed Desinventar data
    Returns
    -------
    df: pd.DataFrame
    """
    return pd.read_csv(EXP_DIR / "processed_desinventar.csv")


def process_desinventar():
    """
    Process Desinventar exported dataset
    """
    df = pd.read_excel(IMPACT_PATH, skiprows=[0], index_col=None)
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    df_clean = df.copy()

    # re-assemble file
    # iterate over rows
    last_valid_index = None
    valid_indices = []
    for index, row in df.loc[2:].iterrows():
        # check if row starts with serial number
        if isinstance(row.iloc[0], int):
            last_valid_index = index
            valid_indices.append(index)
            last_valid_col_i = np.argwhere(~row.isnull()).max()
            last_valid_col = df.columns[last_valid_col_i]
            if last_valid_col_i + 1 < len(df.columns):
                first_empty_col = df.columns[last_valid_col_i + 1]
        else:
            # if row after col A is empty, append to last valid cell of
            # last valid row
            df_clean.loc[last_valid_index, last_valid_col] = df_clean.loc[
                last_valid_index, last_valid_col
            ] + str(row[0])
            # if row has other content, fill in rest of last valid row
            if not row[1:].dropna().empty:
                cells_available = len(
                    df_clean.loc[last_valid_index, first_empty_col:]
                )
                df_clean.loc[last_valid_index, first_empty_col:] = row.values[
                    1 : cells_available + 1
                ]

    df_clean = df_clean.loc[valid_indices]
    df_clean = df_clean[df_clean["Event"] == "TC - Tropical Cyclone"]
    metrics = [
        "Deaths",
        "Injured",
        "Missing",
        "Houses Destroyed",
        "Houses Damaged",
        "Directly affected",
        "Indirectly Affected",
        "Relocated",
        "Evacuated",
        "Losses $USD",
        "Losses $Local",
        "Education centers",
        "Hospitals",
    ]
    df_clean = df_clean.dropna(subset=metrics)
    df_clean[metrics] = df_clean[metrics].astype(int)

    def clean_date(raw_date):
        if isinstance(raw_date, str):
            date_list = [int(x) for x in raw_date.split("/")]
            if date_list[1] == 0:
                date_list[1] = 1
            if date_list[2] == 0:
                date_list[2] = 1
            return datetime(date_list[0], date_list[1], date_list[2])
        return raw_date

    df_clean["datetime"] = df_clean["Date (YMD)"].apply(clean_date)

    df_clean["Season"] = df_clean["datetime"].apply(datetime_to_season)

    def clean_name(raw_name):
        return (
            raw_name.removesuffix("")
            .removeprefix("Tropical Cyclone")
            .removeprefix(" - ")
            .removesuffix("(possibly)")
            .strip()
        )

    df_clean["Cyclone Name"] = df_clean["Description of Cause"].apply(
        clean_name
    )
    df_clean["Name Season"] = (
        df_clean["Cyclone Name"] + " " + df_clean["Season"]
    )

    # clean up errors
    df_clean.loc[df_clean["Name Season"] == "Daman 2007/2008", "Deaths"] = 0

    # keep only relevant columns
    df_clean = df_clean[["Name Season"] + metrics]

    # drop duplicates - Odette
    df_clean = df_clean.drop_duplicates(subset="Name Season", keep="first")

    df_clean.to_csv(EXP_DIR / "processed_desinventar.csv", index=False)


def process_buffer(distance: int = 250):
    """
    Produce and save buffer

    Parameters
    ----------
    distance: int = 250
        Distance from adm0 in km

    Returns
    -------
    None
    """
    filename = f"fji_{distance}km_buffer"
    save_dir = PROC_PATH / filename
    if save_dir.exists():
        print(f"already exists at {save_dir}")
        # return

    gdf_adm0 = gpd.read_file(
        ADM0_PATH, layer="fji_polbnda_adm0_country"
    ).set_crs(3832)
    gdf_buffer = gdf_adm0.buffer(distance * 1000)
    gdf_buffer.to_file(save_dir / f"{filename}.shp")


def load_buffer(distance: int = 250) -> gpd.GeoDataFrame:
    """
    Load buffer file

    Parameters
    ----------
    distance: int = 250
        Distance from adm0 in km

    Returns
    -------

    """
    filename = f"fji_{distance}km_buffer"
    load_path = PROC_PATH / filename / f"{filename}.shp"

    return gpd.read_file(load_path)


def datetime_to_season(date):
    # July 1 (182nd day of the year) is technically the start of the season
    eff_date = date - pd.Timedelta(days=182)
    return f"{eff_date.year}/{eff_date.year + 1}"
