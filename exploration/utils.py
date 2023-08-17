import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv

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
ADM_ID = "ADM3_PCODE"


def load_hindcasts() -> gpd.GeoDataFrame:
    filenames = os.listdir(FCAST_DIR)

    df = pd.DataFrame()
    for filename in filenames:
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
        df_data["name_season"] = (
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
            ]
        ]
        dff = dff.groupby("datetime").first()
        dff = dff.resample("H").interpolate().reset_index()
        dff["Name Season"] = name
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
    cod = cod.set_index(ADM_ID)
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
    missing_events[ADM_ID] = missing_events[cod.index].idxmin(axis=1)
    missing_events = missing_events.drop(columns=cod.index)
    missing_events = missing_events.to_crs(FJI_CRS)

    impact = impact.melt(id_vars=id_cols, var_name=ADM_ID)
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


def load_desinventar() -> pd.DataFrame:
    """
    Load Desinventar exported dataset
    Returns
    -------

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

    df_clean.to_csv(EXP_DIR / "impact_data.csv", index=False)

    return df_clean


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
        return

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
