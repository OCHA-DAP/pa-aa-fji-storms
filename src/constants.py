ISO3 = "fji"  # noqa: F821
FJI_CRS = "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"

# storing specific sids for quick plotting
WINSTON_SID = "2016041S14170"
YASA_SID = "2020346S13168"
HAROLD_SID = "2020092S09155"
MAL_SID = "2023316S09167"
RAE_SID = "2025054S13182"

# from page 2 https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_version4_Technical_Details.pdf noqa: E501
# from 10-min to 1-min wind speed
ONE_TO_TEN_AVG_PERIOD_FACTOR = 1.12

WIND_RADIUS_PARAMS = {
    34: {
        "const": 0.299034,
        "wind_speed_log": 0.587814,
        "lat_abs_log": 0.599866,
    },
    50: {
        "const": -1.926702,
        "wind_speed_log": 0.800029,
        "lat_abs_log": 0.769871,
    },
    64: {
        "const": -1.391375,
        "wind_speed_log": 0.688399,
        "lat_abs_log": 0.546447,
    },
}

WIND_RADIUS_PARAMS_FMS = {
    34: {
        "const": 3.489578,
        "MeanWind_log": 0.166424,
        "lat_abs_log": 0.127185,
    },
    50: {
        "const": -2.438167,
        "MeanWind_log": 0.921004,
        "lat_abs_log": 0.747109,
    },
    64: {
        "const": -3.945538,
        "MeanWind_log": 1.125596,
        "lat_abs_log": 0.789324,
    },
}

NM_TO_M = 1.852 * 1000

ROTUMA2 = "FJ315"

# storms that would have triggered 2023 framework
OLD_TRIG_SIDS = []

NAMESEASON2SID = {
    "Ana 2021": "2021029S16171",
    "Cody (co-dee) 2022": "2022008S17173",
    "Harold 2020": "2020092S09155",
    "Mal 2024": "2023316S09167",
    "Rae [ray] 2025": "2025054S13182",
    "Yasa 2021": "2020346S13168",
}

EASTERN1 = "FJ3"
LAU2 = "FJ305"
WAINIKELI3 = "FJ10308"
QUADS = ["ne", "se", "sw", "nw"]
