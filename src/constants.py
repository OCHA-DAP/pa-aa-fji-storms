ISO3 = "fji"  # noqa: F821
FJI_CRS = "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"

# storing specific sids for quick plotting
WINSTON_SID = "2016041S14170"
YASA_SID = "2020346S13168"
HAROLD_SID = "2020092S09155"
MAL_SID = "2023316S09167"

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

NM_TO_M = 1.852 * 1000

ROTUMA1 = "FJ315"
