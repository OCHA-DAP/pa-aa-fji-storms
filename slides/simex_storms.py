"""Fictional storms for the Fiji AA simulation exercise.

Each scenario is a "true" track (what the storm actually does) plus a list of
forecasts issued along the way. A forecast is the true track from its issue
time onward, displaced by an error that grows with lead time and scaled in
intensity, so early forecasts can be wrong and later ones converge.

The forecasts are written as CSVs in the RSMC Nadi TC-module export format, so
they run through the live trigger code unchanged (and can be used as injects
for a live exercise through the GitHub Action).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Exercise clock: Day 1 00:00 Fiji time (UTC+12) = 2027-01-09 12:00 UTC.
# Slides show "Day N, HH:MM"; the dates only exist to make valid CSVs.
DAY1_UTC = datetime(2027, 1, 9, 12)

LEADS = list(range(0, 121, 6))

# Uncertainty (nm) by lead time, as in RSMC Nadi's 2025-26 forecasts
UNC_LEADS = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
UNC_NM = [60, 75, 90, 105, 120, 136, 152, 169, 185, 209, 232, 256, 280]

FMS_COLUMNS = [
    "Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']",
    "Latitude",
    "Longitude",
    "Symbol",
    "Category",
    "Pressure",
    "PressureOCI",
    "RadiusOCI",
    "Radius1000hPa",
    "RadiusMaxWinds",
    "MeanWind",
    "WindGust",
    "VerticalExtent",
    "Uncertainty",
    "FinalT",
    "CurrentIntensity",
    "GaleRadius",
    "StrongGaleRadius",
    "StormRadius",
    "HurricaneRadius",
    "NEGaleRadius",
    "SEGaleRadius",
    "SWGaleRadius",
    "NWGaleRadius",
    "NEStrongGaleRadius",
    "SEStrongGaleRadius",
    "SWStrongGaleRadius",
    "NWStrongGaleRadius",
    "NEStormRadius",
    "SEStormRadius",
    "SWStormRadius",
    "NWStormRadius",
    "NEHurricaneRadius",
    "SEHurricaneRadius",
    "SWHurricaneRadius",
    "NWHurricaneRadius",
    "P5Wind",
    "DataSource",
    "Land/Water",
    "CycloneStatus",
    "HowLocation",
    "DataTNoDvorak",
    "ModelTNoDvorak",
    "PatternTNoDvorak",
    "UncPressure",
    "HowPressure",
    "P/W_Method",
    "HowMaxWindRadius",
    "HowGaleRadius",
    "HowStormRadius",
    "HowHurricaneRadius",
    "UncMaxWindSpeed",
    "HowMaxWindSpeed",
    "HowGust",
    "EyeRadius",
    "UncEyeRadius",
    "HowEyeRadius",
]


def category(mean_wind_kt: float) -> int:
    """Australian/RSMC Nadi category from 10-min mean wind (knots)."""
    for cat, lower in [(5, 108), (4, 86), (3, 64), (2, 48), (1, 34)]:
        if mean_wind_kt >= lower:
            return cat
    return 0


def wind_radii(w: float) -> dict:
    """Quadrant radii (nm) for a given mean wind, a little larger to the
    south-east as for a storm moving south-east in the South Pacific."""
    out = {}
    for word, threshold, base, slope in [
        ("Gale", 34, 60, 0.8),
        ("Storm", 50, 25, 1.1),
        ("Hurricane", 64, 18, 0.7),
    ]:
        r = base + (w - threshold) * slope if w >= threshold else np.nan
        for quad, f in [("NE", 1.0), ("SE", 1.15), ("SW", 0.9), ("NW", 0.8)]:
            out[f"{quad}{word}Radius"] = round(r * f, 1)
    return out


@dataclass
class Forecast:
    issue_h: int  # hours after Day 1 00:00 Fiji time
    # displacement of the 120-h forecast point from the true position
    # (degrees lat, degrees lon), applied linearly with lead time
    err_120: tuple = (0.0, 0.0)
    wind_factor: float = 1.0


@dataclass
class Scenario:
    key: str
    name: str
    disturbance: str
    # (hours after Day 1 00:00 Fiji time, lat, lon, mean wind kt)
    truth: list
    forecasts: list = field(default_factory=list)

    def true_at(self, h: float) -> tuple:
        t = np.array([p[0] for p in self.truth], dtype=float)
        lat = np.interp(h, t, [p[1] for p in self.truth])
        lon = np.interp(h, t, [p[2] for p in self.truth])
        w = np.interp(h, t, [p[3] for p in self.truth])
        return float(lat), float(lon), float(w)

    def forecast_frame(self, fc: Forecast) -> pd.DataFrame:
        rows = []
        for lead in LEADS:
            h = fc.issue_h + lead
            if h > self.truth[-1][0]:
                break
            lat, lon, w = self.true_at(h)
            frac = lead / 120
            lat += fc.err_120[0] * frac
            lon += fc.err_120[1] * frac
            w = w * (1 + (fc.wind_factor - 1) * min(1, lead / 48))
            w = round(w / 5) * 5
            row = dict.fromkeys(FMS_COLUMNS, np.nan)
            row.update(
                {
                    FMS_COLUMNS[0]: (DAY1_UTC + timedelta(hours=h)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "Latitude": round(lat, 3),
                    "Longitude": round(lon, 3),
                    "Symbol": 4.0,
                    "Category": float(category(w)),
                    "MeanWind": float(w),
                    "WindGust": float(round(w * 1.4 / 5) * 5),
                    "Pressure": float(round(1006 - 0.45 * w)),
                    "Uncertainty": float(np.interp(lead, UNC_LEADS, UNC_NM)),
                    **wind_radii(w),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=FMS_COLUMNS)

    def write_csv(self, fc: Forecast, out_dir: Path) -> Path:
        df = self.forecast_frame(fc)
        base = DAY1_UTC + timedelta(hours=fc.issue_h)
        header = (
            f"# trackType=Official Forecast Track,baseTime="
            f"{base:%Y-%m-%dT%H:%M:%SZ}\n"
            f"# DisturbanceId={self.disturbance}\n"
            f"# CycloneName={self.name.upper()}\n"
            "# MathType of Data\n"
            "(Time->(" + ",".join(FMS_COLUMNS[1:]) + "))\n"
            "# Column Headings\n"
        )
        body = df.to_csv(index=False, na_rep="NaN", float_format="%.3f")
        lines = body.splitlines()
        text = header + lines[0] + "\n# Start of data\n" + "\n".join(lines[1:])
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (
            f"{base:%Y%m%dT%H%M%SZ}_Exercise_Forecast_Track_"
            f"{self.disturbance}_{self.name.upper()}.csv"
        )
        path.write_text(text + "\n")
        return path


def day_label(h: float) -> str:
    """Hours after Day 1 00:00 -> 'Day 2, 18:00'."""
    d, rem = divmod(int(h), 24)
    return f"Day {d + 1}, {rem:02d}:00"


# --- Scenario 1: readiness, then action -----------------------------------
# Forms near Vanuatu, tracks south-east and strikes Viti Levu as a Category 4.
ALPHA = Scenario(
    key="alpha",
    name="Alpha",
    disturbance="SIMEX-01F",
    truth=[
        (0, -13.0, 170.6, 35),
        (24, -13.9, 172.2, 45),
        (48, -14.9, 173.9, 60),
        (72, -15.9, 175.4, 75),
        (96, -17.0, 176.9, 90),
        (108, -17.7, 177.7, 95),
        (120, -18.5, 178.6, 85),
        (144, -20.3, 180.3, 65),
        (180, -23.0, 182.8, 45),
    ],
    forecasts=[
        Forecast(0, err_120=(-1.6, -1.2), wind_factor=0.62),
        Forecast(12, err_120=(0.5, -0.3), wind_factor=1.0),
        Forecast(24, err_120=(0.3, 0.2), wind_factor=1.0),
        Forecast(36, err_120=(-0.2, 0.3), wind_factor=1.0),
        Forecast(72, err_120=(0.0, 0.0), wind_factor=1.0),
    ],
)

# --- Scenario 2: readiness, then nothing ----------------------------------
# Early forecasts take it over Vanua Levu; it recurves east and passes well
# east of Fiji.
BRAVO = Scenario(
    key="bravo",
    name="Bravo",
    disturbance="SIMEX-02F",
    truth=[
        (0, -12.0, 176.8, 45),
        (24, -13.1, 178.0, 60),
        (48, -14.2, 179.8, 75),
        (72, -15.4, 182.0, 80),
        (96, -17.2, 184.0, 70),
        (120, -19.4, 185.6, 55),
        (156, -22.5, 187.0, 40),
    ],
    forecasts=[
        Forecast(0, err_120=(1.2, -5.2), wind_factor=1.1),
        Forecast(12, err_120=(0.6, -3.2), wind_factor=1.05),
        Forecast(24, err_120=(0.3, -1.6), wind_factor=1.0),
        Forecast(36, err_120=(0.1, -0.6), wind_factor=1.0),
        Forecast(60, err_120=(0.0, 0.0), wind_factor=1.0),
    ],
)

# --- Scenario 3: observational trigger only -------------------------------
# A slow tropical storm that never reaches hurricane-force winds, but stalls
# over Viti Levu for two days.
CHARLIE = Scenario(
    key="charlie",
    name="Charlie",
    disturbance="SIMEX-03F",
    truth=[
        (0, -15.2, 175.2, 30),
        (24, -16.2, 176.4, 40),
        (48, -17.0, 177.2, 45),
        (72, -17.5, 177.8, 50),
        (96, -17.9, 178.3, 50),
        (120, -18.7, 179.1, 45),
        (156, -20.3, 180.6, 35),
        (192, -22.0, 182.0, 30),
    ],
    forecasts=[
        Forecast(0, err_120=(0.6, 1.0), wind_factor=1.05),
        Forecast(24, err_120=(0.4, 0.6), wind_factor=1.0),
        Forecast(48, err_120=(0.2, 0.3), wind_factor=1.0),
        Forecast(72, err_120=(0.0, 0.0), wind_factor=1.0),
    ],
)

SCENARIOS = [ALPHA, BRAVO, CHARLIE]
