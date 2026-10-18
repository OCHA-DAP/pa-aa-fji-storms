"""Check that the monitor sees the same forecast from the DB as from the CSV.

For every Official Forecast Track CSV under the given directories whose
exact bytes are the copy stored in storms.fms_forecasts, compare
``parse_fms_forecast(csv)`` (what the GitHub Actions monitor reads) with
``load_fms_forecast_from_db`` (what the Databricks monitor reads). For the
first ``--compute`` of them, also run the trigger calculation both ways
(buffers, 64-kt readiness/action exposure, shifted best/worst exposure) and
compare the numbers.

Read-only. Needs DB read access to ``--db-stage`` (from a laptop: the DB
tunnel, with DSCI_AZ_DB_DEV_HOST=127.0.0.1:15432) and the dev blob.

    python scripts/check_fms_db_equivalence.py --db-stage dev --compute 12 \
        <csv dir> [<csv dir> ...]
"""

import argparse
import hashlib
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd
from pandas.testing import assert_frame_equal
from sqlalchemy import text

from src.constants import FJI_CRS
from src.datasources import codab, worldpop
from src.datasources.fms import (
    calculate_fms_buffers_gdf,
    calculate_shifted_exposures,
    parse_fms_forecast,
)
from src.datasources.fms_db import OFFICIAL_TRACK, load_fms_forecast_from_db
from src.exposure_calc import calculate_single_adm_exposure

SKIP = ("MODIFIED_", "inject_forecast", "_edited", " copy")


def header(data: bytes) -> dict:
    fields = {}
    for line in data.decode("utf-8-sig", "replace").splitlines()[:3]:
        for part in line.lstrip("#").split(","):
            key, sep, value = part.partition("=")
            if sep:
                fields[key.strip()] = value.strip()
    return fields


def trigger_numbers(gdf, da_wp_clip) -> dict:
    """The numbers the monitor's emails and triggers are built from."""
    gdf = gdf.rename(columns={"forecast_time": "valid_time"})
    out = {}
    for name, max_lead in (("readiness", 120), ("action", 72)):
        sub = gdf[gdf["leadtime"] <= max_lead].copy()
        exp = calculate_single_adm_exposure(
            calculate_fms_buffers_gdf(sub.copy()), da_wp_clip
        )
        for speed in (34, 50, 64):
            out[f"{name}_{speed}"] = int(
                exp[exp["buffer_speed"] == speed].iloc[0]["pop_exposed"]
            )
    shifted, _, _ = calculate_shifted_exposures(
        gdf[gdf["leadtime"] <= 120].copy(), da_wp_clip, disable_tqdm=True
    )
    shifted = shifted.sort_values(["shift_deg"]).reset_index(drop=True)
    for col in ("exp_34", "exp_50", "exp_64"):
        out[f"shift_{col}"] = shifted[col].astype(int).tolist()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--db-stage", choices=["dev", "prod"], default="dev")
    parser.add_argument("--compute", type=int, default=0)
    parser.add_argument(
        "--compute-match", default="", help="only files whose name has this"
    )
    args = parser.parse_args()

    engine = stratus.get_engine(stage=args.db_stage)
    with engine.connect() as conn:
        stored = pd.read_sql(
            text(
                "SELECT disturbance_id, track_type, issued_time, "
                "content_sha256 FROM storms.fms_forecasts"
            ),
            conn,
        )
    stored_shas = set(stored["content_sha256"].str.strip())
    print(f"{len(stored)} forecasts in the {args.db_stage} database")

    files = sorted(
        p
        for d in args.dirs
        for p in d.iterdir()
        if p.suffix.lower() == ".csv" and not any(s in p.name for s in SKIP)
    )
    compared, not_stored, other_type, failures = [], 0, 0, []
    seen = set()
    for path in files:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        fields = header(data)
        if fields.get("trackType") != OFFICIAL_TRACK:
            other_type += 1
            continue
        if sha not in stored_shas:
            not_stored += 1
            continue
        expected = parse_fms_forecast(BytesIO(data))
        got = load_fms_forecast_from_db(
            engine, fields["DisturbanceId"], fields["baseTime"]
        )
        try:
            assert list(got.columns) == list(expected.columns), "columns"
            assert_frame_equal(
                pd.DataFrame(got.drop(columns="geometry")),
                pd.DataFrame(expected.drop(columns="geometry")),
                check_exact=False,
                rtol=1e-9,
                check_dtype=False,
            )
            assert got.crs == expected.crs, "crs"
            assert np.allclose(got.geometry.x, expected.geometry.x, rtol=1e-9)
            assert np.allclose(got.geometry.y, expected.geometry.y, rtol=1e-9)
        except AssertionError as e:
            failures.append((path.name, str(e).splitlines()[0][:200]))
            continue
        compared.append((path.name, expected, got))

    print(
        f"{len(compared)} forecasts identical, {len(failures)} differ, "
        f"{not_stored} CSVs not the stored copy, {other_type} not official"
    )
    for name, err in failures:
        print(f"  DIFFER {name}: {err}")

    if args.compute:
        adm3 = codab.load_codab_from_blob(admin_level=3).to_crs(FJI_CRS)
        da_wp = worldpop.load_worldpop_from_blob()
        da_wp = da_wp.assign_coords({"x": ((da_wp.x + 360) % 360)}).sortby("x")
        da_wp_clip = da_wp.rio.clip(adm3.geometry)
        chosen = [c for c in compared if args.compute_match in c[0]]
        for name, expected, got in sorted(chosen)[: args.compute]:
            a = trigger_numbers(expected, da_wp_clip)
            b = trigger_numbers(got, da_wp_clip)
            status = "same" if a == b else "DIFFERENT"
            worst = max(a["shift_exp_64"])
            print(
                f"  {status} {name}: readiness64={a['readiness_64']} "
                f"action64={a['action_64']} worst64={worst}"
            )
            if a != b:
                failures.append((name, f"trigger numbers {a} vs {b}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
