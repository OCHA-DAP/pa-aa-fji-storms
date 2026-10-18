"""Build the two slide decks published under docs/slides/.

    python slides/build_decks.py --fms-dir <folder of RSMC Nadi CSVs>

- 2025-26 season look-back: every RSMC Nadi forecast for Urmil and Vaianu,
  evaluated with the live trigger code (CSVs are RSMC Nadi's, kept out of
  the repo; the Power Automate flow saves them to OneDrive).
- Simulation exercise: three fictional storms (slides/simex_storms.py),
  written as RSMC Nadi-format CSVs to slides/simex_injects/ and evaluated the
  same way.

Each deck is one self-contained HTML page plus its images, a PDF and an
editable PowerPoint file (slides/pptx_export.py).
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).parent))

import pptx_export  # noqa: E402
import rain  # noqa: E402
from figures import RAIN_LEVELS, forecast_map, rain_map  # noqa: E402
from simex_storms import ALPHA, BRAVO, CHARLIE, day_label  # noqa: E402
from trigger_eval import evaluate_forecast, load_static  # noqa: E402

from src.constants import EXP_THRESHOLD_64_KNOTS  # noqa: E402
from src.email.content import render_template  # noqa: E402

HERE = Path(__file__).parent
DOCS = HERE.parent / "docs" / "slides"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

LOOKBACK_EXTENT = (171.6, 182.4, -24.0, -13.6)
URMIL_EXTENT = (164.0, 196.0, -40.0, -12.0)
SIMEX_EXTENT = (172.6, 184.2, -21.6, -12.4)
RAIN_EXTENT = (176.6, 182.2, -19.6, -15.6)

THRESHOLD = EXP_THRESHOLD_64_KNOTS
TITLES = {
    "2025-26-season": "Fiji AA 2025–26 season look-back",
    "simex": "Fiji AA simulation exercise",
}
RAIN_THRESHOLD = 190


def fmt(n) -> str:
    return f"{int(round(n)):,}"


def round_km(d) -> int | None:
    return None if d is None else int(round(d / 5) * 5)


def max_cat(result) -> int:
    return int(result["track"]["Category"].max())


# --------------------------------------------------------------------------
# Emails
# --------------------------------------------------------------------------


def email_png(title: str, body: str, out: Path):
    env = Environment(loader=FileSystemLoader(HERE / "templates"))
    html = env.get_template("email_shell.html").render(title=title, body=body)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "email.html"
        src.write_text(html)
        raw = Path(tmp) / "raw.png"
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                "--window-size=820,1100",
                f"--screenshot={raw}",
                src.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        img = plt.imread(raw)
    # trim the empty page background below the email
    bg = img[-1, -1, :3]
    rows = np.where((np.abs(img[:, :, :3] - bg).sum(axis=2) > 0.05).any(1))[0]
    img = img[: rows.max() + 30]
    plt.imsave(out, img)


def trigger_email(kind: str, cyclone: str, when: str, out: Path):
    body = render_template(
        f"{kind}.html",
        {"cyclone_name": cyclone, "forecast_display_str": when},
    )
    title = (
        f"[EXERCISE] Anticipatory action Fiji: Cyclone {cyclone} "
        f"{kind} trigger ACTIVATED"
    )
    email_png(title, body, out)


def observational_email(cyclone: str, rain_mm: float, period: str, out: Path):
    body = render_template(
        "observational.html",
        {
            "cyclone_name": cyclone,
            "rain_mm": fmt(rain_mm),
            "period_str": period,
        },
        template_dir=HERE / "templates",
    )
    title = (
        f"[EXERCISE] Anticipatory action Fiji: Cyclone {cyclone} "
        "observational trigger ACTIVATED"
    )
    email_png(title, body, out)


# --------------------------------------------------------------------------
# Look-back deck
# --------------------------------------------------------------------------


def build_lookback(fms_dir: Path, out: Path):
    adm3, da_wp = load_static()
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    paths = sorted(fms_dir.glob("2026*.csv"))
    urmil, vaianu = [], []
    for p in paths:
        r = evaluate_forecast(p, with_shifts=True)
        (urmil if r["name"] == "Urmil" else vaianu).append(r)

    # Urmil: one map with every forecast track
    last = urmil[-1]
    forecast_map(
        last,
        adm3,
        da_wp,
        URMIL_EXTENT,
        assets / "urmil.png",
        show_cone=False,
        extra_tracks=[u["track"] for u in urmil[:-1]],
    )

    v_slides = []
    for i, r in enumerate(vaianu):
        fname = f"vaianu_{i + 1}.png"
        forecast_map(r, adm3, da_wp, LOOKBACK_EXTENT, assets / fname)
        v_slides.append(
            {
                "n": i + 1,
                "name": "Tropical disturbance 11F"
                if r["name"] == "Noname"
                else "Vaianu",
                "when": r["issued_fji"].strftime("%a %-d %b, %H:%M"),
                "img": f"assets/{fname}",
                "r64": r["exp_readiness"][64],
                "a64": r["exp_action"][64],
                "worst64": r["worst_64"],
                "d64": round_km(r["dist_64_km"]),
                "dctr": round_km(r["dist_centre_km"]),
                "maxcat": max_cat(r),
            }
        )

    # the real observational emails' rainfall maps
    for f in ["vaianu_obsv_20260407.png", "vaianu_obsv_20260408.png"]:
        src = HERE / "lookback_inputs" / f
        if src.exists():
            shutil.copy(src, assets / f)

    # "how close" summary chart: distance of the 64-kt zone from land
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels = [s["when"].replace(", ", "\n") for s in v_slides]
    vals = [s["d64"] if s["d64"] is not None else 0 for s in v_slides]
    colors = ["#c3ced3" if s["d64"] is None else "#9d372b" for s in v_slides]
    ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    for i, s in enumerate(v_slides):
        txt = "no 64-kt\nwinds" if s["d64"] is None else f"{s['d64']} km"
        ax.text(i, vals[i] + 5, txt, ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(len(vals)), labels, fontsize=8.5)
    ax.set_ylabel("km from land")
    ax.set_ylim(0, max(vals) * 1.3)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(assets / "vaianu_distance.png", dpi=170)
    plt.close(fig)

    data = {
        "urmil": {
            "n": len(urmil),
            "first": urmil[0]["issued_fji"].strftime("%-d"),
            "last": urmil[-1]["issued_fji"].strftime("%-d"),
            "maxcat": max(max_cat(u) for u in urmil),
            "dctr": round_km(min(u["dist_centre_km"] for u in urmil)),
            "r64": max(u["exp_readiness"][64] for u in urmil),
            "worst64": max(u["worst_64"] for u in urmil),
        },
        "vaianu": v_slides,
        "vaianu_summary": {
            "n": len(vaianu),
            "maxcat": max(s["maxcat"] for s in v_slides),
            "dctr": min(s["dctr"] for s in v_slides),
            "d64": min(s["d64"] for s in v_slides if s["d64"] is not None),
            "r64": max(s["r64"] for s in v_slides),
            "worst64": max(s["worst64"] for s in v_slides),
        },
        "rain": [
            {
                "img": "assets/vaianu_obsv_20260407.png",
                "days": "6–7 Apr",
                "mm": 72,
            },
            {
                "img": "assets/vaianu_obsv_20260408.png",
                "days": "7–8 Apr",
                "mm": 42,
            },
        ],
    }
    (HERE / "numbers").mkdir(exist_ok=True)
    (HERE / "numbers" / "lookback.json").write_text(
        json.dumps(data, indent=1, default=str) + "\n"
    )
    return data


# --------------------------------------------------------------------------
# Simulation-exercise deck
# --------------------------------------------------------------------------


def _status(value, threshold, activated_at, now_label):
    """Row state for the trigger table."""
    if activated_at == now_label:
        return "new"
    if activated_at is not None:
        return "done"
    return "notmet" if value < threshold else "new"


def run_scenario(sc, adm3, da_wp, assets: Path):
    """Evaluate every forecast of a scenario; returns slide dicts."""
    slides = []
    act = {"readiness": None, "action": None, "observational": None}
    for i, fc in enumerate(sc.forecasts):
        path = sc.write_csv(fc, HERE / "simex_injects")
        r = evaluate_forecast(path)
        when = day_label(fc.issue_h)
        img = f"assets/{sc.key}_{i + 1}.png"
        forecast_map(r, adm3, da_wp, SIMEX_EXTENT, assets.parent / img)
        r64, a64 = r["exp_readiness"][64], r["exp_action"][64]
        if r64 >= THRESHOLD and act["readiness"] is None:
            act["readiness"] = when
        if a64 >= THRESHOLD and act["action"] is None:
            act["action"] = when
        slides.append(
            {
                "type": "forecast",
                "n": i + 1,
                "of": len(sc.forecasts),
                "when": when,
                "issue_h": fc.issue_h,
                "img": img,
                "r64": r64,
                "a64": a64,
                "r34": r["exp_readiness"][34],
                "d64": round_km(r["dist_64_km"]),
                "dctr": round_km(r["dist_centre_km"]),
                "maxcat": max_cat(r),
                "readiness": _status(r64, THRESHOLD, act["readiness"], when),
                "action": _status(a64, THRESHOLD, act["action"], when),
                "readiness_at": act["readiness"],
                "action_at": act["action"],
                "track": r["track"],
            }
        )
    return slides, act


def rain_check(sc, days, peaks, targets, adm3, assets, key):
    """Synthetic 2-day IMERG check for the given days (1-based)."""
    fields = []
    for d, peak, target in zip(days, peaks, targets):
        centres = [sc.true_at(h)[:2] for h in range((d - 1) * 24, d * 24, 3)]
        da = rain.daily_rain(centres, peak, seed=d * 7 + len(key))
        fields.append(
            da if target is None else rain.scale_to(da, adm3, target)
        )
    da2 = fields[0] + fields[1]
    mm = rain.national_mean(da2, adm3)
    track_hours = range((days[0] - 1) * 24, days[1] * 24 + 1, 6)
    track = {
        "Latitude": [sc.true_at(h)[0] for h in track_hours],
        "Longitude": [sc.true_at(h)[1] for h in track_hours],
    }
    img = f"assets/{key}.png"
    rain_map(
        rain.land_only(da2, adm3),
        adm3,
        RAIN_EXTENT,
        assets.parent / img,
        track=track,
    )
    peak_px = float(da2.rio.clip(adm3.geometry, all_touched=True).max())
    centre_km = min(
        _centre_dist_km(sc.true_at(h)[:2], adm3)
        for h in range((days[0] - 2) * 24, (days[1] + 1) * 24, 3)
    )
    return {
        "type": "rain",
        "img": img,
        "mm": round(mm),
        "peak": int(round(peak_px / 10) * 10),
        "days": f"Days {days[0]} and {days[1]}",
        "centre_km": round_km(centre_km),
    }


def _centre_dist_km(latlon, adm3):
    import geopandas as gpd
    from shapely.geometry import Point

    p = gpd.GeoSeries([Point(latlon[1], latlon[0])], crs=adm3.crs)
    land = adm3.to_crs(3832).union_all()
    return land.distance(p.to_crs(3832).iloc[0]) / 1000


def build_simex(out: Path):
    adm3, da_wp = load_static()
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for f in (HERE / "simex_injects").glob("*.csv"):
        f.unlink()

    scenarios = {}

    # 1 · Alpha: readiness then action
    s, act = run_scenario(ALPHA, adm3, da_wp, assets)
    for kind in ["readiness", "action"]:
        when = act[kind]
        trigger_email(
            kind,
            "Alpha",
            f"{when} (Fiji time)",
            assets / f"alpha_email_{kind}.png",
        )
        idx = next(i for i, x in enumerate(s) if x.get("when") == when)
        s.insert(
            idx + 1,
            {
                "type": "email",
                "kind": kind,
                "when": when,
                "img": f"assets/alpha_email_{kind}.png",
            },
        )
    scenarios["alpha"] = {"slides": s, "act": act}

    # 2 · Bravo: readiness, then nothing
    s, act = run_scenario(BRAVO, adm3, da_wp, assets)
    when = act["readiness"]
    trigger_email(
        "readiness",
        "Bravo",
        f"{when} (Fiji time)",
        assets / "bravo_email_readiness.png",
    )
    idx = next(i for i, x in enumerate(s) if x.get("when") == when)
    s.insert(
        idx + 1,
        {
            "type": "email",
            "kind": "readiness",
            "when": when,
            "img": "assets/bravo_email_readiness.png",
        },
    )
    rc = rain_check(
        BRAVO, (3, 4), (90, 60), (None, None), adm3, assets, "bravo_rain"
    )
    assert rc["mm"] < RAIN_THRESHOLD, rc["mm"]
    rc.update({"when": "Day 5, 12:00", "status": "notmet"})
    s.append(rc)
    scenarios["bravo"] = {"slides": s, "act": act}

    # 3 · Charlie: observational only
    s, act = run_scenario(CHARLIE, adm3, da_wp, assets)
    rc1 = rain_check(
        CHARLIE, (2, 3), (180, 260), (40, 85), adm3, assets, "charlie_rain_1"
    )
    rc1.update({"when": "Day 4, 12:00", "status": "notmet"})
    rc2 = rain_check(
        CHARLIE, (3, 4), (260, 330), (85, 125), adm3, assets, "charlie_rain_2"
    )
    rc2.update(
        {
            "when": "Day 5, 12:00",
            "status": "new" if rc2["mm"] >= RAIN_THRESHOLD else "notmet",
        }
    )
    assert rc1["mm"] < RAIN_THRESHOLD <= rc2["mm"], (rc1["mm"], rc2["mm"])
    act["observational"] = rc2["when"]
    # both rain checks come after the last forecast (Day 4, 00:00)
    s.append(rc1)
    s.append(rc2)
    observational_email(
        "Charlie",
        rc2["mm"],
        "Days 3 and 4",
        assets / "charlie_email_observational.png",
    )
    s.append(
        {
            "type": "email",
            "kind": "observational",
            "when": rc2["when"],
            "img": "assets/charlie_email_observational.png",
        }
    )
    scenarios["charlie"] = {"slides": s, "act": act}

    for sc in scenarios.values():
        for x in sc["slides"]:
            x.pop("track", None)
    (HERE / "numbers").mkdir(exist_ok=True)
    (HERE / "numbers" / "simex.json").write_text(
        json.dumps(scenarios, indent=1, default=str) + "\n"
    )
    return scenarios


# --------------------------------------------------------------------------
# Render + PDF
# --------------------------------------------------------------------------


def render(template: str, out: Path, **ctx):
    env = Environment(
        loader=FileSystemLoader(HERE / "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt"] = fmt
    html = env.get_template(template).render(
        threshold=THRESHOLD,
        rain_threshold=RAIN_THRESHOLD,
        rain_levels=RAIN_LEVELS,
        **ctx,
    )
    # no trailing whitespace / final newline, as the pre-commit hooks want
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (out / "index.html").write_text(html)


def to_pdf(out: Path, pdf_name: str):
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--no-pdf-header-footer",
            "--virtual-time-budget=4000",
            f"--print-to-pdf={out / pdf_name}",
            (out / "index.html").as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def to_pptx(out: Path, pptx_name: str, title: str):
    """Editable PowerPoint, laid out from the rendered HTML."""
    pptx_export.export(out / "index.html", out / pptx_name, title, CHROME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fms-dir", type=Path, required=False)
    ap.add_argument("--only", choices=["lookback", "simex"])
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="re-render HTML/PDF/PowerPoint from slides/numbers/*.json",
    )
    args = ap.parse_args()

    if args.only in (None, "lookback"):
        out = DOCS / "2025-26-season"
        if args.render_only:
            data = json.loads((HERE / "numbers" / "lookback.json").read_text())
        else:
            data = build_lookback(args.fms_dir, out)
        pdf = "fiji-aa-2025-26-season-lookback.pdf"
        pptx = "fiji-aa-2025-26-season-lookback.pptx"
        render("lookback.html.j2", out, d=data, pdf=pdf, pptx=pptx)
        to_pdf(out, pdf)
        to_pptx(out, pptx, TITLES[out.name])

    if args.only in (None, "simex"):
        out = DOCS / "simex"
        if args.render_only:
            data = json.loads((HERE / "numbers" / "simex.json").read_text())
        else:
            data = build_simex(out)
        pdf = "fiji-aa-simulation-exercise.pdf"
        pptx = "fiji-aa-simulation-exercise.pptx"
        render("simex.html.j2", out, sc=data, pdf=pdf, pptx=pptx)
        to_pdf(out, pdf)
        to_pptx(out, pptx, TITLES[out.name])


if __name__ == "__main__":
    main()
