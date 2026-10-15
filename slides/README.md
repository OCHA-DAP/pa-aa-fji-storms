# Slide decks

Two browser slide decks published on the Pages site (`docs/slides/`), each with a
PDF download:

| Deck | URL | What it is |
|---|---|---|
| 2025–26 season look-back | `/slides/2025-26-season/` | Every RSMC Nadi forecast for Urmil (Feb 2026) and Vaianu (Apr 2026) against the trigger, plus the observed rainfall. ~5 minutes. |
| Simulation exercise | `/slides/simex/` | Three fictional cyclones stepped through forecast by forecast: readiness → action (Alpha), readiness → nothing (Bravo), observational only (Charlie). |

Every number on the slides comes from the live trigger code
(`src/datasources/fms.py`, `src/exposure_calc.py`, the same functions
`pipelines/monitor_forecast_trigger.py` uses) — `trigger_eval.py` wraps them.

## Files

- `simex_storms.py` — the three fictional storms: a true track plus forecasts
  whose error shrinks over time, written as RSMC Nadi TC-module CSVs.
- `simex_injects/` — those CSVs. They also work as injects for a hands-on
  exercise: base64 them into the `Check Trigger` workflow with `TEST_EMAIL=True`.
- `rain.py` — synthetic IMERG-like rainfall for the exercise, averaged over the
  country as in `exploration/rain_plot.md`.
- `figures.py` — forecast and rainfall maps.
- `build_decks.py` — computes everything and renders `templates/*.j2` to
  `docs/slides/<deck>/index.html`, then prints each deck to PDF with headless Chrome.
- `numbers/*.json` — the computed numbers, so the HTML/PDF can be re-rendered
  without recomputing.
- `lookback_inputs/` — the two IMERG maps from the observational emails sent
  during Vaianu.

## Rebuild

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r monitoring-requirements.txt jinja2
uv pip install -p .venv/bin/python -e . --no-deps

# everything (needs the RSMC Nadi CSVs — saved by the Power Automate flow to
# OneDrive/fji_fms_forecasts; not committed — and dev blob read access)
.venv/bin/python slides/build_decks.py --fms-dir <folder with the 2026*.csv files>

# one deck, or just re-render HTML + PDF after editing a template
.venv/bin/python slides/build_decks.py --only simex
.venv/bin/python slides/build_decks.py --render-only
```

Needs Google Chrome at the default macOS path for the email images and PDFs.
