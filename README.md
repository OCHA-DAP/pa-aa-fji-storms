# Fiji Anticipatory Action: storms

[![Generic badge](https://img.shields.io/badge/STATUS-ENDORSED-%231EBFB3)](https://shields.io/)

## Background information

Work began on the Fiji Tropical Cyclone AA Pilot in June 2023.

The tropical cyclone (TC) trigger has been developed with Fiji Meteorological
Services (FMS, operators of RSMC Nadi) in
consultation with the National Disaster Management Office (NDMO).

Over August 28-29 2023,
a [workshop](https://www.linkedin.com/feed/update/urn:li:activity:7103241608472514560/)
was held
by OCHA in Suva to build understanding of the framework and finalize the trigger
mechanism.

The proposed trigger has two stages, based on FMS's forecasts:

- Readiness: unofficial internal 120-hr forecast
- Action: official 72-hr forecast

The proposed trigger threshold is a TC that forecast to either:

- Be at Category 4 or greater while within 250 km of any point in Fiji, _or_
- Be at Category 3 or greater while making landfall in Fiji

The anticipatory action framework was formally endorsed by the Emergency Relief Coordinator on Nov 13, 2023.

## Monitoring

The trigger is continuously monitored with the process:

1. FMS produces a new 120-hr forecast of the cyclone track (typically every 6 hours).
2. FMS emails a CSV of the 120-hr forecast to the Centre for Humanitarian Data.
3. A Power Automate flow sends the forecast via a POST request to the `check-trigger.yml` GitHub Action on this repo.
4. The `check-trigger.yml` runs the `src/update_trigger.py` script, which:
    1. Processes the forecast, checking it against the trigger.
    2. If either the _readiness_ or _action_ triggers have been met, send an email to activate the framework.
    3. Produces basic plots of the forecast, and sends them in an informational email (regardless of whether the trigger has been met).

## Overview of analysis

Key analysis in `analysis/`:

- `01_returnperiod.md`: Calculates return period of TCs by strength and distance to Fiji
- `02_historicaltriggers.md`: Checks historical forecasts against trigger
- `03_forecastplots.md`: Creates interactive plots of historical forecasts for simulation exercises
- `04_distances.md`: Calculates distance of historical forecasts and actual tracks to administrative divisions

## Data description

Datasets:

- FMS historical best tracks (private)
  - analysis in this repo is based on file received from FMS, but the
    historical tracks are also publicly available
    on [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
- FMS historical official 72-hr forecasts (private)
  - for Yasa, Harold, and Evan
- NDMO impact data (private)
  - housing destroyed / damaged
  - geolocated infrastructure damage
- Desinventar impact data (public)
- ECMWF track hindcasts (public)

## Directory structure

The code in this repository is organized as follows:

```shell

├── analysis      # Main repository of analytical work for the AA pilot
├── docs          # .Rmd files or other relevant documentation
├── exploration   # Experimental work not intended to be replicated
├── src           # Code to run any relevant data acquisition/processing pipelines
|
├── .gitignore
├── README.md
└── requirements.txt

```

## Reproducing this analysis

Create a directory where you would like the data to be stored,
and point to it using an environment variable called
`AA_DATA_DIR`.

Next create a new virtual environment and install the requirements with:

```shell
pip install -r requirements.txt
```

Finally, install any code in `src` using the command:

```shell
pip install -e .
```

If you would like to instead receive the processed data from our team, please
[contact us](mailto:centrehumdata@un.org).

## Development

All code is formatted according to black and flake8 guidelines.
The repo is set-up to use pre-commit.
Before you start developing in this repository, you will need to run

```shell
pre-commit install
```

The `markdownlint` hook will require
[Ruby](https://www.ruby-lang.org/en/documentation/installation/)
to be installed on your computer.

You can run all hooks against all your files using

```shell
pre-commit run --all-files
```

It is also **strongly** recommended to use `jupytext`
to convert all Jupyter notebooks (`.ipynb`) to Markdown files (`.md`)
before committing them into version control. This will make for
cleaner diffs (and thus easier code reviews) and will ensure that cell outputs
aren't
committed to the repo (which might be problematic if working with sensitive
data).
