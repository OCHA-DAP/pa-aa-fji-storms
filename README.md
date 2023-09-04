# Fiji Anticipatory Action: storms

[![Generic badge](https://img.shields.io/badge/STATUS-UNDER%20DEVELOPMENT-%23007CE0)](https://shields.io/)

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
- Activation: official 72-hr forecast

The proposed trigger threshold is a TC that forecast to either:

- Be at Category 4 or greater while within 250 km of any point in Fiji, _or_
- Be at Category 3 or greater while making landfall in Fiji

## Overview of analysis

The repo currently only contains the analysis of the historical data,
in `exploration`.

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

To run the pipeline that downloads and processes the data, execute:

```shell
python src/main.py
```

To see runtime options, execute:

```shell
python src/main.py -h
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
