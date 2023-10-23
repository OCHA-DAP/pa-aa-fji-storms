import argparse
import logging
import os

from src import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clobber", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    return parser.parse_args()


def run_pipeline(clobber: bool = False):
    utils.download_codab(clobber=clobber)
    utils.process_buffer(distance=250, clobber=clobber)
    forecast_names = [
        x for x in os.listdir(utils.CURRENT_FCAST_DIR) if not x.startswith(".")
    ]
    for forecast_name in forecast_names:
        utils.check_fms_forecast(
            utils.CURRENT_FCAST_DIR / forecast_name, clobber=clobber
        )
    # check trigger
    # write trigger report
    pass


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    run_pipeline(clobber=args.clobber)
