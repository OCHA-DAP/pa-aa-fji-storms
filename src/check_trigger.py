import argparse
import base64
from io import StringIO

import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    return parser.parse_args()


def check_trigger(csv: str):
    """
    Checks trigger, from GitHub Action

    Parameters
    ----------
    csv: str
        Encoded string outputted from Power Automate of
        raw CSV file of forecast

    Returns
    -------

    """
    bytes = csv.encode("ascii") + b"=="
    converted_bytes = base64.b64decode(bytes)
    # print(converted_bytes)
    # csv_json = converted_bytes.decode("utf8")
    # print(csv_json)
    csv_str = converted_bytes.decode("ascii")
    filepath = StringIO(csv_str)
    df = utils.load_fms_forecast(filepath)
    print(df)
    print("checked trigger")


if __name__ == "__main__":
    args = parse_args()
    check_trigger(csv=args.csv)
