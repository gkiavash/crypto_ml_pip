import os
import logging
from dotenv import load_dotenv
from pathlib import Path

from binance.client import Client

logging.basicConfig(
    filename="crypto_ml.log", level=logging.INFO, datefmt="%d-%b-%y %H:%M:%S"
)
dotenv_path = Path("../.env")
load_dotenv(dotenv_path=dotenv_path)


class Config:
    SYMBOL = "EGLDUSDT"
    INTERVAL = Client.KLINE_INTERVAL_15MINUTE
    RAW_DATASET_START_DATETIME = "1 Jan, 2022"
    RAW_DATASET_END_DATETIME = "30 Sep, 2022"

    # RAW_DATASET_START_DATETIME = "1 Oct, 2022"
    # RAW_DATASET_END_DATETIME = "16 Oct, 2022"

    NEW_FILE_NAME_WITHOUT_EXTENSION = "{SYMBOL}_{INTERVAL}_{START}_{END}".format(
        SYMBOL=SYMBOL,
        INTERVAL=INTERVAL,
        START=RAW_DATASET_START_DATETIME,
        END=RAW_DATASET_END_DATETIME,
    )

    RAW_DATASET_DIRECTORY = "datasets"
    RAW_DATASET_FULL_PATH = os.path.join(
        RAW_DATASET_DIRECTORY, f"{NEW_FILE_NAME_WITHOUT_EXTENSION}.csv"
    )
    if not os.path.exists(RAW_DATASET_DIRECTORY):
        os.makedirs(RAW_DATASET_DIRECTORY)

    SIGNAL_MAX_MINUTE_LATER = 6
    SIGNAL_MIN_PERCENT_PROFIT = 0.006

    api_key = os.getenv("api_key")
    api_secret = os.getenv("api_secret")
