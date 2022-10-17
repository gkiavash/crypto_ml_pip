import os
import numpy as np
import pandas as pd

from binance.client import Client

from .config import Config as current_config


def get_data(current_config=current_config):
    client = Client(current_config.api_key, current_config.api_secret)

    klines = client.get_historical_klines(
        current_config.SYMBOL,
        current_config.INTERVAL,
        current_config.RAW_DATASET_START_DATETIME,
        current_config.RAW_DATASET_END_DATETIME,
    )
    klines_ = np.array(klines)
    df = pd.DataFrame(
        {
            "unix": klines_[:, 0],
            "open": klines_[:, 1],
            "high": klines_[:, 2],
            "low": klines_[:, 3],
            "close": klines_[:, 4],
            "volume": klines_[:, 5],
            "num_of_trades": klines_[:, 8],
        }
    )
    df["symbol"] = current_config.SYMBOL
    df = df.round(2)

    df.to_csv(current_config.RAW_DATASET_FULL_PATH, encoding="utf-8", index=False)
