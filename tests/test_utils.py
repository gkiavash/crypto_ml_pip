import os.path

import numpy as np

import crypto_ml
from crypto_ml.utils import *


def test_split_sequence():
    d = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = np.array(d)
    split_sequence(df, 5, 2)
    print(df)


def test_split_series():
    d = [
        [1, 11, 111],
        [2, 22, 222],
        [3, 33, 333],
        [4, 44, 444],
        [5, 55, 555],
        [6, 66, 666],
        [7, 77, 777],
        [8, 88, 888],
    ]
    df = np.array(d)
    x, y = split_series(df, n_past=3, n_future=1, col_output=(0,))
    for i in range(len(x)):
        print(x[i], y[i])


def test_split_series_unix():
    from crypto_ml.utils import split_series
    import pandas as pd

    df = pd.read_csv("tests/test.csv", header=0, delimiter=",")
    print(df.values)
    x, y = split_series(df.values, n_past=3, n_future=1, col_output=None)
    print(x, y)
    for i in range(len(x)):
        print(x[i], y[i])


def test_signal_buy():
    df = pd.read_csv("tests/test.csv", header=0, delimiter=",")
    print(df.head())
    df = signal_buy(
        df=df,
        max_minutes_later=5,
        min_percent_profit=0.006,
        col_names=crypto_ml.col_names,
        lower_bound=False,
    )
    # print(df)
    assert (df["signal_buy"].tolist()) == [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0]


def test_combined():
    row_start = 20

    BASE_PATH = "E:\Projects\crypto_ml\datasets"
    df_smaller = pd.read_csv(os.path.join(BASE_PATH, "EGLDUSDT_5m_1 Jan, 2022_30 Sep, 2022.csv"), header=0, delimiter=",")
    df_bigger = pd.read_csv(os.path.join(BASE_PATH, "EGLDUSDT_15m_1 Jan, 2022_30 Sep, 2022.csv"), header=0, delimiter=",")

    df_smaller = df_smaller[:row_start]
    df_bigger = df_bigger[:row_start]

    df = combine(df_smaller, df_bigger, ["open"])
    print(df.head(25))
    print(df.values)
