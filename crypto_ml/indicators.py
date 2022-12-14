import logging
import pandas as pd
import matplotlib.pyplot as plt

from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, PSARIndicator, MACD

import crypto_ml


def add_indicators_BollingerBands(
    df,
    col_names=crypto_ml.col_names,
    window=21,
    window_dev=2,
    ones_reverse=False,
    is_plot=False,
):
    indicator_bb = BollingerBands(
        close=df[col_names["close"]], window=window, window_dev=window_dev
    )
    # df['bb_bbm'] = indicator_bb.bollinger_mavg().round(2)
    # df['bb_bbh'] = indicator_bb.bollinger_hband().round(2)
    # df['bb_bbl'] = indicator_bb.bollinger_lband().round(2)

    # Add Bollinger Band high indicator
    # df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
    # Add Bollinger Band low indicator
    # df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    # Add Width Size Bollinger Bands
    # df['bb_bbw'] = indicator_bb.bollinger_wband().round(6)
    df["bb_bbp"] = indicator_bb.bollinger_pband().round(4)
    if ones_reverse:
        df.insert(1, "ones", 1)
        df["bb_bbp"] = (
            (df["ones"] / df["bb_bbp"])
            .round(4)
            .abs()
            .apply(lambda x: x if x < 10000 else 10000)
        )
        df.drop(["ones"], axis=1, inplace=True)
    if is_plot:
        print("Done: ", "Boll")
    return df


def add_indicators_RSIIndicator(
    df, col_names=crypto_ml.col_names, window=6, ones_reverse=False, is_plot=False
):
    df["rsi_{}_".format(window)] = (
        RSIIndicator(close=df[col_names["close"]], window=window).rsi().round(2)
    )
    if ones_reverse:
        df.insert(1, "ones", 1)
        df["rsi_{}_".format(window)] = (
            df["ones"] / df["rsi_{}_".format(window)]
        ).round(4)
        df.drop(["ones"], axis=1, inplace=True)
    if is_plot:
        print("Done: ", "RSI")
    return df


def add_indicators_EMAIndicator(
    df, col_names=crypto_ml.col_names, w1=7, w2=25, w3=99, extra=None, is_plot=False
):
    df["ema_{}_".format(w1)] = (
        EMAIndicator(close=df[col_names["close"]], window=w1).ema_indicator().round(2)
    )
    df["ema_{}_".format(w2)] = (
        EMAIndicator(close=df[col_names["close"]], window=w2).ema_indicator().round(2)
    )
    df["ema_{}_".format(w3)] = (
        EMAIndicator(close=df[col_names["close"]], window=w3).ema_indicator().round(2)
    )

    if extra == "binary":
        df["ema_{}_{}".format(w1, w2)] = df.apply(
            lambda row: 1
            if row["ema_{}_".format(w1)] > row["ema_{}_".format(w2)]
            else 0,
            axis=1,
        ).astype("category")
        df["ema_{}_{}".format(w1, w3)] = df.apply(
            lambda row: 1
            if row["ema_{}_".format(w1)] > row["ema_{}_".format(w3)]
            else 0,
            axis=1,
        ).astype("category")
        df["ema_{}_{}".format(w2, w3)] = df.apply(
            lambda row: 1
            if row["ema_{}_".format(w2)] > row["ema_{}_".format(w3)]
            else 0,
            axis=1,
        ).astype("category")

    if extra == "propostional":
        df["ema_{}_{}".format(w1, w2)] = (
            df["ema_{}_".format(w1)] / df["ema_{}_".format(w2)]
        ).round(3)
        df["ema_{}_{}".format(w1, w3)] = (
            df["ema_{}_".format(w1)] / df["ema_{}_".format(w3)]
        ).round(3)
        df["ema_{}_{}".format(w2, w3)] = (
            df["ema_{}_".format(w2)] / df["ema_{}_".format(w3)]
        ).round(3)
    if extra is not None:
        df.drop(
            [
                "ema_7_",
                "ema_25_",
                "ema_99_",
            ],
            axis=1,
            inplace=True,
        )
    if is_plot:
        print("Done: ", "EMA")
    return df


def add_indicators_MACDIndcator(
    df,
    col_names=crypto_ml.col_names,
    window_slow=26,
    window_fast=12,
    window_sign=9,
    is_plot=False,
):
    MACDIndcator = MACD(
        close=df[col_names["close"]],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
    )
    df["macd"] = MACDIndcator.macd().round(2)
    df["macd_diff"] = MACDIndcator.macd_diff().round(2)
    df["macd_signal"] = MACDIndcator.macd_signal().round(2)
    if is_plot:
        print("Done: ", "MACD")
    return df


def add_indicators_PSARIndicator(df, col_names=crypto_ml.col_names, is_plot=False):
    df["sar"] = (
        PSARIndicator(
            high=df[col_names["high"]],
            low=df[col_names["low"]],
            close=df[col_names["close"]],
            step=0.02,
            max_step=0.2,
        )
        .psar()
        .round(2)
    )

    df["sar_"] = df.apply(
        lambda row: 1 if row["sar"] < row[col_names["close"]] else 0, axis=1
    ).astype("category")
    df.drop(["sar"], axis=1, inplace=True)
    if is_plot:
        print("Done: ", "SAR")
    return df


def add_indicators_hour_percent_profit(
    df, col_names=crypto_ml.col_names, is_plot=False
):
    df["hour_percent_profit"] = df.apply(
        lambda row: (row[col_names["close"]] - row[col_names["open"]])
        / row[col_names["open"]],
        axis=1,
    ).round(4)
    if is_plot:
        print("Done: ", "hour_percent_profit")
    return df


def add_indicators_candle(df, col_names=crypto_ml.col_names, is_plot=False):
    df["candle"] = df.apply(
        lambda row: (row[col_names["close"]] - row[col_names["open"]])
        / (row[col_names["high"]] - row[col_names["low"]])
        if row[col_names["high"]] - row[col_names["low"]] != 0
        else 0,
        axis=1,
    ).round(4)
    if is_plot:
        print("Done: ", "candle")
    return df


def add_indicators_v2(df, INPUT_DATASET_PATH, OUTPUT_FULL_PATH, return_df, col_names):
    logging.info("###### " + OUTPUT_FULL_PATH + " ######")
    if df is None:
        df = pd.read_csv(INPUT_DATASET_PATH, header=0, delimiter=",")
        logging.info("dataset found: {}".format(df.columns.tolist()))
    df.dropna(inplace=True)

    df = add_indicators_cs2(df, col_names)

    if return_df:
        return df

    df.set_index("unix", inplace=True)
    df.to_csv(OUTPUT_FULL_PATH, encoding="utf-8", index=True)


def add_indicators_cs1(df, col_names):
    df = add_indicators_BollingerBands(
        df, col_names, window=21, window_dev=2, ones_reverse=False
    )
    df = add_indicators_RSIIndicator(df, col_names, window=6, ones_reverse=False)
    df = add_indicators_EMAIndicator(df, col_names, w1=7, w2=25, w3=99, extra="binary")
    df = add_indicators_MACDIndcator(
        df, col_names, window_slow=26, window_fast=12, window_sign=9
    )
    df = add_indicators_PSARIndicator(df, col_names)
    df = add_indicators_hour_percent_profit(df, col_names)
    df = add_indicators_candle(df, col_names)
    return df


def add_indicators_cs2(df, col_names):
    df = add_indicators_BollingerBands(
        df, col_names, window=21, window_dev=2, ones_reverse=False
    )
    df = add_indicators_RSIIndicator(df, col_names, window=6, ones_reverse=False)
    df = add_indicators_EMAIndicator(df, col_names, w1=7, w2=25, w3=99, extra="binary")
    df = add_indicators_MACDIndcator(
        df, col_names, window_slow=26, window_fast=12, window_sign=9
    )
    df = add_indicators_PSARIndicator(df, col_names)
    df = add_indicators_hour_percent_profit(df, col_names)
    # df = add_indicators_candle(df, col_names)
    return df


def prepare_dataset(df, is_plot=False, is_drop=False):
    df["open"] = (
        df["open"].rolling(window=11, win_type="gaussian", center=True).mean(std=1)
    )
    df["close"] = (
        df["close"].rolling(window=11, win_type="gaussian", center=True).mean(std=1)
    )
    df["high"] = (
        df["high"].rolling(window=11, win_type="gaussian", center=True).mean(std=1)
    )
    df["low"] = (
        df["low"].rolling(window=11, win_type="gaussian", center=True).mean(std=1)
    )

    df = add_indicators_BollingerBands(df, window=21, window_dev=2, ones_reverse=False)
    df = add_indicators_RSIIndicator(df, window=6, ones_reverse=False)
    df = add_indicators_EMAIndicator(df, w1=7, w2=25, w3=99, extra="binary")
    df = add_indicators_MACDIndcator(df, window_slow=26, window_fast=12, window_sign=9)
    df = add_indicators_PSARIndicator(df)
    df = add_indicators_hour_percent_profit(df)

    df["open_smooth"] = (
        df["open"].rolling(window=15, win_type="gaussian", center=True).mean(std=1)
    )
    if is_plot:
        df["open"][0:200].plot()
        df["open_smooth"][0:200].plot()
        plt.show()

    df["rsi_6_smooth"] = (
        df["rsi_6_"].rolling(window=15, win_type="gaussian", center=True).mean(std=1)
    )
    if is_plot:
        df["rsi_6_"][0:200].plot()
        df["rsi_6_smooth"][0:200].plot()
        plt.show()

    df["bb_bbp_smooth"] = (
        df["bb_bbp"].rolling(window=15, win_type="gaussian", center=True).mean(std=1)
    )
    if is_plot:
        df["bb_bbp"][0:200].plot()
        df["bb_bbp_smooth"][0:200].plot()
        plt.show()

    df["hour_percent_profit_smooth"] = (
        df["hour_percent_profit"]
        .rolling(window=15, win_type="gaussian", center=True)
        .mean(std=1)
    )
    if is_plot:
        df["hour_percent_profit"][0:200].plot()
        df["hour_percent_profit_smooth"][0:200].plot()
        plt.show()

    if is_plot:
        print(df.head(50))
        df["hour_percent_profit"].plot()
        plt.show()
        df["rsi_6_"].plot()
        plt.show()
        df["bb_bbp"].plot()
        plt.show()

    if is_drop:
        df.dropna(inplace=True)
        drop_cols = [
            "unix",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "num_of_trades",
            "symbol",
            "rsi_6_",
            "bb_bbp",
            "hour_percent_profit",
            "open_smooth",
        ]
        df.drop(drop_cols, axis=1, inplace=True, errors="ignore")
    return df
