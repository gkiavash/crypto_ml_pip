import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import crypto_ml
from crypto_ml.utils import *
from crypto_ml.strategy import *


def test_strategy():
    df = pd.read_csv("tests/test.csv", header=0, delimiter=",")
    print(df.head())

    # df["open"][0:200].plot()
    # plt.show()
    ts = TestStrategy()
    ts.run(df, df['signal_buy_strategy'])
    print(ts.wallet_btc)
    print(ts.wallet_usdt)
    print(ts.positions)
