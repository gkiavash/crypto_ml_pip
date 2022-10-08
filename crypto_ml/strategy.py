import numpy as np
import pandas as pd
from sklearn import preprocessing

from crypto_ml import utils, indicators

try:
    df_test_path = "tests/BTCUSDT_15m_1 Oct, 2022_9 Oct, 2022.csv"
    df_test = pd.read_csv(df_test_path, header=0, delimiter=",")
    df_test.set_index("unix")
except FileNotFoundError as e:
    pass


class TestStrategy:
    def __init__(self,
                 df_raw,
                 scaler=None,
                 model=None,
                 prob=0.5,
                 n_past=4,
                 wallet_usdt=1000,
                 wallet_btc=0,
                 rate_fee_transaction=0.001,
                 rate_stop_limit=-0.001,
                 rate_sell_profit=0.006,
                 ):
        self.df_raw = df_raw
        if len(self.df_raw) > 200:
            self.df_raw = self.df_raw[-200:]

        if scaler:
            self.scaler = scaler
        else:
            self.scaler = preprocessing.MinMaxScaler()

        self.model = model
        self.prob = prob
        self.n_past = n_past

        self.wallet_usdt = wallet_usdt
        self.wallet_btc = wallet_btc
        self.rate_fee_transaction = rate_fee_transaction
        self.rate_stop_limit = rate_stop_limit
        self.rate_sell_profit = rate_sell_profit

        self.positions = []

    def btc_buy(self, btc_price, btc_qty=0.0001):
        if (self.wallet_usdt - btc_qty * btc_price) < 0:
            raise Exception('Not enough USDT')
        self.wallet_usdt -= btc_qty * btc_price
        self.wallet_btc += ((1 - self.rate_fee_transaction) * btc_qty)
        self.positions.append(
            {
                'btc_price': btc_price,
                'btc_qty': btc_qty * (1 - self.rate_fee_transaction)
             }
        )
        print(f'Bought {btc_qty} BTC with price {btc_price}')

    def btc_sell(self, position, btc_price_cell):
        btc_qty = position['btc_qty']
        if self.wallet_btc - btc_qty < -0.001:
            raise Exception('Not enough BTC')

        self.wallet_btc -= btc_qty
        self.wallet_usdt += ((btc_qty * btc_price_cell) * (1 - self.rate_fee_transaction))
        position["btc_price_cell"] = btc_price_cell
        print(f'Sold {btc_qty} BTC with price {btc_price_cell}')

    def positions_check(self, btc_price):
        for index, position in enumerate(self.positions):
            if "btc_price_cell" in position.keys():
                continue
            if utils.percent_calc(btc_price, position['btc_price']) >= self.rate_sell_profit \
                or utils.percent_calc(btc_price, position['btc_price']) < (self.rate_stop_limit - 2 * self.rate_fee_transaction):
                self.btc_sell(position, btc_price_cell=btc_price)

    def run_(self, new_row):
        self.df_raw.loc[len(self.df_raw.index)] = new_row
        self.df_indicator = self.df_raw.copy(deep=True)

        self.df_indicator = indicators.prepare_dataset(self.df_indicator, is_drop=True)

        X_strategy = self.df_indicator.values[-self.n_past:]
        X_strategy.astype('float64')

        X_strategy = self.scaler.fit_transform(X_strategy)
        X_strategy = np.array([X_strategy])

        if self.model:
            y_hat_strategy = self.model.predict(X_strategy)
        else:
            y_hat_strategy = np.array([[0, 1]])

        y_hat_strategy_classes = y_hat_strategy.argmax(axis=-1)

        y_prob_strategy = y_hat_strategy[0][1]/(y_hat_strategy[0][0]+y_hat_strategy[0][1])

        btc_price = self.df_raw.iloc[-1]["close"]
        # if y_hat_strategy_classes[0] == 1:
        if y_prob_strategy > self.prob:
            self.btc_buy(btc_price=btc_price)
        self.positions_check(btc_price=btc_price)

        return y_hat_strategy, y_hat_strategy_classes
