from crypto_ml import utils, indicators
import crypto_ml


class TestStrategy:
    def __init__(self,
                 df_raw,
                 scaler=None,
                 model=None,
                 wallet_usdt=1000,
                 wallet_btc=0,
                 rate_fee_transaction=0.001,
                 rate_stop_limit=-0.001,
                 rate_sell_profit=0.004,
                 ):
        self.df_raw = df_raw
        if len(self.df_raw) > 200:
            self.df_raw = self.df_raw[-200:]

        self.scaler = scaler,
        self.model = model,

        self.wallet_usdt = wallet_usdt
        self.wallet_btc = wallet_btc
        self.rate_fee_transaction = rate_fee_transaction
        self.rate_stop_limit = rate_stop_limit
        self.rate_sell_profit = rate_sell_profit

        self.positions = []

    def btc_buy(self, btc_price, btc_qty=0.0001):
        print(self.wallet_usdt, btc_price, btc_qty)
        if (self.wallet_usdt - btc_qty * btc_price) < 0:
            raise Exception('Not enough USDT')
        self.wallet_usdt -= btc_qty * btc_price
        self.wallet_btc += btc_qty - (btc_qty * self.rate_fee_transaction)
        self.positions.append(
            {
                'btc_price': btc_price,
                'btc_qty': btc_qty * (1 - self.rate_fee_transaction)
             }
        )

        log_str = 'Bought {} with price {}, wallet_usdt: {}, wallet_btc: {}'.format(btc_qty,
                                                                                    btc_price,
                                                                                    self.wallet_usdt,
                                                                                    self.wallet_btc)
        print(log_str)

    def btc_sell(self, btc_price, btc_qty=0.0001):
        if self.wallet_btc - btc_qty < -0.001:
            raise Exception('Not enough BTC')

        self.wallet_btc -= btc_qty
        self.wallet_usdt += (btc_qty * btc_price) - (btc_qty * btc_price * self.rate_fee_transaction)

    def positions_check(self, btc_price):
        for index, position in enumerate(self.positions):
            if utils.percent_calc(btc_price, position['btc_price']) >= self.rate_sell_profit \
            or utils.percent_calc(btc_price, position['btc_price']) < (self.rate_stop_limit - 2 * self.rate_fee_transaction):
                self.btc_sell(
                    btc_qty=position['btc_qty'],
                    btc_price=btc_price,
                )
                position["btc_price_cell"] = btc_price

    def run(self, dataset, y_hat_classes, col_names=crypto_ml.col_names):
        for index, y_ in enumerate(y_hat_classes):
            btc_prices = dataset[col_names['close']]

            if y_ == 1:
                self.btc_buy(btc_price=btc_prices[index])
            self.positions_check(btc_price=btc_prices[index])

    def run_(self, new_row):
        self.df_raw.loc[len(self.df_raw.index)] = new_row
        self.df_indicator = self.df_raw.copy(deep=True)

        self.df_indicator = indicators.prepare_dataset(self.df_indicator)
        print(self.df_indicator.head())

        X_strategy = self.df_indicator.values[-3:]
        X_strategy.astype('float64')

        X_strategy = self.scaler.fit_transform(X_strategy)

        y_hat_strategy = self.model.predict(X_strategy)
        return y_hat_strategy
