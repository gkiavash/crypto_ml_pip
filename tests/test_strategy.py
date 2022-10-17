from crypto_ml.strategy import *


def test_strategy():
    df_test_path = "tests/BTCUSDT_15m_1 Oct, 2022_9 Oct, 2022.csv"
    df_test = pd.read_csv(df_test_path, header=0, delimiter=",")
    df_test.set_index("unix")

    st = TestStrategy(
        df_raw=df_test[:200], rate_sell_profit=0.007, rate_stop_limit=-0.005
    )

    for i in range(200, 250):
        y_hat_strategy, y_hat_strategy_classes = st.run_(df_test.values[i])
