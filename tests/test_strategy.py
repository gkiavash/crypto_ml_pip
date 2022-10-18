from crypto_ml.strategy import *


def test_strategy():
    df_test_path = "tests/BTCUSDT_15m_1 Oct, 2022_9 Oct, 2022.csv"
    df_test = pd.read_csv(df_test_path, header=0, delimiter=",", index_col="unix")
    # df_test.set_index("unix")

    print(df_test.values.shape)
    print(df_test.head())

    st = TestStrategy(
        df_raw=df_test[:200], rate_sell_profit=0.007, rate_stop_limit=-0.005
    )

    for i in range(200, 220):
        y_hat_strategy, y_hat_strategy_classes = st.run_(df_test.values[i])


def test_rf_strategy():
    df_test_path = "tests/BTCUSDT_15m_1 Oct, 2022_9 Oct, 2022.csv"
    df_test = pd.read_csv(df_test_path, header=0, delimiter=",")
    df_test.set_index("unix")

    model_xg = get_test_rf_model()

    st = RandomForestTestStrategy(
        df_raw=df_test[:200], rate_sell_profit=0.007, rate_stop_limit=-0.005, model=model_xg
    )

    for i in range(200, 220):
        y_hat_strategy, y_hat_strategy_classes = st.run_(df_test.values[i])


def get_test_rf_model():
    # test classification dataset
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=1000, n_features=10)
    print(X.shape, y.shape)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    return model
