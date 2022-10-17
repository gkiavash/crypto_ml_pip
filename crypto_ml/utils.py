import numpy as np
import pandas as pd


def add_previous_rows(df, columns, shifts):
    for col in columns:
        for shift in shifts:
            df["{}_{}".format(col, shift)] = df[col].shift(shift)
    return df


def add_m(df, column, shift):
    df["{}_{}".format(column, shift)] = df[column].shift(shift)
    df["{}_{}_m".format(column, shift)] = df.apply(
        lambda row: row[column] - row["{}_{}".format(column, shift)], axis=1
    ).round(4)
    df.drop(["{}_{}".format(column, shift)], axis=1, inplace=True)
    return df


def percent_calc(price_new, price_base):
    return round((price_new - price_base) / price_base, 6)


def to_supervised(train, n_out=7):  # (x,1,y)
    train = train.tolist()
    for row_num in range(len(train)):
        for row_seq in range(1, n_out):
            ind_adding = row_num - row_seq
            if ind_adding < 0:
                ind_adding = 0
            train[row_num].append(train[ind_adding][0])
    return np.array(train)


def split_sequence(sequence, n_steps_in, n_steps_out):
    """
    :param sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    :param n_steps_in: number of inputs, e.g 6
    :param n_steps_out:  number of outputs, e.g. 2
    :return: sequenced dataframe. For the example above:
    X = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7]
    ],
    y = [
        [6, 7],
        [7, 8],
        [8, 9]
    ]
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_series(series, n_past, n_future, col_output=None):
    """
    :param series: dataset
    :param n_past: number of input steps, e.g. 3: consider 3 previous hours
    :param n_future: number of output steps, e.g. 1: next 1 hours
    :param col_output: index of output column(s) must be iterable. If None then the whole row
    :return: x, y. output for example above:

    x:              y:
    [[  1  11 111]
     [  2  22 222]
     [  3  33 333]] [[4]]
    [[  2  22 222]
     [  3  33 333]
     [  4  44 444]] [[5]] ...
    """
    # n_past ==> no of past observations
    # n_future ==> no of future observations
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future[:, col_output] if col_output is not None else future)
    return np.array(X), np.array(y)


def signal_buy(
    df,
    max_minutes_later,
    min_percent_profit,
    col_names,
    lower_bound=False,
):
    # if need_reverse:
    #     df = df[::-1].reset_index()

    col_close = col_names["close"]
    col_high = col_names["high"]

    col_names_max = [
        "{}_{}".format(col_names["high"], shift)
        for shift in range(1, max_minutes_later + 1)
    ]
    for shift in range(1, max_minutes_later + 1):
        df["{}_{}".format(col_high, shift)] = df[col_high].shift(-shift)
    print("Done: shift")

    df["percent_max_future_rows"] = df.apply(
        lambda row: round(
            (row[col_names_max].max() - row[col_close]) / row[col_close], 4
        ),
        axis=1,
    ).round(4)
    print("Done: percent_max_future_rows")

    if lower_bound:
        print("lower_bound: True")
        col_names_min = [
            "{}_{}".format(col_names["low"], shift)
            for shift in range(1, max_minutes_later + 1)
        ]
        for shift in range(1, max_minutes_later + 1):
            df["{}_{}".format(col_names["low"], shift)] = df[col_names["low"]].shift(
                -shift
            )
        print("Done: shift lower_bound")

        df["percent_min_future_rows"] = df.apply(
            lambda row: round(
                (row[col_names_min].min() - row[col_close]) / row[col_close], 4
            ),
            axis=1,
        ).round(4)
        print("Done: percent_min_future_rows lower_bound")

        df["signal_buy"] = df.apply(
            lambda row: 1
            if row["percent_max_future_rows"] > min_percent_profit
            and row["percent_min_future_rows"] > (-min_percent_profit + 0.002)
            else 0,
            axis=1,
        )
        print("Done: signal_buy with lower bound")
        col_names_max += col_names_min + ["percent_min_future_rows"]
    else:
        df["signal_buy"] = df.apply(
            lambda row: 1 if row["percent_max_future_rows"] > min_percent_profit else 0,
            axis=1,
        )
        print("Done: signal_buy without lower bound")

    col_names_max += [
        "percent_max_future_rows",
    ]

    # if need_reverse:
    #     col_names_max += ['index']

    df.drop(col_names_max, axis=1, inplace=True)

    print(df.head())
    print(df.columns.tolist())
    print(df.groupby(["signal_buy"]).count())

    return df


def confusion_matrix(y_test_classes, y_hat_classes):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(y_hat_classes)):
        if y_hat_classes[i] == 1 and y_test_classes[i] == 1:
            true_pos += 1
        elif y_hat_classes[i] == 0 and y_test_classes[i] == 0:
            true_neg += 1
        elif y_hat_classes[i] == 1 and y_test_classes[i] == 0:
            false_pos += 1
        elif y_hat_classes[i] == 0 and y_test_classes[i] == 1:
            false_neg += 1

    print("true_pos", true_pos)
    print("true_neg", true_neg)
    print("false_pos", false_pos)
    print("false_neg", false_neg)
    return true_pos, true_neg, false_pos, false_neg


def get_y_prob(y_hat, prob_thresh=0.0):
    y_prob = []
    for y_i in y_hat:
        y_prob.append(y_i[1] / (y_i[0] + y_i[1]))

    y_prob = np.array(y_prob)
    # print("y_prob", y_prob)
    # print((y_prob > prob_thresh).sum())
    # print(y_prob[ np.where( y_prob >= prob_thresh ) ])
    y_hat_class = y_hat[np.where(y_prob > prob_thresh)].argmax(axis=-1)

    return y_prob, y_hat_class
