import pandas as pd
import time
import multiprocessing as mp
from backtester import engine, tester
from backtester import API_Interface as api
import numpy as np

training_period = 20  # How far the rolling average takes into calculation
standard_deviations = 3.5  # Number of Standard Deviations from the mean the Bollinger Bands sit
vector_len = 3  # dimensions of the vector
vector_list = []  # list of points as features
vector_value = []  # list of value of features
k = 3  # number of nearest neighbours that contribute to the vote


def logic(account, lookback, sequence):  # Logic function to be used for each time interval in backtest
    """
    Context: Called for every row in the input data.
    Input:  account - the account object
            lookback - the lookback dataframe, containing all data up until this point in time
    Output: none, but the account object will be modified on each call
    """
    today = len(lookback)
    if today >= 20:
        target = lookback['x1'].values

        target = target[today-20:today]  # target sequence of length 20 (1 less than trained)
        mean, var = sequence.compare_local(target, 3)
        print(" > goal: {}".format(target[20 - 1]))
        print("period {}:, mean: {}, var: {}".format(today, mean, var))


def preprocess_data(stock_id):
    """
    Preprocess data by adding columns to the panda dataframe,
        x0 is a zeroth order derivative (typical price),
        x1 is the first order derivative (velocity),
        x2 is the second order derivative (acceleration)
        m is the momentum (acceleration*normalised volume, where volume replaces mass)
    """
    processed_stock_id = []
    for stock in stock_id:
        df = pd.read_csv("data/" + stock + ".csv", parse_dates=[0])
        # generate the underlying function (typical price)
        df['x0'] = (df['open'] + df['low'] + df['close'])/3
        # generate a first order derivative
        df['x1'] = (df['x0'].shift(1) - df['x0'])/df['x0']
        df.at[0, 'x1'] = 0  # remove NaN result from shifting creating a vacancy
        # generate a second order derivative
        df['x2'] = (df['x1'].shift(1) - df['x1'])/df['x1']
        df.at[0, 'x2'] = 0  # remove NaN result from shifting creating a vacancy
        df.at[1, 'x2'] = 0  # remove effect of removing NaN result
        # generate momentum
        df['m'] = df['x2']*df['volume']/df['volume'].sum()
        # save Pandas dataframe to .csv
        df.to_csv("data/" + stock + "_processed.csv", index=False)
        processed_stock_id.append(stock + "_processed")
    return processed_stock_id


if __name__ == "__main__":
    # List of stock data csv's to be tested, located in "data/" folder
    stock_id = ["TSLA_2020-03-09_2022-01-28_15min"]
    # Preprocess the data
    stock_id_proccessed = preprocess_data(stock_id)
    # Run backtest on list of stocks using the logic function
    results = tester.test_array(stock_id_proccessed, logic, chart=True)

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    # Create dataframe of results
    column_headers = ["Buy and Hold", "Strategy", "Longs", "Sells", "Shorts",
                      "Covers", "Stdev_Strategy", "Stdev_Hold", "Stock"]
    df = pd.DataFrame(list(results), columns=column_headers)
    # Save results to csv
    df.to_csv("results/Test_Results.csv", index=False)
