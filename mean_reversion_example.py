import pandas as pd
import time
import multiprocessing as mp
from backtester import engine, tester
from backtester import API_Interface as api
from math import log2

training_period = 200  # How far the rolling average takes into calculation
standard_deviations = 3.5  # Number of Standard Deviations from the mean the Bollinger Bands sit


def logic(account, historical_data):  # Logic function to be used for each time interval in backtest
    """
    Context: Called for every row in the input data.
    Input:  account - the account object
            historical_data (dataframe): contains data points up to the current time
    Output: none, but the account object will be modified on each call
    """
    today = len(historical_data)-1

    # decide on positions after training period
    if today > training_period:
        resistance = historical_data['local_max'][today-1]  # -1 to not include today
        support = historical_data['local_min'][today-1]
        exit_price = log2(historical_data['close'][today]) - log2(historical_data['close'][today-1])
        # enter a long position if price penetrates resistance
        if historical_data['close'][today] > resistance:
            # close short positions
            for position in account.positions:
                if position.type_ == 'short':
                    account.close_position(position, 1, historical_data['close'][today])
            # open a long position
            if account.buying_power > 0:
                account.enter_position('long', account.buying_power, historical_data['close'][today])
        # enter a short position if price penetrates support
        elif historical_data['close'][today] < support:
            # close long positions
            for position in account.positions:
                if position.type_ == 'long':
                    account.close_position(position, 1, historical_data['close'][today])
            # open a short position
            if account.buying_power > 0:
                account.enter_position('short', account.buying_power, historical_data['close'][today])


def preprocess_data(list_of_stocks):
    """
    Context: preprocess data form alphavantage
    Input:  list_of_stocks (list), a list of stock data csvs to be processed
    Output: list_of_stocks_processed - a list of processed stock data csvs
    """
    list_of_stocks_processed = []
    for stock in list_of_stocks:
        data = pd.read_csv("data/" + stock + ".csv", parse_dates=[0])  # read raw .csv from data directory
        data['local_max'] = data['close'].rolling(training_period).max()  # calculate the local maximum
        data['local_min'] = data['close'].rolling(training_period).max()  # calculate the local minimum
        data.to_csv("data/" + stock + "_Processed.csv", index=False)  # write process .csv to data directory
        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed


if __name__ == "__main__":
    # List of stock data csv's to be tested, located in "data/" folder
    list_of_stocks = ["TSLA_2020-03-09_2022-01-28_15min"]
    # Preprocess the data
    list_of_stocks_proccessed = preprocess_data(list_of_stocks)
    # Run backtest on list of stocks using the logic function
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True)

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    # Create dataframe of results
    column_headers = ["Buy and Hold", "Strategy", "Longs", "Sells", "Shorts",
                      "Covers", "Stdev_Strategy", "Stdev_Hold", "Stock"]
    df = pd.DataFrame(list(results), columns=column_headers)
    # Save results to csv
    df.to_csv("results/Test_Results.csv", index=False)
