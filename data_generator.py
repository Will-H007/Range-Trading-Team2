from backtester import API_Interface as api

"""
Common stocks in S&P500:
AAPL
MSFT
AMZN
GOOG
NVDA
UNH
JNJ
FB
JPM
DIS
V
KO
PEP
LLY
TSLA


get_intraday_extended() function:
    Context: Used to get intraday_extended data from Alpha Vantage API.

    Requires: API_Key.txt located in the root folder, Access to the internet

    Input:  symbol: the stock symbol to be returned
            start_date: the starting date for the range of dates to be returned (Can set start date to "all" to return all available data)
            end_date: the ending date for the range of dates to be returned
            interval: the interval of the data to be returned (1min, 5min, 60min, 30min, 60min)
            combine: whether to combine the data into one dataframe or not, if not, saves each timeslice as it's own csv, (combine=True is recommended)

    Output: data: the dataframe containing the intraday_extended data
            OR
            data/SYMBOL_STARTDATE_ENDDATE_INTERVAL.csv: the csv file containing the intraday_extended data

"""


# api.get_intraday_extended('AAPL', 'all', '', '1min', True)
# api.get_intraday_extended('GOOG', 'all', '', '1min', True)
# api.get_intraday_extended('IBM', 'all', '', '1min', True)
# api.get_intraday_extended('MSFT', 'all', '', '1min', True)
# api.get_intraday_extended('AMZN', 'all', '', '1min', True)
# api.get_intraday_extended('TSLA', 'all', '', '1min', True)
other_stocks = ["JNJ","XOM"] #"NVDA","BRK.B","META","UNH",
for stocks in other_stocks:
        api.get_intraday_extended(stocks, 'all', '', '1min', True)
api.get_intraday_extended('TSLA', 'year1month2', 'year1month1', '60min', True, False)
api.get_intraday_extended('TSLA', 'all', 'year1month1', '60min', True, True)