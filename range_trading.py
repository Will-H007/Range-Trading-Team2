from cgi import test
from itertools import count
import pandas as pd
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from joblib import dump, load
# local imports
from backtester import engine, tester
from backtester import API_Interface as api

training_period = 200 # How far the rolling average takes into calculation
standard_deviations = 3 # Number of Standard Deviations from the mean the Bollinger Bands sit

def counts(df):
  # Consecutive days that got above the upper bollinger band
  if df['b1'] > 1 and df['b2'] > 1 and df['b3'] < 1: return 2
  elif df['b1'] > 1 and df['b2'] > 1 and df['b3'] > 1: return 3

    # Consecutive days that got below the lower bollinger band  
  elif df['b1'] < 0 and df['b2'] < 0 and df['b3'] > 0: return -2
  elif df['b1'] < 0 and df['b2'] < 0 and df['b3'] < 0: return -3


  else: return 0


'''
logic() function:
    Context: Called for every row in the input data.

    Input:  account - the account object
            df - the df dataframe, containing all data up until this point in time

    Output: none, but the account object will be modified on each call

'''
def get_adx(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    return plus_di, minus_di, adx_smooth







def logic(account, lookback): # Logic function to be used for each time interval in backtest 
    
    today = len(lookback)-1
    column_names = list(lookback.columns) # ['date', 'open', 'high', 'low', 'close', 'volume']

    # The open is the starting period of trading on a securities exchange or organized over-the-counter market.
    # The close is simply the end of a trading session in the financial markets, however, closing times tend to vary between market and exchange.

    # The high is the highest price at which a stock is traded during a period. 
    # The low is the lowest price of the period. 
    # A stock’s high and low points for the day are often called its intraday high and low.

    # Volume is simply the number of shares traded in a particular stock, index, or other investment over a specific period of time.

    # Approx a day

    '''
    
    Develop Logic Here
    
    '''
    
    if(today > training_period ): # If the lookback is long enough to calculate the Bollinger Bands
    
        if lookback['adx'][today] < 20:
            if lookback['counts'][today] == -2 or lookback['counts'][today] == -3 and lookback['close'][today] > lookback['MA-TP'][today]:
                if(account.buying_power > 0):
                    account.enter_position('long', account.buying_power, lookback['close'][today]) # Enter a long position

            if lookback['b-percent'][today] > 0.8:
                for position in account.positions: # Close all current positions
                    account.close_position(position, 1, lookback['close'][today])
   

'''
preprocess_data() function:
    Context: Called once at the beginning of the backtest. TOTALLY OPTIONAL. 
             Each of these can be calculated at each time interval, however this is likely slower.

    Input:  list_of_stocks - a list of stock data csvs to be processed

    Output: list_of_stocks_processed - a list of processed stock data csvs
'''
def preprocess_data(list_of_stocks):
    list_of_stocks_processed = []
    for stock in list_of_stocks:
        df = pd.read_csv("data/" + stock + ".csv", parse_dates=[0])

        '''
        
        Modify Processing of Data To Suit Personal Requirements.
        
        '''

        # Bollinger Bands
        df['TP'] = (df['close'] + df['low'] + df['high'])/3
        df['std'] = df['TP'].rolling(training_period).std()
        df['MA-TP'] = df['TP'].rolling(training_period).mean()
        df['BOLU'] = df['MA-TP'] + 2*df['std']
        df['BOLD'] = df['MA-TP'] - 2*df['std']
        df['trend'] = df['MA-TP'].pct_change()
        df["b-percent"] = (df['close'] - df['BOLU']) / (df['BOLD'] - df['BOLU'])
        df['b1'] = df["b-percent"].shift(1)
        df['b2'] = df["b-percent"].shift(2)
        df['b3'] = df["b-percent"].shift(3)
        df['counts'] = df.apply(counts, axis = 1)

        # Average True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis = 1)
        true_range = np.max(ranges, axis = 1)
        df['ATR1'] = true_range.rolling(100).sum()/100
     
        # ADX
        df['plus_di'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], 14)[0]).rename(columns = {0:'plus_di'})
        df['minus_di'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], 14)[1]).rename(columns = {0:'minus_di'})
        df['adx'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], 14)[2]).rename(columns = {0:'adx'})
        df = df.dropna()
     

        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed






if __name__ == "__main__":
    stock_symbols = ["JNJ","XOM","AMZN","MSFT","IBM","GOOG","NVDA","META","UNH", "TSLA","PG","JPM","HD","CVX", "ABBV"] 
    # stock_symbols = ["JNJ","XOM","AMZN","MSFT","IBM","GOOG",,"NVDA","META","UNH", "TSLA","PG","JPM","HD","CVX", "ABBV"]
    list_of_stocks = [] # List of stock data csv's to be tested, located in "data/" folder  # "df_2020-03-24_2022-02-12_15min"
    for stock_symbol in stock_symbols:
        # list_of_stocks.append(stock_symbol + "_2020-10-11_2022-09-01_60min")
        # list_of_stocks.append(stock_symbol + "_2020-09-25_2022-08-16_1min")
        list_of_stocks.append(stock_symbol + "_2020-10-09_2022-08-30_15min")
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv