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
from xgboost import XGBRegressor









# local imports
from backtester import engine, tester
from backtester import API_Interface as api

training_period = 20 # How far the rolling average takes into calculation
standard_deviations = 3.5 # Number of Standard Deviations from the mean the Bollinger Bands sit




'''
logic() function:
    Context: Called for every row in the input data.

    Input:  account - the account object
            df - the df dataframe, containing all data up until this point in time

    Output: none, but the account object will be modified on each call

'''

def counts(df):
  if df['fac-1'] > 0 and df['fac-2'] < 0 and df['fac-3'] < 0: return 1
  elif df['fac-1'] > 0 and df['fac-2'] > 0 and df['fac-3'] < 0: return 2
  elif df['fac-1'] > 0 and df['fac-2'] > 0 and df['fac-3'] > 0: return 3
  else: return 0

def geo_ret(ser):
    return (1+ser).prod()-1

def get_features(df):
    df['Pre-Rt'] = df['close'].pct_change()

    df['fac-1'] = df['Pre-Rt'].shift(1)
    df['fac-2'] = df['Pre-Rt'].shift(2)
    df['fac-3'] = df['Pre-Rt'].shift(3)
    df['fac-count'] = df.apply(counts, axis = 1)
    df['fac-4'] = df['Pre-Rt'].rolling(5).apply(geo_ret).shift(1)
    df['fac-5'] = df['Pre-Rt'].rolling(14).apply(geo_ret).shift(1)
    df['fac-6'] = df['Pre-Rt'].rolling(22).apply(geo_ret).shift(1)
    df['fac-7'] = df['Pre-Rt'].rolling(120).apply(geo_ret).shift(1)
    df['fac-8'] = df['Pre-Rt'].rolling(250).apply(geo_ret).shift(1)

    df['OneYearVolume'] = df['volume'].rolling(250).sum().shift(1)

    df['fac-9'] = df['volume'].shift(1)/df['OneYearVolume']
    df['fac-10'] = df['volume'].rolling(2).sum().shift(1)/df['OneYearVolume']
    df['fac-11'] = df['volume'].rolling(5).sum().shift(1)/df['OneYearVolume']
    df['fac-12'] = df['volume'].rolling(14).sum().shift(1)/df['OneYearVolume']
    df['fac-13'] = df['volume'].rolling(22).sum().shift(1)/df['OneYearVolume']
    df['fac-14'] = df['volume'].rolling(120).sum().shift(1)/df['OneYearVolume']
    return df

def risk_management(account):
    # safety_percentage = 1
    if account.buying_power > 6500:
        safety_percentage = 0.5
    elif account.buying_power > 5000 and account.buying_power <= 6500:
        safety_percentage = 0.4
    elif 3500 < account.buying_power and account.buying_power <= 5000:
        safety_percentage = 0.3
    else:
        safety_percentage = 0
    return safety_percentage


def logic(account, lookback): # Logic function to be used for each time interval in backtest 
    
    today = len(lookback)-1
    column_names = list(lookback.columns) # ['date', 'open', 'high', 'low', 'close', 'volume']

    # The open is the starting period of trading on a securities exchange or organized over-the-counter market.
    # The close is simply the end of a trading session in the financial markets, however, closing times tend to vary between market and exchange.

    # The high is the highest price at which a stock is traded during a period. 
    # The low is the lowest price of the period. 
    # A stock’s high and low points for the day are often called its intraday high and low.

    # Volume is simply the number of shares traded in a particular stock, index, or other investment over a specific period of time.
    
  
    today = len(lookback)-1
    close = lookback['close'][today]

    if today > training_period:
        if lookback['EV'][today] + lookback['EV'][today-1]  >= 0.05 and account.b >= 90:
            try: 
                account.enter_position('long', account.buying_power, close)
            except: pass
            account.b = 0
            account.a +=1
        if account.a != 0:
            account.a+=1
        if account.b != 0:
            account.b+=1
        if close > lookback['high'][today-2] and lookback['EV'][today] + lookback['EV'][today-1] < 0 and account.a >= 180:
            for position in account.positions:
                account.close_position(position, 1, close)
            account.a=0
            account.b += 1

    
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
        factors = [
            '2ATR15Min_wav', '2ATRHour_wav',
            '5Candle', '50Candle']

        # BASIC DATA 
        df['15Min_wav'] = df['close'].ewm(alpha = 0.1).mean()
        df['Hour_wav'] = df['close'].ewm(alpha = 0.03).mean()
        df['Day_wav'] = df['close'].ewm(alpha = 0.0015).mean()
        df['5Day_wav'] = df['close'].ewm(alpha = 0.0003).mean()

        df['%15Min_wav'] = (df['close'] - df['15Min_wav'])/df['close']
        df['%Hour_wav'] = (df['close'] - df['Hour_wav'])/df['close']
        df['%Day_wav'] = (df['close'] - df['Day_wav'])/df['close']
        df['%5Day_wav'] = (df['close'] - df['5Day_wav'])/df['close']

        
        df['Candle'] = (df['close'] - df['open'] > 0)*2 - 1
        df['5Candle'] = df['Candle'].rolling(5).sum()
        df['50Candle'] = df['Candle'].rolling(50).sum()
        df['1000Candle'] = df['Candle'].rolling(1000).sum()

        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['1ATR'] = true_range.rolling(100).sum()/10
        df['2ATR'] = true_range.rolling(10000).sum()/1000


        df['1ATR15Min_wav'] = (df['close'] - df['15Min_wav'])/df['1ATR']
        df['1ATRHour_wav'] = (df['close'] - df['Hour_wav'])/df['1ATR']
        df['1ATRDay_wav'] = (df['close'] - df['Day_wav'])/df['1ATR']
        df['1ATR5Day_wav'] = (df['close'] - df['5Day_wav'])/df['1ATR']

        df['2ATR15Min_wav'] = (df['close'] - df['15Min_wav'])/df['2ATR']
        df['2ATRHour_wav'] = (df['close'] - df['Hour_wav'])/df['2ATR']
        df['2ATRDay_wav'] = (df['close'] - df['Day_wav'])/df['2ATR']
        df['2ATR5Day_wav'] = (df['close'] - df['5Day_wav'])/df['2ATR']


        # ML buy/sell signal
        model = XGBRegressor()
        model.load_model("model_sklearn.json")
        df['EV'] = model.predict(df[factors])
       
        

        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV

        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed






if __name__ == "__main__":
    stock_symbols = ["AAPL", "XOM"]
    # stock_symbols = ["JNJ","XOM","AMZN","MSFT","IBM","GOOG","AAPL","NVDA","META","UNH", "TSLA"]
    list_of_stocks = [] # List of stock data csv's to be tested, located in "data/" folder  # "AAPL_2020-03-24_2022-02-12_15min"
    for stock_symbol in stock_symbols:
        # list_of_stocks.append(stock_symbol + "_2020-10-10_2022-08-31_60min")
        list_of_stocks.append(stock_symbol + "_2020-10-09_2022-08-30_15min")
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv