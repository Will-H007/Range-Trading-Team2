from cgi import test
from itertools import count
import pandas as pd
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from joblib import dump, load
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



def geo_ret(ser):
    return (1+ser).prod()-1

def get_features(df):
    df['Pre-Rt'] = df['close'].pct_change()

    df['fac-1'] = df['Pre-Rt'].shift(1)
    df['fac-2'] = df['Pre-Rt'].shift(2)
    df['fac-3'] = (1 + df['fac-1']) * (1 + df['fac-2']) - 1
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

def Z_score(df):
    return (df-df.mean()).div(df.std())

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
    
    period = 300
    # Approx a day

    '''
    
    Develop Logic Here
    
    '''
    #Check every day (64)
    if today % period == 0 and today != 0:
        # lookback = lookback.drop(axis=0, index=lookback.index[today-300])

        # sns.pairplot(lookback.filter(like = 'fac').dropna())
        # plt.show()
   
        factors= lookback.dropna().filter(like='fac').columns.tolist()
        # print(lookback.isna().sum())
    
        scaler = StandardScaler()
        pf = PolynomialFeatures(degree=3)
        Xp = pf.fit_transform(lookback.drop('Pre-Rt',axis = 1)[factors])
        factorsp = pf.get_feature_names(factors)
        Xp = pd.DataFrame(Xp, columns=factorsp, index = lookback.index)



        X_train, X_test, Y_train, Y_test = train_test_split(Xp,lookback['Pre-Rt'], test_size=0.2)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        ridge_model = Ridge(alpha=10)
        ridge_model.fit(X_train,Y_train)
        test_predictions = ridge_model.predict(X_test)

        dump(pf,'poly_converter.joblib')
        dump(ridge_model, 'trading_poly_model.joblib') 

        

        # MAE = mean_absolute_error(Y_test,test_predictions)
        # MSE = mean_squared_error(Y_test,test_predictions)
        # RMSE = np.sqrt(MSE)
        
        # print("MAE: " + str(MAE))
        # print("RMSE: " + str(RMSE))
      
    
    
    if today % period == 0 and today != 0:
        factors= lookback.dropna().filter(like='fac').columns.tolist()
        loaded_poly = load('poly_converter.joblib')
        loaded_model = load('trading_poly_model.joblib')
        lookback_poly = loaded_poly.transform(lookback[factors][today-period:today])
        prediction = loaded_model.predict(lookback_poly) +1
        prediction = lookback['close'][today-period] * np.cumprod(prediction)
        # plt.plot(prediction, label="Model")
        # data = lookback['close'].drop(axis = 0, index = lookback.index[:today-period]).reset_index().drop('index', axis = 1)

        # plt.plot(data,label="Actual Data")
     
        # plt.legend()
        # plt.show()
        protection = risk_management(account)
        slope = (prediction[-1] - prediction[0]) / period
        if(slope > 0): # If current price is below lower Bollinger Band, enter a long position
            for position in account.positions: # Close all current positions
                account.close_position(position, 1, lookback['close'][today])
            if(account.buying_power > 0):
                account.enter_position('long', account.buying_power * protection, lookback['close'][today]) # Enter a long position

        if(slope < 0): # If current price is below lower Bollinger Band, enter a short position
            for position in account.positions: # Close all current positions
                account.close_position(position, 1, lookback['close'][today])
            if(account.buying_power > 0):
                account.enter_position('short', account.buying_power * protection , lookback['close'][today]) # Enter a short position

       

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
        df['Trend'] = df['MA-TP'].pct_change()
        df["percent_b"] = (df['close'] - df['BOLU']) / (df['BOLD'] - df['BOLU'])
        # print(df.describe().transpose())

        #Average True Range
        # days = 14 #The time period employed
        # high_low = df['high'] - df['low']
        # high_close = np.abs(df['high'] - df['close'].shift())
        # low_close = np.abs(df['low'] - df['close'].shift())
        # ranges = pd.concat([high_low, high_close, low_close], axis=1)
        # true_range = np.max(ranges, axis=1)
        # df["atr"] = true_range.rolling(days).sum()/14

     
        df = get_features(df).drop(axis=0, index=df.index[:251])
        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed






if __name__ == "__main__":
    stock_symbols = ["MSFT","IBM"]
    # stock_symbols = ["JNJ","XOM","AMZN","MSFT","IBM","GOOG","AAPL","NVDA","META","UNH"]
    list_of_stocks = [] # List of stock data csv's to be tested, located in "data/" folder  # "AAPL_2020-03-24_2022-02-12_15min"
    for stock_symbol in stock_symbols:
        # list_of_stocks.append(stock_symbol + "_2020-09-19_2022-08-10_15min")
        list_of_stocks.append(stock_symbol + "_2020-09-21_2022-08-12_60min")
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv