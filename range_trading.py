import pandas as pd
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
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
    
    period = 64
    # Approx a day

    '''
    
    Develop Logic Here
    
    '''
    #Check every day (64)
    if today % period == 0 and today >=  training_period*2:

       
        safety_percentage = risk_management(account)

        #Range market
        if lookback['percent_b'][today-64:today].mean() < 0.3:
        
            if(lookback['close'][today] < lookback['BOLD'][today]): # If current price is below lower Bollinger Band, enter a long position
                
                for position in account.positions: # Close all current positions
                    account.close_position(position, 1, lookback['close'][today])
                if(account.buying_power > 0):
                    account.enter_position('long',account.buying_power * safety_percentage, lookback['close'][today]) # Enter a long position

            if(lookback['close'][today] > lookback['BOLU'][today]): # If today's price is above the upper Bollinger Band, enter a short position
                for position in account.positions: # Close all current positions
                    account.close_position(position, 1, lookback['close'][today])
                if(account.buying_power > 0):
                    account.enter_position('short', account.buying_power * safety_percentage, lookback['close'][today]) # Enter a short position


        #Trending market
        else:

            if lookback['Trend'][today-20:today].sum() > 0 and lookback['MA-TP'][today-20:today].mean() <  lookback['close'][today-20:today].mean() and lookback['close'][today] >= lookback['BOLU'][today]:
                for position in account.positions: # Close all current positions
                    account.close_position(position, 1, lookback['close'][today])
                if(account.buying_power > 0):
                    account.enter_position('long',account.buying_power * safety_percentage, lookback['close'][today]) # Enter a long position

            if lookback['Trend'][today-20:today].sum() < 0 and lookback['MA-TP'][today-20:today].mean() >  lookback['close'][today-20:today].mean() and lookback['close'][today] <= lookback['BOLD'][today]:
                for position in account.positions: # Close all current positions
                    account.close_position(position, 1, lookback['close'][today])
                if(account.buying_power > 0):
                    account.enter_position('short', account.buying_power * safety_percentage, lookback['close'][today]) # Enter a short position






        # Plot the Bollinger Bands
   
        # ax = plt.subplots()
        # ax = lookback[['close', 'MA-TP', 'BOLU', 'BOLD']].plot(color=['blue','green' ,'orange', 'yellow'])
        # ax.fill_between(lookback.index, lookback['BOLD'], lookback['BOLU'], facecolor='orange', alpha=0.1)
    

        # plt.show()

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

        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed






if __name__ == "__main__":
    # stock_symbols = ["JNJ","XOM"]
    stock_symbols = ["JNJ","XOM","AMZN","MSFT","IBM","GOOG","AAPL","NVDA","META","UNH"]
    list_of_stocks = [] # List of stock data csv's to be tested, located in "data/" folder  # "AAPL_2020-03-24_2022-02-12_15min"
    for stock_symbol in stock_symbols:
        list_of_stocks.append(stock_symbol + "_2020-09-19_2022-08-10_15min")
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv