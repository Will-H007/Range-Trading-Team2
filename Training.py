import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

factors = [
            '2ATR15Min_wav', '2ATRHour_wav',
            '5Candle', '50Candle']
stock_symbols = ["AMZN","MSFT","IBM","GOOG","AAPL", "TSLA","PG","JPM","HD","CVX", "ABBV"]
list_of_stocks = []
for stock_symbol in stock_symbols:
        list_of_stocks.append(stock_symbol + "_2020-10-09_2022-08-30_60min")
df = pd.read_csv(f'data\{list_of_stocks[0]}_Processed.csv').dropna()
for stock in list_of_stocks[1:]:
    df = pd.concat([df,pd.read_csv(f'data\{stock}_Processed.csv').dropna()])

df.describe()



X = df[factors]
y = df['close']

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

model = XGBRegressor(learning_rate = 0.025, n_estimators=300) #random_state=0, learning_rate = rate, n_estimators=500
model.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            eval_metric='mae')
predictions = model.predict(X_valid)

mean_absolute_error(predictions, y_valid)

model.save_model("model_sklearn.json")

model2 = XGBRegressor()
model2.load_model("model_sklearn.json")