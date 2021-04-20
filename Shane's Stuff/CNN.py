#CNN using graphs as the input to our model

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

STOCKS = ['AAPL']

TIME_RANGE = 20
PRICE_RANGE = 20
VALIDTAION_CUTOFF_DATE = datetime.date(2017, 7, 1)

live_symbols = []
x_live = None
x_train = None
x_valid = None
y_train = []
y_valid = []

for stock in STOCKS:
    print(stock)

    # build image data for this stock
    # stock_data = pdr.get_data_google(stock)

    # download dataframe
    stock_data = pdr.get_data_yahoo(stock, start="2016-01-01", end="2018-01-17")

    values = np.append(stock_data['Open'].values,stock_data['Close'].values).reshape(-1,1)
    scaler_ref = scaler.fit(values)
    stock_data['Symbol'] = stock
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], infer_datetime_format=True)
    stock_data['Date'] = stock_data['Date'].dt.date
    stock_data = stock_data.reset_index(drop=True)
 
    
    stock_closes = scaler.transform((stock_data['Close'].values).reshape(-1,1))
    stock_closes = pd.Series(stock_closes.ravel()).fillna(method='bfill')  
    stock_closes =  list(stock_closes.values)
    stock_opens = scaler.transform((stock_data['Open'].values).reshape(-1,1))
    stock_opens = pd.Series(stock_opens.ravel()).fillna(method='bfill')  
    stock_opens =  list(stock_opens.values)
    
    stock_dates = stock_data['Date'].values 
  
    close_minus_open = list(np.array(stock_closes) - np.array(stock_opens))
