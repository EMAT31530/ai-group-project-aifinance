#CNN using graphs as the input to our model

#From Tutorial at http://amunategui.github.io/unconventional-convolutional-networks/

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from useful_functions import *
import os, sys
import yfinance as yf


def build_data(STOCKS,TIME_RANGE,PRICE_RANGE,VALIDTAION_CUTOFF_DATE):
    
    stock = STOCKS[0]
    budget = 1000
    
    # split image horizontally into two sections - top and bottom sections
    half_scale_size = int(PRICE_RANGE/2)
     
    live_symbols = []
    x_live = None
    x_train = None
    x_valid = None
    y_train = []
    y_valid = []
    for stock in STOCKS:
        print(stock)
        
        def format_date(stock_data):
            stock_data['Symbol'] = stock
            stock_data['Date'] = stock_data.index
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], infer_datetime_format=True)
            stock_data['Date'] = stock_data['Date'].dt.date
            stock_data = stock_data.reset_index(drop=True)
            return stock_data
            
        # build image data for this stock
        # stock_data = pdr.get_data_google(stock)
    
        # download dataframe
        stock_data = yf.download(stock, start="2004-01-01", end="2021-01-01")
        
        #Indicators
        rsi = rsi_calc(stock_data['Close']).fillna(method='bfill')
        macd, macd_signal = macd_calc(stock_data['Close'])[0].fillna(method='bfill'), macd_calc(stock_data['Close'])[1].fillna(method='bfill')
    
        stock_data = format_date(stock_data)
    
        stock_closes = format_data(stock_data['Close'])  
        stock_opens = format_data(stock_data['Close'])
        
        stock_dates = stock_data['Date'].values 
        
    
        for cnt in range(4, len(stock_closes)):
            if (cnt % 500 == 0): print(cnt)
    
            if (cnt >= TIME_RANGE):
    
                #Indicators
                graph_rsi = list(np.round(scale_list(rsi[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
                graph_macd = list(np.round(scale_list(macd[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
                graph_macd_signal = list(np.round(scale_list(macd_signal[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
                
                
               # scale both close and close MA toeghertogether
                close_data_together = list(np.round(scale_list(list(stock_closes[cnt-TIME_RANGE:cnt]), 0, half_scale_size-1),0))
                graph_close = close_data_together[0:PRICE_RANGE]
               
    
                outcome = None
                if (cnt < len(stock_closes) -1):
                    outcome = 0
                    if stock_closes[cnt+1] > stock_closes[cnt]:
                        outcome = 1
    
                blank_matrix_close = np.zeros(shape=(half_scale_size, TIME_RANGE))
                x_ind = 0
                for ma in  graph_close:
                    blank_matrix_close[int(ma), x_ind] = 2 
                    x_ind += 1
    
                # flip x scale dollars so high number is atop, low number at bottom - cosmetic, humans only
                blank_matrix_close = blank_matrix_close[::-1]
                
                
                # store image data into matrix DATA_SIZE*DATA_SIZE
                blank_matrix_diff = np.zeros(shape=(half_scale_size, TIME_RANGE))
                x_ind = 0
                for v,d,e in zip(graph_rsi,graph_macd,graph_macd_signal):
                    blank_matrix_diff[int(v), x_ind] = 3
                    blank_matrix_diff[int(d),x_ind] = 4
                    blank_matrix_diff[int(e),x_ind] = 5
                    x_ind += 1
                # flip x scale so high number is atop, low number at bottom - cosmetic, humans only
                blank_matrix_diff = blank_matrix_diff[::-1]
    
                blank_matrix = np.vstack([blank_matrix_close, blank_matrix_diff]) 
                
                if cnt==500:
                    # graphed on matrix
                    plt.imshow(blank_matrix)
                    plt.show()
                    # straight timeseries 
                    plt.plot(graph_close, color='black')
                    plt.show()
    
                
    
    
                if (outcome == None):
                    # live data
                    if x_live is None:
                        x_live =[blank_matrix]
                    else:
                        x_live = np.vstack([x_live, [blank_matrix]])
                    live_symbols.append(stock)
    
                elif (stock_dates[cnt] >= VALIDTAION_CUTOFF_DATE):
                    # validation data
                    if x_valid is None:
                        x_valid = [blank_matrix]
                    else:
                        x_valid = np.vstack([x_valid, [blank_matrix]])
                    y_valid.append(outcome)
    
                else:
                    # training data
                    if x_train is None:
                        x_train = [blank_matrix]
                    else:
                        x_train = np.vstack([x_train, [blank_matrix]])
                    y_train.append(outcome)

    return x_train, y_train,x_valid, y_valid
