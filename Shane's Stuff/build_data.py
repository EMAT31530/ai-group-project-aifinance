#CNN using graphs as the input to our model
#Build data for model
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from useful_functions import *
import os, sys
import yfinance as yf




def build_data(STOCKS,TIME_RANGE,PRICE_RANGE,VALIDTAION_CUTOFF_DATE):
    
    #STOCKS : List of stock Tickers
    
    stock = STOCKS[0]#Only using one in the end, initially planned for many stocks
    budget = 1000
    
    # split image horizontally into two sections - top and bottom sections
    half_scale_size = int(PRICE_RANGE/2)
     
    live_symbols = []
    x_live = None
    x_train = None
    x_valid = None
    y_train = []
    y_valid = []
        
        def format_date(stock_data):
            #Reformat date
            stock_data['Symbol'] = stock
            stock_data['Date'] = stock_data.index
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], infer_datetime_format=True)
            stock_data['Date'] = stock_data['Date'].dt.date
            stock_data = stock_data.reset_index(drop=True)
            return stock_data
    
        # download dataframe for stock
        stock_data = yf.download(stock, start="2010-01-01", end="2021-05-09")
        
        #Indicators we want to base our trading strategy off
        rsi = rsi_calc(stock_data['Close']).fillna(method='bfill')
        macd, macd_signal = macd_calc(stock_data['Close'])[0].fillna(method='bfill'), macd_calc(stock_data['Close'])[1].fillna(method='bfill')
        
        #Reformat Date
        stock_data = format_date(stock_data)
        #Closing Prices
        stock_closes = format_data(stock_data['Close'])  
        stock_opens = format_data(stock_data['Close'])
        
        stock_dates = stock_data['Date'].values 
        
    
        for cnt in range(4, len(stock_closes)):
    
            if (cnt >= TIME_RANGE):
    
                # Scale  Indicators for graph
                graph_rsi = list(np.round(scale_list(rsi[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
                graph_macd = list(np.round(scale_list(macd[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
                graph_macd_signal = list(np.round(scale_list(macd_signal[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
                
                
               # scale closing prices
                close_data_together = list(np.round(scale_list(list(stock_closes[cnt-TIME_RANGE:cnt]), 0, half_scale_size-1),0))
                graph_close = close_data_together[0:PRICE_RANGE]
               
    
                outcome = None
                if (cnt < len(stock_closes) -1):
                    outcome = 0
                    if stock_closes[cnt+1] > stock_closes[cnt]: #If stock price goes up --> update
                        outcome = 1
    
                blank_matrix_close = np.zeros(shape=(half_scale_size, TIME_RANGE)) #Canvas for our image
                x_ind = 0
                for ma in  graph_close:
                    blank_matrix_close[int(ma), x_ind] = 2 #Colour closing price pixels
                    x_ind += 1
    
                # If you want to view an image this switches the axes
                blank_matrix_close = blank_matrix_close[::-1]
                
                
                blank_matrix_diff = np.zeros(shape=(half_scale_size, TIME_RANGE)) #canvas for second graph
                x_ind = 0
                for v,d,e in zip(graph_rsi,graph_macd,graph_macd_signal):
                    #Colour indicator pixels
                    blank_matrix_diff[int(v), x_ind] = 3
                    blank_matrix_diff[int(d),x_ind] = 4
                    blank_matrix_diff[int(e),x_ind] = 5
                    x_ind += 1
                # If you want to view an image this switches the axes
                blank_matrix_diff = blank_matrix_diff[::-1]
    
                blank_matrix = np.vstack([blank_matrix_close, blank_matrix_diff]) #Put graphs together
                
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
