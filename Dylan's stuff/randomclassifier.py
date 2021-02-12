#!pip install yfinance
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn import tree
import yfinance as yf
import datetime

#returns dataframe of a given stock/ticker from a given start date up to the current date 
def yfinance_data(ticker_symbol, start_date):
    ticker_data = yf.Ticker(ticker_symbol)
    today = datetime.datetime.today().isoformat()
    ticker_DF = ticker_data.history(perod = '1d', start = start_date, end = today[:10])
    return ticker_DF

#import ticker data
DF = yfinance_data("NKE", "2018-1-1")
del DF['Stock Splits']
del DF['Dividends']
#assign buy/sell label to each date in the training set, and add labels column to dataframe
labels=[]
for i in range(len(DF)-1):
    if DF.values[i, 3] > DF.values[i+1, 3]:
        labels.append('sell')
    else:
        labels.append('buy')
labels.append('-')

test_size = int(round(len(DF)*0.25))
test_labels = labels[-test_size:]
#len(test_labels) == test_size

#generate a list of random 'buy' / 'sell' signals, the same length as the list of true test_labels
random_signal = ['buy', 'sell']
random_labels = []
for i in range(len(test_labels)):
    random_labels.append(random.choice(random_signal))

len(random_labels) == len(test_labels)

random_model_accuracy = accuracy_score(test_labels, random_labels) * 100
random_model_accuracy
