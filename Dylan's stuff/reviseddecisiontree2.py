# -*- coding: utf-8 -*-
"""RevisedDecisionTree2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NIbguwvTb6miEMDauZ-CUEMzuq8R1s5q
"""

#!pip install yfinance
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import yfinance as yf
import datetime

#returns dataframe of a given stock/ticker from a given start date up to the current date - start_date should be in format "2020-03-21" and the ticker symbol also needs to be in ""
def yfinance_data(ticker_symbol, start_date):
    ticker_data = yf.Ticker(ticker_symbol)
    today = datetime.datetime.today().isoformat()
    ticker_DF = ticker_data.history(perod = '1d', start = start_date, end = today[:10])
    return ticker_DF

#returns RSI values for each date, as a dataframe
def rsi_calc(close):
    #calculate difference in close price in 1 day
    diff = close.diff(1)
    
    # this preserves dimensions of diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative difference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]

    up_chg_avg = up_chg.ewm(com=14 - 1, min_periods=14).mean()
    down_chg_avg = down_chg.ewm(com=14 - 1, min_periods=14).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# returns the macd and macd signal as data frames for a ticker
def macd_calc(close):  
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return macd, macd_signal

#locates dates where the macd line moves above/below (or neither) the macd signal line
def macd_diff(macd_signal, macd):
    macd_diff = macd_signal - macd
    return macd_diff.values

#returns whether the MACD indicates whether we should buy/sell or neither
#append "+1" for a buy signal (if signal line goes above macd line) and "-1" for a sell signal. Append "0" if no signal (if the signal line does not cross the macd line)
def macd_signals(signals, macs):
    macdiff = macd_diff(signals, macs)
    crossings = ['-', '-']

    c = 2
    while c < len(macdiff):
        if np.sign(macdiff[c]) != np.sign(macdiff[c-1]):
            if macdiff[c] > 0:
                crossings.append(1)
            else:
                crossings.append(-1)
        else:
            crossings.append(0)
        c += 1
    return(crossings)

#appends on-balance volume column to the dataframe DF
def obv_calc(DF, labels):
    volumes = DF.values[:,4]
    OBV = []
    day = 0
    while day < len(volumes):
        if day < 4:
            obv = '-'
        else:
            obv = 0
            i = 1
            while i<4:
                d = day - i
                if labels[d-1] == 'sell':
                    obv -= volumes[d]
                else:
                    obv += volumes[d]
                i += 1

        OBV.append(obv)
        day += 1
            
    DF['OBV'] = OBV
    return DF

def DecisionTree(ticker_symbol, start_date):
    #import ticker data
    DF = yfinance_data(ticker_symbol, start_date)
    del DF['Stock Splits']
    del DF['Dividends']
    #dates = DF.index

    #assign buy/sell label to each date in the training set, and add labels column to dataframe
    labels=[]
    for i in range(len(DF)-1):
        if DF.values[i, 3] > DF.values[i+1, 3]:
            labels.append('sell')
        else:
            labels.append('buy')
    labels.append('-')
    DF['Labels'] = labels

    #calculate RSI values for each date 
    closes = pd.DataFrame({'closes': DF.values[:,3]})  
    rsivalues = rsi_calc(closes)
    DF['RSI'] = rsivalues.values[:,0]

    #append OBV values to the dataframe
    obv_calc(DF, labels)

    #use relevant functions to determine whether MACD indicates 'buy', 'sell' or neither on a given day
    macs, signals = macd_calc(closes)
    mac_signals = macd_signals(signals, macs)
    DF['MACD'] = mac_signals

    #remove the first 14 days, as we don't have RSI values for these dates
    df = DF[14:]
    #remove the last day (today), as we don't yet have a label for this date
    last_row = len(df) - 1
    df = df.drop(df.index[last_row])

    X = df.values[:, 6:9]
    Y = df.values[:, 5]

    #split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 100)
    #create our tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
    tree.fit(X_train, Y_train);

    #predict 'buy'/'sell' labels for our test set and compare with Y-test to determine accuracy of the decisio tree
    y_pred = tree.predict(X_test)
    model_accuracy = accuracy_score(Y_test, y_pred) * 100

    print("model_accuracy:", model_accuracy)
    return y_pred

DecisionTree("NKE", "2019-1-1");