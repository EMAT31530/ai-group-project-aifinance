import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product

#takes a stock ticker, a start and end date, then returns training and test data!
def get_train_test(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    close = pd.DataFrame(raw["Adj Close"]) #we only want closing prices as a dataframe object
    split = int(len(close) * 0.5) #the split will occur halfway through the data
    train = close.iloc[:split].copy() #get our training data
    test = close.iloc[split:].copy() #get our test data
    return train, test

def test_SMAs(SMA_short, SMA_long, test):
    for SMAs, SMAl, in product(SMA_short, SMA_long): #Iterate through all possible combos
        test.dropna(inplace=True)
        test['returns'] = np.log(test / test.shift(1)) #find log of day to day returns
        test['SMAs'] = test.rolling(SMAs).mean() #get short moving av
        test['SMAl'] = test.rolling(SMAl).mean() #get long moving av
        test.dropna(inplace=True)
        test['Position'] = np.where(test['SMAs'] > test['SMAl'], 1, 0)
        test['Strat'] = test['Position'].shift(1) * test['Returns']
        test.dropna(inplace=True)
        performance = np.exp(test[['Returns', 'Strat']].sum())
        results = results.append(pd.DataFrame(
            {'SMAs': SMAs, 'SMAl': SMAl,
             'MARKET': performance['Returns'],
             'STRAT': performance['Strat'],
             'OUT': performance['Strat'] - performance['Returns']},
            index=[0]
        ), ignore_index=True)
    return results


train, test = get_train_test("AAPL", "2010-01-01", "2020-01-01") #Get your train and test data made up
print(train.head())#take a look at the start of the training
print(test.head())#take a look at the start of the testing

SMA_short = range(20, 61, 4) #set parameter values for short period sma... range(start, stop, step)
SMA_long = range(180, 281, 10) #set parameter values for long period sma

results = test_SMAs(SMA_short, SMA_long, train)
results.info()
