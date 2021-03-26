import yfinance as yf
import pandas as pd
import numpy as np
from pylab import mpl, plt
from datetime import date
import logging
import gym
from sklearn.cluster import KMeans

#Comment this line to avoid debugging messages
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(filename)s - %(message)s')
memory_len = 10
vision_len = 5


#takes a stock ticker, a start and end date, then returns training and test data!
def get_price_data(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    data = pd.DataFrame(raw["Adj Close"]).dropna() #we only want closing prices as a dataframe object
    data[f'{vision_len} Day Rewards'] = (data['Adj Close'] - data['Adj Close'].shift(vision_len)) / data['Adj Close'].shift(vision_len) * 100
    return data


def split_df(df):
    split = int(len(df) * 0.5) #the split will occur halfway through the data
    train = df.iloc[:split].copy() #get our training data
    test = df.iloc[split:].copy() #get our test data
    return train, test


def make_memory(df, memory_len):
    df['Memory'] = [df['Adj Close'].shift(i for i in range(0, memory_len))]
    return df


def __main__():
    train, test = split_df(get_price_data('AAPL', '2013-01-01', '2016-01-01'))
    train = make_memory(train, memory_len)

    logging.debug(train.tail())


if __name__ == '__main__':
        __main__()
