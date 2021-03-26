from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import ta

class timestamp(object):
    def __init__(self, now):
        # Previous midnight datetime object
        midnight_today = datetime(int(now.strftime('%Y')), int(now.strftime('%m')), int(now.strftime('%d')), 0, 0, 0)
        # Start of the training data
        self.start_day = str(midnight_today - timedelta(hours=2*24)).split(' ')[0]
        self.yesterday = str(midnight_today - timedelta(hours=24)).split(' ')[0]

def get_recent_prices(ticker, start, freq):
    # Import here as its an API - will need to re-establish connection
    import yfinance as yf
    raw = yf.download(ticker, start=start, interval=freq).dropna() #grab some data
    data = pd.DataFrame(raw[["Open", "High", "Low", "Close"]]).iloc[:-1].dropna() #we only want closing prices as a dataframe object
    data.reset_index(drop=True, inplace=True)
    return data

def heiken(df):
    df['HK Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    df['HK High'] = df['High']
    df['HK Low'] = df['Low']
    df['HK Close'] = (df['Low'] + df['High'] + df['Open'] + df['Close']) / 4
    df['HK Colour'] = np.where(df['HK Open'] < df['HK Close'], 1, 0)
    df.dropna(inplace=True)

def add_indicators(df):
    df['MACD'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd()
    df['MACD Sig'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_signal()
    df['MACD Hist'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
    # df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

def create_lags(df, feature_names, vision_len=8):
    global lag_feature_names
    lag_feature_names = []

    for feature in feature_names:
        for lag in range(0, vision_len):
            title = feature + '_lag_{}'.format(lag)
            lag_feature_names.append(title)
            df[title] = df[feature].shift(lag)

def create_bins(df, lag_feature_names, bins=[0]):

    global lag_bin_names
    lag_bin_names = []

    for col in lag_feature_names:
        title = col + '_bin'
        lag_bin_names.append(title)
        df[title] = np.digitize(df[col], bins=bins)
