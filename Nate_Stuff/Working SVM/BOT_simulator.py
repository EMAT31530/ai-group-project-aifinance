from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.svm import SVC
import ta
import openpyxl
from colorama import Fore, Style

class timestamp(object):
    def __init__(self, now):
        # Previous midnight datetime object
        midnight_today = datetime(int(now.strftime('%Y')), int(now.strftime('%m')), int(now.strftime('%d')), 0, 0, 0)
        # Start of the training data
        self.start_day = str(midnight_today - timedelta(hours=24*20)).split(' ')[0]

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

def create_lags(df, feature_names, vision_len=7):
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

def update_log(df):
    pass

@logger.catch
def main():
    # Figure out what time it is
    timestamps = timestamp(datetime.now())
    # Get a recent df for training
    df = get_recent_prices('BTC-USD', timestamps.start_day, '5m')

    # Adding indics here:
    heiken(df)
    #add_indicators(df)

    # We may also just want to use price history as a feature
    df['Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)

    # Define a list of the features for the AI to look at
    feature_names = ['Ret', 'HK Colour']

    # create lags so at each point the AI can see 'vision_len' back
    create_lags(df, feature_names)
    df.dropna(inplace=True)

    # convert every lag to binary variables
    create_bins(df, lag_feature_names)
    df.dropna(inplace=True)

    # Now for the prize - the correct decision
    df['Correct Call'] = np.sign(np.log(df['Close'].shift(-1) / df['Close']))
    df.dropna(inplace=True)
    df['Correct Call'] = np.where(df['Correct Call'] == 1.0, 1, -1)
    df.dropna(inplace=True)

    # split the df
    df_train = df.iloc[:len(df)//2]
    df_test = df.iloc[len(df)//2:]

    C = 1 # sets number of hyperplanes

    correct_count = 0
    incorrect_count = 0

    test_results = pd.DataFrame()

    for i in range(len(df_test)):
        # fit the model
        mfit = SVC(C=C).fit(df_train[lag_bin_names], df_train['Correct Call'])

        # make a copy of the next row and delete the oldest row of the test data
        next_row = df_test.copy().head(1)
        df_test = df_test.iloc[1:]

        # Add this next row to the training set and delete the oldest row
        df_train = pd.concat([df_train, next_row.copy()])
        df_train = df_train.iloc[1:]

        # use it to predict the final row:
        next_row['Call'] = mfit.predict(next_row[lag_bin_names])
        test_results = pd.concat([test_results, next_row])

        # Count if its right or wrong
        if next_row['Call'].iloc[-1] == next_row['Correct Call'].iloc[-1]:
            correct_count += 1
        else:
            incorrect_count += 1

    print('Correct Calls:   ', correct_count)
    print('Incorrect Calls: ', incorrect_count)

    print('Win rate: ', np.round(100 * correct_count / (correct_count + incorrect_count)), '%')

if __name__ == '__main__':
    main()
