from loguru import logger
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.svm import SVC
import ta

def get_price_data(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    data = pd.DataFrame(raw[["Open", "High", "Low", "Close"]]).dropna() #we only want closing prices as a dataframe object
    data.reset_index(drop=True, inplace=True)
    return data

def get_today_prices(ticker, start, freq):
    raw = yf.download(ticker, start=start, interval=freq).dropna() #grab some data
    data = pd.DataFrame(raw[["Open", "High", "Low", "Close"]]).dropna() #we only want closing prices as a dataframe object
    data.reset_index(drop=True, inplace=True)
    return data

def heiken(df):
    df['HK Open'] = df['Open'].shift(1) + df['Close'].shift(1)
    df['HK High'] = df['High']
    df['HK Low'] = df['Low']
    df['HK Close'] = (df['Low'] + df['High'] + df['Open'] + df['Close']) / 4
    df['HK Colour'] = np.where(df['HK Open'] < df['HK Close'], 1, -1)
    df.dropna(inplace=True)

def get_percs(data, vision=10):
    data['Past Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    return data.dropna()

# in this example we use lagged returns as features
def create_price_lags(df):
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        df[col] = df['Returns'].shift(lag)
        cols.append(col)

def create_RSI_lags(df):
    for lag in range(1, lags+1):
        col = 'RSI_lag_{}'.format(lag)
        df[col] = df['RSI'].shift(lag)
        cols.append(col)

def create_macd_hist(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    # df['MACD'] = exp1 - exp2
    # df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # df['MACD hist'] = df['MACD'] - df['Signal']
    MACD = exp1 - exp2
    Signal = MACD.ewm(span=9, adjust=False).mean()
    df['MACD hist'] = MACD - Signal

def create_macd_lags(df):
    for lag in range(1, lags+1):
        col = 'hist_lag_{}'.format(lag)
        df[col] = df['MACD hist'].shift(lag)
        cols.append(col)

# create binary features for buy/sell for each feature
def create_bins(df, bins=[0]):
    global cols_bin
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        df[col_bin] = np.digitize(df[col], bins=bins)
        cols_bin.append(col_bin)

@logger.catch
def main():

    global lags
    lags = 8

    # Get today's recent prices:
    df = get_today_prices('BTC-USD', '2021-03-10', '5m')
    # heiken(df)

    # Get prices from a range of days
    # df = get_price_data('BTC-USD', '2020-01-01', str(datetime.today()).split(' ')[0])
    # heiken(df)

    # Create the return percentages
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)

    # Turn the returns into directions
    df['Direction'] = np.sign(df['Returns']).astype(int)
    #print(df.tail())

    global cols
    cols = []

    # Create lagged prices
    create_price_lags(df)
    df.dropna(inplace=True)

    create_macd_hist(df)
    #df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df.dropna(inplace=True)

    create_macd_lags(df)
    df.dropna(inplace=True)

    # Create binary variable for each new feature
    create_bins(df)
    print(df.tail(60))

    # split into train and test
    dfa, dfb = np.array_split(df, 2)

    ###########  SOLVE THE SYSTEM  ###########

    C = 1 # We only want one hyperplane for buy/sell

    # fit the SVC model of direction from binary variables
    mfit = SVC(C=C).fit(dfa[cols_bin], dfa['Direction'])

    # show what the prediction was through the solve
    dfb['pos'] = mfit.predict(dfb[cols_bin])

    dfb['Correct'] = np.where(dfb['pos'] == dfb['Direction'], '$$$', '-')
    dfb['Passive'] = np.where(dfb['Direction'] == 1, '$$$', '-')

    #print(dfb[['Close', 'Direction', 'Passive', 'pos', 'Correct']].tail(50))

    dfb['Hold'] = (dfb['Close'] - dfb['Close'].shift(1))
    dfb['Strat'] = (dfb['Close'] - dfb['Close'].shift(1)) * dfb['pos']

    print(dfb[['Close', 'Hold', 'pos', 'Strat']].tail(50))

    hold_prof = dfb['Hold'].sum()
    strat_prof = dfb['Strat'].sum()

    print('\nHodl: ', hold_prof)
    print('Bot: ', strat_prof)

    ##########################################

if __name__ == '__main__':
    main()
