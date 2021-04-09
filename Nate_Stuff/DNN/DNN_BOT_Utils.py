from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import ta
from scipy import stats
import cryptocompare as cc

# CRYPTO COMPARE
# Minute data is only available for the past 7 days
# Takes about 2.5s to load
def get_cc_minute_prices(coin, days=7):
    # This will return a list of dicts! Whoopyyyy
    day_stamps = [datetime.now() - timedelta(hours=24 * i) for i in range(days)]
    df = pd.DataFrame(cc.get_historical_price_minute(coin, 'USDT', exchange='Kraken', toTs = day_stamps[0]))

    for j in range(1, days):
        old_day = pd.DataFrame(cc.get_historical_price_minute(coin, 'USDT', exchange='Kraken', toTs = day_stamps[j]))
        df = pd.concat([old_day, df])

    times = df['time'].tolist()
    newtimes = []
    for timeval in times:
        newtimes.append(datetime.utcfromtimestamp(timeval).strftime('%Y-%m-%d %H:%M:%S'))
    df['time'] = newtimes
    df['volume'] = df['volumeto']
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    return df

# Hour data is available for a very long time
# Takes a fair while to load but who cares when you have an hour for each prediction
# I know for a fact that you can get 5 years of data in roughly 15 seconds
def get_cc_hour_prices(coin, calls=15):
    # This will return a list of dicts! Whoopyyyy
    call_stamps = [datetime.now() - timedelta(hours=1440 * i) for i in range(calls)]
    df = pd.DataFrame(cc.get_historical_price_hour(coin, 'BUSD', exchange='Binance', toTs = call_stamps[0]))

    for j in range(1, calls):
        old_call = pd.DataFrame(cc.get_historical_price_hour(coin, 'BUSD', exchange='Binance', toTs = call_stamps[j]))
        df = pd.concat([old_call, df])

    times = df['time'].tolist()
    newtimes = []
    for timeval in times:
        newtimes.append(datetime.utcfromtimestamp(timeval).strftime('%Y-%m-%d %H:%M:%S'))
    df['time'] = newtimes
    df['volume'] = df['volumeto']
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    return df

def heiken(df, to_delay):
    df['HK open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    df['HK high'] = df['high']
    df['HK low'] = df['low']
    df['HK close'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
    df['HK color'] = np.where(df['HK open'] < df['HK close'], 1., 0.)
    to_delay.append('HK color')
    df.dropna(inplace=True)

def add_returns(df, to_encode):
    df['return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df.dropna(inplace=True)
    to_encode.append('return')

def add_indicators_svm(df, to_encode, to_delay):
    ############## MOMENTUM ###################
    # RSI momentum indicator:
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    to_encode.append('rsi')

    # ################# TREND #####################
    # # Not bothering with macd for now as a trender:
    # df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd()
    # df['macd sig'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd_signal()
    # df['macd hist'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
    # to_encode.append('macd hist')
    # to_encode.append('macd')

    # # using ema8 and ema20 crossover as a trend strategy here:
    # df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    # df['ema8'] = ta.trend.ema_indicator(df['close'], window=8)
    # df['ema dist'] = df['ema8'] - df['ema20']
    # df['ema bull'] = np.where(df['ema8'] > df['ema20'], 1, 0)
    # to_delay.append('ema bull')
    # to_encode.append('ema dist')

    # ############  VOLATILITY #######################
    # # average true range gives a volatility measure
    # df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    # df['atr diff'] = (df['atr'] - df['atr'].shift(1)) / df['atr'].shift(1) * 100
    # to_encode.append('atr diff')
    #
    # # ############## VOLUME #########################
    # # the ease of movement indicator is exactly what it sounds like
    # df['eom'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume'], window=14).ease_of_movement()
    # to_encode.append('eom')

def add_indicators_dnn(df, to_delay):
    ############## MOMENTUM ###################
    # RSI momentum indicator:
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    to_delay.append('rsi')

    # ################# TREND #####################
    # # Not bothering with macd for now as a trender:
    # df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd()
    # df['macd sig'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd_signal()
    # df['macd hist'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
    # to_delay.append('macd hist')
    # to_delay.append('macd')

    # # using ema8 and ema20 crossover as a trend strategy here:
    # df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    # df['ema8'] = ta.trend.ema_indicator(df['close'], window=8)
    # df['ema dist'] = df['ema8'] - df['ema20']
    # df['ema bull'] = np.where(df['ema8'] > df['ema20'], 1, 0)
    # to_delay.append('ema bull')
    # to_delay.append('ema dist')

    # ############  VOLATILITY #######################
    # # average true range gives a volatility measure
    # df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    # df['atr diff'] = (df['atr'] - df['atr'].shift(1)) / df['atr'].shift(1) * 100
    # to_delay.append('atr diff')
    #
    # # ############## VOLUME #########################
    # # the ease of movement indicator is exactly what it sounds like
    # df['eom'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume'], window=14).ease_of_movement()
    # to_delay.append('eom')

def interval(col, confidence):
    m = np.mean(col)
    std = np.std(col)
    h = std * stats.t.ppf((1 + confidence) / 2., len(col)-1)
    return m+h

# This needs work - will have to include stds and means etc
def encode_class(df, to_encode, to_delay, bin_no=4):

    for name in to_encode:

        # Set up the original data bins
        bins = [-1* float('inf'), float('inf')]
        col = df[name].copy()

        if bin_no % 2 == 0:
            confs = [(2*(1+i)*(1/bin_no)) for i in range(0, int((bin_no/2)-1))]
            bins = bins + [np.mean(col)] + [interval(col, i) for i in confs] + [(-1 * interval(col, i)) for i in confs]
        else:
            confs = [((1/bin_no) + 2*(1/bin_no)*j) for j in range(0, int((bin_no-1)/2))]
            bins = bins + [interval(col, i) for i in confs] + [(-1 * interval(col, i)) for i in confs]

        bins = sorted(bins)
        #print('bins:', bins)
        df['code ' + name] = np.digitize(col, bins)
        to_delay.append('code ' + name)

def create_lags(df, to_delay, features_list, vision_len=5):

    for feature in to_delay:
        for lag in range(0, vision_len):
            title = feature + '_lag_{}'.format(lag)
            features_list.append(title)
            df[title] = df[feature].shift(lag)

def create_bins(df, lag_feature_names, bins=[0]):

    global lag_bin_names
    lag_bin_names = []

    for col in lag_feature_names:
        title = col + '_bin'
        lag_bin_names.append(title)
        df[title] = np.digitize(df[col], bins=bins)

def reformat_y(list):
    newlist=[]
    for item in list:
        if item == 'LONG' or item == 1.0 :
            newlist.append([1., 0.])
        else:
            newlist.append([0., 1.])
    return np.array(newlist)
