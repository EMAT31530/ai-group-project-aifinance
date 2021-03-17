# The first version of the SVM trade call bot!

#       - Every 5 minutes pulls 2 days of data plus todays up to the current time
#       - SVM predicts the color of the next candle then re-fits after each candle
#       - Turns every indication into a BIN variable for fitting
#           - Might change this in next version...

from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.svm import SVC
import ta
from colorama import Fore, Style

class timestamp(object):
    def __init__(self, now, train_days=2):
        # Previous midnight datetime object
        midnight_today = datetime(int(now.strftime('%Y')), int(now.strftime('%m')), int(now.strftime('%d')), 0, 0, 0)
        # Start of the training data
        self.start_day = str(midnight_today - timedelta(hours=train_days*24)).split(' ')[0]

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

def loop():

    # Figure out the set of times we want to train on
    timestamps = timestamp(datetime.now(), 2)

    # Get the recent df for training on a 5m interval
    df = get_recent_prices('BTC-USD', timestamps.start_day, '5m')
    # Tell us what time it finished downloading data
    print(f'\n{Fore.CYAN}TIME:          ' + str(datetime.now()).split(' ')[1][:-4] + f'{Style.RESET_ALL}')

    # Adding indics here:
    heiken(df)
    add_indicators(df)

    # We may also just want to use changes in price history as a feature
    df['Ret'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df.dropna(inplace=True)

    # Define a list of the features for the AI to consider and create lags for
    feature_names = ['Ret', 'HK Colour', 'MACD', 'MACD Sig', 'MACD Hist']

    # create lags so at each point the AI can see 'vision_len' back, default is
    # currently set to 8 periods
    create_lags(df, feature_names, 8)
    df.dropna(inplace=True)

    # convert every lag to binary variables - may normalise eventually
    create_bins(df, lag_feature_names)
    df.dropna(inplace=True)

    # Here we grab our final row to make the prediction from
    df_final_row = df.copy()
    df_final_row = df_final_row.tail(1)

    # Now for the prize - the correct decision
    df['Correct Call'] = np.sign(np.log(df['Close'].shift(-1) / df['Close']))
    df.dropna(inplace=True)
    df['Correct Call'] = np.where(df['Correct Call'] == 1.0, 1, -1)
    df.dropna(inplace=True)

    C = 1 # sets number of hyperplanes

    # fit the model
    mfit = SVC(C=C).fit(df[lag_bin_names], df['Correct Call'])

    # use it to predict the final row:
    df_final_row['Call'] = mfit.predict(df_final_row[lag_bin_names])

    # Set the color of the output of price and call and what the color is
    price_quote_color = np.where(df['Close'].iloc[-1] > df['Close'].iloc[-2], Fore.GREEN, Fore.RED)
    call_color = np.where(df_final_row['Call'].iloc[-1] == 1, Fore.GREEN, Fore.RED)
    call = np.where(df_final_row['Call'].iloc[-1] == 1, 'LONG', 'SHORT')

    # Print these to the UI
    print(f'{Fore.CYAN}PRICE:         {Style.RESET_ALL}' + f'{price_quote_color}' + str(df_final_row['Close'].iloc[-1]) + f'{Style.RESET_ALL}')
    print(f'{Fore.CYAN}CALL:          {Style.RESET_ALL}' + f'{call_color}' + str(call) + f'{Style.RESET_ALL}')

    # Decide whether the price went up or down from the previous period
    if df['Ret'].iloc[-1] > 0:
        up = True
    else:
        up = False

    return call, up

@logger.catch
def main():

    # Initialise the previous call and whether the price went up or down
    prev_call = ''
    up = False

    # Initialise the number of correct calls and all calls at zero
    correct_calls = 0
    call_count = 0

    while 1:
        # sleep until the time is right
        time.sleep(0.5)
        now = datetime.now()
        now_min_end = int(now.strftime('%M')) % 10
        now_sec = int(now.strftime('%S'))

        # When the time is right, we begin
        if (now_min_end == 4 or now_min_end == 9) and now_sec == 57:
            # Try is necessary so that if the Yahoo API fails the program does
            # not stop
            try:
                # Figure out the time
                start_time = datetime.now()
                # Complete the loop, predict the next price and tell us if the
                # price went up
                call, up = loop()

                # If it did go up and the prev call was buy or vice versa for
                # sell, count this as a correct call
                if (up == True and prev_call == 'LONG') or (up == False and prev_call == 'SHORT'):
                    correct_calls += 1
                    call_count += 1
                else: # If not, just increase the number of calls made
                    call_count += 1

                # Carry the next call forward
                prev_call = call

                # Quote the process time
                print('\nPROCESS TIME: ', datetime.now()-start_time, '\n')

                # Display the cumulative success rate only if more than one call
                # has been made
                if call_count > 1:
                    win_rate = np.round(100*correct_calls/(call_count-1))
                    print('Correct Calls: ' , correct_calls , '/' , call_count-1, ' : ', win_rate, '% wins')
                    time.sleep(1)
            except:
                pass

if __name__ == '__main__':
    main()
