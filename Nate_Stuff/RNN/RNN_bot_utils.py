from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import ta
from scipy import stats
import cryptocompare as cc
import matplotlib.pyplot as plt

# We have a few options for scaling:
from sklearn.preprocessing import StandardScaler #--- Better if the data is normally distributed
from sklearn.preprocessing import MinMaxScaler

# ###############  RNN stuff:  ########################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

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
    return df

def create_lags(df, to_delay, features_list, vision_len=5):

    for feature in to_delay:
        for lag in range(1, vision_len):
            title = feature + '_lag_{}'.format(lag)
            features_list.append(title)
            df[title] = df[feature].shift(lag)

def get_train_test(df, test_size=.1):
    # define variables to delay and a list of the features to use:
    features_list = []

    df['next close'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    features_list.append('close')

    df = df[features_list + ['next close']]
    df['correct call'] = np.where(df['next close'] > df['close'], 'LONG', 'SHORT')
    df.dropna(inplace=True)

    # Define a point at which to split the data into training and testing
    test_size = int(np.round(test_size * len(df)))

    # Split into train and test
    training = df.copy().iloc[:-test_size]
    test = df.copy().iloc[-test_size+1:]

    return training, test, features_list

def scale_and_lag(training, test, features_list, VISION):
    # Iterate through the column names in the feature list and scale individually
    for col_name in features_list + ['next close']:
        scaler = MinMaxScaler()
        training[col_name] = scaler.fit_transform(training[col_name].to_numpy().reshape(-1, 1))
        test[col_name] = scaler.transform(test[col_name].to_numpy().reshape(-1, 1))

    lagged_feature_list = []
    create_lags(training, features_list, lagged_feature_list, vision_len=VISION)
    create_lags(test, features_list, lagged_feature_list, vision_len=VISION)
    training.dropna(inplace=True)
    test.dropna(inplace=True)

    return lagged_feature_list

def numpy_ify(training, test, features_list):
    x_training = np.array(training[features_list])
    x_test = np.array(test[features_list])

    y_training = np.array(training['next close'])
    y_test = np.array(test['next close'])

    #Reshaping the NumPy array to meet TensorFlow standards
    x_training = np.reshape(x_training, (x_training.shape[0], x_training.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_training, x_test, y_training, y_test

def make_rnn(x_training):
    # Initialise RNN
    rnn = Sequential()
    # Add LSTM layer:
    rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_training.shape[1], 1)))
    # Perform dropout regularisation
    rnn.add(Dropout(0.2))
    # Adding three more LSTM layers with dropout regularization
    # No return sequence on the last one
    for i in [True, True, False]:
        rnn.add(LSTM(units = 45, return_sequences = i))
        rnn.add(Dropout(0.2))
    # Add output layer
    rnn.add(Dense(units = 1))
    # Compile RNN
    rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return rnn

def make_preds(rnn, x_test):
    # Generating our predicted values
    preds_df = pd.DataFrame(rnn.predict(x_test))
    preds_df.columns = ['prediction']
    preds_df['current'] = preds_df['prediction'].shift(1)
    return preds_df

def add_stats(preds_df, df, test_size, STRAT):

    # Define a point at which to split the data into training and testing
    test_size = int(np.round(test_size * len(df)))

    # Now we need to generate a DataFrame to assess the success of the method
    test = df.iloc[-test_size+1:]
    test.reset_index(drop=True, inplace=True)

    # Grab close and next close unscaled values
    preds_df['close'] = test['close']
    preds_df['next close'] = test['next close']
    preds_df.dropna(inplace=True)

    # Put the correct call in next to the one made by the algo
    # remebering the algo is trying to predict scaled prices so we have to manipulate that a bit
    preds_df['correct call'] = np.where(preds_df['close'] < preds_df['next close'], 'LONG', 'SHORT')
    preds_df['call'] = np.where(preds_df['prediction'] > preds_df['current'], 'LONG', 'SHORT')


    preds_df['prize'] = preds_df['next close'] - preds_df['close']
    preds_df['prize perc'] = (preds_df['next close'] - preds_df['close']) / preds_df['close']

    preds_df['call prizes'] = np.where(preds_df['call'] == 'LONG', preds_df['prize'], STRAT * preds_df['prize'])
    preds_df['call prize perc'] = np.where(preds_df['call'] == 'LONG', preds_df['prize perc'], STRAT * preds_df['prize perc'])

    preds_df.dropna(inplace=True)

    # The algo is correct where it makes the same call as the colour of the next candle
    preds_df['Correct?'] = np.where(preds_df['correct call'] == preds_df['call'], '$$$$', '-')

def simulate(preds_df, RISK, START_BAL):

    balance = START_BAL
    prizes = preds_df['call prize perc'].tolist()
    cum_balance = []
    for i in range(len(prizes)):
        balance = (1-RISK) * balance + RISK * balance* (1 + prizes[i])
        cum_balance.append(balance)
    preds_df['rnn balance'] = cum_balance

    hold_balance = START_BAL
    hold_prizes = preds_df['prize perc'].tolist()
    cum_hold_balance = []
    for i in range(len(hold_prizes)):
        hold_balance = hold_balance * (1 + hold_prizes[i])
        cum_hold_balance.append(hold_balance)
    preds_df['hold balance'] = cum_hold_balance

def summarise(preds_df, START_BAL):
    # Count how many times it predicts the right price move
    correct_list = preds_df['Correct?'].tolist()
    correct_answers = correct_list.count('$$$$')
    accuracy = np.round(correct_answers / len(correct_list) * 100)

    # Figure out the profit from passive investment
    passive_prof = np.round(preds_df['hold balance'].iloc[-1] - START_BAL)

    print('Correct calls: ', correct_answers, '/', len(correct_list), '   Accuracy: ', accuracy, '%')

    call_earnings = np.round(preds_df['rnn balance'].iloc[-1] - START_BAL)

    print(preds_df[['current', 'prediction', 'correct call', 'call', 'hold balance', 'rnn balance']].head(60))
    print('\n\n')
    print(preds_df[['current', 'prediction', 'correct call', 'call', 'hold balance', 'rnn balance']].tail(60))

    print('Passive earnings: £', passive_prof)
    print('Bot earnings:     £', call_earnings)

    print('Passive gain: ', np.round(((passive_prof+START_BAL) / START_BAL - 1) * 100), '%')
    print('Bot gain:     ', np.round(((call_earnings+START_BAL) / START_BAL - 1) * 100), '%')

def plot_competition(preds_df, STRAT, RISK, START_BAL):
    # Plotting our bot performance

    for i in range(len(preds_df)-1):
        if preds_df['call'].iloc[i] == 'SHORT':
            if STRAT == -1:
                plt.axvspan(i, i+1, alpha=0.1, color='red')
        else:
            plt.axvspan(i, i+1, alpha=0.2, color='green')

    plt.plot(preds_df['rnn balance'], label='rnn balance', color='magenta')
    plt.plot(preds_df['hold balance'], label='hold balance', color='black')
    plt.title('Strategy: ' + str(STRAT) + '    Risk: ' + str(RISK) + '    Start Balance: $' + str(START_BAL))
    plt.grid()
    plt.legend()
    plt.show()
