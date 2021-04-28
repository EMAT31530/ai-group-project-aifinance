# This version will use data from the CryptoCompare API and will therefore
# work exclusively on cryptoassets
# it will make calls every hour... might have to change that eventually

# Stops debug messages from tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
# We have a few options for scaling:
from sklearn.preprocessing import StandardScaler #--- Better if the data is normally distributed
from sklearn.preprocessing import MinMaxScaler
# Need a train / test splitter
from sklearn.model_selection import train_test_split
# ###############  DNN stuff:  ########################################
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import SGD
# ###############  RNN stuff:  ########################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
# Pedantics
from colorama import Fore, Style
# import other work
from RNN_bot_utils import *

#set up the API:
KEY = "b14d953abbd87596c1c1c75e5c2318eea939bc1ddb17a113355b8a4cca312488"
cc.cryptocompare._set_api_key_parameter(KEY)

def loop():
    pass

@logger.catch
def main():

    # First, we import the data:
    df = get_cc_hour_prices('BTC', 6)

    # Initialise list of features to delay
    to_delay, features_list = [], []

    ###################  ADD FEATURES  #######################################
    # Get heiken candles
    add_returns(df, to_delay, 3)
    heiken(df, to_delay) # We will see how it performs on just heiken candles first
    df.dropna(inplace=True)
    # print(df)
    ##########################################################################

    ################## GET DELAYS ############################################
    # Lag the features to have, at each point, a vision of what recently occurred
    create_lags(df, to_delay, features_list, vision_len=10)
    df.dropna(inplace=True)
    ##########################################################################

    ############ ADD CORRECT CALLS ###########################################
    # These are the prizes, the colour of the next candle
    df['correct call'] = np.where(df['return future'] > 0, 1., 0.)
    df.dropna(inplace=True)
    # print(df[['close', 'return', 'return future', 'correct call']])
    ##########################################################################

    ################## Data Manipulation  ####################################
    x = df[features_list].copy()
    y = df['correct call'].copy()

    test_size = int(np.round(.1 * len(x)))

    x_train = x.iloc[:-test_size].to_numpy()
    x_test = x.iloc[-test_size+1:].to_numpy()

    y_train = y.iloc[:-test_size].to_numpy()
    y_test = y.iloc[-test_size+1:]
    y_test.reset_index(drop=True, inplace=True)

    # Next we have to scale stuff:
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #Reshaping the NumPy array to meet TensorFlow standards
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Printing the new shape of x_training_data
    print(x_train.shape)
    print(y_train.shape)
    ##########################################################################

    ##################  MODEL CONSTRUCTION  ##################################
    # Initialise RNN
    rnn = Sequential()
    # Add LSTM layer:
    rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_train.shape[1], 1)))
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
    #########################################################################

    ##############  TRAINING   ##############################################
    rnn.fit(x_train, y_train, epochs = 20, batch_size = 32)
    #########################################################################

    ####################    PREDICTION    ###################################
    y_pred = pd.DataFrame(rnn.predict(x_test))
    y_pred = pd.concat((y_pred, y_test), axis=1)
    y_pred['call'] = np.where(y_pred[0] > 0.5, 1., 0.)
    y_pred['match'] = np.where(y_pred['correct call'] == y_pred['call'], 'MATCH', '-')
    match_no = y_pred['match'].tolist().count('MATCH')
    print(y_pred.tail(60))
    print('Matches: ', match_no, '/', len(y_pred), ' - ', np.round(match_no/len(y_pred) * 100), '% success')
    #########################################################################


if __name__ == '__main__':
    main()
