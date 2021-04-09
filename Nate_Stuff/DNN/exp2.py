# This version will use data from the CryptoCompare API and will therefore
# work exclusively on cryptoassets
# it will make calls every hour... might have to change that eventually

from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Stops debug messages from tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from colorama import Fore, Style

# import other work
from DNN_BOT_Utils import *

#set up the API:
KEY = "b14d953abbd87596c1c1c75e5c2318eea939bc1ddb17a113355b8a4cca312488"
cc.cryptocompare._set_api_key_parameter(KEY)

def loop():
    pass

@logger.catch
def main():

    # First, we import the training data:
    df = get_cc_hour_prices('BTC', 5)

    # establish lists of features we want to digitize
    # and some to delay
    to_delay = []
    features_list = []

    ###################  ADD FEATURES  #######################################
    add_returns(df, to_delay)
    # Get heiken candles
    #heiken(df, to_delay)
    #add_indicators_dnn(df, to_delay)
    df.dropna(inplace=True)
    ##########################################################################

    ################## GET DELAYS ############################################
    create_lags(df, to_delay, features_list, vision_len=40)
    df.dropna(inplace=True)
    ##########################################################################

    ############ ADD CORRECT CALLS ###########################################
    df['correct call'] = np.where(df['return'].shift(-1) > 0, 1., 0.)
    df.dropna(inplace=True)
    # print(df[['time', 'close', 'correct call'] + features_list])
    ##########################################################################

    ################## ML ####################################################
    x_train, x_test, y_train, y_test = train_test_split(df[features_list].to_numpy(), df['correct call'].to_numpy(), test_size = 0.05)
    y_train = reformat_y(y_train)
    y_test = reformat_y(y_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = Sequential() # Init model
    num_classes = 2 # buy or sell

    # put together the DNN layers
    model.add(Dense(32, input_dim=len(features_list), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile and fit model
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100)

    actual = pd.DataFrame(model.predict(x_test))
    actual['Call'] = np.where(actual[1] > 0.5, 'LONG', 'SHORT')
    actual['close'] = df['close']
    actual['correct call'] = df['correct call']
    print(actual.tail(60))

    #find loss and accuracy
    loss, accuracy = model.evaluate(x_test, y_test)

    ##########################################################################

if __name__ == '__main__':
    main()
