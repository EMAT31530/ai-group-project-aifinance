# This version will use data from the CryptoCompare API and will therefore
# work exclusively on cryptoassets
# it will make calls every minute... might have to change that eventually

from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from sklearn.svm import SVC

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
    df = get_cc_hour_prices('BTC', 15)

    # establish lists of features we want to digitize
    # and some to delay
    to_encode = []
    to_delay = []
    features_list = []

    ###################  ADD FEATURES  #######################################
    add_returns(df, to_encode)
    # Get heiken candles
    #heiken(df, to_delay)
    #add_indicators(df, to_encode, to_delay)
    df.dropna(inplace=True)
    ##########################################################################

    ######################### MANIPULATE FEATURES ############################
    encode_class(df, to_encode, to_delay, bin_no=4)
    #print(df[['time', 'close', 'digi return', 'digi rsi', 'digi atr', 'digi eom']].tail(60))
    # might binarize some features as well
    ##########################################################################

    ################## GET DELAYS ############################################
    create_lags(df, to_delay, features_list)
    df.dropna(inplace=True)
    #print(df[['time', 'close'] + features_list])
    ##########################################################################

    ############ ADD CORRECT CALLS ###########################################
    df['correct call'] = np.where(df['return'].shift(-1) > 0, 'LONG', 'SHORT')
    df.dropna(inplace=True)
    #print(df[['time', 'close', 'correct call']])
    ##########################################################################

    # grab last few rows to predict from:
    train = df.copy().iloc[:-61]
    test = df.copy().iloc[-61:-1]

    # Now we are ready to fit the model
    fit = SVC(C=1).fit(train[features_list], train['correct call'])
    print('MODEL TRAINED')

    test['call'] = fit.predict(test[features_list])
    test['prediction'] = np.where(test['call'] == test['correct call'], 'YES', '-')

    print(test[['time', 'close', 'prediction'] + features_list])
    successes = test['prediction'].tolist().count('YES')
    success_rate = np.round(successes / len(test) * 100, 2)
    print('Success rate: ', success_rate, '%')


if __name__ == '__main__':
    main()
