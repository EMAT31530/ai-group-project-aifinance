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
import csv
# Pedantics
from colorama import Fore, Style
# import other work
from RNN_bot_utils import *

import tensorflow as tf

#set up the API:
KEY = "b14d953abbd87596c1c1c75e5c2318eea939bc1ddb17a113355b8a4cca312488"
cc.cryptocompare._set_api_key_parameter(KEY)

@logger.catch
def main():

    VISION = 12
    epoch_no = 20
    # based on the assumption it is sensible to retrain AI every month
    test_size = 732

    # # second arg is number of hours * ~1400
    # df = get_cc_hour_prices('BTC', 5)

    df = pd.read_csv('kraken_btc_usd_hour.csv')
    train_size = 10000 # set the number of hours we want to train the AI on
    df = df[-train_size-test_size:]
    df.reset_index(drop=True, inplace=True)

    # Split to training and test data
    training, test, features_list = get_train_test(df, test_size=test_size)
    # Get lagged prices
    lagged_features = scale_and_lag(training, test, features_list, VISION)
    # Add lagged prices to features list
    features_list = sorted(list(set(features_list + lagged_features)))
    # Split to test and train
    x_training, x_test, y_training, y_test = numpy_ify(training, test, features_list)

    # # MODEL CONSTRUCTION
    # rnn = make_rnn(x_training)
    # # TRAINING
    # rnn.fit(x_training, y_training, epochs = epoch_no)
    # # Save rnn
    # model_name = 'v' + str(VISION) + '_t10k_e' + str(epoch_no)
    # rnn.save('models/' + model_name)

    rnn = tf.keras.models.load_model('models/v' + str(VISION) + '_t10k_e' + str(epoch_no))

    # Generating our predicted values
    preds_df = make_preds(rnn, x_test)

    # STRAT = 0 : buy-side investment, STRAT = -1 means shorting as well
    STRAT = 0
    # Risk is the proportion of account balance to risk in a single trade
    RISK = .9
    # Start balance is the amount in each account before investing
    START_BAL = 1000

    add_stats(preds_df, df, test_size, STRAT)
    # SIMULATE INVESTMENT
    simulate(preds_df, RISK, START_BAL)
    summarise(preds_df, START_BAL)

    plot_competition(preds_df, STRAT, RISK, START_BAL)

if __name__ == '__main__':
    main()
