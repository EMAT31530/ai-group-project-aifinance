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
import matplotlib.pyplot as plt
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

@logger.catch
def main():

    VISION = 40

    # First, we import the data:
    df = get_cc_hour_prices('BTC', 6)

    # Add our indicators and heiken candles
    heiken(df)
    add_indicators(df)

    all_data = df['close'].values

    test_size = .1
    test_size = int(np.round(test_size * len(all_data)))

    training_data = all_data[:-test_size]
    test_data = np.array(all_data[-test_size+1:]) # We'll need this as a numpy array later

    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data.reshape(-1, 1))

    # Initialize our x_training_data and y_training_data variables
    # as empty Python lists
    x_training_data = []
    y_training_data =[]

    # Populate the Python lists using 40 timesteps
    # We go to len minus one as we want a prize to be the future price at next
    # timestamp
    for i in range(VISION, len(training_data)-1):
        x_training_data.append(training_data[i-VISION:i, 0])
        y_training_data.append(training_data[i+1, 0])

    # Transform to NumPy arrays
    x_training_data = np.array(x_training_data)
    y_training_data = np.array(y_training_data)

    #Reshaping the NumPy array to meet TensorFlow standards
    x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], x_training_data.shape[1], 1))

    # Verifying the shape of the NumPy arrays
    print(x_training_data.shape)
    print(y_training_data.shape)

    ##################  MODEL CONSTRUCTION  ##################################
    # Initialise RNN
    rnn = Sequential()
    # Add LSTM layer:
    rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))
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
    rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)
    #########################################################################

    # Grab the unscaled test data again and convert to numpy
    print(test_data.shape)
    #Plot the test data
    plt.plot(test_data)
    plt.show()

    # make the x test data similar to the x train by putting in its last 40 points
    x_test_data = all_data[len(all_data) - len(test_data) - VISION:]
    x_test_data = np.reshape(x_test_data, (-1, 1))

    #Scale the test data
    x_test_data = scaler.transform(x_test_data)

    #Grouping our test data so each point can see 40 back
    final_x_test_data = []
    for i in range(VISION, len(x_test_data-1)):
        final_x_test_data.append(x_test_data[i-VISION:i, 0])
    final_x_test_data = np.array(final_x_test_data)

    # Reshaping the NumPy array to meet TensorFlow standards
    final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], final_x_test_data.shape[1], 1))

    # Generating our predicted values
    predictions = rnn.predict(final_x_test_data)

    # Plotting our predicted values
    plt.plot(predictions)
    plt.show()

    #Unscaling the predicted values and re-plotting the data
    unscaled_predictions = scaler.inverse_transform(predictions)

    plt.plot(unscaled_predictions)
    plt.show()

    #Plotting the predicted values against Facebook's actual stock price
    plt.plot(unscaled_predictions, color = '#135485', label = "Predictions")
    plt.plot(test_data, color = 'black', label = "Real Data")
    plt.title('Price Predictions')
    plt.show()


if __name__ == '__main__':
    main()
