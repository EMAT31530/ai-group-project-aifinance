#CNN using graphs as the input to our model

#From Tutorial at http://amunategui.github.io/unconventional-convolutional-networks/

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from utils import *


def scale_list(l, to_min, to_max):
    def scale_number(unscaled, to_min, to_max, from_min, from_max):
        return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min

    if len(set(l)) == 1:
        return [np.floor((to_max + to_min)/2)] * len(l)
    else:
        return [scale_number(i, to_min, to_max, min(l), max(l)) for i in l]

def format_date(stock_data):
    stock_data['Symbol'] = stock
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], infer_datetime_format=True)
    stock_data['Date'] = stock_data['Date'].dt.date
    stock_data = stock_data.reset_index(drop=True)
    return stock_data

def format_data(stock_data_column):
    stock_new = stock_data_column
    stock_new = stock_new.fillna(method='bfill')  
    stock_new =  list(stock_new.values)
    return stock_new

def clean_predictions(predictions,threshold):
    new_preds = {}
    for i in range(0,len(predictions)):
        if predictions[i] > threshold:
            new_preds[i] = 1
        else:
            new_preds[i] = -1 
    return new_preds
    
 

STOCKS = ['AAPL']

TIME_RANGE = 20
PRICE_RANGE = 20
VALIDTAION_CUTOFF_DATE = datetime.date(2019, 1, 1)

# split image horizontally into two sections - top and bottom sections
half_scale_size = int(PRICE_RANGE/2)
 
live_symbols = []
x_live = None
x_train = None
x_valid = None
y_train = []
y_valid = []

# xgboost lists
live_data_xgboost = []
validation_data_xgboost = []
train_data_xgboost = []

for stock in STOCKS:
    print(stock)

    # build image data for this stock
    # stock_data = pdr.get_data_google(stock)

    # download dataframe
    stock_data = pdr.get_data_yahoo(stock, start="2016-01-01", end="2020-01-01")

    stock_data = format_date(stock_data)

    stock_closes = format_data(stock_data['Close'])  
    stock_opens = format_data(stock_data['Close'])
    
    stock_dates = stock_data['Date'].values 
  
    close_minus_open = list(np.array(stock_closes) - np.array(stock_opens))

    # lets add a rolling average as an overlay indicator - back fill the missing
    # first five values with the first available avg price
    longer_ma_smoother = 6
    stock_closes_rolling_avg = pd.Series(stock_data['Close']).rolling(window=longer_ma_smoother).mean() 
    stock_closes_rolling_avg = stock_closes_rolling_avg.fillna(method='bfill')  
    stock_closes_rolling_avg =  list(stock_closes_rolling_avg.values)

    for cnt in range(4, len(stock_closes)):
        if (cnt % 500 == 0): print(cnt)

        if (cnt >= TIME_RANGE):
            # start making images
            graph_open = list(np.round(scale_list(stock_opens[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
            graph_close_minus_open = list(np.round(scale_list(close_minus_open[cnt-TIME_RANGE:cnt], 0, half_scale_size-1),0))
            # scale both close and close MA toeghertogether
            close_data_together = list(np.round(scale_list(list(stock_closes[cnt-TIME_RANGE:cnt]) + 
                list(stock_closes_rolling_avg[cnt-TIME_RANGE:cnt]), 0, half_scale_size-1),0))
            graph_close = close_data_together[0:PRICE_RANGE]
            graph_close_ma = close_data_together[PRICE_RANGE:] 

            outcome = None
            if (cnt < len(stock_closes) -1):
                outcome = 0
                if stock_closes[cnt+1] > stock_closes[cnt]:
                    outcome = 1

            blank_matrix_close = np.zeros(shape=(half_scale_size, TIME_RANGE))
            x_ind = 0
            for ma, c in zip(graph_close_ma, graph_close):
                blank_matrix_close[int(ma), x_ind] = 1 
                blank_matrix_close[int(c), x_ind] = 2  
                x_ind += 1

            # flip x scale dollars so high number is atop, low number at bottom - cosmetic, humans only
            blank_matrix_close = blank_matrix_close[::-1]

            # store image data into matrix DATA_SIZE*DATA_SIZE
            blank_matrix_diff = np.zeros(shape=(half_scale_size, TIME_RANGE))
            x_ind = 0
            for v in graph_close_minus_open:
                blank_matrix_diff[int(v), x_ind] = 3  
                x_ind += 1
            # flip x scale so high number is atop, low number at bottom - cosmetic, humans only
            blank_matrix_diff = blank_matrix_diff[::-1]

            blank_matrix = np.vstack([blank_matrix_close, blank_matrix_diff]) 

            if 1==2:
                # graphed on matrix
                plt.imshow(blank_matrix)
                plt.show()

                # straight timeseries 
                plt.plot(graph_close, color='black')
                plt.show()

            if (outcome == None):
                # live data
                if x_live is None:
                    x_live =[blank_matrix]
                else:
                    x_live = np.vstack([x_live, [blank_matrix]])
                live_symbols.append(stock)

                live_data_xgboost.append(graph_close_ma + graph_close + graph_close_minus_open + [0])

            elif (stock_dates[cnt] >= VALIDTAION_CUTOFF_DATE):
                # validation data
                if x_valid is None:
                    x_valid = [blank_matrix]
                else:
                    x_valid = np.vstack([x_valid, [blank_matrix]])
                y_valid.append(outcome)

                validation_data_xgboost.append(graph_close_ma + graph_close + graph_close_minus_open + [outcome])

            else:
                # training data
                if x_train is None:
                    x_train = [blank_matrix]
                else:
                    x_train = np.vstack([x_train, [blank_matrix]])
                y_train.append(outcome)

                train_data_xgboost.append(graph_close_ma + graph_close + graph_close_minus_open + [outcome])





####################################################################
# Run simple keras CNN model
####################################################################

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 1000
num_classes = 2
epochs = 20
 
# input image dimensions
img_rows, img_cols = TIME_RANGE, PRICE_RANGE

# add fake depth channel 
x_train_mod = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
input_shape = (TIME_RANGE, PRICE_RANGE, 1)

x_train_mod = x_train_mod.astype('float32')
x_valid = x_valid.astype('float32')

print('x_train_mod shape:', x_train_mod.shape)
print('x_valid shape:', x_valid.shape)
 
y_train_mod = keras.utils.to_categorical(y_train, num_classes)
y_valid_mod = keras.utils.to_categorical(y_valid, num_classes)

model = Sequential()
model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(10, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


######################## testing
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

#################################


model.fit(x_train_mod, y_train_mod,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_train_mod, y_train_mod))
 

score = model.evaluate(x_train_mod, y_train_mod, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
 
predictions_cnn = model.predict(x_valid)

# run an accuracy or auc test
from sklearn.metrics import roc_curve, auc, accuracy_score
 
# balance
print('Outcome balance %f' % np.mean(y_train_mod[:,1]))

# print('Model accuracy: ', accuracy_score(y_valid_mod[:,1], temp_predictions,'%'))
fpr, tpr, thresholds = roc_curve(y_valid_mod[:,1], predictions_cnn[:,1])
roc_auc = auc(fpr, tpr)
print('AUC: %f' % roc_auc)
from sklearn.metrics import roc_auc_score
 
####################################################################
# Play around with thresholds to pick the best predictions
####################################################################

# pick top of class to find best bets 
actuals = y_valid_mod[:,1]
preds = predictions_cnn[:,1]
from sklearn.metrics import accuracy_score
print ('Accuracy on all data:', accuracy_score(actuals,[1 if x >= 0.5 else 0 for x in preds]))
 
threshold = 0.5
preds = predictions_cnn[:,1][predictions_cnn[:,1] >= threshold]
actuals = y_valid_mod[:,1][predictions_cnn[:,1] >= threshold]
from sklearn.metrics import accuracy_score
print ('Accuracy on higher threshold:', accuracy_score(actuals,[1 if x > 0.5 else 0 for x in preds]))
print('Returns:',len(actuals))


##Simulate an investment 
real_predictions = clean_predictions(predictions_cnn[:,1],0.5)
stock_period = pdr.get_data_yahoo(stock, start="2019-01-01", end="2020-01-01")

a = profit_calc(1000, correct_action_list(real_predictions),stock_period['Close'])
