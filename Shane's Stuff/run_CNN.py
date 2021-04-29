####################################################################
# Run simple keras CNN model
####################################################################

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from build_data import*



 
    
TIME_RANGE = 30
PRICE_RANGE = 30
VALIDTAION_CUTOFF_DATE = datetime.date(2019, 1, 1)
STOCKS = ['EURUSD=X']
batch_size = 1000
num_classes = 2
epochs = 30
budget = 1000


x_train, y_train, x_valid, y_valid = build_data(STOCKS,TIME_RANGE,PRICE_RANGE,VALIDTAION_CUTOFF_DATE)
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
 
threshold = 0.6
preds = predictions_cnn[:,1][predictions_cnn[:,1] >= threshold]
actuals = y_valid_mod[:,1][predictions_cnn[:,1] >= threshold]
from sklearn.metrics import accuracy_score
print ('Accuracy on higher threshold:', accuracy_score(actuals,[1 if x > 0.5 else 0 for x in preds]))
print('Returns:',len(actuals))


##Simulate an investment 
real_predictions = clean_predictions(predictions_cnn[:,1],0.5)
stock_period = yf.download(STOCKS[0], start="2019-01-01", end="2020-01-01")

a = profit_calc(budget, correct_action_list(real_predictions),stock_period['Close'])
