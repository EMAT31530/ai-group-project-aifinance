# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:59:28 2020

@author: Shane
"""
#Recurrent Neural Network (LSTM) to predict stock price of an asset based on previous 60 days using closing price (working on adding MACD,RSI and OBV)


import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import math

def locate_action(pred_prices):
  action_dict={}
  for i in range(0,len(pred_prices)-1):
    if pred_prices[i+1]>=pred_prices[i]:
      action_dict[pred_prices.keys()[i]]= "buy"
    else:
      action_dict[pred_prices.keys()[i]]= "sell"
  return action_dict

def correct_action_list(action_index_dict): # Processes the actions to remove: consecutive actions; starting with a sell; ending with a buy.
    prev = 'sell'
    delete = []
    last_a = ''
    last_i = ''
    for k, v in action_index_dict.items():
        if prev == v:
            delete.append(k)
        elif k == 0:
            delete.append(k)
        else:
            prev = v
    for i in delete:
        del action_index_dict[i]
    if len(action_index_dict) !=0:
        for k, v in action_index_dict.items():
            last_a = v
            last_i = k
        if last_a == 'buy':
            del action_index_dict[last_i]

    return action_index_dict

plt.style.use("fivethirtyeight")

df = web.DataReader("AAPL", data_source="yahoo",start = "2010-01-01",end = "2020-12-04")
#New df with just closing prices

data = df.filter(["Close"])
dataset = data.values

#Selecting training set
training_data_len = math.ceil(len(dataset)*0.8)
#Scale data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)

#Create and scale traning data set
train_data = scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])

#Convert to numpy arrays

x_train,y_train = np.array(x_train),np.array(y_train)

#Reshape data
x_train= np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Build LSTM model

model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile model

model.compile(optimizer = "adam", loss = "mean_squared_error")

#Train the model
model.fit(x_train,y_train,batch_size = 1, epochs = 1)

#Create and scale test data set
test_data = scaled_data[training_data_len-60:,:]
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])
  
#Convert to numpy array
x_test = np.array(x_test)

#Reshape data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get model predicted values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = data[:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions

#Visualize model performance
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel("Date",fontsize=18)
plt.ylabel("Close Price USD $",fontsize = 18)
for key in correct_action_list(locate_action(valid["Predictions"])):
  if correct_action_list(locate_action(valid["Predictions"]))[key] == "buy":
    plt.axvline(key,color = "g", linewidth = 1)
  else:
    plt.axvline(key,color = "r", linewidth = 1)


plt.plot(valid[["Close","Predictions"]])
plt.legend(["Valid","Close","Predictions"],loc = "lower right")
plt.show()
