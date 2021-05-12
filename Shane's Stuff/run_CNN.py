####################################################################
# Run simple keras CNN model
####################################################################

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from build_data import*

TIME_RANGE = 100
PRICE_RANGE = 100
VALIDTAION_CUTOFF_DATE = datetime.date(2020, 7, 7)
STOCKS = ['NFLX']
batch_size = 1000
num_classes = 2
epochs = 10
budget = 1000

#Get all the data
x_train, y_train, x_valid, y_valid = build_data(STOCKS,TIME_RANGE,PRICE_RANGE,VALIDTAION_CUTOFF_DATE)
# input image dimensions
img_rows, img_cols = TIME_RANGE, PRICE_RANGE

# add fake depth channel 
x_train_mod = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
input_shape = (TIME_RANGE, PRICE_RANGE, 1)

#Convert to float 32
x_train_mod = x_train_mod.astype('float32')
x_valid = x_valid.astype('float32')

#Convert to keras format
y_train_mod = keras.utils.to_categorical(y_train, num_classes)
y_valid_mod = keras.utils.to_categorical(y_valid, num_classes)

#Create Neural Network
model = Sequential()
model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(10, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Display model summary
model.summary()

#Fit model to training data
model.fit(x_train_mod, y_train_mod,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_train_mod, y_train_mod))
		  
#Get predictions for test set
predictions_cnn = model.predict(x_valid)

##Simulate an investment 
real_predictions = clean_predictions(predictions_cnn[:,1],0.50) #50% threshold

#Get data for test period
stock_period = yf.download(STOCKS[0], start="2020-07-07", end="2021-05-09") 

profit_list = []
passive_list = []
higher_confidence = []
for i in range(1,len(stock_period['Close'])):
  real_predictions = clean_predictions(predictions_cnn[:i,1],0.5) #All predictions over 50% are converted to buy signals
  real_predictions1 = clean_predictions(predictions_cnn[:i,1],0.6)#All predictions over 60% are converted to buy signals
  #Calculate profit for strategies
  a = profit_calc(budget, correct_action_list(real_predictions),stock_period['Close'][:i]) 
  b = profit_calc(budget, correct_action_list(real_predictions1),stock_period['Close'][:i])
  #Append results for plotting
  profit_list.append(budget + a[0])
  passive_list.append(budget + a[1])
  higher_confidence.append(budget + b[0])

import matplotlib.pyplot as plt
index = []
for i in range(len(profit_list)): #x axis representing dates
  index.append(i)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(index, profit_list, color='tab:blue')
ax.plot(index, passive_list, color='tab:orange')
ax.plot(index, higher_confidence, color='tab:red')
plt.show()

