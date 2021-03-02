#!pip install yfinance
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import datetime


#returns dataframe of a given stock/ticker from a given start date up to the current date - start_date should be in format "2020-03-21" and the ticker symbol also needs to be in ""
def yfinance_data(ticker_symbol, start_date):
    ticker_data = yf.Ticker(ticker_symbol)
    today = datetime.datetime.today().isoformat()
    ticker_DF = ticker_data.history(perod = '1d', start = start_date, end = today[:10])
    return ticker_DF

#returns RSI values for each date, as a dataframe
def rsi_calc(close):
    #calculate difference in close price in 1 day
    diff = close.diff(1)
    
    # this preserves dimensions of diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative difference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]

    up_chg_avg = up_chg.ewm(com=14 - 1, min_periods=14).mean()
    down_chg_avg = down_chg.ewm(com=14 - 1, min_periods=14).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# returns the macd and macd signal as data frames for a ticker
def macd_calc(close):  
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return macd, macd_signal

#locates dates where the macd line moves above/below (or neither) the macd signal line
def macd_diff(macd_signal, macd):
    macd_diff = macd_signal - macd
    return macd_diff.values

#returns whether the MACD indicates whether we should buy/sell or neither
#append "+1" for a buy signal (if signal line goes above macd line) and "-1" for a sell signal. Append "0" if no signal (if the signal line does not cross the macd line)
def macd_signals(signals, macs):
    macdiff = macd_diff(signals, macs)
    crossings = ['-', '-']

    c = 2
    while c < len(macdiff):
        if np.sign(macdiff[c]) != np.sign(macdiff[c-1]):
            if macdiff[c] > 0:
                crossings.append(1)
            else:
                crossings.append(-1)
        else:
            crossings.append(0)
        c += 1
    return(crossings)

#appends on-balance volume column to the dataframe DF
def obv_calc(DF, labels):
    volumes = DF.values[:,4]
    OBV = []
    day = 0
    while day < len(volumes):
        if day < 4:
            obv = '-'
        else:
            obv = 0
            i = 1
            while i<4:
                d = day - i
                if labels[d-1] == 'sell':
                    obv -= volumes[d]
                else:
                    obv += volumes[d]
                i += 1
        OBV.append(obv)
        day += 1
            
    DF['OBV'] = OBV
    return DF


    #import ticker data
DF = yfinance_data('BB', "2018-1-1")
del DF['Stock Splits']
del DF['Dividends']
#assign buy/sell label to each date in the training set, and add labels column to dataframe
labels=[]
for i in range(len(DF)-1):
    if DF.values[i, 3] > DF.values[i+1, 3]:
        labels.append('sell')
    else:
        labels.append('buy')
labels.append('-')
    
#calculate RSI values for each date 
closes = pd.DataFrame({'closes': DF.values[:,3]})  
rsivalues = rsi_calc(closes)
DF['RSI'] = rsivalues.values[:,0]

#append OBV values to the dataframe
obv_calc(DF, labels)

#use relevant functions to determine whether MACD indicates 'buy', 'sell' or neither on a given day
macs, signals = macd_calc(closes)
mac_signals = macd_signals(signals, macs)
DF['MACD'] = mac_signals

#add labels column to the dataframe
DF['Labels'] = labels
    
#remove the first 14 days, as we don't have RSI values for these dates
df = DF[14:]
#remove the last day (today), as we don't yet have a label for this date
last_row = len(df) - 1
df = df.drop(df.index[last_row])

#remove the follwing columns from the dataset, after we have used them to in the calculation of relevenat financial indicators
Cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in Cols:
    del df[col]

test_size = int(round(len(df)*0.25))
train_size = len(df) - test_size

#obtain your initial training set, from which bootstrap samples will be generated for the ensemble method,
Train_Data = df.head(train_size)
#and the test set
Test_Data = df.tail(test_size)

from scipy import stats
##function which takes Train_Data and Test_Data dataframes as arguments
test_dates = Test_Data.index
X_test = Test_Data.values[:,[0,1,2]]
Y_test = Test_Data.values[:,3]

Label_Predictions = []
counter = 0
while counter < 500:
    #Generate bootstrap train datasets, with replacement, from Train_Data
    #Each bootstrap train set will be 2/3 the size of Train_Data
    boot_df = Train_Data.sample(frac = 0.67)
    X_train = boot_df.values[:,[0,1,2]]
    Y_train = boot_df.values[:,3]

    #train the decision tree on this bootstrap train set
    Boot_Tree = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=2)
    Boot_Tree.fit(X_train, Y_train)

    y_pred = Boot_Tree.predict(X_test).tolist()
    #model_accuracy = accuracy_score(Y_test, y_pred) * 100
    Label_Predictions.append(y_pred)
    counter += 1

#Find and store the most common predicition for each date, of the 100 bootstrap models
#do this using the stats.mode() function
Final_Predictions=[]
for i in range(len(test_dates)):
    p = [item[i] for item in Label_Predictions]
    Final_Predictions.append(stats.mode(p)[0][0])
    
model_accuracy = accuracy_score(Y_test, Final_Predictions)*100
model_accuracy
