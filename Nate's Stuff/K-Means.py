import yfinance as yf
import pandas as pd
import numpy as np
from pylab import mpl, plt
from datetime import date
import mpld3
import logging

#Comment this line to avoid debugging messages
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(filename)s - %(message)s')

#Set styles up
plt.style.use("seaborn")
mpl.rcParams["font.family"] = "serif"
today = date.today()

#takes a stock ticker, a start and end date, then returns training and test data!
def get_price_data(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    close = pd.DataFrame(raw["Adj Close"]) #we only want closing prices as a dataframe object
    return close

def add_MAs(df, ShortPeriod=20, LongPeriod=50):
    df['SMAshort'] = df['Adj Close'].rolling(ShortPeriod).mean()
    df['SMAlong'] = df['Adj Close'].rolling(LongPeriod).mean()
    return df

def add_MACD(df):
    EMA12 = df['Adj Close'].ewm(span=12).mean()
    EMA26 = df['Adj Close'].ewm(span=26).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    return df

def add_RSI(df, period=14):

    return df

#START OF PROGRAM
logging.debug('PROGRAM EXECUTION INITIATED')
ShortPeriod = 20
LongPeriod = 50

data = get_price_data('AAPL', '2020-01-10', today)
#Add moving Averages:
data = add_MAs(data, ShortPeriod, LongPeriod)
#Add MACD:
data = add_MACD(data)
#Add RSI:
data = add_RSI(data)

#Set up plotting grid
fig = plt.figure(figsize=(15, 9))
grid = plt.GridSpec(14, 8, hspace=0.2)

#sort out main axis
main_ax = fig.add_subplot(grid[:-3, 0:])
main_ax.plot(data['Adj Close'], label='Price')
main_ax.plot(data['SMAlong'], label='Long SMA')
main_ax.plot(data['SMAshort'], label='Short SMA')

#sort out MACD axis
macd_ax = fig.add_subplot(grid[-3:-1, 0:], sharex=main_ax)
macd_ax.plot(data['MACD'], label='MACD')
macd_ax.plot(data['Signal'], label='Signal')
plt.legend()

#Disable logging before plot so we don't get silly information about fonts
logging.disable(logging.DEBUG)
plt.show()

logging.debug('PROGRAM TERMINATED')
