import yfinance as yf
import pandas as pd
import numpy as np
from pylab import mpl, plt
from datetime import date
import mpld3
import logging

#Comment this line to avoid debugging messages
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(filename)s - %(message)s')
#stop mpl giving us debug messages:
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

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
    dif = df['Adj Close'].diff()
    UpDays = dif.copy()
    UpDays[dif <= 0] = 0.0
    DownDays = abs(dif.copy())
    DownDays[dif > 0] = 0.0
    RSUp = UpDays.rolling(period).mean()
    RSDown = DownDays.rolling(period).mean()
    rsi = 100-100/(1 + RSUp/RSDown)
    df['RSI'] = rsi
    return df

def plot_chart(df):
    fig = plt.figure(figsize=(15, 9))
    grid = plt.GridSpec(14, 8, hspace=0.2)
    ax_closes = fig.add_subplot(grid[:-5, 0:])
    ax_macd = fig.add_subplot(grid[-5:-3, 0:], sharex=ax_closes)
    ax_rsi = fig.add_subplot(grid[-3:-1, 0:], sharex=ax_closes)
    ax_closes.xaxis_date()
    ax_closes.plot(df['Adj Close'], label='Close Price')
    ax_closes.plot(df['SMAshort'], label='Short SMA')
    ax_closes.plot(df['SMAlong'], label='Long SMA')
    ax_closes.legend()
    ax_macd.plot(df['MACD'], label='MACD')
    ax_macd.plot(df['Signal'], label='Signal')
    ax_macd.legend()
    ax_rsi.set_ylabel("(%)")
    ax_rsi.plot(df.index, [70]*len(df.index), label='Overbought')
    ax_rsi.plot(df.index, [30]*len(df.index), label='Oversold')
    ax_rsi.plot(df['RSI'], label='RSI')
    ax_rsi.legend()
    plt.show()

def split_df(df):
    split = int(len(df) * 0.5) #the split will occur halfway through the data
    train = df.iloc[:split].copy() #get our training data
    test = df.iloc[split:].copy() #get our test data
    return train, test

def get_signals(df):
    df['SMAsig'] = np.where(df['SMAshort'] > df['SMAlong'], 1, 0)
    df['MACDsig'] = np.where(df['MACD'] > df['Signal'], 1, 0)
    df['RSIsig'] = np.where(df['RSI'] < 30 and df['RSI'].shift(1) > 30, 1, 0)
    return df

#START OF PROGRAM
logging.debug('PROGRAM EXECUTION INITIATED')
ShortPeriod = 10
LongPeriod = 35

#Get some data
data = get_price_data('AAPL', '2020-01-01', today)
#Add moving Averages:
data = add_MAs(data, ShortPeriod, LongPeriod)
#Add MACD:
data = add_MACD(data)
#Add RSI:
data = add_RSI(data).dropna()

logging.debug(data)
plot_chart(data)

#Now it is time to split into Training and Test Data:
train, test = split_df(data)
logging.debug(train.tail())
logging.debug(test.head())

train = get_signals(train)
logging.debug(train.tail())

logging.debug('PROGRAM TERMINATED')
