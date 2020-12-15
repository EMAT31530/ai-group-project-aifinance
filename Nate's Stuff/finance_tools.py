import yfinance as yf
import pandas as pd

#takes a stock ticker, a start and end date, then returns training and test data!
def get_train_test(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    close = pd.Datafram(raw["Adj Close"]) #we only want closing prices as a dataframe object
    split = int(len(close) * 0.5) #the split will occur halfway through the data
    train = close.iloc[:split].copy() #get our training data
    test = close.iloc[split:].copy() #get our test data
    return train, test
