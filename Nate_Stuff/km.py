from loguru import logger
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def get_price_data(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    data = pd.DataFrame(raw[["Open", "High", "Low", "Close"]]).dropna() #we only want closing prices as a dataframe object
    data.reset_index(drop=True, inplace=True)
    return data

def get_percs(data, vision=10):
    data['Past Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    return data.dropna()

def create_lags(df):
    global cols
    lags = 2
    cols = []
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        df[col] = df['Returns'].shift(lag)
        cols.append(col)

def plot_feature(data, param):
    plt.scatter(data[param], data['Future Ret'])
    plt.title('10 point returns versus ' + param)

    plt.xlabel(param)
    plt.ylabel('10 point returns')

    plt.axhline(0)
    plt.axvline(0)

    plt.grid()
    plt.show()

@logger.catch
def main():
    df = get_price_data('EURUSD=X', '2015-01-01', '2021-03-05')
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    df['Direction'] = np.sign(df['Returns']).astype(int)


    create_lags(df)
    df.dropna(inplace=True)
    print(df.head())    

    model = KMeans(n_clusters=2, random_state=0)
    model.fit(df[cols])

    df['pos clus'] = model.predict(df[cols])
    df['pos clus'] = np.where(df['pos clus'] == 1, -1, 1)

    plt.scatter(df[cols].iloc[:,0], df[cols].iloc[:,1], c=df['pos clus'], cmap='coolwarm')
    plt.show()

    #plot_feature(df, 'Past Ret')

if __name__ == '__main__':
    main()
