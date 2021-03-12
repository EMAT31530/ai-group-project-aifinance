from loguru import logger
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

def get_price_data(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    data = pd.DataFrame(raw[["Open", "High", "Low", "Close"]]).dropna() #we only want closing prices as a dataframe object
    data.reset_index(drop=True, inplace=True)
    return data

def get_percs(data, vision=10):
    data['Past Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    return data.dropna()

# in this example we use lagged returns as features
def create_lags(df):
    global cols
    lags = 7
    cols = []
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        df[col] = df['Returns'].shift(lag)
        cols.append(col)

# create binary features for buy/sell for each feature
def create_bins(df, bins=[0]):
    global cols_bin
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        df[col_bin] = np.digitize(df[col], bins=bins)
        cols_bin.append(col_bin)

# fit models using the bins cols and the price direction after
def fit_models(df, models):
    mfit = {model: models[model].fit(df[cols_bin], df['Direction'])
            for model in models.keys()}

# get position values from fitted models
def derive_positions(df, models):
    for model in models.keys():
        df['pos_' + model] = models[model].predict(df[cols_bin])

# evaluates the effectiveness of each strategy
def eval_strats(df, models):
    global sel
    sel = []
    for model in models.keys():
        col = 'strat_' + model
        df[col] = df['pos_' + model] * df['Returns']
        sel.append(col)
    sel.insert(0, 'Returns')

@logger.catch
def main():
    # Get price data and set up returns and direction
    df = get_price_data('ETH-USD', '2015-01-01', '2021-03-05')
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    df['Direction'] = np.sign(df['Returns']).astype(int)
    print(df.head())

    # this algo uses lagged log returns as features
    create_lags(df)
    df.dropna(inplace=True)
    print(df.head())

    # create binary features for dataframe
    create_bins(df)

    # visualise the data
    df.plot.scatter(x='lag_1', y='lag_2', c='Returns', cmap='coolwarm')

    plt.show()

    ############### SVM ################################################

    # how many hyperplanes?
    C = 1
    # define a dictionary of models:
    models = {'svm': SVC(C=C)}
    # fit the model to the data
    fit_models(df, models)
    derive_positions(df, models)
    eval_strats(df, models)

    # print comparative returns
    print(df[sel].sum().apply(np.exp))

    # visualise returns
    df[sel].cumsum().apply(np.exp).plot()
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()
