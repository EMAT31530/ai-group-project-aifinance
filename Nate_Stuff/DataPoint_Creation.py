import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DataPoint(object):
    def __init__(self, df, day, vision_len):
        self.price_hist = df[['Open', 'High', 'Low', 'Close']].iloc[day-vision_len:day]
        self.next_perc = df['Perc'].iloc[day+1]
        self.mean_vol = df['Volume'].iloc[day-4:day].mean()
        # self.recent_rsi = df['RSI'].iloc[day-4:day]
        # self.recent_hist = df['hist'].iloc[day-4:day]

        self.recent_rsi = df['RSI'].iloc[day-4]
        self.recent_hist = df['hist'].iloc[day-4]

        self.vision_change = (self.price_hist['Close'].iloc[-1] - self.price_hist['Close'].iloc[0]) / self.price_hist['Close'].iloc[0]
        self.polyfit_grad = get_current_grad(self.price_hist['Close'].tolist())

def get_current_grad(list):
    t = [i for i in range(len(list))]
    fit_errs = {}

    for i in range(2):
        fit_coords = poly_to_coords(t, np.polyfit(t, list, i))
        fit_errs[get_err(list, fit_coords)] = i

    opt_ord = fit_errs[min(fit_errs.keys())]
    best_fit = np.polyfit(t, list, opt_ord)
    return val_at_point(list[-1], np.polyder(best_fit))
def poly_to_coords(t, coefs):
    coords = []
    for i in range(len(t)):
        x = 0
        for coef in coefs:
            x += coef * x
        coords.append(x)
    return coords
def get_err(price_list, fit):
    err = 0
    for i in range(len(price_list)):
        err += abs(price_list[i] - fit[i])
    return err
def val_at_point(point, coefs):
    point_val = 0
    for i in coefs:
        point_val += i * point
    return point_val

# get price data and chuck some logs and signals in there
def get_price_data(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    data = pd.DataFrame(raw[["Open", "High", "Low", "Close", "Volume"]]).dropna() #we only want closing prices as a dataframe object
    # for item in data:
    #     data['log ' + item] = np.log(data[item])
    data.reset_index(inplace=True, drop=True)
    return data
def add_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['hist'] = df['macd'] - df['sig']
def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]
def get_RSI(df):
    n = 14
    df['change'] = df['Close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)
    df['avg_gain'] = rma(df.gain[n+1:].to_numpy(), n, np.nansum(df.gain.to_numpy()[:n+1])/n)
    df['avg_loss'] = rma(df.loss[n+1:].to_numpy(), n, np.nansum(df.loss.to_numpy()[:n+1])/n)
    df['rs'] = df.avg_gain / df.avg_loss
    df['RSI'] = round(100 - (100 / (1 + df.rs)), 3)
    for i in ['change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs']:
        del df[i]
# not using yet
def plot_param(ind, dep):
    for point in points:
        plt.scatter(ind, dep)
    plt.ylabel = 'Dep'
    plt.xlabel = 'Ind'
    plt.show()

# for a given date, returns a df of the past 10 days and a single line df of the reward
def get_state_reward_pair(df, t, vision_len):
    state_data = df.iloc[t-vision_len:vision_len]
    val = df.iloc[t+1]
    return state_data, val

start_date = '2020-08-01'
end_date = '2021-01-26'
vision_len = 10

df = get_price_data('BTC-USD', start_date, end_date)
df['Perc'] = (df['Close'] - df['Close'].shift(vision_len)) / df['Close'].shift(vision_len)
add_macd(df)
get_RSI(df)

print(df.tail())

points = []
for i in range(vision_len, len(df)-1):
    points.append(DataPoint(df, i, vision_len))

for point in points:
    plt.scatter(point.vision_change, point.next_perc)
plt.ylabel('Next perc')
plt.xlabel('Ind Var')
plt.grid()
plt.show()
