import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

class ActiveDates(object):
    def __init__(self, start, end):
        self.active_start_date = datetime.strftime(datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=vision_len), '%Y-%m-%d')
        self.active_end_date = datetime.strftime(datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1), '%Y-%m-%d')
# get price data and chuck some logs in there
def get_price_data(ticker, start, end):
    raw = yf.download(ticker, start, end).dropna() #grab some data
    data = pd.DataFrame(raw[["Open", "High", "Low", "Close"]]).dropna() #we only want closing prices as a dataframe object
    for item in data:
        data['log ' + item] = np.log(data[item])
    data.reset_index(inplace=True, drop=False)
    data['Date'] = pd.to_datetime(data['Date'])
    return data
# for a given date, returns a df of the past 10 days and a single line df of the reward
def get_state_reward_pair(df, date_current, vision_len):
    date_back = datetime.strftime(date_current - timedelta(days=vision_len+1), '%Y-%m-%d')
    date_forward = datetime.strftime(date_current + timedelta(days=1), '%Y-%m-%d')
    state_data = df.loc[(df['Date'] >= date_back) & (df['Date'] <= date_current)]
    prize = df.loc[(df['Date'] == date_forward)]
    return datetime.strftime(date_current, '%Y-%m-%d'), state_data, prize

start_date = '2017-01-01'
end_date = '2021-01-01'
vision_len = 10

df = get_price_data('AAPL', start_date, end_date)
state_space = {}

active = ActiveDates(start_date, end_date)

datelist = pd.date_range(start=active.active_start_date, end=active.active_end_date)

for date in datelist:
    date, ind, val = get_state_reward_pair(df, date, vision_len)
    state_space[date] = (ind, val)

print(state_space['2018-06-01'])
