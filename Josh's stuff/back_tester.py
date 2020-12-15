import yfinance as yf
import datetime
from Basic_macd_trader import basic_trader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Returns ticker_DF (Ticker is another name for stock)
def yfinance_data(ticker_symbol, start_date): # returns data frame of prices for given ticker
    ticker_data = yf.Ticker(ticker_symbol)
    today = datetime.datetime.today().isoformat()
    ticker_DF = ticker_data.history(perod='1d', start=start_date, end=today[:10])
    return ticker_DF


# Returns macd, macd_signal
def macd_calc(close):  # returns the macd and macd signal as data frames for a ticker
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return macd, macd_signal


# Returns rsi
def rsi_calc(close):  # assumed period of 14, returns dataframe for rsi.
    diff = close.diff(1).dropna()  # diff in one field(one day)
    # this preservers dimensions off diff values
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


# Returns action_index_dict
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


# Returns profit
def profit_calc(action_index_dict, budget, close): # calculates profit using action dictionary
    profit = budget
    num_shares = 0
    for k, v in action_index_dict.items():
        if v == 'sell':
            share_price = close[k]
            profit = share_price * num_shares
            num_shares = 0 # sold all shares
        else:
            share_price = close[k]
            num_shares = profit/share_price # using share price and max budget to calculate num of shares brought
            profit = 0 # all money used to buy stocks
    profit -= budget

    return profit


# Prints and returns profit
def profit_for_stock(ticker_symbol, start_date): # calculates profit for a given stock
    budget = 1000
    df = yfinance_data(ticker_symbol, start_date)
    close = df['Close']

    macd, macd_signal = macd_calc(close)

    action_index_dict = basic_trader(macd, macd_signal)
    action_index_dict = correct_action_list(action_index_dict)

    profit = profit_calc(action_index_dict, budget, close)
    print('The profit for ' + ticker_symbol + ' was ' + str(profit))

    return profit


ticker_symbol = 'TSLA'
start_date = '2020-1-1' # 'year-month-day'

profit_for_stock(ticker_symbol, start_date)
