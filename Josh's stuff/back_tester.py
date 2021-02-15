import yfinance as yf
import datetime
import time
import pandas as pd
from genetic_algorithm import *
from rsi_sma_macd_trader import final_dataframe
from trading_ann import MLP_fit
import random


# Returns ticker_DF (Ticker is another name for stock)
def yfinance_data(ticker_symbol, start_date, period):  # returns data frame of prices for given ticker
    ticker_data = yf.Ticker(ticker_symbol)
    today = datetime.datetime.today().isoformat()
    ticker_DF = ticker_data.history(perod=period, start=start_date, end=today[:10])
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
    up_chg = 0 * diff  # this preservers dimensions off diff values
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


# Returns a 50 day sma
def short_sma(close):
    sma_period = 20
    fast_sma = close.rolling(window=sma_period).mean()
    return fast_sma

# Returns a 100 day sma
def long_sma(close):
    sma_period = 50
    slow_sma = close.rolling(window=sma_period).mean()
    return slow_sma


def sell(old_profit, loss_counter, row, num_shares, sell_counter):
    prev_act = -1
    sell_counter += 1
    profit = row['Close'] * num_shares
    if old_profit > profit:
        loss_counter += 1
    return profit, prev_act, loss_counter, sell_counter


def buy(profit, row, num_shares):
    num_shares = profit/row['Close']  # using share price and max budget to calculate num of shares brought
    prev_act = 1
    return num_shares, prev_act

# Returns profit
def profit_calc(budget, df2):  # calculates profit using action dictionary
    profit = budget
    prev_act = -1
    sell_counter, loss_counter, num_shares = 0, 0, 0
    for index, row in df2.iterrows():
        if row['predictions'] == -1 and prev_act != -1:
            profit, prev_act, loss_counter, sell_counter = sell(profit, loss_counter, row, num_shares, sell_counter)
        elif row['predictions'] == 1 and prev_act != 1:
            num_shares, prev_act = buy(profit, row, num_shares)
    profit -= budget
    print('The algorithm made a loss ' + str(loss_counter) + ' times.')
    print('The algorithm sold stock ' + str(sell_counter) + ' times.')
    print('The algorithm made a profit of ' + str(profit))
    return profit


def get_features(df):
    close = df['Close']

    rsi = rsi_calc(close)
    rsi.name = 'rsi'
    rsi = rsi.round(0)

    macd, macd_signal = macd_calc(close)
    macd_hist = (macd_signal - macd)
    macd_hist.name = 'macd_hist'
    macd_hist = macd_hist.round(1)

    quick, slow = short_sma(close), long_sma(close)
    sma_diff = (quick - slow)
    sma_diff.name = 'sma_diff'
    sma_diff = sma_diff.round(1)

    indicator_list = [rsi, sma_diff, macd_hist]  # Input indicators
    headers = []
    for indicator in indicator_list:  # Create list of headers for DF
        headers.append(indicator.name)
    return close, indicator_list, headers


def print_ga(pop_buy, pop_sell, pop_hold):
    print('---------- BUY -------------')
    for chromo in pop_buy.chromosomes:
        print(chromo.genes_val)
    print('---------- HOLD -------------')
    for chromo in pop_hold.chromosomes:
        print(chromo.genes_val)
    print('---------- SELL -------------')
    for chromo in pop_sell.chromosomes:
        print(chromo.genes_val)

# Prints and returns profit
def profit_for_stock(ticker_symbol, start_date, budget, period): # calculates profit for a given stock
    print('----------------------------- ' + ticker_symbol + ' ----------------------------------------------- ')
    df = yfinance_data(ticker_symbol, start_date, period)
    close, indicator_list, headers = get_features(df)
    df2 = final_dataframe(close, indicator_list)
    # --------- Genetic Algorithm ------------------------------ #
    pop_size = 20
    it_num = 50
    pop_buy, pop_sell, pop_hold = create_pops(pop_size, it_num, indicator_list, df2)
    dfs = create_train_df(pop_buy, pop_sell, pop_hold, headers)
    # ---------------------------------------------------------- #
    train_df = pd.concat(dfs)
    df2 = MLP_fit(df2, train_df)
    profit = profit_calc(budget, df2)
    print_ga(pop_buy, pop_sell, pop_hold)
    return profit


ticker_symbol = 'NKE'
start_date = '2005-1-1'  # 'year-month-day'
budget = 1000
period = '1d'  # Periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max intervals: 1m,2m,5m,15m,30m,60m,90m,1h

profit_for_stock(ticker_symbol, start_date, budget, period)
