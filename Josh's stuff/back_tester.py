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
    sma_period = 50
    fast_sma = close.rolling(window=sma_period).mean()
    return fast_sma

# Returns a 100 day sma
def long_sma(close):
    sma_period = 100
    slow_sma = close.rolling(window=sma_period).mean()
    return slow_sma


# Returns profit
def profit_calc(budget, df2):  # calculates profit using action dictionary
    profit = budget
    share_price = 0
    num_shares = 0
    prev_act = -1
    prev_share = 0
    sell_counter = 0
    loss_counter = 0
    for index, row in df2.iterrows():
        if row['predictions'] == -1 and prev_act != -1:
            share_price = row['Close']
            profit = share_price * num_shares
            num_shares = 0  # sold all shares
            prev_act = -1
            if prev_share > share_price:
                loss_counter += 1
            # print('Sell: ' + str(share_price) + ' Profit: ' + str(profit) + ' Shares: ' + str(num_shares))
        elif row['predictions'] == 1 and prev_act != 1:
            sell_counter += 1
            share_price = row['Close']
            num_shares = profit/share_price  # using share price and max budget to calculate num of shares brought
            profit = 0  # all money used to buy stocks
            prev_act = 1
            prev_share = share_price
            # print('Buy: ' + str(share_price) + ' Profit: ' + str(profit) + ' Shares: ' + str(num_shares))
    if num_shares != 0:
        profit = num_shares * share_price
    profit -= budget
    print('The algorithm made a loss ' + str(loss_counter) + ' times.')
    print('The algorithm sold stock ' + str(sell_counter) + ' times.')
    print('The algorithm made a profit of ' + str(profit))
    return profit


# Prints and returns profit
def profit_for_stock(ticker_symbol, start_date, budget, period): # calculates profit for a given stock
    print('----------------------------- ' + ticker_symbol + ' ----------------------------------------------- ')
    df = yfinance_data(ticker_symbol, start_date, period)
    close = df['Close']
    rsi = rsi_calc(close)
    rsi.name = 'rsi'
    macd, macd_signal = macd_calc(close)
    macd_hist = (macd_signal - macd)
    macd_hist.name = 'macd_hist'
    rsi = rsi.round(0)
    macd_hist = macd_hist.round(1)
    indicator_list = [rsi, macd_hist]  # Input indicators
    df2 = final_dataframe(close, indicator_list)
    # --------- Genetic Algorithm ------------------------------ #
    pop_size = 20
    it_num = 50
    headers = []
    for indicator in indicator_list:  # Create list of headers for DF
        headers.append(indicator.name)
    pop_buy, pop_sell, pop_hold = create_pops(pop_size, it_num, indicator_list, df2)
    dfs = create_train_df(pop_buy, pop_sell, pop_hold, headers)
    # ---------------------------------------------------------- #
    train_df = pd.concat(dfs)
    df2 = MLP_fit(df2, train_df)
    profit = profit_calc(budget, df2)
    print('Length of buy pop is ' + str(len(pop_buy.chromosomes)))
    print('Length of hold pop is ' + str(len(pop_hold.chromosomes)))
    print('Length of sell pop is ' + str(len(pop_sell.chromosomes)))
    return profit


ticker_symbol = 'NKE'
start_date = '2005-1-1'  # 'year-month-day'
budget = 1000
period = '1d'  # Periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max intervals: 1m,2m,5m,15m,30m,60m,90m,1h

profit_for_stock(ticker_symbol, start_date, budget, period)

