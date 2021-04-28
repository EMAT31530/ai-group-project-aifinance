from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import yfinance as yf
import datetime
import ta
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import time
import tensorflow as tf


# Returns ticker_DF (Ticker is another name for stock)
def yfinance_data(ticker_symbol, start_date, period):  # returns data frame of prices for given ticker
    ticker_data = yf.Ticker(ticker_symbol)
    today = datetime.datetime.today().isoformat()
    ticker_DF = ticker_data.history(perod=period, start=start_date, end=today[:10])
    return ticker_DF


def create_df(start_date):
    df = yfinance_data('NKE', start_date, '1d')
    df = df.drop(['Dividends', 'Stock Splits', 'Open', 'Volume', 'High', 'Low'], axis=1)
    df = df.reset_index(drop=True, inplace=False)
    df['Macd'] = ta.trend.macd_diff(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df['Rsi'] = ta.momentum.rsi(df['Close'], window=14, fillna=False)
    df = df.dropna()
    return df


# ----- Q Table stuff --------- #
def create_q_table(discrete_os_size, actions):
    q_table = np.random.uniform(low=-100, high=100, size=(discrete_os_size + [actions]))
    return q_table


# (10x7 state - 10x7 low value) / 10x7 win size
def get_discrete_state(state, column_low_val_array, discrete_win_size_array):
    state = state.to_numpy()
    discrete_state = (state - column_low_val_array) / discrete_win_size_array
    discrete_state = discrete_state.astype(np.int)
    return tuple(map(tuple, discrete_state))


class Trading(Env):
    def __init__(self, df, lookback_win=1, initial_balance=1000):
        self.df = df.dropna().reset_index()
        self.df = self.df.drop(['index'], axis=1)
        self.lookback_win = lookback_win
        self.current_step = self.lookback_win  # Initial point
        self.state = self.df.iloc[
                     self.current_step - self.lookback_win: self.current_step]  # State is a selection of 10 points
        self.end_step = len(self.df) - self.lookback_win
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.stock_held = 0
        self.net_worth = self.balance
        self.prev_net_worth = self.balance
        self.buys = 0
        self.sells = 0
        self.holds = 0
        self.dupe = 0

        self.action_space = Discrete(3)  # Actions we can take 0 = hold, 1 = buy, 2 = sell
        self.state_size = (self.lookback_win, len(df.columns))

    def reset(self):
        self.current_step = self.lookback_win  # Initial point
        self.state = self.df.iloc[
                     self.current_step - self.lookback_win: self.current_step]  # State is a selection of 10 points
        self.end_step = len(self.df) - self.lookback_win
        self.balance = self.initial_balance
        self.stock_held = 0
        self.net_worth = self.balance
        self.prev_net_worth = self.balance
        self.buys = 0
        self.sells = 0
        self.holds = 0
        self.dupe = 0

        return self.state

    def step(self, action):
        self.current_step += 1
        self.state = self.df.iloc[self.current_step - self.lookback_win: self.current_step]
        current_price = self.df.loc[self.current_step, 'Close']

        self.prev_net_worth = self.net_worth
        self.net_worth = self.stock_held * current_price + self.balance

        # Actions
        if action == 0:
            self.holds += 1
            # calculating reward
            reward = self.net_worth - self.prev_net_worth

        elif action == 1 and self.balance > 0:  # BUY
            self.buys += 1
            self.stock_bought = self.balance / current_price
            self.balance = 0
            self.stock_held += self.stock_bought

            # calculating reward
            reward = self.prev_net_worth - self.net_worth
            info = 'buy'

        elif action == 2 and self.stock_held > 0:  # SELL
            self.sells += 1
            self.balance += self.stock_held * current_price
            self.stock_held = 0

            # calculating reward
            reward = self.net_worth - self.prev_net_worth
            info = 'sell'

        else:
            self.dupe += 1
            reward = -10

        if self.current_step >= self.end_step:  # Check if shower is done
            done = True
        else:
            done = False

        info = []  # Set info placeholder

        return self.state, reward, done, info  # Return step information

    def render(self):
        pass


# def train_q_table():
#     df = create_df()
#     env = Trading(df)
#
#     state_shape = env.state_size
#     actions = env.action_space.n
#
#     segements = [15]
#     discrete_os_size = segements * state_shape[1]  # 15x7
#     discrete_win_size_array = []
#     column_low_val_array = []
#     for column in df:
#         high = df[column].max()
#         low = df[column].min()
#         column_low_val_array.append(low)
#         discrete_win_size_array.append((high - low) / segements[0] + 1)
#
#     q_table = create_q_table(discrete_os_size, actions)
#
#     learning_rate = 0.1
#     discount = 0.95
#     episodes = 500
#     SHOW_EVERY = 50
#     epsilon = 0.5
#     start_epsilon_decaying = 1
#     end_epsilon_decaying = episodes // 2
#
#     epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)
#
#     discrete_state = get_discrete_state(env.reset(), column_low_val_array, discrete_win_size_array)
#
#     for episode in range(1, episodes + 1):
#         state = env.reset()
#         done = False
#         discrete_state = get_discrete_state(state, column_low_val_array, discrete_win_size_array)
#
#         while not done:
#
#             if np.random.random() > epsilon:
#                 action = np.argmax(q_table[discrete_state[0]])
#             else:
#                 action = np.random.randint(3)
#             n_state, reward, done, info = env.step(action)
#             new_discrete_state = get_discrete_state(n_state, column_low_val_array, discrete_win_size_array)
#
#             if not done:
#                 max_future_q = np.max(q_table[new_discrete_state[0]])
#                 current_q = q_table[discrete_state[0] + (action,)]
#                 new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
#                 q_table[discrete_state[0] + (action,)] = new_q
#
#             discrete_state = new_discrete_state
#         if end_epsilon_decaying >= episode >= start_epsilon_decaying:
#             epsilon -= epsilon_decay_value
#
#         if episode % SHOW_EVERY == 0:
#             print('Episode:{} Net_worth:{} Buys:{} Sells:{} Holds:{} Duplicates:{}'.format(episode, env.net_worth,
#                                                                                            env.buys, env.sells,
#                                                                                            env.holds, env.dupe))