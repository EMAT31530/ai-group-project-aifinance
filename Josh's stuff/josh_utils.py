from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import yfinance as yf
import ta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Returns ticker_DF (Ticker is another name for stock)
def yfinance_data(ticker_symbol, start_date, end_date, period):  # returns data frame of prices for given ticker
    ticker_data = yf.Ticker(ticker_symbol)
    ticker_DF = ticker_data.history(perod=period, start=start_date, end=end_date)
    return ticker_DF


def create_df(start_date, end_date):
    df = yfinance_data('BTC-USD', start_date, end_date, '1d')
    df = df.drop(['Dividends', 'Stock Splits', 'Open', 'Volume', 'High', 'Low'], axis=1)
    df = df.reset_index(drop=True, inplace=False)
    # df['Macd'] = ta.trend.macd_diff(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    # df['Rsi'] = ta.momentum.rsi(df['Close'], window=14, fillna=False)
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


class Trading_3_action(Env):
    def __init__(self, df, lookback_win=5, initial_balance=1000):
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
        self.action_space_size = 3
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

        self.net_worth = self.stock_held * current_price + self.balance
        reward = self.net_worth - self.prev_net_worth  # reward calc here as net_worth doesn't change after action
        self.prev_net_worth = self.net_worth
        # Actions
        if action == 0:
            self.holds += 1
            # calculating reward

        elif action == 1 and self.balance > 0:  # BUY
            self.buys += 1
            self.stock_bought = self.balance / current_price
            self.balance = 0
            self.stock_held += self.stock_bought

            # calculating reward
            info = 'buy'

        elif action == 2 and self.stock_held > 0:  # SELL
            self.sells += 1
            self.balance += self.stock_held * current_price
            self.stock_held = 0

            # calculating reward
            info = 'sell'

        else:
            self.dupe += 1


        if self.current_step >= self.end_step:  # Check if shower is done
            done = True
        else:
            done = False

        info = []  # Set info placeholder

        return self.state, reward, done, info  # Return step information

    def render(self, close):
        pass


class Trading_2_action(Env):
    def __init__(self, df, lookback_win=30, initial_balance=1000):
        self.df = df.dropna().reset_index()
        self.df = self.df.drop(['index'], axis=1)
        self.lookback_win = lookback_win
        self.current_step = self.lookback_win  # Initial point
        self.state = self.df.iloc[
                     self.current_step - self.lookback_win: self.current_step]  # State is a selection of 10 points
        self.end_step = len(self.df) - self.lookback_win
        self.initial_balance = initial_balance
        self.balance = 0
        self.stock_held = self.initial_balance / self.df.loc[self.current_step, 'Close']
        self.net_worth = self.initial_balance
        self.buys = 0
        self.sells = 0
        self.buy_dupe = 0
        self.sell_dupe = 0
        self.last_a = -1
        self.last_price = self.df.loc[self.current_step, 'Close']
        self.action_space = Discrete(2)  # Actions we can take 0 = buy, 1 = sell
        self.action_space_size = 2
        self.state_size = (self.lookback_win, len(df.columns))
        self.step_list_l = []
        self.price_list_l = []
        self.step_list_s = []
        self.price_list_s = []

    def reset(self):
        self.current_step = self.lookback_win  # Initial point
        self.state = self.df.iloc[
                     self.current_step - self.lookback_win: self.current_step]  # State is a selection of 10 points
        self.end_step = len(self.df) - self.lookback_win
        self.balance = 0
        self.stock_held = self.initial_balance / self.df.loc[self.current_step, 'Close']
        self.net_worth = self.initial_balance
        self.buys = 0
        self.sells = 0
        self.buy_dupe = 0
        self.sell_dupe = 0
        self.last_a = -1  # neither buy or sell
        self.last_price = self.df.loc[self.current_step, 'Close']
        self.step_list_l = []
        self.price_list_l = []
        self.step_list_s = []
        self.price_list_s = []

        return self.state

    def step(self, action):
        self.current_step += 1
        self.state = self.df.iloc[self.current_step - self.lookback_win: self.current_step]
        current_price = self.df.loc[self.current_step, 'Close']
        reward = 0  # Removes warning

        if action == 0 and self.last_a != 0:  # BUY (switching position)
            reward = self.last_price - current_price
            self.buys += 1
            self.balance = self.stock_held * (2 * self.last_price - current_price)
            self.stock_held = self.balance / current_price
            self.balance = 0
            self.net_worth = self.balance + self.stock_held * current_price
            self.last_price = current_price
            self.last_a = 0  # change last action to a long
            self.step_list_l.append(self.current_step)
            self.price_list_l.append(current_price)

        elif action == 1 and self.last_a != 1:  # SELL (switching position)
            reward = current_price - self.last_price
            self.sells += 1
            self.balance = 0
            self.net_worth = self.balance + self.stock_held * (2 * self.last_price - current_price)
            self.last_price = current_price
            self.last_a = 1  # change last action to a short
            self.step_list_s.append(self.current_step)
            self.price_list_s.append(current_price)

        elif action == 0:  # BUY (holding position)
            reward = current_price - self.last_price
            self.net_worth = self.balance + self.stock_held * current_price
            self.buy_dupe += 1

        elif action == 1:  # SELL (holding position)
            reward = self.last_price - current_price
            self.net_worth = self.balance + self.stock_held * (2 * self.last_price - current_price)
            self.sell_dupe += 1

        if self.current_step >= self.end_step:  # Check if shower is done
            done = True
        else:
            done = False

        info = []  # Set info placeholder

        return self.state, reward, done, info  # Return step information

    def render(self, close):
        # fig = px.line(self.df, x=range(len(self.df)), y="Close", text=self.action_list)

        fig = make_subplots()
        # Add traces
        fig.add_trace(
            go.Scatter(x=list(range(len(self.df))), y=self.df['Close'], mode='lines', name='Close')
        )
        fig.add_trace(
            go.Scatter(x=self.step_list_s, y=self.price_list_s, mode='markers', name='Short')
        )
        fig.add_trace(
            go.Scatter(x=self.step_list_l, y=self.price_list_l, mode='markers', name='Long')
        )
        fig.update_yaxes(title_text="Price")
        fig.update_xaxes(title_text="Days")
        fig.show()


class Trading_2_action_simple(Env):
    def __init__(self, df, lookback_win=30, initial_balance=1000):
        self.df = df.dropna().reset_index()
        self.df = self.df.drop(['index'], axis=1)
        self.lookback_win = lookback_win
        self.current_step = self.lookback_win  # Initial point
        self.current_price = self.df.loc[self.current_step, 'Close']
        self.state = self.df.iloc[
                     self.current_step - self.lookback_win: self.current_step]  # State is a selection of 10 points
        self.end_step = len(self.df) - self.lookback_win
        self.initial_balance = initial_balance
        self.balance = 0
        self.stock_held = self.initial_balance / self.df.loc[self.current_step, 'Close']
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.buys = 0
        self.sells = 0
        self.buy_dupe = 0
        self.sell_dupe = 0
        self.last_a = -1
        self.last_price = self.df.loc[self.current_step, 'Close']
        self.action_space = Discrete(2)  # Actions we can take 0 = buy, 1 = sell
        self.action_space_size = 2
        self.state_size = (self.lookback_win, len(df.columns))
        self.step_list_l = []
        self.price_list_l = []
        self.step_list_s = []
        self.price_list_s = []
        self.net_worth_list = []
        self.buynhold = []
        self.starting_bit = self.stock_held

    def reset(self):
        self.current_step = self.lookback_win  # Initial point
        self.state = self.df.iloc[
                     self.current_step - self.lookback_win: self.current_step]  # State is a selection of 10 points
        self.end_step = len(self.df) - self.lookback_win
        self.current_price = self.df.loc[self.current_step, 'Close']
        self.balance = 0
        self.stock_held = self.initial_balance / self.df.loc[self.current_step, 'Close']
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.buys = 0
        self.sells = 0
        self.buy_dupe = 0
        self.sell_dupe = 0
        self.last_a = -1  # neither buy or sell
        self.last_price = self.df.loc[self.current_step, 'Close']
        self.step_list_l = []
        self.price_list_l = []
        self.step_list_s = []
        self.price_list_s = []
        self.net_worth_list = []
        self.buynhold = []
        self.starting_bit = self.stock_held

        return self.state

    def step(self, action):
        if action == 0 and self.last_a != 0:  # BUY (switching position)
            self.buys += 1
            self.balance = self.stock_held * (2 * self.last_price - self.current_price)
            self.stock_held = self.balance / self.current_price
            self.balance = 0
            self.last_price = self.current_price
            self.last_a = 0  # change last action to a long
            self.step_list_l.append(self.current_step)
            self.price_list_l.append(self.current_price)

        elif action == 1 and self.last_a != 1:  # SELL (switching position)
            self.sells += 1
            self.balance = 0
            self.last_price = self.current_price
            self.last_a = 1  # change last action to a short
            self.step_list_s.append(self.current_step)
            self.price_list_s.append(self.current_price)

        elif action == 0:  # BUY (holding position)
            self.buy_dupe += 1

        elif action == 1:  # SELL (holding position)
            self.sell_dupe += 1

        self.current_step += 1
        self.state = self.df.iloc[self.current_step - self.lookback_win: self.current_step]
        self.current_price = self.df.loc[self.current_step, 'Close']

        if action == 0:  # BUY (holding position)
            self.net_worth = self.balance + self.stock_held * self.current_price

        elif action == 1:  # SELL (holding position)
            self.net_worth = self.balance + self.stock_held * (2 * self.last_price - self.current_price)

        reward = self.net_worth - self.prev_net_worth  # Removes warning
        self.prev_net_worth = self.net_worth

        self.buynhold.append(self.starting_bit * self.current_price)
        self.net_worth_list.append(self.net_worth)
        if self.current_step >= self.end_step:  # Check if shower is done
            done = True
        else:
            done = False

        info = []  # Set info placeholder

        return self.state, reward, done, info  # Return step information

    def render(self, close):
        # fig = px.line(self.df, x=range(len(self.df)), y="Close", text=self.action_list)

        fig = make_subplots()
        # Add traces
        fig.add_trace(
            go.Scatter(x=list(range(len(self.df))), y=self.df['Close'], mode='lines', name='Close')
        )
        fig.add_trace(
            go.Scatter(x=self.step_list_s, y=self.price_list_s, mode='markers', name='Short')
        )
        fig.add_trace(
            go.Scatter(x=self.step_list_l, y=self.price_list_l, mode='markers', name='Long')
        )
        fig.update_yaxes(title_text="Price")
        fig.update_xaxes(title_text="Days")
        fig.show()


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