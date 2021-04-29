from josh_utils import *
from keras.models import Sequential, load_model
import keras


class Trading(Env):
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

        # Actions
        if action == 0:
            self.holds += 1
            # calculating reward

        elif action == 1 and self.balance > 0:  # BUY
            self.buys += 1
            self.stock_bought = self.balance / current_price
            self.balance = 0
            self.stock_held += self.stock_bought
            self.prev_net_worth = self.net_worth  # only change prev_networth when opening a new position

            # calculating reward
            info = 'buy'

        elif action == 2 and self.stock_held > 0:  # SELL
            self.sells += 1
            self.balance += self.stock_held * current_price
            self.stock_held = 0
            self.prev_net_worth = self.net_worth  # only change prev_networth when opening a new position

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


start_date = '2018-1-1'
df = create_df(start_date)
env = Trading(df)

state_shape = env.state_size
actions = env.action_space.n

# Making my own agent #
discount = 0.99
replay_memory_size = 50000
min_replay_memory_size = 1000
minibatch_size = 64
update_target_every = 2
model_name = '256x2'
min_reward = -200 # for model save
episodes = 2
epsilon = 1
epsilon_decay = 0.9999
min_epsilon = 0.001
SHOW_EVERY = 1


class DQNAgent:
    def __init__(self):
        # Main model (gets trained every step)
        self.model = self.create_model()

        # Target model (this is what we predict against every step)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(50, input_shape=env.state_size))
        model.add(Dropout(0.2))
        model.add(Dense(50))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(env.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]  # Add divide by max (scale results)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch])  # Add divide by max (scale results)
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])  # Add divide by max (scale results)
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size = minibatch_size, verbose=0, shuffle=False
                       # callbacks=[self.tensorboard] if terminal_state else None
                        )

        # updating to determine if we want to update target model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def save_model(self):
        self.model.save("C:/Users/Joshg/PycharmProjects/pythonProject/ai-group-project-aifinance/Josh's stuff/"
                        "reinforcement_models/first_model.h5")

    def load_model(self, model_name):
        self.model = keras.models.load_model("C:/Users/Joshg/PycharmProjects/pythonProject/ai-group-project-aifinance/"
                                             "Josh's stuff/reinforcement_models/{}.h5".format(model_name))


agent = DQNAgent()
agent.load_model('first_model')
for episode in range(1, episodes+1):
    start_time = datetime.datetime.now()
    episode_reward = 0
    step = 1
    current_state = env.reset()
    rand_cntr = 0
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            rand_cntr += 1
            action = np.random.randint(0, env.action_space_size)

        new_state, reward, done, info = env.step(action)
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            epsilon = max(min_epsilon, epsilon)

    if episode % SHOW_EVERY == 0:
        print('Episode:{} Net_worth:{} Buys:{} Sells:{} Holds:{} Duplicates:{} Epsilon:{} Random:{}'.format(episode,
                                                                                    env.net_worth, env.buys, env.sells,
                                                                                    env.holds, env.dupe, epsilon,
                                                                                    rand_cntr))
    print('Episode took {}.'.format(datetime.datetime.now() - start_time))

print('Saving agent')
agent.save_model()
