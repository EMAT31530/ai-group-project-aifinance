from josh_utils import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from collections import deque
import random


class DQNAgent:
    def __init__(self, env):
        # Main model (gets trained every step)
        self.model = self.create_model()

        # Target model (this is what we predict against every step)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

        self.action_space = env.action_space_size
        self.epsilon = 1

        # self.time_dict = {'current_states': 0, 'new_current_states': 0, 'fit': 0}

    def create_model(self):
        model = Sequential()
        model.add(Dense(256, input_shape=env.state_size))
        model.add(Dropout(0.2))
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(env.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        # current_state, action, reward, new_state, done
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]  # Add divide by max (scale results)

    def act(self, current_state):
        if np.random.random() > self.epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, self.action_space)

        # Decay Epsilon
        if self.epsilon > min_epsilon:
            self.epsilon *= epsilon_decay
            self.epsilon = max(min_epsilon, self.epsilon)
        return action

    def train(self, terminal_state):
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

        self.model.fit(np.array(x), np.array(y), batch_size=minibatch_size, verbose=0, shuffle=False)

        # updating to determine if we want to update target model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def render(self, reward_arr):
        fig = go.Figure(data=go.Scatter(x=list(range(1,len(reward_arr)+1)), y=reward_arr, mode='lines', name='Reward'))
        fig.show()

    def save_model(self, episodes, update_target_every, minibatch_size, start_date, actions, replay_memory_size, min_replay_memory_size, lookback_win):
        self.model.save("C:/Users/Joshg/PycharmProjects/pythonProject/ai-group-project-aifinance/Josh's stuff/"
                        "reinforcement_models/ep{}_ute{}_mb{}_sd{}_ac{}_rms{}_mrm{}_lw{}.h5".format(episodes, update_target_every,
                                                                                   minibatch_size, start_date, actions,
                                                                                   replay_memory_size, min_replay_memory_size, lookback_win))
        print('Agent saved as ep{}_ute{}_mb{}_sd{}_ac{}_rms{}_mrm{}_lw{}.h5'.format(episodes, update_target_every,
                                                                                   minibatch_size, start_date, actions,
                                                                                   replay_memory_size, min_replay_memory_size, lookback_win))

    def load_model(self, model_name):
        self.model = keras.models.load_model("C:/Users/Joshg/PycharmProjects/pythonProject/ai-group-project-aifinance/"
                                             "Josh's stuff/reinforcement_models/{}.h5".format(model_name))


start_date = '2019-1-1'
end_date = '2020-1-1'
df = create_df(start_date, end_date)
env = Trading_2_action_simple(df)
state_shape = env.state_size
actions = env.action_space.n

# Making my own agent #
discount = 0.99
replay_memory_size = 150
min_replay_memory_size = 100
minibatch_size = 32
update_target_every = 10
model_name = '256x2'
min_reward = -200  # for model save
episodes = 5
epsilon = 1
epsilon_decay = 0.9999
min_epsilon = 0.001
SHOW_EVERY = 5
reward_arr = []

agent = DQNAgent(env)
# for episode in range(1, episodes+1):
#     ep_start_time = datetime.datetime.now()
#     step = 1
#     current_state = env.reset()
#     done = False
#     temp_reward_tot = 0
#
#     while not done:
#         action = agent.act(current_state)
#
#         new_state, reward, done, info = env.step(action)
#         agent.update_replay_memory((current_state, action, reward, new_state, done))
#         temp_reward_tot += reward
#         agent.train(done)
#         current_state = new_state
#         step += 1
#
#     reward_arr.append(temp_reward_tot)
#     if episode % SHOW_EVERY == 0:
#         if env.action_space_size == 3:
#             print('Episode:{} Net_worth:{} Buys:{} Sells:{} Holds:{} Duplicates:{} Epsilon:{}'.format(episode,
#                                                                                     env.net_worth, env.buys, env.sells,
#                                                                                     env.holds, env.dupe, epsilon,
#                                                                                     ))
#         else:
#             print('Episode:{} Net_worth:{} Buys:{} Sells:{} Buy Dupe:{} Sell Dupe:{} Epsilon:{}'.format(episode,
#                                                                                                                 env.net_worth,
#                                                                                                                 env.buys,
#                                                                                                                 env.sells,
#                                                                                                                 env.buy_dupe,
#                                                                                                                 env.sell_dupe,
#                                                                                                                 epsilon,
#                                                                                                                 ))
#     print('Episode {} took {}.'.format(episode, datetime.datetime.now() - ep_start_time))
#
# agent.render(reward_arr)
# print('Saving agent')
# agent.save_model(episodes, update_target_every, minibatch_size, start_date, env.action_space_size, replay_memory_size,
#                  min_replay_memory_size, env.lookback_win)

agent.load_model('bit_ep400_ute10_mb32_sd2017-1-1_ac2_rms150_mrm100_lw30')

done = False
step = 1
current_state = env.reset()
while not done:
    action = agent.act(current_state)

    new_state, reward, done, info = env.step(action)

    current_state = new_state
    step += 1

fig = make_subplots()
# Add traces
fig.add_trace(
    go.Scatter(x=list(range(len(env.net_worth_list))), y=env.net_worth_list, mode='lines', name='Net worth')
)
fig.add_trace(
    go.Scatter(x=list(range(len(env.net_worth_list))), y=env.buynhold, mode='lines', name='Buy and hold')
)
fig.update_yaxes(title_text="Price")
fig.update_xaxes(title_text="Days")
fig.show()
