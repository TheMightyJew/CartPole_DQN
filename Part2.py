import datetime
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import time
import random
from tqdm import tqdm
import os
from ModifiedTensorBoard import ModifiedTensorBoard


def init_variables():
    global ENV_NAME, env, observation_space, action_space, three_layers, currTime, timestr, DISCOUNT, REPLAY_MEMORY_SIZE
    global MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, C, MODEL_NAME, EPISODES, epsilon, EPSILON_DECAY, MIN_EPSILON, LEARNING_RATE
    global AGGREGATE_STATS_EVERY, SHOW_PREVIEW
    ENV_NAME = "CartPole-v1"
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    three_layers = True
    currTime = datetime.datetime.now()
    timestr = str(currTime.hour) + "_" + str(currTime.minute) + "_" + str(currTime.second)
    DISCOUNT = 1
    REPLAY_MEMORY_SIZE = 2000  # How many last steps to keep for model training
    MIN_REPLAY_MEMORY_SIZE = 400  # Minimum number of steps in a memory to start training
    MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
    C = 16  # Terminal states (end of episodes)
    MODEL_NAME = None
    # Environment settings
    EPISODES = 20000
    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    EPSILON_DECAY = 0.9995
    MIN_EPSILON = 0.01
    LEARNING_RATE = 0.001
    #  Stats settings
    AGGREGATE_STATS_EVERY = 50  # episodes
    SHOW_PREVIEW = False


# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.get_model()

        # Target network
        self.target_model = self.get_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def get_model(self):
        if three_layers:
            return self.get_three_layer_model(observation_space)
        else:
            return self.get_five_layer_model(observation_space)

    def get_three_layer_model(self, observation_space):
        global MODEL_NAME
        model = Sequential()
        model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(action_space, activation="linear"))
        MODEL_NAME = 'Model3-' + timestr
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        return model

    def get_five_layer_model(self, observation_space):
        global MODEL_NAME
        model = Sequential()
        model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(action_space, activation="linear"))
        MODEL_NAME = 'Model5-' + timestr
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > C:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


def run_program():
    global epsilon
    agent = DQNAgent()
    curr_moving_avg = 0
    last_100_rewards = deque(maxlen=100)
    max_until_now = -1
    passed_475_before = False
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1
        # Reset environment and get initial state
        current_state = env.reset()
        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, action_space)
            new_state, reward, done, info = env.step(action)
            # Transform new continues state to new discrete state and count reward
            episode_reward += reward
            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()
            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)
            current_state = new_state
            step += 1
        last_100_rewards.append(episode_reward)
        curr_moving_avg = np.average(last_100_rewards)
        agent.tensorboard.update_stats(moving_reward_average=curr_moving_avg, epsilon=epsilon, reward=episode_reward)

        if curr_moving_avg >= 475:
            if not passed_475_before:
                passed_475_before = True
                print("Model: " + str(MODEL_NAME) + ", passed 475 avg reward at episode: " + str(episode))
            if curr_moving_avg > max_until_now + 1:
                # +1 so it will be less models that we will save
                # we will save the best one that is significantly better
                max_until_now = curr_moving_avg
                agent.model.save(f'models/{MODEL_NAME}__episodes__{EPISODES}__avg__{curr_moving_avg}__min__{int(time.time())}.model')
            else:  # for stopping when reaching 475
                break
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        episode += 1

    # save the model:
    agent.model.save(f'models/{MODEL_NAME}__episodes__{EPISODES}__avg__{curr_moving_avg}__min__{int(time.time())}.model')


#five layers:
init_variables()
three_layers = False
run_program()

#three layers:
init_variables()
run_program()
