import datetime
import random
import time

import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from ModifiedTensorBoard import ModifiedTensorBoard


#from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.9
LEARNING_RATE = 0.001

MEMORY_SIZE = 2000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9995

currTime = datetime.datetime.now()
timestr = str(currTime.hour) + "_" + str(currTime.minute) + "_" + str(currTime.second)


class DQNSolver:

    def __init__(self, observation_space, action_space,threeLayers):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        if threeLayers:
            self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.8))
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.8))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.8))
            self.model.add(Dense(self.action_space, activation="linear"))
            MODEL_NAME = 'Model3-' + timestr

        else:
            self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.8))
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.8))
            self.model.add(Dense(512, activation="relu"))
            self.model.add(Dropout(0.8))
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.8))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.8))
            self.model.add(Dense(self.action_space, activation="linear"))
            MODEL_NAME = 'Model5-' + timestr
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0,callbacks=[self.tensorboard] if terminal else None)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole(threeLayers):
    env = gym.make(ENV_NAME)
    #score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space,threeLayers)
    run = 0

    sum = 0
    while True:
        run += 1

        # Update tensorboard step every episode
        dqn_solver.tensorboard.step = run

        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                #score_logger.add_score(step, run)

                sum += step
                averageSize = 1
                if run%averageSize == 0:
                    dqn_solver.tensorboard.update_stats(reward_avg_of_current_century= sum / (run-100*(int((run-1)/100))),epsilon=dqn_solver.exploration_rate,reward=step)
                if run%100 == 0:
                    sum = 0


                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole(threeLayers=False)
