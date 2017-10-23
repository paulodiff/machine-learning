# -*- coding: utf-8 -*-
# https://keon.io/deep-q-learning/

# CartPole-v0 - https://github.com/openai/gym/wiki/CartPole-v0
# 0	Cart Position	-2.4	2.4
# 1	Cart Velocity	-Inf	Inf
# 2	Pole Angle	~ -41.8°	~ 41.8°
# 3	Pole Velocity At Tip	-Inf	Inf
# The system is controlled by applying a force of +1 or -1 to the cart.
# The pendulum starts upright, and the goal is to prevent it from falling over.
# A reward of +1 is provided for every timestep that the pole remains upright.
# The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000) # FIFO len 2000
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        # For a mean squared error regression problem
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Agente agisce

        if np.random.rand() <= self.epsilon:
            retvalue = random.randrange(self.action_size)
            # print(retvalue)
            # print("DQNAgent:act R {}".format(retvalue))
            return retvalue

        act_values = self.model.predict(state)
        retvalue = np.argmax(act_values[0])
        # print("DQNAgent:act P {}".format(retvalue))
        return retvalue  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # print("MODEL FIT ", state[0][0], state[0][2], target_f, target, done )


            # train with state and target_f
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




if __name__ == "__main__":

    # enable plot update
    plt.ion()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_xlabel('GameN')
    ax1.set_ylabel('Score')
    ax1.set_title('Q-Learning')
    ax1.set_xlim([0, EPISODES])


    x_data = []
    y_data = []
    #ax1.axis(0,0,100,100)

    #line, = ax1.plot([], [], lw=2)
    #line.set_data([], [])

    # self.line1 = Line2D([], [], color='black')

    ax2 = fig.add_subplot(212)
    ax2.set_xlim([0, EPISODES])




    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("state_size: {}, action_size: {}".format(state_size,action_size))
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    ax1.set_ylim([0, agent.epsilon])

    for e in range(EPISODES):


        state = env.reset()
        state = np.reshape(state, [1, state_size])

        # print(state)

        print("START EPISODE {}".format(e), state)

        for time in range(500):
            # env.render()

            # sceglie l'azione
            action = agent.act(state)

            # applica l'azione
            next_state, reward, done, _ = env.step(action)

            # print(time, action, done, reward, next_state)

            reward = reward if not done else -10

            next_state = np.reshape(next_state, [1, state_size])

            # add to memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            # se finito stampa i dati
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))

                #ax1.plot(e, time, 'ro')
                #plt.plot(e, time, 'ro')
                #xdata.append(e)
                #ydata.append(time)
                #line.set_xdata(xdata)
                #line.set_ydata(ydata)
                #plt.draw()
                #plt.show()
                #plt.pause(1e-17)

                ax1.plot(e, agent.epsilon, 'ro')
                x_data.append(e)
                y_data.append(time)
                ax2.plot(x_data, y_data, 'g-')

                # ax1.set_xlabel('x step: {0} cost: {1}'.format(step, c1))
                # plt.xlim(-2, 2)

                # plt.ylim(0.1, 0.6)
                # plt.ylabel('y {0} '.format(step))
                plt.pause(1e-17)

                #plt.legend()
                plt.show()

                #plt.pause(0.1)
                #plt.legend()
                #plt.show()
                break

        # se la memoria è maggiore della dimensione del batcj esegue un replay per il training
        if len(agent.memory) > batch_size:
            # print("REPLAY ", len(agent.memory),batch_size)
            agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
    fig.savefig('dqnOutput.png')