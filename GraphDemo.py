import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import gym
import numpy as np

EPISODES = 1000

x_data = []
y_data = []

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('fff')
ax1.set_xlim([0, EPISODES])
ax1.set_ylim([0, 500])
ax2 = fig.add_subplot(212)

for e in range(1000):
    y = random.randint(0,500)
    print(e, y)
    ax1.plot(e, y, 'ko')
    x_data.append(e)
    y_data.append(y)
    ax2.plot(x_data, y_data, 'b-')
    #line.set_data(e, y)
    plt.pause(0.05)

while True:
    plt.pause(0.05)