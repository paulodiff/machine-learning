import gym
env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()
    print("Reset episode n. %d" % i_episode)
    for t in range(100):
        env.render()
        # print(observation)
        action = 1
        print(action)
        observation, reward, done, info = env.step(action)
        print(observation)
        print(reward)
        print(done)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break