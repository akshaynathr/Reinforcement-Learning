import gym, sys, numpy as np


env = gym.make('FrozenLake-v0')

print("Observation space:")
print(env.observation_space)

print("Action Space:")
print(env.action_space)

q_learning_table =np.zeros([env.observation_space.n, env.action_space.n])



# -- hyper parameters

num_epis = 5000
num_iter = 2000
learning_rate = .03
discount = 0.8


for epis in range(num_epis):
    state =env.reset()

    for iter in range(num_iter):
        action = np.argmax(q_learning_table[state,:]+np.random.randn(1,4))
        state_new , reward,done ,_ =env.step(action)
        q_learning_table[state,action] = (1-learning_rate)* q_learning_table[state,action] + learning_rate*(reward + discount*np.max(q_learning_table[state_new,:]))
        state = state_new

        if done: break



print(np.argmax(q_learning_table,axis=1))
print(np.around(q_learning_table,6))
 