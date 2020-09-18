import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import math
from matplotlib import pyplot as plt
from sfc import ServiceChainEnv

# Hyper Parameters
BATCH_SIZE = 32
EPISODE = 200
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 20   # target update frequency
MEMORY_CAPACITY = 100
N_ACTIONS = 8
N_STATES = 31


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 50)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, count):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            while True:
                action = torch.max(actions_value, 1)[1].data.numpy()
                if state[action * 2] >= sc.vnf[count * 2] and state[action * 2 + 1] >= sc.vnf[count * 2 + 1]:
                    action = action[0]  # return the argmax index
                    break
                else:
                    actions_value[0, action[0]] = -1000
        else:   # random
            while True:
                action = np.random.randint(0, N_ACTIONS)
                if state[action * 2] >= sc.vnf[count * 2] and state[action * 2 + 1] >= sc.vnf[count * 2 + 1]:
                    action = action
                    break
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
sc = ServiceChainEnv()
print('\nCollecting experience...')

reward = []
for i_episode in range(EPISODE):
    s = sc.reset()
    A = [1]
    ep_r = 0
    for count_step in range(5):
        # env.render()
        a = dqn.choose_action(s, count_step)
        A.append(a)

        # take action
        s_, r, done, info = sc.step(a, count_step)

        # modify the reward
        # x, x_dot, theta, theta_dot = s_
        if not done:
            r1 = sc.process_delay[a]
            r2 = sc.propagation_delay[A[count_step], A[count_step+1]]
            r = 1/(r1 + r2)
        else:
            r1 = sc.process_delay[a]
            r2 = sc.propagation_delay[A[count_step], A[count_step+1]] + sc.propagation_delay[A[count_step+1], 7]
            r = 1/(r1 + r2)

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(float(ep_r), 2), '| Action:', A)

        if done:
            reward.append(ep_r)
            break
        s = s_


episode = np.arange(EPISODE)
plt.plot(episode, reward)
plt.show()
