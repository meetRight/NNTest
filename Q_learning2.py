import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import os

np.random.seed(2)  # reproducible


world_R, world_C = 10, 10
# world_C = 3
# N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right', 'up', 'down']     # available actions
# EPSILON = 0.3   # greedy police
ALPHA = 0.9     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 100   # maximum episodes
FRESH_TIME = 0.1    # fresh time for one move
end_pos_x = 9
end_pos_y = 8


def build_q_table(world_R, world_C, actions):
    k = 0
    I = np.zeros((world_R * world_C, 2), int)
    for i in range(world_R):
        for j in range(world_C):
            I[k, 0] = i
            I[k, 1] = j
            k += 1
    I = np.transpose(I).tolist()
    table = pd.DataFrame(np.zeros((world_R * world_C, len(actions))), index=I, columns=actions)
    print(table)    # show table
    return table


def choose_action(pos_x, pos_y, q_table, EPSILON):
    # This is how to choose an action
    pos_actions = q_table.loc[(pos_x, pos_y), :]
    if (np.random.uniform() < EPSILON) or ((pos_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = pos_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(pos_x, pos_y, A):
    # This is how agent will interact with the environment
    if A == 'up':
        if pos_y == end_pos_y and pos_x == end_pos_x + 1:
            next_pos_x = 'end'
            next_pos_y = 'end'
            Reward = 100
        elif pos_x == 0:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = 0
        else:
            next_pos_x = pos_x - 1
            next_pos_y = pos_y
            Reward = 0
    elif A == 'down':
        if pos_y == end_pos_y and pos_x == end_pos_x - 1:
            next_pos_x = 'end'
            next_pos_y = 'end'
            Reward = 100
        elif pos_x == world_R - 1:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = 0
        else:
            next_pos_x = pos_x + 1
            next_pos_y = pos_y
            Reward = 0
    elif A == 'left':
        if pos_x == end_pos_x and pos_y == end_pos_y + 1:
            next_pos_x = 'end'
            next_pos_y = 'end'
            Reward = 100
        elif pos_y == 0:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = 0
        else:
            next_pos_x = pos_x
            next_pos_y = pos_y - 1
            Reward = 0
    elif A == 'right':
        if pos_x == end_pos_x and pos_y == end_pos_y - 1:
            next_pos_x = 'end'
            next_pos_y = 'end'
            Reward = 100
        elif pos_y == world_C - 1:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = 0
        else:
            next_pos_x = pos_x
            next_pos_y = pos_y + 1
            Reward = 0
    return next_pos_x, next_pos_y, Reward


def update_env(pos_x, pos_y, episode, step_counter):
    # This is how environment be updated
    if pos_x == 'end' and pos_y == 'end':
        # os.system("cls")
        print('\rEpisode %s: total_steps = %s' % (episode+1, step_counter))
        total_step.append(step_counter)
        time.sleep(1)
    # else:
    #     os.system('cls')
    #     for i in range(world_R):
    #         env = ['+'] * world_C
    #         if i == end_pos_x:
    #             env[end_pos_y] = 'O'
    #         if i == pos_x:
    #             env[pos_y] = '@'
    #         a = ''.join(env)
    #         print('\r{}'.format(a))
    #         time.sleep(FRESH_TIME)
    # print('\n')


def rl():
    # main part of RL loop
    EPSILON = 0.3
    q_table = build_q_table(world_R, world_C, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        pos_x = 0
        pos_y = 0
        is_terminated = False
        update_env(pos_x, pos_y, episode, step_counter)
        while not is_terminated:
            A = choose_action(pos_x, pos_y, q_table, EPSILON)
            next_pos_x, next_pos_y, Reward = get_env_feedback(pos_x, pos_y, A)  # take action & get next state and reward
            q_predict = q_table.loc[(pos_x, pos_y), A]
            if next_pos_x == 'end' and next_pos_y == 'end':
                q_target = Reward  # next state is terminal
                if step_counter <= min(total_step):
                    EPSILON = EPSILON * 0.9
                is_terminated = True  # terminate this episode
            else:
                q_target = Reward + GAMMA * q_table.loc[(next_pos_x, next_pos_y), :].max()
            q_table.loc[(pos_x, pos_y), A] += ALPHA * (q_target - q_predict)  # update
            pos_x = next_pos_x  # move to next state
            pos_y = next_pos_y
            update_env(pos_x, pos_y, episode, step_counter+1)
            step_counter += 1
        # print('\n')
        # print(q_table)
    return q_table


if __name__ == "__main__":
    total_step = [500]
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
    plt.plot(np.arange(MAX_EPISODES+1), total_step)
    plt.show()

