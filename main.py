import os
import time
from multiprocessing import Process
from threading import Event

import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import DQN
from stable_baselines_master.stable_baselines.common.evaluation import evaluate_policy


if __name__ == '__main__':

    n_steps, best_mean_reward = 0, -np.inf

    # to print timesteps during learning
    def callback(_locals, _globals):
        global n_steps, best_mean_reward
        if (n_steps % 100) == 0:
            print('training at timestep {}...'.format(n_steps))
        n_steps += 1

    env_names = ['MsPacmanNoFrameskip-v4', 'MsPacmanNoFrameskip-v4']
    envs = []

    for env_name in env_names:
        env = make_atari_env(env_name, num_env=1, seed=0) # num_env might need to be 1 for WSL ubuntu
        env = VecFrameStack(env, n_stack=4)
        envs.append(env)

    models = [None, None]

    def create_model_then_learn(idx, policy_type, env, learning_starts=1000, prioritized_replay=False, verbose=0):
        models[idx] = DQN(policy_type, env, learning_starts=learning_starts, prioritized_replay=False, verbose=verbose)
        print("model {} created".format(idx))
        print("model {} begins learning...".format(idx))
        models[idx].learn(total_timesteps=1100, callback=callback)
        print("model {} done learning!".format(idx))

    indiv_processes = []
    multi_processes = []
    n_indiv = 2
    n_multi = 0

    # events to synchronize indiv and multi agent replay buffer accesses
    indiv_rb_dones = [Event for indiv_num in range(n_indiv)]
    multi_rb_dones = [Event for multi_num in range(n_multi)]

    # set to true so indiv agents can go first
    for multi_rb_done in multi_rb_dones:
        multi_rb_done.set()

    replay_buffers = [None for indiv_num in range(n_indiv)] # TODO make this visible to agents

    # spawn indiv agent processes
    for indiv_num in range(n_indiv):
        p = Process(target=create_model_then_learn, args=( (indiv_num, 'CnnPolicy', envs[indiv_num], 1000, True, 1) ) )
        print('indiv process made')
        indiv_processes.append(p)

    # spawn multi agent processes
    for multi_num in range(n_multi):
        p = Process(target=create_model_then_learn, args=( (indiv_num, 'CnnPolicy', envs[multi_num], 1000, True, 1) ) ) # TODO modify learn() to differentiate b/t indiv and multi agent
        print('mt process made')
        multi_processes.append(p)

    # start indiv agent processes
    for indiv_process in indiv_processes:
        print('indiv start')
        indiv_process.start()
        time.sleep(1) 

    # start multi agent processes
    for multi_process in multi_processes:
        multi_process.start()
        time.sleep(1)

    for indiv_process in indiv_processes:
        indiv_process.join()

    for multi_process in multi_processes:
        multi_process.join()


    # DQN('CnnPolicy', env, learning_starts=5000, prioritized_replay=True, verbose=1)
    # model.learn(total_timesteps=10000, callback=callback)

    # episode_rewards, n_steps = evaluate_policy(model, env, n_eval_episodes=50, return_episode_rewards=True)
    # print(episode_rewards)
    # plt.hist(episode_rewards)
    # plt.show()

    # obs = env.reset()
    # for i in range(10000):
    #     action, _states = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()
    #     env.render()