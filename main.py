import os
import time
from multiprocessing import Process

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

    def callback(_locals, _globals):
        global n_steps, best_mean_reward
        if (n_steps % 100) == 0:
            print('training at timestep {}...'.format(n_steps))
        n_steps += 1

    env_1 = make_atari_env('MsPacmanNoFrameskip-v4', num_env=1, seed=0)
    env_1 = VecFrameStack(env_1, n_stack=4)

    env_2 = make_atari_env('BoxingNoFrameskip-v4', num_env=1, seed=0)
    env_2 = VecFrameStack(env_2, n_stack=4)

    models = [None, None]

    def create_model_then_learn(idx, policy_type, env, learning_starts=1000, prioritized_replay=False, verbose=0):
        models[idx] = DQN(policy_type, env, learning_starts=learning_starts, prioritized_replay=prioritized_replay, verbose=verbose)
        print("model {} created".format(idx))
        print("model {} begins learning...".format(idx))
        models[idx].learn(total_timesteps=1500, callback=callback)
        print("model {} done learning!".format(idx))


    processes = []
    p_1 = Process(target=create_model_then_learn, args=( (0, 'CnnPolicy', env_1, 1000, True, 1) ) )
    p_2 = Process(target=create_model_then_learn, args=( (1, 'CnnPolicy', env_2, 1000, True, 1) ) )
    processes.append(p_1)
    processes.append(p_2)

    p_1.start()
    time.sleep(1)    
    p_2.start()
    time.sleep(1)

    for p_x in processes:
        p_x.join()


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