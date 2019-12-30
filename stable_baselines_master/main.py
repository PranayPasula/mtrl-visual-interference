import os
import time
import threading
import copy

import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.deepq.dqn import DQN

import mysettings
from mylogger import Logger

import vec_monitor


if __name__ == '__main__':

    n_steps, best_mean_reward = 0, -np.inf

    n_indiv = 4
    n_multi = 1
    
    # to print timesteps during learning
    def callback(_locals, _globals):
        global n_steps, best_mean_reward
        if (n_steps % (100 * n_indiv)) == 0:
            print('training at timestep {}...'.format(n_steps // n_indiv))
        n_steps += 1

    env_names = ['MsPacmanNoFrameskip-v4' for i in range(n_indiv)]
    multi_env_names = ['MsPacmanNoFrameskip-v4' for i in range(n_multi)]

    # Share replay buffers, envs, Events that indicate whether a task is done, 
    # an Event that indicates whether indiv task agents can proceed,
    # an Event that indicates whether multitask agents can proceed,
    # and an Event that indicates that all indiv task agents have begun learning.
    shared_stuff = dict(indiv_replay_buffers=[None for i in range(n_indiv)],
                        indiv_timesteps=[0 for i in range(n_indiv)],
                        multi_timesteps=[0 for i in range(n_multi)],
                        unwrapped_indiv_envs=[],
                        indiv_agent_dones = [threading.Event() for i in range(n_indiv)],
                        multi_agent_dones = [threading.Event() for i in range(n_multi)],
                        indiv_agent_step_dones = [threading.Event() for i in range(n_indiv)],
                        multi_agent_step_dones = [threading.Event() for i in range(n_multi)],
                        goto_next_step = threading.Event(),
                        goto_next_barrier = threading.Barrier(n_indiv + n_multi),
                        all_timesteps_same= threading.Event(),
                        indiv_allow = threading.Event(),
                        multi_allow = threading.Event(),
                        learning_starts_ev = [threading.Event() for i in range(n_indiv)],
                        barrier_never_broken = True)

    # Set Events so that indiv and multitask agents are properly synchronized
    for i in range(n_indiv):
        shared_stuff['indiv_agent_dones'][i].clear()
        shared_stuff['indiv_agent_step_dones'][i].clear()
    for i in range(n_multi):
        shared_stuff['multi_agent_step_dones'][i].clear()
    shared_stuff['all_timesteps_same'].set()
    shared_stuff['indiv_allow'].set()


    indiv_envs = []
    multi_envs = []

    for i, env_name in enumerate(env_names):
        env = make_atari_env(env_name, num_env=1, seed=0) # num_env might need to be 1 for WSL ubuntu. will throw multithreading error otherwise
        shared_stuff['unwrapped_indiv_envs'].append(copy.deepcopy(env))
        env = VecFrameStack(env, n_stack=4)
        indiv_envs.append(env)
    shared_stuff['indiv_envs'] = indiv_envs

    for i, env_name in enumerate(multi_env_names):
        env = make_atari_env(env_name, num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        multi_envs.append(env)
    shared_stuff['multi_envs'] = multi_envs

    indiv_models = []
    multi_models = []

    # init model and then run training process.
    # thread function
    def create_model_then_learn(shared_stuff, model_type, model_num, policy_type, env, 
                            learning_starts=1000, prioritized_replay=False, batch_size=32, verbose=0):
        global logdirs
        assert model_type == 'i' or 'm', "invalid model type"
        if model_type == 'm':
            batch_size = n_indiv * batch_size
        print(type(env))
        model = DQN(policy_type, env, learning_starts=learning_starts, prioritized_replay=prioritized_replay, 
                    batch_size=batch_size, verbose=verbose, target_network_update_freq=5000, buffer_size=50000, shared_stuff=shared_stuff)
        model.model_type = model_type
        model.model_num = model_num

        if model_type == 'i':
            model.indiv_logger = Logger(logdirs['indiv'][model_num])
        elif model_type == 'm':
            for indiv_num in range(n_indiv):
                model.multi_loggers[indiv_num] = Logger(logdirs['multi'][model_num][indiv_num])

        model_type_str = 'indiv' if model_type == 'i' else 'multi'
        print("{} task DQN {} created".format(model_type_str, model_num))
        print("{} task DQN {} begins learning...".format(model_type_str, model_num))

        model.learn(total_timesteps=5000000, callback=callback, tb_log_name="DQN_{}_{}".format(model_type, model_num))

        print("{} task DQN {} done learning!".format(model_type_str, model_num))

        # TODO the following block isn't used
        if model_type == 'i':
            indiv_models.append(model)
        else:
            multi_models.append(model)


    # create directories to log and set up loggers
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    run_dir = os.path.join(data_path, time.strftime("%d-%m-%Y_%H-%M-%S"))
    logdirs = {'indiv' : [None for i in range(n_indiv)], 
               'multi' : [[None for i in range(n_indiv)] for j in range(n_multi)]}
    for indiv_num in range(n_indiv):
        logdir = "DQN_indiv_{}_{}".format(indiv_num, env_names[indiv_num])
        logdir = os.path.join(run_dir, logdir)
        logdirs['indiv'][indiv_num] = logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    for multi_num in range(n_multi):
        for indiv_num in range(n_indiv):
            above_logdir = "DQN_multi_{}".format(multi_num)
            logdir = "indiv_task_{}_{}".format(indiv_num, env_names[indiv_num])
            logdir = os.path.join(run_dir, above_logdir, logdir)
            logdirs['multi'][multi_num][indiv_num] = logdir
            if not os.path.exists(logdir):
                os.makedirs(logdir)

    indiv_threads = []
    multi_threads = [] 

    args = {'learning_starts': 25000,
            'prioritized_replay': False,
            'batch_size': 16,
            'verbose': 2}

    # spawn indiv task model threads
    for indiv_num in range(n_indiv):
        t = threading.Thread(target=create_model_then_learn, 
                        args=(shared_stuff, 'i', 
                            indiv_num, 'CnnPolicy', indiv_envs[indiv_num]), kwargs=args)
        print('indiv thread made')
        indiv_threads.append(t)

    # spawn multitask model threads
    for multi_num in range(n_multi):
        t = threading.Thread(target=create_model_then_learn,
                        args=(shared_stuff, 'm',
                            multi_num, 'CnnPolicy', multi_envs[multi_num]), kwargs=args)
        print('mt thread made')
        multi_threads.append(t)

    # start indiv task model threads
    for indiv_thread in indiv_threads:
        print('indiv start')
        indiv_thread.start()
        time.sleep(1) 

    time.sleep(3)

    # start multitask model threads
    for multi_thread in multi_threads:
        print('mt start')
        multi_thread.start()
        time.sleep(1)

    for indiv_thread in indiv_threads:
        indiv_thread.join()

    for multi_thread in multi_threads:
        multi_thread.join()


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