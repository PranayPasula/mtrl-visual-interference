import os
import time
from multiprocessing import Manager, Process

import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# from stable_baselines_master.stable_baselines.common.cmd_util import make_atari_env
# from stable_baselines_master.stable_baselines.common.vec_env import VecFrameStack
# from stable_baselines_master.stable_baselines.deepq.dqn import DQN
# from stable_baselines_master.stable_baselines.common.evaluation import evaluate_policy

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.deepq.dqn import DQN

import mysettings
from mylogger import Logger

if __name__ == '__main__':

    n_steps, best_mean_reward = 0, -np.inf

    # to print timesteps during learning
    def callback(_locals, _globals):
        global n_steps, best_mean_reward
        if (n_steps % 100) == 0:
            print('training at timestep {}...'.format(n_steps))
        n_steps += 1

    n_indiv = 2
    n_multi = 1

    env_names = ['MsPacmanNoFrameskip-v4' for i in range(n_indiv)]
    multi_env_names = ['MsPacmanNoFrameskip-v4' for i in range(n_multi)]

    # # initialize replay buffer array for future storage and access by models
    manager = Manager()
    shared_stuff = manager.dict(unwrapped_indiv_envs=manager.list([None for i in range(n_indiv)]),
                                indiv_replay_buffers=manager.list([None for i in range(n_indiv)]),
                                indiv_allow=None,
                                multi_allow=None,
                                indiv_buffer_dones=None,
                                multi_buffer_dones=None)

    indiv_envs = []
    multi_envs = []

    for env_name in env_names:
        env = make_atari_env(env_name, num_env=1, seed=0) # num_env might need to be 1 for WSL ubuntu. will throw multiprocessing error otherwise
        # unwrapped_indiv_envs.append(env) # must be before VecFrameStack bc can't pickle VecFrameStack object
        env = VecFrameStack(env, n_stack=4)
        indiv_envs.append(env)

    for env_name in multi_env_names:
        env = make_atari_env(env_name, num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        multi_envs.append(env)

    """
        Events for synchronizing access to replay buffers.

        Access requirements are
            for individual task DQNs: indiv_allow is T and multi_buffer_dones are all T
            for multitask DQNs:       multi_allow is T and indiv_buffer_dones are all T

        Initializations are
            indiv_allow = T
            multi_allow = F
            indiv_buffer_dones = T
            multi_buffer_dones = T,
        and are set so that individual task DQNs always access the buffers first.
    """
    indiv_allow = manager.Event()
    indiv_allow.set()
    multi_allow = manager.Event()
    indiv_buffer_dones = manager.list()
    multi_buffer_dones = manager.list()
    for i in range(n_indiv):
        indiv_buffer_done = manager.Event()
        indiv_buffer_done.set()
        indiv_buffer_dones.append(indiv_buffer_done)
    for i in range(n_multi):
        multi_buffer_done = manager.Event()
        multi_buffer_done.set()
        multi_buffer_dones.append(multi_buffer_done)
    import pdb; pdb.set_trace()

    # buffer_sync_events = {'indiv_allow': []
    #                       }    
    # indiv_allow = manager.Event()
    # indiv_allow.set()
    # multi_allow = manager.Event()
    # indiv_buffer_dones = manager.list([manager.Event() for i in range(n_indiv)])
    # multi_buffer_dones = manager.list([manager.Event() for i in range(n_multi)])
    # import pdb; pdb.set_trace()
    # for i in range(n_indiv):
    #     indiv_buffer_dones[i].set()
    # for i in range(n_multi):
    #     multi_buffer_dones[i].set()
    # shared_stuff['indiv_allow'] = indiv_allow
    # shared_stuff['multi_allow'] = multi_allow
    # shared_stuff['indiv_buffer_dones'] = indiv_buffer_dones
    # shared_stuff['multi_buffer_dones'] = multi_buffer_dones

    indiv_models = []
    multi_models = []

    def create_model_then_learn(shared_stuff, model_type, model_num, policy_type, env, 
                            learning_starts=100, prioritized_replay=False, batch_size=32, verbose=0):
        global logdirs
        assert model_type == 'i' or 'm', "invalid model type"
        if model_type == 'm':
            batch_size = n_indiv * batch_size
            
        model = DQN(policy_type, env, learning_starts=learning_starts, prioritized_replay=prioritized_replay, 
                    batch_size=batch_size, verbose=verbose, shared_stuff=shared_stuff)
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

        model.learn(total_timesteps=1000, callback=callback, tb_log_name="DQN_{}_{}".format(model_type, model_num))

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

    indiv_processes = []
    multi_processes = [] 

    args = {'learning_starts': 100,
            'prioritized_replay': False,
            'batch_size': 32,
            'verbose': 1}

    # spawn indiv task model processes
    for indiv_num in range(n_indiv):
        p = Process(target=create_model_then_learn, 
                        args=(shared_stuff, 'i', 
                            indiv_num, 'CnnPolicy', indiv_envs[indiv_num]), kwargs=args)
        print('indiv process made')
        indiv_processes.append(p)

    # spawn multitask model processes
    for multi_num in range(n_multi):
        p = Process(target=create_model_then_learn,
                        args=(shared_stuff, 'm',
                            multi_num, 'CnnPolicy', multi_envs[multi_num]), kwargs=args)
        print('mt process made')
        multi_processes.append(p)

    # start indiv task model processes
    for indiv_process in indiv_processes:
        print('indiv start')
        indiv_process.start()
        time.sleep(1) 

    time.sleep(5)

    # start multitask model processes
    for multi_process in multi_processes:
        print('mt start')
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