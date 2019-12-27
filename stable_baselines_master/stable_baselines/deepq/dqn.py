from functools import partial
from collections import OrderedDict

import threading
import sys
import time
import tensorflow as tf
import numpy as np
import gym
import copy


from stable_baselines import logger, deepq
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
# from stable_baselines.common.evaluation import evaluate_policy # MYEDIT
from stable_baselines.common.vec_env import VecEnv, VecFrameStack

from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger

import mysettings # for global replay buffers


def transform_obs(obs, model_num):

    if model_num == 0:
        return obs
    elif model_num == 1:
        return 255 - obs
    elif model_num == 2:
        return np.floor(np.sqrt(obs) * np.sqrt(255.))
    elif model_num == 3:
        return np.floor( (obs ** 2.) / 255. )


class DQN(OffPolicyRLModel):
    """
    The DQN model class.
    DQN paper: https://arxiv.org/abs/1312.5602
    Dueling DQN: https://arxiv.org/abs/1511.06581
    Double-Q Learning: https://arxiv.org/abs/1509.06461
    Prioritized Experience Replay: https://arxiv.org/abs/1511.05952

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param double_q: (bool) Whether to enable Double-Q learning or not.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float)alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, train_freq=1, batch_size=32, double_q=True,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, shared_stuff=None):

        # TODO: replay_buffer refactoring
        super(DQN, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, policy_base=DQNPolicy,
                                  requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.model_type = None # MYEDIT 'i' for individual task model or 'm' for multitask model
        self.model_num = None # MYEDIT a value in {0, 1, 2,..., total # of this type model}

        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q

        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None
        self.episode_reward = None

        # MYEDIT hacky inits
        self.shared_stuff = shared_stuff
        self.indiv_mean_10ep_reward = -np.inf
        self.indiv_mean_10ep_reward_actual = -np.inf
        self.multi_mean_10ep_rewards = [-np.inf for i in range(len(self.shared_stuff['indiv_replay_buffers']))]
        self.multi_mean_10ep_rewards_actual = [-np.inf for i in range(len(self.shared_stuff['indiv_replay_buffers']))]
        self.indiv_best_mean_10ep_reward = -np.inf
        self.indiv_best_mean_10ep_reward_actual = -np.inf
        self.multi_best_mean_10ep_rewards = [-np.inf for i in range(len(self.shared_stuff['indiv_replay_buffers']))]
        self.multi_best_mean_10ep_rewards_actual = [-np.inf for i in range(len(self.shared_stuff['indiv_replay_buffers']))]
        self.indiv_logger = None
        self.multi_loggers = [None for i in range(len(self.shared_stuff['indiv_replay_buffers']))]

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.step_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def setup_model(self):
        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DQN cannot output a gym.spaces.Box action space."  

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DQN model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.act, self._train_step, self.update_target, self.step_model = deepq.build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise,
                    sess=self.sess,
                    full_tensorboard_log=self.full_tensorboard_log,
                    double_q=self.double_q
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("deepq")

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            if self.model_type == 'i': # MYEDIT skip steps in the DQN algorithm if multitask model
                # Create the replay buffer
                if self.prioritized_replay:
                    self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                    if self.prioritized_replay_beta_iters is None:
                        prioritized_replay_beta_iters = total_timesteps
                    else:
                        prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                    self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                        initial_p=self.prioritized_replay_beta0,
                                                        final_p=1.0)
                else:
                    self.replay_buffer = ReplayBuffer(self.buffer_size)
                    self.beta_schedule = None

                self.shared_stuff['indiv_replay_buffers'][self.model_num] = self.replay_buffer # MYEDIT

                if replay_wrapper is not None:
                    assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                    self.replay_buffer = replay_wrapper(self.replay_buffer)


                # Create the schedule for exploration starting from 1.
                self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                initial_p=1.0,
                                                final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_rewards_actual = []
            episode_successes = []
            print(type(self.env))
            obs = self.env.reset()
            # Induce visual dissimilarity
            obs = transform_obs(obs, self.model_num)
            reset = True
            self.episode_reward = np.zeros((1,))
            start_time = time.time()
            for _ in range(total_timesteps):
                
                if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break

                if self.model_type == 'i': # MYEDIT skip taking action, updating exploration, and adding to replay buffer if multitask model
                    # if callback is not None:
                    #     # Only stop training if return value is False, not when it is None. This is for backwards
                    #     # compatibility with callbacks that have no return statement.
                    #     if callback(locals(), globals()) is False:
                    #         break
                    # Take action and update exploration to the newest value
                    kwargs = {}
                    if not self.param_noise:
                        update_eps = self.exploration.value(self.num_timesteps)
                        update_param_noise_threshold = 0.
                    else:
                        update_eps = 0.
                        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                        # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                        # for detailed explanation.
                        update_param_noise_threshold = \
                            -np.log(1. - self.exploration.value(self.num_timesteps) +
                                    self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                        kwargs['reset'] = reset
                        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                        kwargs['update_param_noise_scale'] = True
                    with self.sess.as_default():
                        action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                    env_action = action
                    reset = False
                    new_obs, rew, done, info = self.env.step(env_action)
                    # Store episode reward once episode ends
                    if info is not None:
                        episode_info = info.get('episode')
                        # if self.num_timesteps > 150 and done:
                            # import pdb; pdb.set_trace()
                        if episode_info is not None:
                            print(episode_info)
                            # import pdb; pdb.set_trace()
                            episode_rewards_actual.append(episode_info['r'])
                    # Induce visual dissimilarity
                    new_obs = transform_obs(new_obs, self.model_num)

                    # Start main indiv/multitask agent synchronization cycle 
                    # after all indiv task agents start learning.
                    if self.num_timesteps > self.learning_starts:
                        self.shared_stuff['indiv_allow'].wait()
                        self.shared_stuff['multi_allow'].clear()
                        self.shared_stuff['indiv_agent_dones'][self.model_num].clear()

                    # Update self.shared_stuff['unwrapped_indiv_envs'] just before multitask dqns are evaluated
                    if self.num_timesteps % 5000 == 0:
                        self.shared_stuff['unwrapped_indiv_envs'][self.model_num] = copy.deepcopy(self.env.unwrapped)
      
                    # Store transition in the replay buffer.
                    self.replay_buffer.add(obs, action, rew, new_obs, float(done))

                    obs = new_obs

                    # TODO Change inside this block to capture actual episode rewards
                    # if writer is not None:
                    #     ep_rew = np.array([rew]).reshape((1, -1))
                    #     ep_done = np.array([done]).reshape((1, -1))
                    #     self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                    #                                                     self.num_timesteps)

                    episode_rewards[-1] += rew
                    if done:
                        # MYEDIT no need to track successes
                        # maybe_is_success = info.get('is_success')
                        # if maybe_is_success is not None:
                        #     episode_successes.append(float(maybe_is_success))
                        if not isinstance(self.env, VecEnv):
                            import pdb; pdb.set_trace()
                            obs = self.env.reset()
                            # Induce visual dissimilarity
                            obs = transform_obs(obs, self.model_num)
                        episode_rewards.append(0.0)
                        reset = True
                
                    if self.num_timesteps > self.learning_starts:
                        self.shared_stuff['learning_starts_ev'][self.model_num].set()
                        for i in self.shared_stuff['learning_starts_ev']:
                            i.wait()
                        self.shared_stuff['indiv_agent_dones'][self.model_num].set()
                        for i in self.shared_stuff['indiv_agent_dones']:
                            i.wait()
                        self.shared_stuff['indiv_allow'].clear()
                        if self.shared_stuff['barrier_never_broken'] == True:
                            self.shared_stuff['indiv_allow'].set()
                        self.shared_stuff['multi_allow'].set()


                # obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights, batch_idxes = [False for i in range(8)]
                if self.model_type == 'i': # MYEDIT different replay buffer sampling for indiv vs multi models
                    # Do not train if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    can_sample = self.replay_buffer.can_sample(self.batch_size)
                    if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:
                        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                        if self.prioritized_replay:
                            experience = self.replay_buffer.sample(self.batch_size,
                                                                beta=self.beta_schedule.value(self.num_timesteps))
                            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        else:
                            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                            weights, batch_idxes = np.ones_like(rewards), None

                        # # bug with last line: "local variable 'obses_t' referenced before assignment"
                        # if writer is not None:
                        #     # run loss backprop with summary, but once every 100 steps save the metadata
                        #     # (memory, compute time, ...)
                        #     if (1 + self.num_timesteps) % 100 == 0:
                        #         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        #         run_metadata = tf.RunMetadata()
                        #         summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                        #                                                 dones, weights, sess=self.sess, options=run_options,
                        #                                                 run_metadata=run_metadata)
                        #         writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        #     else:
                        #         summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                        #                                                 dones, weights, sess=self.sess)
                        #     writer.add_summary(summary, self.num_timesteps)
                        # else:
                        _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                    if self.prioritized_replay:
                            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                    if can_sample and self.num_timesteps > self.learning_starts and \
                            self.num_timesteps % self.target_network_update_freq == 0:
                        # Update target network periodically.
                        self.update_target(sess=self.sess)

                elif self.model_type == 'm':
                    # don't let multitask agents start learning until the indiv tasks all start learning
                    if not all([i.is_set() for i in self.shared_stuff['learning_starts_ev']]):
                        self.shared_stuff['multi_allow'].wait()
                        self.num_timesteps = self.learning_starts + 1
                    elif self.num_timesteps <= self.learning_starts:
                        self.num_timesteps = self.learning_starts + 1
                    n_indiv = len(self.shared_stuff['indiv_replay_buffers'])
                    indiv_model_batch_size = int(self.batch_size / n_indiv) # bc (multi model batch size) = n_indiv * (indiv model batch size)
                    can_samples = [self.shared_stuff['indiv_replay_buffers'][i].can_sample(indiv_model_batch_size) for i in range(n_indiv)]

                    # don't train if warmup phase is not over or if there aren't enough samples in replay buffers.
                    # all(can_sample) is hacky but simplifies code and shouldn't affect results.
                    if all(can_samples) and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:

                        # randomly sample across indiv model replay buffers.
                        sample_idxes = np.random.randint(0, n_indiv, size=self.batch_size)
                        buffers_to_sample, sample_sizes = np.unique(sample_idxes, return_counts=True)

                        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                        experience = np.array([])
                        if self.prioritized_replay:

                            self.shared_stuff['multi_allow'].wait()
                            self.shared_stuff['indiv_allow'].clear()
                            self.shared_stuff['multi_agent_dones'][self.model_num].clear()

                            for (buffer_idx, sample_size) in zip(buffers_to_sample, sample_sizes):
                                exp_buff = self.shared_stuff['indiv_replay_buffers'][buffer_idx].sample(sample_size,
                                                                                beta=self.beta_schedule.value(self.num_timesteps))
                                exp_buff = np.asarray(exp_buff) # bc prioritized replay buffer sample() returns tuple
                                if experience.shape[0] == 0:
                                    experience = exp_buff
                                else:
                                    experience = np.concatenate((experience, exp_buff), axis=1)
                            
                            self.shared_stuff['multi_agent_dones'][self.model_num].set()
                            for i in self.shared_stuff['multi_agent_dones']:
                                i.wait()
                            self.shared_stuff['multi_allow'].clear()
                            self.shared_stuff['indiv_allow'].set()

                            # since we sorted by replay buffer index, randomize the order of experiences
                            assert experience.shape[1] == self.batch_size, "error: number of multitask model experiences != multitask model batch size"
                            randomized_idxes = np.random.choice(experience.shape[1], size=experience.shape[1], replace=False)
                            experience = experience[:, randomized_idxes]
                            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        else:
                            obses_t, actions, rewards, obses_tp1, dones, weights = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
                            
                            self.shared_stuff['multi_allow'].wait()
                            self.shared_stuff['indiv_allow'].clear()
                            self.shared_stuff['multi_agent_dones'][self.model_num].clear()

                            for (buffer_idx, sample_size) in zip(buffers_to_sample, sample_sizes):
                                obses_t_temp, actions_temp, rewards_temp, obses_tp1_temp, dones_temp = self.shared_stuff['indiv_replay_buffers'][buffer_idx].sample(sample_size)
                                weights_temp, batch_idxes = np.ones_like(rewards_temp), None
                                
                                if obses_t.shape[0] == 0:
                                    obses_t = obses_t_temp
                                    actions = actions_temp
                                    rewards = rewards_temp
                                    obses_tp1 = obses_tp1_temp
                                    dones = dones_temp
                                    weights = weights_temp
                                else:
                                    obses_t = np.concatenate((obses_t, obses_t_temp), axis=0)
                                    actions = np.concatenate((actions, actions_temp), axis=0)
                                    rewards = np.concatenate((rewards, rewards_temp), axis=0)
                                    obses_tp1 = np.concatenate((obses_tp1, obses_tp1_temp), axis=0)
                                    dones = np.concatenate((dones, dones_temp), axis=0)
                                    weights = np.concatenate((weights, weights_temp), axis=0)

                                # exp_buff = np.asarray([obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes])
                                # if experience.shape[0] == 0:
                                #     experience = exp_buff
                                # else:
                                #     experience = np.concatenate((experience, exp_buff), axis=1)

                            # self.shared_stuff['multi_agent_dones'][self.model_num].set()
                            # for i in self.shared_stuff['multi_agent_dones']:
                            #     i.wait()
                            # self.shared_stuff['multi_allow'].clear()
                            # self.shared_stuff['indiv_allow'].set()
                            
                            assert obses_t.shape[0] == self.batch_size, "error: number of multitask model experiences != multitask model batch size"
                            # since we sorted by replay buffer index, randomize the order of experiences
                            randomized_idxes = np.random.choice(self.batch_size, size=self.batch_size, replace=False)
                            obses_t = obses_t[randomized_idxes]
                            actions = actions[randomized_idxes]
                            rewards = rewards[randomized_idxes]
                            obses_tp1 = obses_tp1[randomized_idxes]
                            dones = dones[randomized_idxes]
                            weights = weights[randomized_idxes]
                            # obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes = np.split(experience, experience.shape[0])

                        # if writer is not None:
                        #     # run loss backprop with summary, but once every 100 steps save the metadata
                        #     # (memory, compute time, ...)
                        #     if (1 + self.num_timesteps) % 100 == 0:
                        #         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        #         run_metadata = tf.RunMetadata()
                        #         summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                        #                                             dones, weights, sess=self.sess, options=run_options,
                        #                                             run_metadata=run_metadata)
                        #         writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        #     else:
                        #         summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                        #                                             dones, weights, sess=self.sess)
                        #     writer.add_summary(summary, self.num_timesteps)
                        # else:                            
                        _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                        if self.prioritized_replay:
                            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                            # TODO
                            # below line won't work correctly for multitask models because 
                            # batch_idxes alone doesn't differentiate source replay buffers
                            raise NotImplementedError
                            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                    if all(can_samples) and self.num_timesteps > self.learning_starts and \
                            self.num_timesteps % self.target_network_update_freq == 0:
                        # Update target network periodically.
                        self.update_target(sess=self.sess)

                # For indiv-task agent, log mean return over last 10 ep.
                # For multi-task agent, evaluate policy on each task and log mean returns over some # ep.
                if (self.num_timesteps > self.learning_starts) and (self.num_timesteps) % 5000 == 0:

                    print(str(time.time() - start_time) + " sec")

                    mean_10ep_reward_actual = 0
                    if self.model_type == 'i':
                        print(len(episode_rewards_actual))
                        if len(episode_rewards_actual) == 0:
                            mean_10ep_reward_actual = -np.inf
                        else:
                            mean_10ep_reward_actual = round(float(np.mean(episode_rewards_actual[-10:])), 1)
                        self.indiv_mean_10ep_reward_actual = mean_10ep_reward_actual
                        self.indiv_best_mean_10ep_reward_actual = max(self.indiv_best_mean_10ep_reward_actual, mean_10ep_reward_actual)
                        print("indiv agent {} has avg reward {}".format(self.model_num, mean_10ep_reward_actual))

                    elif self.model_type =='m':
                        for i in self.shared_stuff['learning_starts_ev']:
                            i.wait()

                        print("multi_agent {}".format(self.model_num))

                        # self.shared_stuff['multi_allow'].wait()
                        # self.shared_stuff['indiv_allow'].clear()
                        # self.shared_stuff['multi_agent_dones'][self.model_num].clear()
                        
                        for indiv_task_num in range(len(self.shared_stuff['indiv_replay_buffers'])):

                            # envs in shared_stuff['unwrapped_indiv_envs'] are unwrapped, so wrap before evaluating policy
                            eval_env = VecFrameStack(copy.deepcopy(self.shared_stuff['unwrapped_indiv_envs'][indiv_task_num]), n_stack=4)

                            # calling a function inside of a method by passing in self feels like bad practice
                            episode_rewards, episode_rewards_actual, _ = evaluate_policy(self, indiv_task_num, eval_env, n_eval_episodes=20, return_episode_rewards=True)
                            if len(episode_rewards_actual) > 0:
                                self.multi_mean_10ep_rewards_actual[indiv_task_num] = round(float(np.mean(episode_rewards_actual)), 1)
                            else:
                                self.multi_mean_10ep_rewards_actual[indiv_task_num] = 0

                            print("multi agent {} has avg reward {} on task {}".format(self.model_num, self.multi_mean_10ep_rewards_actual[indiv_task_num], indiv_task_num))

                    # if self.model_type == 'i':
                    #     num_episodes = len(episode_rewards_actual)
                    #     if self.verbose >= 1:
                    #         logger.record_tabular("steps", self.num_timesteps)
                    #         logger.record_tabular("episodes", num_episodes)
                    #         logger.record_tabular("mean 10 ep actual reward", mean_10ep_reward_actual)
                    #         logger.record_tabular("% time spent exploring",
                    #                             int(100 * self.exploration.value(self.num_timesteps)))
                    #         logger.dump_tabular()

                    if self.model_type == 'i':
                        logs = OrderedDict()
                        logs["Train_EnvstepsSoFar"] = self.num_timesteps
                        print("Timestep %d" % (self.num_timesteps,))
                        if self.indiv_mean_10ep_reward_actual > -1000:
                            logs["Train_AverageReturn"] = self.indiv_mean_10ep_reward_actual
                            print("mean reward (10 episodes) %f" % self.indiv_mean_10ep_reward_actual)
                        if self.indiv_best_mean_10ep_reward_actual > -1000:
                            logs["Train_BestReturn"] = self.indiv_best_mean_10ep_reward_actual
                            print("best mean reward %f" % self.indiv_best_mean_10ep_reward_actual)
                        if start_time is not None:
                            time_since_start = (time.time() - start_time)
                            print("running time %f" % time_since_start)
                            logs["TimeSinceStart"] = time_since_start

                        sys.stdout.flush()

                        for key, value in logs.items():
                            print('{} : {}'.format(key, value))
                            self.indiv_logger.log_scalar(value, key, self.num_timesteps)
                            print('Done logging...\n\n')

                        self.indiv_logger.flush()
                    else:
                        for indiv_task_num in range(len(self.shared_stuff['indiv_replay_buffers'])):
                            logs = OrderedDict()
                            logs["Train_EnvstepsSoFar"] = self.num_timesteps
                            print("Timestep %d" % (self.num_timesteps,))
                            if self.multi_mean_10ep_rewards_actual[indiv_task_num] > -1000:
                                logs["Train_AverageReturn"] = self.multi_mean_10ep_rewards_actual[indiv_task_num]
                                print("mean reward (10 episodes) %f" % self.multi_mean_10ep_rewards_actual[indiv_task_num])
                            if self.multi_best_mean_10ep_rewards_actual[indiv_task_num] > -1000:
                                logs["Train_BestReturn"] = self.multi_best_mean_10ep_rewards_actual[indiv_task_num]
                                print("best mean reward %f" % self.multi_best_mean_10ep_rewards_actual[indiv_task_num])
                            if start_time is not None:
                                time_since_start = (time.time() - start_time)
                                print("running time %f" % time_since_start)
                                logs["TimeSinceStart"] = time_since_start

                            sys.stdout.flush()

                            for key, value in logs.items():
                                print('{} : {}'.format(key, value))
                                self.multi_loggers[indiv_task_num].log_scalar(value, key, self.num_timesteps)
                                print('Done logging...\n\n')

                            self.multi_loggers[indiv_task_num].flush()

                if self.model_type == 'm': 
                    self.shared_stuff['multi_agent_dones'][self.model_num].set()
                    for i in self.shared_stuff['multi_agent_dones']:
                        i.wait()
                    self.shared_stuff['multi_allow'].clear()
                    self.shared_stuff['indiv_allow'].set()
                    
                # num_episodes = len(episode_rewards)
                # if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                #     logger.record_tabular("steps", self.num_timesteps)
                #     logger.record_tabular("episodes", num_episodes)
                #     # MYEDIT no need to track successes
                #     # if len(episode_successes) > 0:
                #     #     logger.logkv("success rate", np.mean(episode_successes[-10:]))
                #     logger.record_tabular("mean 10 episode reward", mean_10ep_reward)
                #     logger.record_tabular("% time spent exploring",
                #                         int(100 * self.exploration.value(self.num_timesteps)))
                #     logger.dump_tabular()

                # num_episodes = len(episode_rewards_actual)
                # if self.model_type == 'i':
                #     if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards_actual) % log_interval == 0:
                #         logger.record_tabular("steps", self.num_timesteps)
                #         logger.record_tabular("episodes", num_episodes)
                #         logger.record_tabular("mean 10 ep actual reward", mean_10ep_reward_actual)
                #         logger.record_tabular("% time spent exploring",
                #                             int(100 * self.exploration.value(self.num_timesteps)))
                #         logger.dump_tabular()
               
               
                self.num_timesteps += 1
                if self.num_timesteps % 1000 == 0:
                    print("{} {} {}".format(self.model_type, self.model_num, self.num_timesteps))
                if all([i.is_set() for i in self.shared_stuff['learning_starts_ev']]):
                    
                    # if self.model_type == 'i':
                    #     self.shared_stuff['indiv_timesteps'][self.model_num] = self.num_timesteps
                    # else:
                    #     self.shared_stuff['multi_timesteps'][self.model_num] = self.num_timesteps

                    # if self.model_type == 'i':
                    #     if min(self.shared_stuff['indiv_timesteps']) < max(self.shared_stuff['multi_timesteps']):
                    #         self.shared_stuff['timestep_comparator'].clear()
                    #     else:
                    #         self.shared_stuff['timestep_comparator'].set()
                    # else:
                    #     self.shared_stuff['timestep_comparator'].wait()

                    # self.shared_stuff['all_timesteps_same'].wait()
                    # diff_flag = 0
                    # for indiv_timestep in self.shared_stuff['indiv_timesteps']:
                    #     for multi_timestep in self.shared_stuff['multi_timesteps']:
                    #         if indiv_timestep != multi_timestep:
                    #             diff_flag = 1
                    # if diff_flag == 0:
                    #     self.shared_stuff['all_timesteps_same'].set()
                    # else:
                    #     self.shared_stuff['all_timesteps_same'].clear()
                    #     self.shared_stuff['all_timesteps_same'].wait()

                    # self.shared_stuff['goto_next_step'].clear()

                    # if self.model_type == 'i':
                    #     self.shared_stuff['indiv_agent_step_dones'][self.model_num].set()
                    # else:
                    #     self.shared_stuff['multi_agent_step_dones'][self.model_num].set()

                    # if all([indiv_step_done.is_set() for indiv_step_done in self.shared_stuff['indiv_agent_step_dones']]) and \
                    #             all([multi_step_done.is_set() for multi_step_done in self.shared_stuff['multi_agent_step_dones']]):

                    #     self.shared_stuff['goto_next_step'].set()
                    #     if self.model_type == 'i':
                    #         self.shared_stuff['indiv_agent_step_dones'][self.model_num].clear()
                    #     else:
                    #         self.shared_stuff['multi_agent_step_dones'][self.model_num].clear()
                    # else:
                    #     self.shared_stuff['goto_next_step'].wait()
                    
#                     if self.shared_stuff['goto_next_barrier'].n_waiting == 2:
#                         self.shared_stuff['indiv_allow'].set()
                    print(self.shared_stuff['goto_next_barrier'].n_waiting)
                    self.shared_stuff['goto_next_barrier'].wait()
                    if (self.model_type == 'i') and (self.model_num == '0'):
                        self.shared_stuff['goto_next_barrier'] = threading.Barrier(5)
                    self.shared_stuff['barrier_never_broken'] = False

        return self

    # MYEDIT TODO (make sure of this) predict is used for evaluation only,
    # so don't need to do anything with shared_stuff['unwrapped_indiv_envs'] monitor list here
    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            assert isinstance(self.action_space, gym.spaces.Discrete)
            actions = actions.reshape((-1,))
            assert observation.shape[0] == actions.shape[0], "Error: batch sizes differ for actions and observations."
            actions_proba = actions_proba[np.arange(actions.shape[0]), actions]
            # normalize action proba shape
            actions_proba = actions_proba.reshape((-1, 1))
            if logp:
                actions_proba = np.log(actions_proba)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def get_parameter_list(self):
        return self.params

    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "double_q": self.double_q,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

def evaluate_policy(model, indiv_task_num, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False):
        """
        Runs policy for `n_eval_episodes` episodes and returns average reward.
        This is made to work only with one env.

        :param model: (BaseRLModel) The RL agent you want to evaluate.
        :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
            this must contain only one environment.
        :param n_eval_episodes: (int) Number of episode to evaluate the agent
        :param deterministic: (bool) Whether to use deterministic or stochastic actions
        :param render: (bool) Whether to render the environement or not
        :param callback: (callable) callback function to do additional checks,
            called after each step.
        :param reward_threshold: (float) Minimum expected reward per episode,
            this will raise an error if the performance is not met
        :param return_episode_rewards: (bool) If True, a list of reward per episode
            will be returned instead of the mean.
        :return: (float, int) Mean reward per episode, total number of steps
            returns ([float], int) when `return_episode_rewards` is True
        """
        if isinstance(env, VecEnv):
            assert env.num_envs == 1, "You must pass only one environment when using this function"

        episode_rewards, n_steps = [], 0
        episode_rewards_actual = []
        for _ in range(n_eval_episodes):
            obs = env.reset()
            # Apply visual transformation that matches that of corresponding indiv task
            obs = transform_obs(obs, indiv_task_num)
            done, state = False, None
            episode_reward = 0.0
            while not done:
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, _info = env.step(action)
                # Apply visual transformation that matches that of corresponding indiv task
                obs = transform_obs(obs, indiv_task_num)
                episode_reward += reward
                if callback is not None:
                    callback(locals(), globals())
                n_steps += 1
                if render:
                    env.render()
            # Store clipped episode rewards and actual episode rewards
            episode_rewards.append(episode_reward)
            if _info[0].get('episode') is not None:
                episode_reward_actual = _info[0]['episode']['r']
                episode_rewards_actual.append(episode_reward_actual)
        mean_reward = np.mean(episode_rewards)
        mean_reward_actual = np.mean(episode_rewards_actual)
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, 'Mean reward below threshold: '\
                                            '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
        if return_episode_rewards:
            return episode_rewards, episode_rewards_actual, n_steps
        return mean_reward, mean_reward_actual, n_steps
