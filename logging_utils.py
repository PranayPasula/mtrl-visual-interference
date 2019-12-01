import time
import sys
import numpy as np
from collections import OrderedDict
from logger import Logger

def perform_logging(agent, tag):
    episode_rewards = get_wrapper_by_name(agent.env, "Monitor").get_episode_rewards()
    if len(episode_rewards) > 0:
        agent.mean_episode_reward = np.mean(episode_rewards[-100:])
    if len(episode_rewards) > 100:
        agent.best_mean_episode_reward = max(agent.best_mean_episode_reward, agent.mean_episode_reward)
 
    logs = OrderedDict()

    logs["Train_EnvstepsSoFar"] = agent.timesteps
    print("Timestep %d" % (agent.timesteps,))
    if agent.mean_episode_reward > -5000:
        logs["Train_AverageReturn"] = np.mean(agent.mean_episode_reward)
    print("mean reward (100 episodes) %f" % agent.mean_episode_reward)
    if agent.best_mean_episode_reward > -5000:
        logs["Train_BestReturn"] = np.mean(agent.best_mean_episode_reward)
    print("best mean reward %f" % agent.best_mean_episode_reward)

    if agent.start_time is not None:
        time_since_start = (time.time() - agent.start_time)
        print("running time %f" % time_since_start)
        logs["TimeSinceStart"] = time_since_start

    sys.stdout.flush()

    for key, value in logs.items():
        print('{} : {}'.format(key, value))
        agent.logger.log_scalar(value, key, agent.t)
    print('Done logging...\n\n')

    agent.logger.flush()