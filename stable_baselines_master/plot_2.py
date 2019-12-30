import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def parse_tf_events_file(filename):
    train_returns = []
    for e in tf.train.summary_iterator(filename): # changed from file to filename
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                train_returns.append(v.simple_value)

    return train_returns


if __name__=='__main__':

    filenames_1 = ["data/26-12-2019_12-17-26/DQN_indiv_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577362662.winter-19-gpu-works-vm",
                  "data/26-12-2019_12-17-26/DQN_indiv_1_MsPacmanNoFrameskip-v4/events.out.tfevents.1577362662.winter-19-gpu-works-vm",
                  "data/26-12-2019_12-17-26/DQN_multi_0/indiv_task_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577362670.winter-19-gpu-works-vm",
                  "data/26-12-2019_12-17-26/DQN_multi_0/indiv_task_1_MsPacmanNoFrameskip-v4/events.out.tfevents.1577362670.winter-19-gpu-works-vm"]
    
    filenames_2 = ["data/27-12-2019_05-39-46/DQN_indiv_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425199.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-39-46/DQN_indiv_1_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425198.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-39-46/DQN_multi_0/indiv_task_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425199.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-39-46/DQN_multi_0/indiv_task_1_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425199.winter-19-gpu-works-vm"]
    
    filenames_3 = ["data/27-12-2019_05-46-40/DQN_indiv_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425623.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-46-40/DQN_indiv_1_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425624.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-46-40/DQN_indiv_2_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425625.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-46-40/DQN_indiv_3_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425625.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-46-40/DQN_multi_0/indiv_task_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425625.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-46-40/DQN_multi_0/indiv_task_1_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425625.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-46-40/DQN_multi_0/indiv_task_2_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425625.winter-19-gpu-works-vm",
                  "data/27-12-2019_05-46-40/DQN_multi_0/indiv_task_3_MsPacmanNoFrameskip-v4/events.out.tfevents.1577425625.winter-19-gpu-works-vm"]

    for filename in filenames_1:
        train_returns = parse_tf_events_file(filename)
        plt.plot(train_returns, linewidth=0.5)
    
    plt.legend(['1', '2', '1_1', '1_2'])
    plt.savefig('first.png', tight='bbox_inches')
    plt.close()
    
    for filename in filenames_2:
        train_returns = parse_tf_events_file(filename)
        plt.plot(train_returns, linewidth=0.5)
    
    plt.legend(['1', '2', '1_1', '1_2'])
    plt.savefig('second.png', tight='bbox_inches')
    plt.close()
    
    for filename in filenames_3:
        train_returns = parse_tf_events_file(filename)
        plt.plot(train_returns, linewidth=0.5)
    
    plt.legend(['1', '2', '3', '4', '1_1', '1_2', '1_3', '1_4'])
    plt.savefig('third.png', tight='bbox_inches')
    plt.close()