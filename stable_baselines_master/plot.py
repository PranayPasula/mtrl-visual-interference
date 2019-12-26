# import matplotlib
# matplotlib.use('TkAgg')

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

    filenames = ["data/26-12-2019_03-46-07/DQN_indiv_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353584.DESKTOP-VOR7CBL",
        "data/26-12-2019_03-46-07/DQN_indiv_2_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353587.DESKTOP-VOR7CBL",
        "data/26-12-2019_03-46-07/DQN_indiv_2_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353587.DESKTOP-VOR7CBL",
        "data/26-12-2019_03-46-07/DQN_multi_0/indiv_task_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353589.DESKTOP-VOR7CBL",
        "data/26-12-2019_03-46-07/DQN_multi_0/indiv_task_1_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353589.DESKTOP-VOR7CBL",
        "data/26-12-2019_03-46-07/DQN_multi_0/indiv_task_2_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353589.DESKTOP-VOR7CBL",
        "data/26-12-2019_03-46-07/DQN_multi_1/indiv_task_0_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353589.DESKTOP-VOR7CBL",
        "data/26-12-2019_03-46-07/DQN_multi_1/indiv_task_1_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353589.DESKTOP-VOR7CBL",
        "data/26-12-2019_03-46-07/DQN_multi_1/indiv_task_2_MsPacmanNoFrameskip-v4/events.out.tfevents.1577353589.DESKTOP-VOR7CBL"]

    for filename in filenames:
        train_returns = parse_tf_events_file(filename)
        plt.plot(train_returns)
    
    plt.legend(['1', '2', '3', '1_1', '1_2', '1_3', '2_1', '2_2', '2_3'])
    plt.show()