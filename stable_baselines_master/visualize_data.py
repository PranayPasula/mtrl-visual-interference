# Adapted from Luuk Derksen

from __future__ import print_function
import time
import os

import numpy as np
import pandas as pd
import cv2

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns


# This loads and returns image data.
def load_data(data_path, is_color=True):

    data = [cv2.imread(os.path.join(data_path, f), is_color) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    data = np.array(data)
    img_shape = data[0].shape

    # data = data.astype('float32') / 255.0   
        
    return data


# This applies same transformations to data 
# as visual interference experiment does.
def transform_data(data):
    
    data1 = data.astype(np.uint8)
    data2 = (255 - data).astype(np.uint8)
    data3 = np.floor(np.sqrt(data) / np.sqrt(255.0)).astype(np.uint8)
    data4 = np.floor(data ** 2.0 / 255.0).astype(np.uint8)
    
    return data1, data2, data3, data4

if __name__=='__main__':
    
    N = 1000

    # Load data and apply same transformations as visual interference experiment does.
    data_path = 'beamrider_frames/'
    data = load_data(data_path)[:N]
    data = data.reshape(data.shape[0], -1)
    data1, data2, data3, data4 = transform_data(data)


    X = np.vstack([data1, data2, data3, data4])
    num_transforms = int(X.shape[0] / data.shape[0])
    assert num_transforms == X.shape[0] / data.shape[0], "Error: num_transforms must be integer-valued (i.e. decimal part = 0)" 
    y = np.vstack([np.full((data1.shape[0], 1), i) for i in np.linspace(1, num_transforms, num=num_transforms)])
    y = np.squeeze(y)
    print(X.shape, y.shape)

    # Load data into dataframe for easier plotting later
    feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    X, y = None, None

    print('Size of the dataframe: {}'.format(df.shape))

    # # PCA
    # pca = PCA(n_components=5)
    # pca_result = pca.fit_transform(df[feat_cols].values)

    # df['pca-one'] = pca_result[:,0]
    # df['pca-two'] = pca_result[:,1] 
    # df['pca-three'] = pca_result[:,2]
    
    # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


    np.random.seed(0)
    rand_perm_idxes = np.random.permutation(df.shape[0])

    # plt.figure(figsize=(16,10))
    # sns.scatterplot(
    #     x="pca-one", 
    #     y="pca-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", num_transforms),
    #     data=df.loc[rand_perm_idxes, :],
    #     legend="full",
    #     alpha=0.7
    # )
    # plt.show()

    # ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    # ax.scatter(
    #     xs=df.loc[rand_perm_idxes,:]["pca-one"], 
    #     ys=df.loc[rand_perm_idxes,:]["pca-two"], 
    #     zs=df.loc[rand_perm_idxes,:]["pca-three"], 
    #     c=df.loc[rand_perm_idxes,:]["y"], 
    #     cmap='tab10'
    # )
    # ax.set_xlabel('pca-one')
    # ax.set_ylabel('pca-two')
    # ax.set_zlabel('pca-three')
    # plt.show()

    # t-SNE with PCA input
    df_subset = df.loc[rand_perm_idxes[:N], :].copy()
    data_subset = df_subset[feat_cols].values

    pca_10 = PCA(n_components=10)
    pca_result_10 = pca_10.fit_transform(data_subset)

    df_subset['pca-one'] = pca_result_10[:, 0]
    df_subset['pca-two'] = pca_result_10[:, 1] 
    df_subset['pca-three'] = pca_result_10[:, 2]

    print('Cumulative explained variation for 10 principal components: {}'.format(np.sum(pca_10.explained_variance_ratio_)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_10)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-pca10-one'] = tsne_pca_results[:, 0]
    df_subset['tsne-pca10-two'] = tsne_pca_results[:, 1]

    plt.figure(figsize=(16,8))
    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", num_transforms),
        data=df_subset,
        legend="full",
        alpha=0.6,
        ax=ax1
    )
    ax3 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-pca10-one", y="tsne-pca10-two",
        hue="y",
        palette=sns.color_palette("hls", num_transforms),
        data=df_subset,
        legend="full",
        alpha=0.6,
        ax=ax3
    )
    plt.show()