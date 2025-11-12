import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

import utils
from shrink import shrink
from clustering import project_clustering, plotCluster, plotCluster3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default='../../result/', type=str, help='the folder to save results')
    parser.add_argument('--dir', default='../../data/', type=str, help='the folder of datasets')
    parser.add_argument('--data_name', default='WinnipegDataset', type=str,
                        help='dataset name, one of [overlap1, overlap2, birch1, '
                             'birch2, iris, breast, iris, wine, htru, knowledge, worms, urbanland, WinnipegDataset]')
    parser.add_argument('--ratio', default=0.1, type=float, help='missing ratio')
    parser.add_argument('--NoEnhance', default=False, type=bool, help='Whether to use a Non-Enhancement version of SDC')

    args = parser.parse_args()

    ###################### # some preliminary works # ##################################
    choose_dims = None
    dir_path = args.save_dir + args.data_name
    if os.path.isdir(dir_path) is False:
        os.mkdir(dir_path)
    dims_path = args.dir + args.data_name + '/candidate_dim.txt'
    if os.path.isfile(dims_path) is True:
        choose_dims = np.loadtxt(dims_path)
        choose_dims = choose_dims.astype(int).tolist()
        print('dimensions to be analysed: ', choose_dims)
    total_time = 0
    file_path = args.dir + args.data_name + '/miss_' + str(args.ratio) + '.csv'
    complete_path = args.dir + args.data_name + '/original.csv'
    file_type = file_path.split(".")[-1]
    pca = PCA(n_components=2)
    if os.path.isdir(args.save_dir) is False:
        os.mkdir(args.save_dir)

    ########## # load dataset # ############
    if file_type == "csv":
        data = pd.read_csv(file_path)
        label = data['y'].values.astype(int)
        data = data.iloc[:, :-1].values
        complete_data = pd.read_csv(complete_path).values[:, :-1]
    print('data shape: ', data.shape)
    estimate_clusters = len(set(label))
    print("estimate_clusters: ", estimate_clusters)

    ###################### # normalization # ##################################
    for j in range(data.shape[1]):
        max_ = np.nanmax(data[:, j])
        min_ = np.nanmin(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            if np.isnan(data[i][j]) == False:
                data[i][j] = (data[i][j] - min_) / (max_ - min_)

    ####### # divide complete objects and objects with missing values into two sub-datasets # #########
    ####### # incomplete_data: objects with missing values, incomplete_label: labels of objects with missing values
    ####### # data: complete objects, label: labels of complete objects
    incomplete_idx = []
    for i in range(data.shape[0]):
        if np.isnan(data[i]).any() == True:
            incomplete_idx.append(i)
    incomplete_data = data[incomplete_idx]
    incomplete_label = label[incomplete_idx]
    data = np.delete(data, incomplete_idx, axis=0)
    complete_data_incomplete = complete_data[incomplete_idx]
    complete_data = np.delete(complete_data, incomplete_idx, axis=0)
    complete_data = np.concatenate([complete_data, complete_data_incomplete], axis=0)
    label = np.delete(label, incomplete_idx)
    np.savetxt(args.dir + args.data_name + '/miss_' + str(args.ratio) + '_complete_objects.txt', data)

    ########### # visuaualize original dataset # #############
    # dataPCA = data.copy()
    # if data.shape[1] > 2:
    #     dataPCA = pca.fit_transform(data)
    # plotCluster(dataPCA, label, "Original", args)

    ##### # generate virtual object # #######
    point = []
    for i in range(data.shape[1]):
        point.append(-9999)
    point = np.array(point).reshape((-1,))

    ######## # calculate r # #########
    start = time.time()
    tree = KDTree(data)
    density = np.zeros((data.shape[0],))
    dis, neighbor_idx = tree.query(data, k=6)  # round(data.shape[0] * 0.02)-1)
    r = np.mean(dis[:, 1]) * 5
    print("r: ", r)
    total_time += time.time() - start

    ########## # the enhancement progress # ###########
    if args.data_name in ['WinnipegDataset', 'urbanland']:
        start = time.time()
        for i in range(data.shape[0]):
            idx = tree.query_ball_point(data[i], r)
            density[i] = len(idx)
        total_time += time.time() - start
    else:
        N, D = data.shape
        command = "density_compute.exe " + args.dir + " " + args.data_name + " " + str(args.ratio) + " " + str(N) + " " + str(D) + " " + str(r)
        os.system(command)  # call the function to calculate density
        density = np.loadtxt(args.dir + args.data_name + "/dense_" + str(args.ratio) + ".txt", dtype=int).reshape(-1, )
        density_time = density[-1]
        total_time += density_time / 1000
        density = density[:-1]

    start = time.time()
    # ori_data = np.concatenate((data, incomplete_data), axis=0)
    if args.NoEnhance is False:
        data = shrink(data, density, neighbor_idx, 5, 1)  # infor
    total_time += time.time() - start

    ########### # visualize dataset after enhancement # ##########
    # if args.NoEnhance is False:
    #     dataPCA = data.copy()
    #     if data.shape[1] > 2:
    #         dataPCA = pca.fit_transform(data)
    #     plotCluster(dataPCA, label, "Enhancement", args)

    ########## # merge objects with missing values and complete objects # #############
    data = np.concatenate((data, incomplete_data), axis=0)
    label = np.concatenate((label, incomplete_label), axis=0).reshape((-1,))

    ########## # clustering # ##############
    cluster, cluster_time = project_clustering(data, args, r, choose_dims=choose_dims)
    total_time += cluster_time

    ########## # save clustering results # ##############
    np.savetxt(args.save_dir + args.data_name + "/miss_" + str(args.ratio) + "_clustering_result.txt", cluster)

    ########## # estimation of metrics # #############
    utils.estimation(label, cluster, args, total_time)

    ####### # visualize dataset with clustering label # ######
    # data = np.nan_to_num(data)
    # if data.shape[1] > 2:
    #     data = pca.fit_transform(data)
    # if args.NoEnhance is False:
    #     plotCluster(complete_data, cluster, "Clustering-SDC", args)
    # if args.NoEnhance is True:
    #     plotCluster(complete_data, cluster, "Clustering-SDC-NoEnhance", args)


