import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def computeVar(data: np.ndarray) -> list:
    nn = data.shape[0]
    dd = data.shape[1]
    res = [0 for i in range(dd)]
    for i in range(dd):
        var = np.nanvar(data[:, i])
        res[i] = [var]
    return res

def denseGraph(project:np.ndarray, r:float):
    # project_s = np.sort(project)
    min_ = np.nanmin(project)
    max_ = np.nanmax(project)
    dc = (max_ - min_) / 100
    dense = np.zeros((100, ))
    for i in range(len(project)):
        if np.isnan(project[i]) == True:
            continue
        inter = int(np.floor((project[i] - min_) / dc))
        if inter >= 100:
            inter = 99
        dense[inter] += 1
    print(dense)
    return dense, min_, dc

def plotDense(dense_graph: np.ndarray, dc:float, min_:float) -> list:
    dense_graph = dense_graph.reshape(-1, )
    x = [i for i in range(len(dense_graph))]
    fig, ax = plt.subplots()
    ax.plot(x, dense_graph)
    ax.set_title("density decision")
    print("请点击坐标点，按任意键退出...")
    points = plt.ginput(n=-1, timeout=-1)
    plt.show()
    border = []
    for point in points:
        pos = point[0] * dc + min_
        border.append(pos)
    return border

def getDenseGraph(project: np.ndarray, r: float) -> np.ndarray:
    nn = project.shape[0]
    dense_graph = np.zeros((nn, ))
    project_sorted = np.sort(project)
    # print(project_sorted)
    head = 0
    tail = 1
    tail_old = 0
    count = 0
    flag = False
    while tail < nn:
        if head == tail:
            tail_old = tail
            count = 0
            tail += 1
        elif project_sorted[tail] - project_sorted[head] <= r:
            count += 1
            tail += 1
            flag = True
        else:
            if flag == True:
                dense_graph[head:tail] += count
                if tail_old > head:
                    dense_graph[tail_old:tail] += tail_old - head - 1
                tail_old = tail
                flag = False
            head += 1
            count = 0
            if np.isnan(project_sorted[tail]) == True:
                break
    if flag == True:
        dense_graph[head:tail] += count
        dense_graph[tail_old:tail] += tail_old - head - 1

    # dense_2 = np.zeros((nn, ))
    # for i in range(nn):
    #     for j in range(nn):
    #         if i != j and abs(project_sorted[i] - project_sorted[j]) <= r:
    #             dense_2[i] += 1
    # print(np.sum(dense_graph == dense_2))
    # print(dense_graph)
    # print(dense_2)
    return dense_graph, project_sorted

def plotDenseGraph(dense_graph: np.ndarray, sorted_project: np.ndarray, dim: int, args) -> list:
    x = sorted_project.reshape((-1, ))
    fig, ax = plt.subplots()
    ax.plot(x, dense_graph, linewidth=1.5)
    ax.set_title("SDC (overlap1)" if args.NoEnhance is False else "SDC-NoEnhan (overlap1)", size=30)
    # ax.set_xlabel(size=30)
    # ax.set_ylabel(size=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.savefig(args.save_dir + args.data_name + "/" + ("SDC-" if args.NoEnhance is False else "SDC-NH-") + str(dim) + ".png", dpi=300, bbox_inches='tight')
    border = []
    # if args.plot_fig_only is False:
    print("请点击坐标点，按任意键退出...")
    points = plt.ginput(n=-1, timeout=-1)
    plt.show()
    for point in points:
        border.append(point[0])
    return border

def getProjectCluster(project: np.ndarray, border: list,  project_cluster: np.ndarray):
    nn = project.shape[0]
    border = sorted(border)
    for i in range(nn):
        if np.isnan(project[i]) == True:
            continue
        for j in range(len(border)):
            if project[i] <= border[j]:
                project_cluster[i] = j
                break
            else:
                project_cluster[i] = j+1

def project_clustering(data: np.ndarray, args, rr:float, choose_dims=None, estimate_clusters=None):
    '''

    :param data:
    :param args:
    :param rr: radius
    :param choose_dims: A list contains dimensions indexes. Choose dims to split single dimensional dataset to rough
            cluster-partition. The default value is 'None', which means all the dimensions need to be analysed. By determining
            'choose_dims', you can stop the executing process of SDC in advance.
    :param estimate_clusters: the estimated number of clusters in dataset, if you want to
            stop running the SDC in advance, you can set the estimate_clusters closer to the real cluster number.
            The default value is None, which means the SDC can end normally without setting 'estimate_clusters'.
    :return: Clustering results and clustering time.
    '''
    dd = data.shape[1]
    nn = data.shape[0]
    vars = computeVar(data)
    vars = np.array(vars).reshape((-1, ))
    dims = np.argsort(-vars)
    print(dims)
    if choose_dims is not None:
        dims = choose_dims
    print(dims)
    project_cluster_pre = -np.ones((nn, ), dtype=int)
    cluster_time = 0
    r = rr / dd#np.sqrt(dd)
    tmp = False
    for i, dim in enumerate(dims):
        start = time.time()
        project = data[:, dim].reshape(-1,)
        # t1 = time.time()
        dense_graph, sorted_project = getDenseGraph(project, r)
        # dense_graph, min_, dc = denseGraph(project, r)
        # print("dense graph: ", time.time() - t1)
        cluster_time += time.time() - start
        border = plotDenseGraph(dense_graph, sorted_project, dim, args)
        # border = plotDense(dense_graph, min_, dc)

        # if args.plot_fig_only:
        #     continue

        if len(border) == 0:
            continue
        print(border)
        start = time.time()
        if i == 0 or tmp == False:
            if len(border) > 0:
                tmp = True
            # t1 = time.time()
            getProjectCluster(project, border, project_cluster_pre)
            # print("project: ", time.time() - t1)
            continue
        else:
            project_cluster_lat = -np.ones((nn,), dtype=int)
            # t1 = time.time()
            getProjectCluster(project, border, project_cluster_lat)
            # print("project: ", time.time() - t1)
            cluster_map = {}# 簇的映射关系
            cluster_pre = {}# 第i次的投影簇，其中只存在在两维度上都不缺失的点
            cluster_lat = {}# 第i+1次的投影簇，其中只存在在两维度上都不缺失的点
            incomplete_pre = {}# 第i次的投影簇，只有第i维不缺失，第i+1维缺失的点
            incomplete_lat = {}# 第i次的投影簇，只有第i+1维不缺失，第i维缺失的点
            count = 0# 融合之后的簇的编号
            # print("project cluster pre: ", project_cluster_pre)
            # print("project cluster lat: ", project_cluster_lat)
            t1 = time.time()
            ###### # 在两个维度上不缺失点的聚类 # ########
            for j, item in enumerate(zip(project_cluster_pre, project_cluster_lat)):
                if item[0] != -1 and item[1] != -1:
                    key = str(item[0]) + str(item[1])
                    if cluster_map.get(key) is None:
                        cluster_map[key] = count
                        count += 1
                    project_cluster_pre[j] = cluster_map[key]
                    if cluster_pre.get(item[0]) is None:
                        cluster_pre[item[0]] = []
                    if cluster_lat.get(item[1]) is None:
                        cluster_lat[item[1]] = []
                    cluster_pre[item[0]].append(j)
                    cluster_lat[item[1]].append(j)
                elif item[0] == -1 and item[1] != -1:
                    if incomplete_lat.get(item[1]) is None:
                        incomplete_lat[item[1]] = []
                    incomplete_lat[item[1]].append(j)
                elif item[0] != -1 and item[1] == -1:
                    if incomplete_pre.get(item[0]) is None:
                        incomplete_pre[item[0]] = []
                    incomplete_pre[item[0]].append(j)

            ###### # 在两个维度上其中一个维度存在缺失的点的聚类 # ########

            ######### # 概率最大的情况 # ##########
            for key, val in incomplete_pre.items():
                if cluster_pre.get(key) == None:
                    continue
                num_dict = {}
                cluster = cluster_pre[key]
                for item in cluster:
                    cluster_number = project_cluster_pre[item]
                    num_dict[cluster_number] = num_dict.get(cluster_number, 0) + 1
                max_num = 0
                for kkey, vval in num_dict.items():
                    if vval > max_num:
                        max_num = vval
                        cls = kkey
                project_cluster_pre[val] = cls

            for key, val in incomplete_lat.items():
                if cluster_pre.get(key) == None:
                    continue
                num_dict = {}
                cluster = cluster_lat[key]
                for item in cluster:
                    cluster_number = project_cluster_pre[item]
                    num_dict[cluster_number] = num_dict.get(cluster_number, 0) + 1
                max_num = 0
                for kkey, vval in num_dict.items():
                    if vval > max_num:
                        max_num = vval
                        cls = kkey
                project_cluster_pre[val] = cls
            print("clustering time: ", time.time() - t1)
        # print("after cluster each project: ", project_cluster_pre)
        cluster_time += time.time() - start
            # print(cluster_map)
        if estimate_clusters is not None and len(cluster_map) >= estimate_clusters:
            break
    project_cluster_pre[np.where(project_cluster_pre == -1)] = 0
    return project_cluster_pre, cluster_time

def plotCluster3d(data: np.ndarray, labels: np.ndarray, title: str, args):
    fig = plt.figure(figsize=(7, 5))
    ax = Axes3D(fig)
    sortNumMax = np.max(labels)
    sortNumMin = np.min(labels)
    color = [
        '#E55604',
        '#B0578D',
        "#8B7355",  # 赭色
        '#66CDAA',
        '#4E4FEB',
        '#e84118',

        '#e77f67',
        "#900C3F",  # 紫红色
        "#006400",  # 深绿色

        "#4B0082",  # 靛青色
        "#FF4500",  # 橙红色
        "#FF1493",  # 深粉色

        "#008B8B",  # 深青色
        "#4682B4",  # 钢蓝色
        '#C71585',
        "#A9A9A9",  # 暗灰色
        "#556B2F",  # 暗绿色

        "#9370DB",  # 中紫色

        "#FFD700",  # 库金色
        "#2E8B57",  # 海洋绿色
        "#BDB76B",  # 黄褐色

        "#654321",  # 深棕色
        "#9400D3",  # 暗紫色
        "#CD5C5C",  # 褐红色
        '#B22222',

        "#FF7F50",  # 珊瑚色
        '#006400',
        '#556B2F',
        '#808000',
        '#BDB76B',
        '#98FB98',
        '#00FA9A',
        '#7FFFD4',
        '#FF8C00',
        '#2F4F4F',
        '#00CED1',
        '#FAFAD2',
        '#FFE4E1',
        '#DC143C',
        '#E0FFFF',
        '#7B68EE',
        '#000080',
        '#FF00FF',
        '#FFC0CB',
        '#FFF5EE',
        '#DB7093',

    ]
    for i in set(labels):
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        fontSize = 30
        colorNum = (i - sortNumMin) % len(color)
        formNum = 0
        ax.scatter(Together[:, 0], Together[:, 1], Together[:, 2], s=fontSize, c=color[colorNum])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
    # plt.title(title, fontsize=20)
    plt.savefig(
        args.save_dir + args.data_name + "/" + ("SDC-" if args.NoEnhance is False else "SDC-NH-") + title + ".png",
        dpi=300, bbox_inches='tight')
    plt.show()

def plotCluster(data: np.ndarray, labels: np.ndarray, title: str, args):
    fig, ax = plt.subplots(figsize=(6, 5))
    sortNumMax = np.max(labels)
    sortNumMin = np.min(labels)
    color = [
        '#E55604',
        '#4E4FEB',
        '#B0578D',
        "#008B8B",  # 深青色
        "#8B7355",  # 赭色
        "#900C3F",  # 紫红色
        '#e84118',
        '#e77f67',
        "#006400",  # 深绿色

        "#4B0082",  # 靛青色
        "#FF4500",  # 橙红色
        "#FF1493",  # 深粉色

        "#FF7F50",  # 珊瑚色
        "#4682B4",  # 钢蓝色

        "#A9A9A9",  # 暗灰色
        "#556B2F",  # 暗绿色
        '#FF8C00',
        "#9370DB",  # 中紫色

        "#FFD700",  # 库金色
        "#2E8B57",  # 海洋绿色
        "#BDB76B",  # 黄褐色

        "#654321",  # 深棕色
        "#9400D3",  # 暗紫色
        '#66CDAA',
        "#CD5C5C",  # 褐红色
        '#B22222',
        '#006400',
        '#556B2F',
        '#C71585',
        '#808000',
        '#BDB76B',
        '#98FB98',
        '#00FA9A',
        '#7FFFD4',
        '#66CDAA',
        '#2F4F4F',
        '#00CED1',
        '#FAFAD2',
        '#FFE4E1',
        '#DC143C',
        '#E0FFFF',
        '#7B68EE',
        '#000080',
        '#FF00FF',
        '#FFC0CB',
        '#FFF5EE',
        '#DB7093',

    ]
    # color = ['#125B50', '#4D96FF', '#FFD93D', '#FF6363', '#CE49BF', '#22577E', '#4700D8', '#F900DF', '#95CD41',
    #          '#FF5F00', '#40DFEF', '#8E3200', '#001E6C', '#C36A2D', '#B91646']
    lineform = ['o']
    for i in range(sortNumMin, sortNumMax + 1):
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        fontSize = 10
        colorNum = (i - sortNumMin) % len(color)
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.title(title, fontsize=20)
    plt.savefig(args.save_dir + args.data_name + "/" + ("SDC-" if args.NoEnhance is False else "SDC-NH-") + title + ".png", dpi=300, bbox_inches='tight')
    plt.show()
