import argparse
import collections
import os.path
import random

import numpy as np
import pandas as pd


def generateMissing(data:np.ndarray, missing_ratio=0.1)->(np.ndarray, np.ndarray):
    random.seed(602)
    np.random.seed(602)
    n, d = data.shape  # number of observations, dimension
    x_miss = np.copy(data)
    x_miss = x_miss.reshape(-1, )
    miss_index = np.random.choice(n*d, np.floor(n*d * missing_ratio).astype(int), replace=False).reshape(-1, )
    # print(miss_index.shape)
    # x_miss[miss_index[:miss_index.shape[0]//2], 0] = np.nan
    # x_miss[miss_index[miss_index.shape[0]//2:], 1] = np.nan
    x_miss[miss_index] = np.nan
    x_miss = x_miss.reshape(-1, d)
    mask = np.isfinite(x_miss)  # binary mask that indicates which values are missing
    return x_miss, mask

def generateMissingData(data:np.ndarray, missing_ratio=0.1)->(np.ndarray, np.ndarray):
    random.seed(602)
    np.random.seed(602)
    n, d = data.shape  # number of observations, dimension
    x_miss = np.copy(data)
    miss_index = np.random.choice(n, np.floor(n * missing_ratio).astype(int), replace=False).reshape(-1, )
    # x_miss[miss_index[:miss_index.shape[0]//2], 0] = np.nan
    # x_miss[miss_index[miss_index.shape[0]//2:], 1] = np.nan
    for idx in miss_index:
        dim = np.random.choice(d, random.randint(1, d // 2))
        x_miss[idx, dim] = np.nan
    mask = np.isfinite(x_miss)  # binary mask that indicates which values are missing
    return x_miss, mask

def load_data(file_path:str)->(np.ndarray, np.ndarray):
    data_type = file_path.split(".")[-1]
    if data_type == 'txt':
        data = np.loadtxt(file_path)
        label = data[:, -1].reshape(-1, 1)
        data = data[:, :-1]
    elif data_type == 'csv':
        data = pd.read_csv(file_path)
        label = data.iloc[:, -1].values.reshape(-1, 1)
        data = data.iloc[:, :-1].values
    else:
        return "type dose not support"
    return data, label

def save_data(data:np.ndarray, label:np.ndarray, mask:np.ndarray, miss:np.ndarray, data_name:str, missing_ratio:float):
    columns = []
    for i in range(data.shape[1]):
        columns.append('dim' + str(i))
    df_mask = pd.DataFrame(mask, columns=columns)
    columns.append('y')
    tmp = np.concatenate((data, label), axis=1)
    df = pd.DataFrame(tmp, columns=columns)
    tmp = np.concatenate((miss, label), axis=1)
    df_miss = pd.DataFrame(tmp, columns=columns)
    df_miss.to_csv("../../data/" + data_name + "/miss_" + str(missing_ratio) + ".csv", index=False)
    df_mask.to_csv("../../data/" + data_name + "/mask_" + str(missing_ratio) + ".csv", index=False)
    df.to_csv("../../data/" + data_name + "/original.csv", index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result/', type=str, help='the director to save results')
    parser.add_argument('--dir', default='../../data/', type=str, help='the director of dataset')
    parser.add_argument('--data_name', default='breast', type=str,
                        help='dataset name, one of {overlap1, overlap2, birch1, '
                             'birch2, iris, breast, iris, wine, htru, knowledge}')
    parser.add_argument('--ratio', default=0.1, type=float, help='missing ratio')
    args = parser.parse_args()
    file_path = "../../data/" + args.data_name
    if os.path.isdir(file_path) == False:
        os.mkdir(file_path)
    file_path = file_path + '/original.csv'
    missing_rate_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for missing_rate in missing_rate_set:
        data, label = load_data(file_path)
        data = data.astype(float)
        # print(data)
        # label_count = collections.Counter(label.reshape(-1, ).tolist())
        # print(label_count)
        data_miss, mask = generateMissingData(data, missing_rate)
        # print(np.isnan(data_miss[:, 0]).sum())
        # print(np.isnan(data_miss[:, 1]).sum())
        # print(data_miss)
        save_data(data, label, mask, data_miss, args.data_name, missing_rate)
