import os

import numpy as np
import pandas as pd
import time
def computePurity(labels_true, labels_pred):
  clusters = np.unique(labels_pred)
  labels_true = np.reshape(labels_true, (-1, ))
  labels_pred = np.reshape(labels_pred, (-1, ))
  count = []
  for c in clusters:
    idx = np.where(labels_pred == c)
    labels_tmp = labels_true[idx].reshape((-1, )).astype(int)
    count.append(np.bincount(labels_tmp).max())
  return np.sum(count) / labels_true.shape[0]

def estimation(labels, pred, args, clustering_time):
    # clustering by kmeans
    from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, \
    fowlkes_mallows_score as FMI, homogeneity_completeness_v_measure as HCV
    ari = ARI(labels, pred)
    nmi = (AMI(labels, pred, average_method="arithmetic"))
    fmi = (FMI(labels, pred))
    purity = (computePurity(labels, pred))
    hcv = HCV(labels, pred)
    homogeneity = (hcv[0])
    completeness = (hcv[1])
    vm = (hcv[2])
    my_metrics = [[ari, nmi, purity, vm, homogeneity, completeness, clustering_time]]
    my_metrics = pd.DataFrame(my_metrics,
                            columns=['ari', 'nmi','purity', 'vm', 'homogeneity', 'completeness', 'time'])
    print(my_metrics)
    save_path = args.save_dir + args.data_name
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    if args.NoEnhance == False:
        save_path = save_path + "/result_SDC_" + str(args.ratio) + ".csv"
    else:
        save_path = save_path + "/result_SDC-NoEnhan_" + str(args.ratio) + ".csv"
    my_metrics.to_csv(save_path, index=False)