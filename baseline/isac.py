# ISAC
# clusters the meta-train datasets based on meta-features.
# Given a new dataset,
# it identifies its closest cluster and selects the best model
# with largest avg. performance on the cluster’s datasets.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def isac(train_idx, test_idx, num_clusters=5):
    p_df = pd.read_csv('../pmatrix_full_copy.csv')
    p = p_df.iloc[1:, 1:]
    p = p.to_numpy().astype(float).transpose() # P
    p_test = p[test_idx]
    
    # meta features
    # full_landmarker_msp = np.load('../full_landmarker_msp.npy')
    full_landmarker_msp = np.load('../full_mf.npy')
    # full_landmarker_msp = np.load('../full_stats.npy')

    meta_features_train = full_landmarker_msp[train_idx]
    meta_features_test = full_landmarker_msp[test_idx]

    methods = pd.read_csv('../pytorch_ood.csv')['Unnamed: 0'].to_list()[:-1]

    # Step 1: Clustering
    kmeans = KMeans(n_clusters=num_clusters)
    A = np.nan_to_num(meta_features_train, nan=0.0)
    clusters = kmeans.fit_predict(A)

    pred_score = []
    for x in meta_features_test:
        i = 0
    # Step 2: Assign new row to the nearest cluster
        x = np.nan_to_num(x, nan=0.0).reshape((1,len(x)))
        new_row_cluster = kmeans.predict(x)
        print('test row cluster: ', new_row_cluster)
        # Step 3: Calculate distances within the assigned cluster
        # for i in range(num_clusters):
        #     print(np.where(clusters == i)[0])
        # print()
        cluster_indices = np.where(clusters == new_row_cluster)[0]
        print('cluster indices: ', cluster_indices)
        p_cluster = p[cluster_indices,:]
        # print(p_cluster.shape)
        # print(p_cluster)
        # print('test cluster shape', p_cluster.shape)
        avg_performance = np.mean(p_cluster, axis=0)
        # print('test avg perf shape', avg_performance.shape)

        # print(avg_performance.shape,'avg_performance', avg_performance)
        closet_method = np.argmax(avg_performance)
        # print(cluster_indices)
        print('test',methods[closet_method])
        # print(methods[cluster_indices[closet_method]])
        pred_score.append(p_test[i][closet_method])
        print()
        i = i+1

    return np.array(pred_score)

train_idx = np.r_[17:22, 29:34, 41:46]
test_idx = np.r_[0:6]
# train_idx = np.r_[0:6, 11:17]
# test_idx = np.r_[29:34]
print(isac(train_idx, test_idx, num_clusters=3))