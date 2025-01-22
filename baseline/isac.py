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
    full_landmarker_msp = np.load('../full_mf.npy')

    meta_features_train = full_landmarker_msp[train_idx]
    meta_features_test = full_landmarker_msp[test_idx]

    # methods = pd.read_csv('../pytorch_ood.csv')['Unnamed: 0'].to_list()[:-1]

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
        cluster_indices = np.where(clusters == new_row_cluster)[0]
        print('cluster indices: ', cluster_indices)
        p_cluster = p[cluster_indices,:]
        avg_performance = np.mean(p_cluster, axis=0)
        closet_method = np.argmax(avg_performance)
        pred_score.append(p_test[i][closet_method])
        print()
        i = i+1
    return np.array(pred_score)

