# ARGOSMART
# finds the closest meta-train dataset (1NN) to a given test dataset,
# based on meta-feature similarity, and selects the model
# with the best performance on the 1NN dataset

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def argosmart(train_idx, test_idx):
    p_df = pd.read_csv('../pmatrix_full_copy.csv')
    p = p_df.iloc[1:, 1:]
    p = p.to_numpy().astype(float).transpose() # P
    p_test = p[test_idx]
    p_train = p[train_idx]
    max_indices = np.argmax(p_train, axis=1)
    # print(max_indices)

    # meta features
    # full_landmarker_msp = np.load('../full_landmarker_msp.npy')
    full_landmarker_msp = np.load('../full_mf.npy')
    meta_features_train = full_landmarker_msp[train_idx]
    meta_features_test = full_landmarker_msp[test_idx]


    methods = pd.read_csv('../pytorch_ood.csv')['Unnamed: 0'].to_list()[:-1]
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    meta_features_train = np.nan_to_num(meta_features_train, nan=0.0)
    knn.fit(meta_features_train)

    # Find the most similar row in meta_features_train to the meta_features_test
    pred_score = []
    i=0
    for x in meta_features_test:
        x = np.nan_to_num(x, nan=0.0).reshape((1,len(x)))
        distance, index = knn.kneighbors(x)
        # print(distance, index)
        print(f"The most similar row index: {index[0][0]}")
        print(f"Methods: {methods[max_indices[index[0][0]]]}")
        pred_score.append(p_test[i][max_indices[index[0][0]]])
        i=i+1
        print()
    return np.array(pred_score)

# train_idx = np.r_[17:22, 29:34, 41:46]
# test_idx = np.r_[0:6]
# print(argosmart(train_idx, test_idx))