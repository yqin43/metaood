# global best
# the simplest meta-learner that
# selects the model with the largest avg. performance
# across all train datasets, without using meta-features.

import pandas as pd
import numpy as np

def gb(train_idx, test_idx):
    p_df = pd.read_csv('../pmatrix_full_copy.csv')
    p = p_df.iloc[1:, 1:]
    p = p.to_numpy().astype(float).transpose() # P
    p_test = p[test_idx]

    p_train = p[train_idx]
    row_averages = np.mean(p_train, axis=0)
    pred_idx = np.argmax(row_averages)

    #### test ####
    methods = ["Openmax", "MCD", "ODIN", "Mahalanobis", "EnergyBased", "Entropy", "MaxLogit", "KML", "ViM", "MSP", "KNN"]
    print(methods[pred_idx])
    ##############
    pred_score = p_test[:, pred_idx]
    # true_score = np.argsort(p_test)[:,::-1]
    # pred_score = np.argsort(row_averages)[::-1]

    # pred_scores = np.tile(pred_score, true_score.shape[0]).reshape(np.shape(true_score))
    # from sklearn.metrics import dcg_score, ndcg_score
    # print(ndcg_score(true_score, pred_scores))
    return pred_score


train_idxs = [np.r_[17:22, 29:34, 41:46],
              np.r_[0:6, 11:17],
              np.r_[6:11, 29:34, 41:46],
              np.r_[6:11, 17:22, 29:34]
              ]
test_idxs = [np.r_[0:6], #test on cifar10
             np.r_[29:34], #test on imagenet
             np.r_[11:17], #test on cifar100
             np.r_[34:41] #test on fashionMNIST
             ]

gb_lst = []
for i in range(4):
    train_idx = train_idxs[i]
    test_idx = test_idxs[i]
    gb_lst.extend(gb(train_idx, test_idx))

gb_arr = np.array(gb_lst)
with open('gb.npy', 'wb') as f:
    np.save(f, gb_arr)
# train_idx = np.r_[17:22, 29:34, 41:46]
# test_idx = np.r_[0:6]
# print(gb(train_idx, test_idx))
