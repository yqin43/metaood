# mega ensemble
# averages outlier scores from all the models for a given dataset.
# ME does not perform model selection but rather uses all the models.

import pandas as pd
import numpy as np

def me(train_idx, test_idx):
    p_df = pd.read_csv('../pmatrix_full_copy.csv')
    p = p_df.iloc[1:, 1:]
    p = p.to_numpy().astype(float).transpose() # P
    
    test_p = p[test_idx]
    # print(test_p.shape)
    predict_score = np.mean(test_p, axis=1)
    return predict_score
    # print(predict_score.shape)
    
    # max_idx = np.argmax(predict_score)
    # print(max_idx)
    # max_values = [test_p[i][max_idx] for i in range(test_p.shape[0])]
    # return max_values


# train_idx = np.r_[17:22, 29:34, 41:46]
# test_idx = np.r_[0:6]
# print(me(train_idx, test_idx))