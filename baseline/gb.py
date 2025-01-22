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
    pred_score = p_test[:, pred_idx]
    return pred_score
