import pandas as pd
import numpy as np

num_methods = 11

def random_s(train_idx, test_idx):
    np.random.seed(42)
    
    pred_idx = np.random.randint(0, num_methods, size=test_idx.shape[0])

    p_df = pd.read_csv('../pmatrix_full_copy.csv')
    p = p_df.iloc[1:, 1:]
    p = p.to_numpy().astype(float).transpose() # P
    p_test = p[test_idx]

    pred_score = []
    for i in range(test_idx.shape[0]):
        pred_score.append(p_test[i][pred_idx[i]])
    return pred_score