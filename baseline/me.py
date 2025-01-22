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
    predict_score = np.mean(test_p, axis=1)
    return predict_score