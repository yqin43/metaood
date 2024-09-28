import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor

# mf_rank = 10 # todo

def ksvd(ia_matrix, k):
    '''
        Apply k-singular value decomposition (SVD)
        - Return matrices with k dimensions

        :param ia_matrix: numpy 2D array - (instance, algorithm) matrix
        :param k: int - SVD dimension
        :return: numpy 2D array, numpy array, numpy 2D array - Uk matrix representing rows,
                                                                 sk matrix (array) for singular values,
                                                                 Vk matrix representing columns
    '''
    max_k = min(len(ia_matrix), len(ia_matrix[0]))
    if k > max_k:
        k = max_k

    U, s, V = np.linalg.svd(ia_matrix, full_matrices=False)
    Uk = U[:,0:k]
    sk = s[0:k] #only diagonal values
    Vk = V[0:k,:]

    return Uk, sk, Vk, s

def train(ia_matrix, i_ft_matrix, mf_rank):
        
        max_rank = np.min(ia_matrix.shape)
        if mf_rank > max_rank:  
            mf_rank = max_rank
        
        Uk, sk, Vk, s = ksvd(ia_matrix, mf_rank)
        # Uk, sk, Vk, s = apply_mf(ia_matrix, self.mf_type, mf_rank)  
    
        map_method = RandomForestRegressor(n_estimators=10, random_state=42)
        map_method.fit(i_ft_matrix, Uk)

        return map_method, sk, Vk

def test(regr_model, i_ft_matrix):
    return regr_model.predict(i_ft_matrix)

def predict(train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix, mf_rank, test_idx):
    p_df = pd.read_csv('../pmatrix_full_copy.csv')
    p = p_df.iloc[1:, 1:]
    p = p.to_numpy().astype(float).transpose() # P
    # p_train = p[train_idx]
    p_test = p[test_idx]

    regr_model, sk, Vk = train(train_ia_rank_matrix, train_i_ft_matrix, mf_rank)
    Uk = test(regr_model, test_i_ft_matrix)
        
    if mf_rank == 1: ## just convert array (x,) to (x,1) not to have any error while multiplication
        Uk = np.reshape(Uk, (-1,1))    
    
    pred = np.dot(Uk, np.dot(np.diag(sk), Vk))
    #np.max(pred, axis=1)
    max_idx = np.argmax(pred, axis=1)
    print(pred.shape, max_idx.shape, p_test.shape)
    print(max_idx)
    max_values = [p_test[i][max_idx[i]] for i in range(p_test.shape[0])]
    return max_values#





# train_idx = np.r_[17:22, 29:34, 41:46]
# test_idx = np.r_[0:6]

# p_df = pd.read_csv('../pmatrix_full_copy.csv')
# p = p_df.iloc[1:, 1:]
# p = p.to_numpy().astype(float).transpose() # P
# p_train = p[train_idx]
# p_test = p[test_idx]

# # meta features
# # full_landmarker_msp = np.load('../full_landmarker_msp.npy')
# full_landmarker_msp = np.load('../full_mf.npy')
# # full_landmarker_msp = np.load('../full_stats.npy')

# meta_features_train = full_landmarker_msp[train_idx]
# meta_features_test = full_landmarker_msp[test_idx]

# result = predict(p_train, meta_features_train, meta_features_test, 10)

# print(result.shape)
# print(result)