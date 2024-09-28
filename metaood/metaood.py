import time
start_time = time.time()

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

num_methods = 11
train_idxs = [np.r_[17*num_methods:22*num_methods, 29*num_methods:34*num_methods, 41*num_methods:46*num_methods],
              np.r_[0:6*num_methods, 11*num_methods:17*num_methods],
              np.r_[6*num_methods:11*num_methods, 29*num_methods:34*num_methods, 41*num_methods:46*num_methods],
              np.r_[6*num_methods:11*num_methods, 17*num_methods:22*num_methods, 29*num_methods:34*num_methods]
              ]
test_idxs = [np.r_[0:6*num_methods], #test on cifar10
             np.r_[29*num_methods:34*num_methods], #test on imagenet
             np.r_[11*num_methods:17*num_methods], #test on cifar100
             np.r_[34*num_methods:41*num_methods] #test on fashionMNIST
             ]

dataset_emb = np.load('d_emb.npy')
method_emb = np.load('m_emb.npy')

combined_f = []
for i in range(dataset_emb.shape[0]):
  for j in range(num_methods):
    combined_f.append(np.concatenate((dataset_emb[i], method_emb[j]), axis=0))
combined_f = np.array(combined_f)#506 2662

p_df = pd.read_csv('data/pmatrix_full_copy.csv')
p = p_df.iloc[1:, 1:]
performances = p.to_numpy().astype(float).transpose() # P

performances = performances.flatten().reshape(combined_f.shape[0],1)

num_rounds = 100  # Number of boosting rounds
params = {"max_leaves": 120, "min_child_weight": 24, "learning_rate": 0.3}

xgboost_result = []
for i in range(len(train_idxs)):
    train_idx = train_idxs[i]
    test_idx = test_idxs[i]

    X_train = np.nan_to_num(combined_f[train_idx], nan=0)
    X_test = np.nan_to_num(combined_f[test_idx], nan=0)

    #preprocessing
    scaler_meta_features = MinMaxScaler(clip=True).fit(np.unique(X_train, axis=0))
    X_train = scaler_meta_features.transform(X_train)
    scaler_meta_features = MinMaxScaler(clip=True).fit(np.unique(X_test, axis=0))
    X_test = scaler_meta_features.transform(X_test)

    y_train = performances[train_idx]
    y_test = performances[test_idx]

    # Convert the data into DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train the model
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]

    model = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=10)

    # Make predictions
    y_pred = model.predict(dtest)
    y_pred = y_pred.reshape(int(y_test.shape[0]/num_methods), num_methods)
    max_idx = np.argmax(y_pred, axis=1)

    y_test = y_test.reshape(y_pred.shape)
    max_values = [y_test[i][max_idx[i]] for i in range(y_test.shape[0])]

    xgboost_result.extend(max_values)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse:.2f}")


xgboost_result = np.array(xgboost_result)

end_time = time.time()
execution_time = end_time - start_time
print(f"Runtime: {execution_time} seconds")

with open('metaood_result.npy', 'wb') as f:
    np.save(f, xgboost_result)