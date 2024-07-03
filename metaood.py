import torch
from torch import nn
import random
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

num_methods = 11
mf_dim = 358

class meta_predictor(nn.Module):
    def __init__(self, n_col, n_per_col, embedding_dim=3):
        super(meta_predictor, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(int(n_per_col[i]), embedding_dim) for i in range(n_col)])
        self.classifier = nn.Sequential(
            nn.Linear(mf_dim + 1 + n_col * embedding_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid())

    def forward(self, meta_features=None, nla=None, components=None):
        assert components.shape[1] == len(self.embeddings)

        embedding_list = []
        for i, e in enumerate(self.embeddings):
            embedding_list.append(e(components[:, i].long()))

        embedding = torch.cat(embedding_list, dim=1)
        embedding = torch.cat((meta_features, nla, embedding), dim=1)
        pred = self.classifier(embedding)

        return embedding, pred

class Utils():
    def __init__(self):
        pass

    # remove randomness
    def set_seed(self, seed):
        # basic seed
        np.random.seed(seed)
        random.seed(seed)

        # pytorch seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # generate unique value
    def unique(self, a, b):
        u = 0.5 * (a + b) * (a + b + 1) + b
        return int(u)

    def get_device(self, gpu_specific=True):
        if gpu_specific:
            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                print(f'number of gpu: {n_gpu}')
                print(f'cuda name: {torch.cuda.get_device_name(0)}')
                print('GPU is on')
            else:
                print('GPU is off')

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        return device

    def criterion(self, y_true, y_pred, mode=None):
        assert torch.is_tensor(y_true) and torch.is_tensor(y_pred)
        if mode == 'pearson':
            x = y_pred
            y = y_true
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            metric = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        elif mode == 'ranknet':
            n = y_pred.size(0)

            assert y_true.ndim == 1 and y_pred.ndim == 1
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

            mask = ~torch.eye(n, dtype=torch.bool)
            p_ij = torch.sign(y_true - y_true.T)
            p_ij[p_ij == -1] = 0
            s_ij = torch.sigmoid((y_pred - y_pred.T) * 100)

            p_ij = p_ij[mask].view(n, n - 1)
            s_ij = s_ij[mask].view(n, n - 1)

            metric = -F.binary_cross_entropy(s_ij, p_ij)

        elif mode == 'mse':
            criterion = nn.MSELoss()
            metric = -criterion(y_pred, y_true)

        elif mode == 'weighted_mse':
            start = 1.00
            end = 0.01
            decay_factor = 0.5

            t = torch.linspace(0, 1, y_pred.shape[0])

            exponential_decay = torch.exp(torch.log(torch.tensor(end / start)) * decay_factor * t) * start
            exponential_decay = exponential_decay.to(y_pred.device)

            idx_sort = torch.argsort(0.8 * y_true + 0.2 * y_pred)
            y_pred = y_pred[idx_sort]
            y_true = y_true[idx_sort]

            metric = torch.sum((torch.pow((y_pred - y_true), 2) * exponential_decay))
            metric = -metric

        else:
            raise NotImplementedError

        return metric


def fit(train_loader, model, optimizer, epochs, loss_name=None, val_loader=None, es=False, tol: int = 5):
    utils = Utils()
    best_metric = -9999; t = 0
    loss_epoch = []
    for i in range(epochs):
        loss_batch = []
        for batch in train_loader:
            batch_meta_features, batch_la, batch_components, batch_y = batch[:,:mf_dim], batch[:,mf_dim:mf_dim+1], batch[:,mf_dim+1:mf_dim+1+num_methods], batch[:,mf_dim+1+num_methods:]
            batch_la, batch_y = batch_la.squeeze(1), batch_y.squeeze(1)

            # clear grad
            model.zero_grad()

            # loss forward
            _,pred = model(batch_meta_features, batch_la.unsqueeze(1), batch_components)
            loss = -utils.criterion(y_pred=pred.squeeze(), y_true=batch_y, mode=loss_name)

            # loss backward
            loss.backward()

            # update
            optimizer.step()
            loss_batch.append(loss.item())

        loss_epoch.append(np.mean(loss_batch))

        if val_loader is not None and es:
            val_metric = utils.evaluate(model, val_loader=val_loader, device=batch_y.device, mode=loss_name)
            print(f'Epoch: {i}--Training Loss: {round(np.mean(loss_batch), 4)}---Validation Metric: {round(val_metric.item(), 4)}')
            if val_metric > best_metric:
                best_metric = val_metric
                t = 0
            else:
                t += 1

            if t > tol:
                print(f'Early stopping at epoch: {i}!')
                break
        else:
            print(f'Epoch: {i}--Loss: {round(np.mean(loss_batch), 4)}')

    return i

def run(train_idx, test_idx):
    full_landmarker_msp = np.load('/home/yuehanqi/ood/full_mf.npy')
    meta_features= full_landmarker_msp[train_idx]
    # components.shape # 11 methods, 358 dim meta_feature
    meta_features= np.nan_to_num(meta_features, nan=0)

    
    num_train = len(train_idx)
    components = np.eye(num_methods)
    p_df = pd.read_csv('/home/yuehanqi/ood/pmatrix_full_copy.csv')
    p = p_df.iloc[1:, 1:]
    performances = p.to_numpy().astype(float).transpose()[train_idx] # P

    meta_features1 = np.repeat(meta_features, repeats=num_methods, axis=0)

    components1 = np.vstack([components]*num_train)
    performances1 = []
    for i in range(num_train):
        for j in range(num_methods):
            performances1.append(performances[i][j])
    performances1 = torch.tensor(performances1).unsqueeze(1).float()

    # fitting meta predictor
    print('fitting meta predictor...')
    epochs = 20
    las = torch.full((meta_features1.shape[0], 1), 5).int()

    # min-max scaling for meta-features
    scaler_meta_features = MinMaxScaler(clip=True).fit(np.unique(meta_features1, axis=0))
    meta_features1 = scaler_meta_features.transform(meta_features1)
    # min-max scaling for la
    las = np.array(las).reshape(-1, 1)
    scaler_las = MinMaxScaler(clip=True).fit(np.unique(las, axis=0))
    las = scaler_las.transform(las)

    meta_features1, components1, las = torch.tensor(meta_features1).float(), torch.tensor(components1).float(), torch.tensor(las).float()

    train_loader = DataLoader(torch.concatenate((meta_features1, las, components1, performances1), axis=1),
                                        batch_size=1, shuffle=True, drop_last=True)

    # initialize meta predictor
    model = meta_predictor(n_col=components1.shape[1],
                            n_per_col=[max(components1[:, i]).item() + 1 for i in range(components1.shape[1])])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fit(train_loader, model, optimizer, epochs=epochs, loss_name='mse') 

    meta_features_test= full_landmarker_msp[test_idx]
    meta_features_test = np.nan_to_num(meta_features_test, nan=0)
    meta_features_t1 = np.repeat(meta_features_test, num_methods, axis=0)

    performances_test = p.to_numpy().astype(float).transpose()[test_idx] # P
    components_t1 = np.vstack([components]*meta_features_test.shape[0])
    performances_t1 = []
    
    for i in range(meta_features_test.shape[0]):
        for j in range(num_methods):
            performances_t1.append(performances_test[i][j])
    performances_t1 = torch.tensor(performances_t1).unsqueeze(1).float()

    scaler_meta_features = MinMaxScaler(clip=True).fit(np.unique(meta_features_t1, axis=0))
    meta_features_t1 = scaler_meta_features.transform(meta_features_t1)
    # min-max scaling for la
    las = torch.full((meta_features_t1.shape[0], 1), 5).float()

    las = np.array(las).reshape(-1, 1)
    scaler_las = MinMaxScaler(clip=True).fit(np.unique(las, axis=0))
    las = scaler_las.transform(las)

    meta_features_t1, components_t1,las = torch.tensor(meta_features_t1).float(), torch.tensor(components_t1).float(), torch.tensor(las).float()

    # eval
    model.eval()
    with torch.no_grad():
        print(meta_features_t1.shape, components_t1.shape, las.shape)
        _, pred = model(meta_features_t1, las, components_t1)
    print(pred.shape)
    pred_matrix = pred.reshape((test_idx.shape[0], num_methods))
    print(type(pred_matrix))

    p = p.to_numpy().astype(float).transpose() # P
    # get actual corresponding performance of the test samples
    p_test = p[test_idx]

    max_idx = np.argmax(pred_matrix.numpy(), axis=1)
    max_values = [p_test[i][max_idx[i]] for i in range(p_test.shape[0])]
    return max_values