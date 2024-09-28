import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=10):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        
        # Optional: additional layers can be added here
        self.fc = nn.Linear(latent_dim, 1, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        # print(user_indices.shape, item_indices.shape, user_vec.shape, item_vec.shape)
        # interaction = torch.mul(user_vec, item_vec)#.sum(dim=1, keepdim=True)
        interaction = user_vec*item_vec#.sum(dim=1, keepdim=True)
        # print(interaction.shape)
        output = self.fc(interaction)
        output = torch.sigmoid(output)
        return output.squeeze()

def train(model, optimizer, criterion, data_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for user_indices, item_indices, ratings in data_loader:
            optimizer.zero_grad()
            # print(user_indices.shape, item_indices.shape, ratings.shape) 64*1
            predictions = model(user_indices, item_indices)
            loss = criterion(predictions, ratings.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

def test(model, data_loader, p_test):
    model.eval()
    with torch.no_grad():
        # errors = []
        pred = []
        for user_indices, item_indices, ratings in data_loader:
            predictions = model(user_indices, item_indices)
            pred.append(predictions)
            # shape 66
            # max_values, max_indices = torch.max(predictions, dim=1)
            # print("Highest ratings per user:", max_values)
            # print("Indices of highest rated items:", max_indices)
    pred_torch = torch.cat(pred, dim=0)#.numpy()
    pred_torch = pred_torch.view(p_test.shape)
    max_values, max_indices = torch.max(pred_torch, dim=1)
    
    # pred_torch = pred_torch.view(p_test.shape[1], p_test.shape[0])
    # max_values, max_indices = torch.max(pred_torch, dim=0)
    # print(max_indices)

    print("Highest ratings per user:", max_values)
    print("Indices of highest rated items:", max_indices)
    max_values = [p_test[i][max_indices[i]] for i in range(p_test.shape[0])]
    return max_values
    # print(pred_torch.numpy())
    # return pred_torch.numpy()
    #         error = F.mse_loss(predictions, ratings.float(), reduction='none')
    #         errors.extend(error.tolist())
    #     mse = np.mean(errors)
    # print(f'Test MSE: {mse}')

def run(train_idx, test_idx):
    p_df = pd.read_csv('../pmatrix_full_copy.csv')
    p = p_df.iloc[1:, 1:]
    p = p.to_numpy().astype(float).transpose() # P
    P_train = p[train_idx]
    P_test = p[test_idx]

    # Convert numpy arrays to tensors
    P_train_tensor = torch.tensor(P_train, dtype=torch.float32)
    P_test_tensor = torch.tensor(P_test, dtype=torch.float32)

    # Create DataLoaders
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(torch.arange(0, len(train_idx)).repeat(11), torch.tile(torch.arange(0, 11), (len(train_idx),)), P_train_tensor.flatten())
    test_dataset = TensorDataset(torch.arange(0, len(test_idx)).repeat(11), torch.tile(torch.arange(0, 11), (len(test_idx),)), P_test_tensor.flatten())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = MatrixFactorization(num_users=len(train_idx), num_items=11)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Train the model
    train(model, optimizer, criterion, train_loader)

    # Test the model
    # user: torch.arange(0, len(test_idx)).repeat(11)
    # item: torch.tile(torch.arange(0, 11), (len(test_idx),))
    return test(model, test_loader, P_test)

    
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
data = []
for i in range(4):
    run(train_idxs[i], test_idxs[i])
    data.extend(run(train_idxs[i], test_idxs[i]))
data_arr = np.array(data)
print()
print(data_arr)
# with open('ncf.npy', 'wb') as f:
#     np.save(f, data_arr)