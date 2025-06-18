import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, Adagrad, RMSprop, SGD
from model import Model
from optimizer import CustomAdam
from dataset import IrisDataset

SEED = 89    

def seed_env(SEED):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_tests(optimizer, n_fts, n_cls, train_data, test_data, seed, lr):
    seed_env(seed)
    epochs = 100
    model = Model(n_fts, n_cls, 512).to(device)
    train_loader = DataLoader(train_data, 64, shuffle=True)
    test_loader  = DataLoader(test_data, 16, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    if optimizer == 'custom':
        optim = CustomAdam(model.parameters(), lr)
    elif optimizer == 'adam':
        optim = Adam(model.parameters(), lr)
    elif optimizer == 'rms':
        optim = RMSprop(model.parameters(), lr)
    elif optimizer == 'ada':
        optim = Adagrad(model.parameters(), lr)
    elif optimizer == 'sgd':
        optim = SGD(model.parameters(), lr)

    train_losses = []
    test_losses = []
    for i in range(epochs):
        train_loss = 0
        test_loss = 0
        model.train()
        for _, batch in enumerate(train_loader):
            optim.zero_grad()
            x, y = batch['x'].float().to(device), batch['y'].to(device)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        model.eval()
        for _, batch in enumerate(test_loader):
            optim.zero_grad()
            x, y = batch['x'].float().to(device), batch['y'].to(device)
            out = model(x)
            loss = loss_fn(out, y)
            test_loss += loss.item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    return train_losses, test_losses
        
        
seed_env(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('iris.csv')
le = LabelEncoder()
df['Species'] = le.fit_transform(df[['Species']])
X, y = df.drop(columns=['Species']), df['Species']

dataset = IrisDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train, test = run_tests('custom', X.shape[-1], y.nunique(), train_dataset, test_dataset, 50, 1e-4)
print(train)
print(test)
    
    
