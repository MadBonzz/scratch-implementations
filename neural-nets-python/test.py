import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import Model
from optimizer import CustomAdam
from dataset import IrisDataset

SEED = 89
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('iris.csv')
le = LabelEncoder()
df['Species'] = le.fit_transform(df[['Species']])
X, y = df.drop(columns=['Species']), df['Species']

dataset = IrisDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

model = Model(X.shape[-1], y.nunique(), 512).to(device)
optim = CustomAdam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
train_loader = DataLoader(train_dataset, 64, shuffle=True)
test_loader  = DataLoader(test_dataset, 16, shuffle=True)
epochs = 1

for i in range(epochs):
    optim.zero_grad()
    data = next(iter(train_loader))
    x, y = data['x'].float().to(device), data['y'].to(device)
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optim.step()
    print(loss.item())
    
    
