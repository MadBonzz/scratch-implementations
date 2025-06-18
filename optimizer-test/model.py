import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_features, n_cls, hidden_size, bias=False):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(n_features, hidden_size, bias=bias)
        self.layer2 = nn.Linear(hidden_size, n_cls, bias=bias)
        self.act_fn = nn.ReLU()
        nn.init.uniform_(self.layer1.weight)
        nn.init.uniform_(self.layer2.weight)

    def forward(self, x):
        x = self.act_fn(self.layer1(x))
        return self.layer2(x)
        