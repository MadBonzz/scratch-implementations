from torch.utils.data import Dataset
import torch

class IrisDataset(Dataset):
    def __init__(self, x, y):
        super(IrisDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x.iloc[idx]
        y = self.y[idx]
        return {'x' : torch.tensor(x), 'y' : torch.tensor(y)}