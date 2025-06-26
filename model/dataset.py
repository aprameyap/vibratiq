import torch
from torch.utils.data import Dataset

class BearingDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=False)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x, x  # input == target for autoencoder