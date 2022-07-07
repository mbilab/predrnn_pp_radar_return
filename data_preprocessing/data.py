from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class RadarDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.len = len(self.data)
    def __getitem__(self, idx):
        radar_input = (torch.from_numpy(self.data[idx])).float()
        return radar_input
    
    def __len__(self):
        return self.len
