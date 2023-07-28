import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob

class DatasetFromLatent(Dataset):
    def __init__(self, path, sex='female'):
        self.path = path
        self.sex = sex
        
        self.filename = glob(self.path + f'/{sex}/*.pt')
        
    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, idx):
        return torch.load(self.filename[idx]).to(dtype=torch.float16)[0][0], 1