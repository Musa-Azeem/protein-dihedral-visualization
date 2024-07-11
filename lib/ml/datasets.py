from torch.utils.data import Dataset
import torch
from pathlib import Path

def get_dataset(lengths, path):
    path = Path(path+'-'.join([str(l) for l in lengths]))
    samples = [f.stem for f in path.iterdir()]

class ProteinDataset(Dataset):
    def __init__(self, id, path):
        self.id = id
        self.path = path

        self.X, self.y, self.xres = torch.load(self.path / f'{id}.pt')
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.xres[i], self.y[i]