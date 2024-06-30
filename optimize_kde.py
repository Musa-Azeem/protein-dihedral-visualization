from lib import DihedralAdherence
from lib import PDBMineQuery
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import gaussian_kde
from pathlib import Path
PDBMINE_URL = os.getenv("PDBMINE_URL")
PROJECT_DIR = 'tests'
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

class ProteinDataset(Dataset):
    def __init__(self, id, path):
        self.id = id
        self.path = path

        self.X, self.y, self.xres, self.af = torch.load(self.path / f'{id}.pt')
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.xres[i], self.af[i], self.y[i]
    

lengths = [4096, 512, 256, 256]
path = Path('ml_samples/'+'-'.join([str(l) for l in lengths]))
samples = [f.stem for f in path.iterdir()]

from lib.retrieve_data import retrieve_target_list
ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
targetlist = retrieve_target_list()
skip = [targetlist.loc[id, 'pdb_code'].upper() for id in ids]
samples = sorted(list(set(samples) - set(skip)))

train, test = train_test_split(samples, test_size=0.35, random_state=42)
torch.save((train, test), 'ml_data/split.pt')
# train, test = to ch.load('ml_data/split.pt')
train_dataset = ConcatDataset([ProteinDataset(s, path) for s in train])
test_dataset = ConcatDataset([ProteinDataset(s, path) for s in test])
trainloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
len(train_dataset), len(test_dataset), len(train_dataset) + len(test_dataset)


phi_grid, psi_grid = np.meshgrid(np.linspace(-180, 180, 100), np.linspace(-180, 180, 100))
grid = np.vstack([phi_grid.ravel(), psi_grid.ravel()])
diff = lambda x1, x2: min(abs(x1 - x2), 360 - abs(x1 - x2))
h = 0.5
results = []
for i in range(0,10,2):
    wi = np.power(2,i)
    for j in range(0,10,2):
        wj = np.power(2,j)
        for k in range(0,10,2):
            wk = np.power(2,k)
            for l in tqdm(range(0,10,2)):
                wl = np.power(2,l)
                ws = [wi,wj,wk,wl]
                weights = np.concatenate([np.array([w]*l) for w,l in zip(ws, lengths)])
                for i in np.random.choice(len(train_dataset), 16):
                    x, xres, af, y = train_dataset[i]
                    x = x.numpy()
                    y = y.numpy()
                    kde = gaussian_kde(x, weights=weights, bw_method=h)
                    probs = kde(grid).reshape(phi_grid.shape)
                    kdepeak = grid[:,probs.argmax()]
                    dist = np.sqrt(diff(y[0], kdepeak[0])**2 + diff(y[1], kdepeak[1])**2)
                    results.append([wi,wj,wk,wl,dist])
                    with open ('results_.csv', 'a') as f:
                        f.write(f'{wi},{wj},{wk},{wl},{dist}\n')
df = pd.DataFrame(results, columns=['w4', 'w5', 'w6', 'w7', 'da'])
df.to_csv('results.csv', index=False)