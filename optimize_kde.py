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
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular

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
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# len(train_dataset), len(test_dataset), len(train_dataset) + len(test_dataset)

s = [sum(lengths[:i]) for i,l in enumerate(lengths)]

def run_kde_grid_search(idxs, fn, values, winsizes):
    phi_psi_grid_size = 50
    def get_preweights(X):
        phi_grid, psi_grid = np.meshgrid(np.linspace(-180, 180, phi_psi_grid_size), np.linspace(-180, 180, phi_psi_grid_size))
        grid = np.vstack([phi_grid.ravel(), psi_grid.ravel()])

        mask = (X[0] == 0) & (X[1] == 0)
        X = X[:,~mask]
        n_eff = X.shape[1]
        if n_eff == 0:
            return torch.zeros(*phi_grid.shape, 1), 0

        h=0.5
        cov = np.cov(X, rowvar=True)
        cho_cov = torch.tensor(cholesky(cov, lower=True)) * h # lower triangular
        h_det = cho_cov.diag().prod() # torch.det(cho_cov) # product of diagonal

        def K(x):
            # 2 dimensional standard normal distribution
            # return torch.exp(-0.5 * x.pow(2).sum(dim=1) / h) / (2 * np.pi * torch.sqrt(h_det))
            return torch.exp(-0.5 * x.pow(2).sum(dim=1)) / (2 * np.pi * h_det)
        def kde_pre_weights(xi):
            xi = torch.tensor(xi) if not isinstance(xi, torch.Tensor) else xi
            if xi.ndim == 1:
                xi = xi.unsqueeze(1)
            points = torch.tensor(solve_triangular(cho_cov, X, lower=True))
            xi = torch.tensor(solve_triangular(cho_cov, xi, lower=True))
            xi = xi.T.unsqueeze(-1)
            likelihood = K(points.unsqueeze(0) - xi)# * weights
            # likelihood = likelihood.sum(dim=1) / weights.sum()
            return likelihood

        preweights = kde_pre_weights(grid).reshape(*phi_grid.shape,-1)
        return preweights, n_eff

    diff = lambda x1, x2: min(abs(x1 - x2), 360 - abs(x1 - x2))
    s = [sum(lengths[:i]) for i,l in enumerate(lengths)]

    import itertools
    combinations = [comb for comb in itertools.product(values, repeat=len(winsizes)) if sum(comb) == 1]
    if len(winsizes) == 4:
        combinations = [tuple(np.array([1,32,64,128]) / np.array([1,32,64,128]).sum())] + combinations
    print(len(combinations), combinations)

    results = []
    for i in tqdm(idxs):
        X,xres,af,y = train_dataset[i]
        try:
            preweights = []
            n_effs = []
            for i in range(len(winsizes)):
                preweight, n_eff = get_preweights(X[:,s[i]:(s[i+1] if i < 3 else None)])
                # print(preweight.shape, n_eff)
                preweights.append(preweight)
                n_effs.append(n_eff)
            n_effs.append(n_eff)

            phi_grid, psi_grid = np.meshgrid(np.linspace(-180, 180, phi_psi_grid_size), np.linspace(-180, 180, phi_psi_grid_size))
            grid = np.vstack([phi_grid.ravel(), psi_grid.ravel()])
        except Exception as e:
            print(e)
            continue
        
        for ws in combinations:
            kdews = torch.tensor(ws)
            n = [w*n for w,n in zip(kdews, n_effs)]
            probs = torch.stack([(p * kdews[i]).sum(dim=-1) for i,p in enumerate(preweights)]).sum(dim=0) / sum(n)
            kdepeak = grid[:,probs.argmax()]
            dist = np.sqrt(diff(y[0], kdepeak[0])**2 + diff(y[1], kdepeak[1])**2)
            # print(y, kdepeak, probs.max(), dist)
            results.append([*ws,dist.item()])
        
        if i % 100 == 0:
            results_df = pd.DataFrame(results, columns=[*[f'w{w}' for w in winsizes], 'da'])
            results_df.to_csv(fn, index=False)
    results_df = pd.DataFrame(results, columns=[*[f'w{w}' for w in winsizes], 'da'])
    results_df.to_csv(fn, index=False)


idxs = [[],[],[],[]] # idxs with matches for winsize 4,5,6,7
for i,(X,xres,af,y) in enumerate(train_dataset):
    
    # winsize 7 - if there are matches for 7, there are matches for all
    x = X[:,s[3]:]
    if x.sum() > 0:
        idxs[3].append(i)
        idxs[2].append(i)
        idxs[1].append(i)
        idxs[0].append(i)
        continue

    # winsize 6 - if there are matches for 6, there are matches for 5 and 4
    x = X[:,s[2]:s[3]]
    if x.sum() > 0:
        idxs[2].append(i)
        idxs[1].append(i)
        idxs[0].append(i)
        continue

    # winsize 5 - if there are matches for 5, there are matches for 4
    x = X[:,s[1]:s[2]]
    if x.sum() > 0:
        idxs[1].append(i)
        idxs[0].append(i)
        continue
    
    # winsize 4 - always have matches for w4
    idxs[0].append(i)

# all windows:
grid_size = 6
values = np.linspace(0, 1, grid_size).round(5)
run_kde_grid_search(idxs[3][:1000], 'results_w7.csv', values, [4,5,6,7])

# 6 and below:
run_kde_grid_search(idxs[2][:1000], 'results_w6.csv', values, [4,5,6])

# 5 and 4:
run_kde_grid_search(idxs[1][:1000], 'results_w5.csv', values, [4,5])