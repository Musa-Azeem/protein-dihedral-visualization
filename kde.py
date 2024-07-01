from lib import DihedralAdherence
from lib import PDBMineQuery, MultiWindowQuery
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict
from dotenv import load_dotenv
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from lib.constants import AMINO_ACID_MAP, AMINO_ACID_MAP_INV
PDBMINE_URL = os.getenv("PDBMINE_URL")
PROJECT_DIR = 'ml_data'

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
s = [sum(lengths[:i]) for i,l in enumerate(lengths)]

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


from scipy.stats import gaussian_kde
from lib.utils import find_kdepeak_af
def get_kdepeak(x, af):
    kdews = [1,32,64,128]
    Xi = x.cpu().numpy().copy()
    kdes = []
    af = pd.DataFrame(af, columns=['phi', 'psi'])
    for i in range(Xi.shape[0]):
        mask = (Xi[i,0] == 0) & (Xi[i,1] == 0)
        Xi[i,:,mask] = np.nan
        x1 = Xi[i,:,s[0]:s[1]]
        x1 = x1[:,~np.isnan(x1).any(axis=0)]
        w1 = np.full(x1.shape[1], kdews[0])
        x2 = Xi[i,:,s[1]:s[2]]
        x2 = x2[:,~np.isnan(x2).any(axis=0)]
        w2 = np.full(x2.shape[1], kdews[1])
        x3 = Xi[i,:,s[2]:s[3]]
        x3 = x3[:,~np.isnan(x3).any(axis=0)]
        w3 = np.full(x3.shape[1], kdews[2])
        x4 = Xi[i,:,s[3]:]
        x4 = x4[:,~np.isnan(x4).any(axis=0)]
        w4 = np.full(x4.shape[1], kdews[3])

        afi = af.loc[af.index == i]

        x = np.concatenate([x1,x2,x3,x4], axis=1)
        w = np.concatenate([w1,w2,w3,w4])
        phi_psi_dist = pd.DataFrame(np.concatenate([x,w.reshape(1,-1)]).T, columns=['phi', 'psi', 'weight'])


        if x.shape[1] == 0:
            kdes.append(np.full(2, np.nan))
            continue
        try:
            kdepeak = find_kdepeak_af(phi_psi_dist, None, afi).values
            print('KDEPEAK', kdepeak)
        except Exception as e:
            print(e)
            kdes.append(np.full(2, np.nan))
            continue
        kdes.append(kdepeak)
    return np.stack(kdes)

# Eucledian distance
def diff(x1, x2):
    d = np.abs(x1 - x2)
    return np.minimum(d, 360-d)

kdes = []
kdes_loss = []
for X,xres,af,y in tqdm(testloader):
    kde = get_kdepeak(X, af)
    kdes.append(kde)
kdes = np.concatenate(kdes)
torch.save(kdes, 'ml_data/kdes.pt')