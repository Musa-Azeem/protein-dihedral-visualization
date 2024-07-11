from torch import nn
import torch
import torch.nn.functional as F
from lib.constants import AMINO_ACID_MAP
import numpy as np
import pandas as pd

def save_model(model, path):
    if type(model) == nn.DataParallel:
        model = model.module
    torch.save(model.state_dict(), path)
def load_model(model, path):
    if type(model) == nn.DataParallel:
        model = model.module
    model.load_state_dict(torch.load(path))
    return model

def get_ml_pred(peaks, res, af, ml):
    xres = AMINO_ACID_MAP[res]
    xres = F.one_hot(torch.tensor(xres, dtype=torch.int64), num_classes=20).unsqueeze(0)
    X = torch.tensor(peaks).unsqueeze(0).to(torch.float32)
    af = torch.tensor(af).unsqueeze(0).to(torch.float32)
    pred = ml(X, xres, af).squeeze().numpy()
    return pred

# def get_ml_pred(phi_psi_dist, winsizes, res, af, ml):
#     phis = []
#     psis = []
#     xres = AMINO_ACID_MAP[res]
#     lengths_dict = {w:l for w,l in zip(winsizes, ml.lengths)}
#     for w in winsizes:
#         phi, psi = phi_psi_dist[phi_psi_dist.winsize == w][['phi', 'psi']].values.T
#         if phi.shape[0] > lengths_dict[w]:
#             phi = np.random.choice(phi, lengths_dict[w], replace=False)
#             psi = np.random.choice(psi, lengths_dict[w], replace=False)
#         else:
#             phi = np.pad(phi, (0, lengths_dict[w] - phi.shape[0]))
#             psi = np.pad(psi, (0, lengths_dict[w] - psi.shape[0]))
#         phis.append(phi)
#         psis.append(psi)
#     phis = np.concatenate(phis)
#     psis = np.concatenate(psis)

#     xres = F.one_hot(torch.tensor(xres, dtype=torch.int64), num_classes=20).unsqueeze(0)
#     X = torch.tensor(np.stack([phis, psis]), dtype=torch.float32).unsqueeze(0)
#     pred = ml(X, xres).squeeze().numpy()
#     return pd.Series({'phi': pred[0], 'psi': pred[1]})