from lib import DihedralAdherence, MultiWindowQuery
from lib.retrieve_data import retrieve_target_list
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import json
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
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from lib.constants import AMINO_ACID_MAP, AMINO_ACID_MAP_INV
PDBMINE_URL = os.getenv("PDBMINE_URL")
PROJECT_DIR = 'ml_data'

# PDBMINE_URL = os.getenv("PDBMINE_URL")
# winsizes = [4,5,6,7]
# # ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
# # winsizes = [4,5,6,7]
# # PROJECT_DIR = 'tests'
# # for id in ids:
# #     da = DihedralAdherence(id, winsizes, PDBMINE_URL, PROJECT_DIR)
# #     outdir = Path('ml_data') / (da.pdb_code.upper() + '_' + da.outdir.name.split('_')[1])
# #     outdir.mkdir(exist_ok=True)
# #     os.system(f'cp {da.outdir}/xray_phi_psi.csv {outdir}/xray_phi_psi.csv')
# #     for winsize in winsizes:
# #         os.system(f'cp {da.outdir}/phi_psi_mined_win{winsize}.csv {outdir}/phi_psi_mined_win{winsize}.csv')

# ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
# targetlist = retrieve_target_list()
# skip = [targetlist.loc[id, 'pdb_code'].upper() for id in ids]

# PROJECT_DIR = 'ml_data'
# proteins = json.load(open('proteins.json'))
# pdb_codes = proteins[:1000]
# for pdb_code in pdb_codes:
#     da = MultiWindowQuery(pdb_code, winsizes, PDBMINE_URL, PROJECT_DIR)
#     try:
#         da.compute_structure()
#     except Exception as e:
#         print(f'Error {pdb_code}: {e}')
#         os.system(f'rm -r {da.outdir}')
#         continue
#     da.query_pdbmine()

PDBMINE_URL = os.getenv("PDBMINE_URL")
PROJECT_DIR = 'ml_data'
# casp_protein_ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
# pdb_codes = ['6T1Z', '7UM1', '7D2O', '7CN6', '7W6B', '7ABW', '6Y4F', '6POO', '6YJ1', '6YA2', '6UV6', '6VQP']
pdb_codes = [f.name.split('_')[0] for f in Path(PROJECT_DIR).iterdir() if f.is_dir()]
winsizes = [4,5,6,7]
outdir = Path('ml_samples/kde')
outdir.mkdir(exist_ok=True, parents=True)
n_matches = defaultdict(list)
# pdb_codes = [pdb_codes[0]]
for id in pdb_codes:
    if (outdir / f'{id}.pt').exists():
        continue
    try:
        da = MultiWindowQuery(id, winsizes, PDBMINE_URL, PROJECT_DIR)
        # da.compute_structure()
        da.load_results()
    except FileNotFoundError as e:
        print(e)
        continue
    if da.af_phi_psi is None:
        continue
    seqs = pd.merge(
        da.xray_phi_psi[['seq_ctxt', 'res', 'phi', 'psi']], 
        da.af_phi_psi[['seq_ctxt', 'phi', 'psi']], 
        on='seq_ctxt', suffixes=('', '_af')
    ).rename(columns={'seq_ctxt': 'seq'})
    if seqs.shape[0] == 0:
        print('No sequences for', id)
        continue

    print(seqs.shape, seqs.seq.nunique())
    X = []
    y = []
    x_res = []
    af_phi_psi = []
    for i,row in tqdm(seqs.iterrows()):
        kdepeaks = []
        if np.isnan(row.phi) or np.isnan(row.psi) or np.isnan(row.phi_af) or np.isnan(row.psi_af):
            print('NaNs for', row.seq)
            continue
        for q in da.queries:
            inner_seq = q.get_subseq(row.seq)
            # matches = q.results[q.results.seq == inner_seq][['seq', 'phi', 'psi']]
            matches = q.results[q.results.seq == inner_seq]
            if matches.shape[0] < 2:
                kdepeaks.append(torch.zeros(2))
                continue
            phi = matches.phi.values
            psi = matches.psi.values
            x = np.stack([phi, psi])
            try:
                kde = gaussian_kde(x, bw_method=0.5)
                phi_grid, psi_grid = np.meshgrid(np.linspace(-180, 180, 180), np.linspace(-180, 180, 180))
                grid = np.vstack([phi_grid.ravel(), psi_grid.ravel()])
                probs = kde(grid).reshape(phi_grid.shape)
                kdepeak = grid[:,probs.argmax()]
                kdepeaks.append(torch.tensor(kdepeak))
            except np.linalg.LinAlgError as e:
                kdepeaks.append(torch.zeros(2))
        kdepeaks = torch.stack(kdepeaks)
        if torch.sum(kdepeaks) == 0:
            print('No matches for', row.seq)
            continue
        X.append(kdepeaks)
        y.append(torch.tensor([row.phi, row.psi]))
        x_res.append(AMINO_ACID_MAP[row.res])
        af_phi_psi.append(torch.tensor([row.phi_af, row.psi_af]))
    if len(X) == 0:
        print('No matches for', id)
        continue
    X = torch.stack(X)
    y = torch.stack(y)
    x_res = F.one_hot(torch.tensor(x_res).to(torch.int64), num_classes=20)
    af_phi_psi = torch.stack(af_phi_psi)
    torch.save((X, x_res, af_phi_psi, y), outdir / f'{id}.pt')