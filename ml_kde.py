from lib import MultiWindowQuery
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv
import torch
from torch.nn import functional as F
from scipy.stats import gaussian_kde
from lib.constants import AMINO_ACID_MAP, AMINO_ACID_MAP_INV

PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")
PROJECT_DIR = 'ml_data'
pdb_codes = [f.name.split('_')[0] for f in Path(PROJECT_DIR).iterdir() if f.is_dir()]
winsizes = [4,5,6,7]
outdir = Path('ml_samples/kde')
outdir.mkdir(exist_ok=True, parents=True)
for id in pdb_codes:
    if (outdir / f'{id}.pt').exists():
        print('Already processed', id)
        continue
    try:
        da = MultiWindowQuery(id, winsizes, PDBMINE_URL, PROJECT_DIR)
        da.load_results()
    except FileNotFoundError as e:
        print(e)
        continue
    if da.af_phi_psi is None:
        print('No af_phi_psi for', id)
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
            matches = q.results[q.results.seq == inner_seq]
            if matches.shape[0] < 2:
                kdepeaks.append(torch.zeros(2))
                continue
            phi = matches.phi.values
            psi = matches.psi.values
            x = np.stack([phi, psi])
            try:
                kde = gaussian_kde(x, bw_method=0.5)
                phi_grid, psi_grid = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-180, 180, 360))
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