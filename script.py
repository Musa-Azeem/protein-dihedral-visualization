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

PDBMINE_URL = os.getenv("PDBMINE_URL")
winsizes = [4,5,6,7]
# ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
# winsizes = [4,5,6,7]
# PROJECT_DIR = 'tests'
# for id in ids:
#     da = DihedralAdherence(id, winsizes, PDBMINE_URL, PROJECT_DIR)
#     outdir = Path('ml_data') / (da.pdb_code.upper() + '_' + da.outdir.name.split('_')[1])
#     outdir.mkdir(exist_ok=True)
#     os.system(f'cp {da.outdir}/xray_phi_psi.csv {outdir}/xray_phi_psi.csv')
#     for winsize in winsizes:
#         os.system(f'cp {da.outdir}/phi_psi_mined_win{winsize}.csv {outdir}/phi_psi_mined_win{winsize}.csv')

ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
targetlist = retrieve_target_list()
skip = [targetlist.loc[id, 'pdb_code'].upper() for id in ids]

PROJECT_DIR = 'ml_data'
proteins = json.load(open('proteins.json'))
pdb_codes = proteins[:1000]
for pdb_code in pdb_codes:
    da = MultiWindowQuery(pdb_code, winsizes, PDBMINE_URL, PROJECT_DIR)
    try:
        da.compute_structure()
    except Exception as e:
        print(f'Error {pdb_code}: {e}')
        os.system(f'rm -r {da.outdir}')
        continue
    da.query_pdbmine()