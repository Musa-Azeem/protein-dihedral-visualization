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

winsizes = [4,5,6,7]
PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")
PROJECT_DIR = 'ml_data_10.23.24'

targetlist = retrieve_target_list()
ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
proteins = json.load(open('proteins.json'))
skip = [targetlist.loc[id, 'pdb_code'].upper() for id in ids]

for pdb_code in proteins:
    if pdb_code in skip:
        continue
    da = MultiWindowQuery(pdb_code, winsizes, PDBMINE_URL, PROJECT_DIR, match_outdir='/Data/cache')
    if len(list(da.outdir.iterdir())) > 0:
        print('Results already exist')
        continue
    try:
        da.compute_structure()
    except Exception as e:
        print(f'Error {pdb_code}: {e}')
        os.system(f'rm -r {da.outdir}')
        continue
    da.query_pdbmine()
