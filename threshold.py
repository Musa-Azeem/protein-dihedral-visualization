from lib import DihedralAdherence
from lib import PDBMineQuery
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from tqdm import tqdm

PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")
PROJECT_DIR = 'casp_da'
proteins = ['T1024', 'T1096', 'T1091', 'T1030', 'T1038', 'T1030-D2', 'T1024-D1', 'T1032-D1', 'T1053-D1', 'T1027-D1', 'T1029-D1']

import itertools
for n in range(1,6):
    results = []
    combinations = list(itertools.combinations(np.linspace(16,256,7), n))
    for thresholds in combinations:
        gdts = []
        scores = []
        for j,protein in enumerate(proteins):
            da = DihedralAdherence(protein, [4,5,6,7], PDBMINE_URL, PROJECT_DIR, kdews=[1,32,64,128], 
                                mode='ml', weights_file='ml_data/best_model_kde_64-64_390.pt', device='cuda')
            da.load_results_da()
            da.filter_nas(0.8)
            for id in da.protein_ids:
                pred = da.phi_psi_predictions[da.phi_psi_predictions.protein_id == id].dropna()
                if pred.shape[0] == 0:
                    continue
                gdt = da.grouped_preds[da.grouped_preds.protein_id == id].GDT_TS.values
                if len(gdt) == 0:
                    continue
                gdt = gdt[0]
                score = [(pred.da < thresholds[0]).sum() / pred.shape[0]]
                for i in range(1, n):
                    score.append(((pred.da > thresholds[i-1]) & (pred.da < thresholds[i])).sum() / pred.shape[0])
                scores.append(score)
                gdts.append(gdt)
        x = np.array(scores)
        y = np.array(gdts)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        results.append([thresholds, model.rsquared])
        results_df = pd.DataFrame(results, columns=['thresholds', 'rsquared'])
        results_df.to_csv(f'results_threshold_{n}.csv', index=False)