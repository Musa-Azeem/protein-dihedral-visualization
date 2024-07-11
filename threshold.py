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
proteins = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']

results = []
for thresh in np.linspace(0, 250, 100):
    thresh = round(thresh, 2)
    scores = []
    rmsds = []
    protein_ids = []
    for protein in proteins:
        da = DihedralAdherence(protein, [4,5,6,7], PDBMINE_URL, PROJECT_DIR, kdews=[1,32,64,128], 
                            mode='kde_af')
        da.load_results_da()
        pbar = tqdm(da.protein_ids)
        pbar.set_description(f'Protein {protein} Threshold {thresh}')
        for id in pbar:
            preds = da.phi_psi_predictions[da.phi_psi_predictions.protein_id == id].dropna()
            if preds.shape[0] == 0:
                continue
            rmsd = da.grouped_preds[da.grouped_preds.protein_id == id].RMS_CA.values
            if len(rmsd) == 0:
                continue
            rmsd = rmsd[0]
            score = (preds.da < thresh).sum() / preds.shape[0]
            scores.append(score)
            rmsds.append(rmsd)
            protein_ids.append(protein)
        
    model = stats.linregress(rmsds, scores)
    results.append((thresh, model.slope, model.intercept, model.rvalue**2, model.pvalue))

    results_df = pd.DataFrame(results, columns=['threshold', 'slope', 'intercept', 'rsquared', 'pvalue'])
    results_df.to_csv('results_threshold_combined.csv', index=False)