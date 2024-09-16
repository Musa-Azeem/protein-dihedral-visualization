from lib import DihedralAdherence
from lib import PDBMineQuery
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import json
from matplotlib.patches import ConnectionPatch
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import defaultdict

PDBMINE_URL = os.getenv("PDBMINE_URL")
PROJECT_DIR = 'casp_da'

proteins = [
  'T1024', 'T1030', 'T1030-D2', 'T1024-D1', 'T1032-D1', 'T1053-D1', 'T1027-D1', 'T1029-D1',
  'T1025-D1', 'T1028-D1', 'T1030-D1', 'T1053-D2', 'T1057-D1','T1058-D1', 'T1058-D2'
]
da = DihedralAdherence(proteins[0], [4,5,6,7], PDBMINE_URL, PROJECT_DIR, kdews=[1,1,1,1], 
                      mode='full_window', weights_file='ml_runs/best_model-kde_16-32_383.pt', device='cpu')
                    #   mode='ml', weights_file='ml_runs/best_model-kde_16-32_383.pt', device='cpu')

da.load_results_da()
center_idx_ctxt = da.queries[-1].get_center_idx()
winsize_ctxt = da.queries[-1].winsize
if center_idx_ctxt < 0:
    center_idx_ctxt = winsize_ctxt + center_idx_ctxt
da.seqs_for_window = da.seqs[center_idx_ctxt:-(winsize_ctxt - center_idx_ctxt - 1)]

def diff(x1, x2):
    d = np.abs(x1 - x2)
    return np.minimum(d, 360-d)

def get_phi_psi_dist(q, seq_ctxt):
    seq = q.get_subseq(seq_ctxt)
    phi_psi_dist = q.results_window[q.results_window.seq == seq]
    phi_psi_dist = phi_psi_dist[['match_id', 'window_pos', 'phi', 'psi']].pivot(index='match_id', columns='window_pos', values=['phi', 'psi'])
    phi_psi_dist.columns = [f'{c[0]}_{c[1]}' for c in phi_psi_dist.columns.to_flat_index()]
    return phi_psi_dist

def get_xrays(da, q, seq_ctxt):
    center_idx = q.get_center_idx_pos()
    xray_pos = da.xray_phi_psi[da.xray_phi_psi.seq_ctxt == seq_ctxt].pos.iloc[0]
    xrays = da.xray_phi_psi[(da.xray_phi_psi.pos >= xray_pos-center_idx) & (da.xray_phi_psi.pos < xray_pos-center_idx+q.winsize)]
    xray_point = np.concatenate([xrays['phi'].values, xrays['psi'].values])
    return xrays, xray_point

def xray_sil_score(phi_psi_dist, xray_point):
    cluster_aves = phi_psi_dist.groupby('cluster').mean()
    nearest_cluster_idx = np.linalg.norm(diff(cluster_aves.values, xray_point), axis=1).argmin()
    nearest_cluster = cluster_aves.iloc[nearest_cluster_idx].name
    
    xray_sil = silhouette_samples(
        np.vstack([phi_psi_dist.iloc[:,:-1], xray_point[np.newaxis,:]]), 
        np.append(phi_psi_dist.cluster.values, nearest_cluster)
    )[-1]
    return xray_sil

def assign_clusters(phi_psi_dist, eps=75):
    def diff(x1, x2):
        d = np.abs(x1 - x2)
        return np.minimum(d, 360-d)
    precomputed_dists = np.linalg.norm(diff(phi_psi_dist.values[:,np.newaxis], phi_psi_dist.values), axis=2)
    phi_psi_dist['cluster'] = DBSCAN(eps=eps, min_samples=5, metric='precomputed').fit(precomputed_dists).labels_
    n_clusters = len(phi_psi_dist.cluster.unique())
    return n_clusters


results = []
q = da.queries[0]
from tqdm import tqdm
for seq_ctxt in tqdm(da.seqs_for_window):
    phi_psi_dist = get_phi_psi_dist(q, seq_ctxt)
    xrays, xray_point = get_xrays(da, q, seq_ctxt)
    if phi_psi_dist.shape[0] == 0:
        continue
    if xrays.shape[0] != q.winsize:
        continue
    n_clusters = assign_clusters(phi_psi_dist)
    phi_psi_dist = phi_psi_dist[phi_psi_dist.cluster != -1]

    sil_score = silhouette_score(phi_psi_dist.iloc[:,:-1], phi_psi_dist.cluster)
    xray_sil = xray_sil_score(phi_psi_dist, xray_point)

    n_samples = phi_psi_dist.shape[0]
    results.append([da.casp_protein_id, seq_ctxt, n_samples, n_clusters, sil_score, xray_sil])
    results_df = pd.DataFrame(results, columns=['protein', 'seq_ctxt', 'n_samples', 'n_clusters', 'sil_score', 'xray_sil'])
    results_df.to_csv('8d_cluster_results.csv', index=False)