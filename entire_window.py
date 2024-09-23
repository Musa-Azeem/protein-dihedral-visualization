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
from tqdm import tqdm

PDBMINE_URL = os.getenv("PDBMINE_URL")
PROJECT_DIR = 'casp_da'

proteins = [
  'T1024', 'T1030', 'T1030-D2', 'T1024-D1', 'T1032-D1', 'T1053-D1', 'T1027-D1', 'T1029-D1',
  'T1025-D1', 'T1028-D1', 'T1030-D1', 'T1053-D2', 'T1057-D1','T1058-D1', 'T1058-D2'
]
da = DihedralAdherence(proteins[0], [4,5,6,7], PDBMINE_URL, PROJECT_DIR, kdews=[1,1,1,1], 
                      mode='full_window', weights_file='ml_runs/best_model-kde_16-32_383.pt', device='cpu')
                    #   mode='ml', weights_file='ml_runs/best_model-kde_16-32_383.pt', device='cpu')

# da.load_results_da()
da.load_results()
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
    phi_psi_dist = phi_psi_dist.dropna(axis=0)
    return phi_psi_dist

def get_xrays(ins, q, seq_ctxt, return_df=False):
    center_idx = q.get_center_idx_pos()
    xray_pos = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq_ctxt].pos.iloc[0]
    xrays = ins.xray_phi_psi[(ins.xray_phi_psi.pos >= xray_pos-center_idx) & (ins.xray_phi_psi.pos < xray_pos-center_idx+q.winsize)].copy()
    xray_point = np.concatenate([xrays['phi'].values, xrays['psi'].values])
    if return_df:
        return xray_point, xrays
    return xray_point

def get_preds(ins, q, seq_ctxt):
    center_idx = q.get_center_idx_pos()
    pred_pos = ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq_ctxt].pos.unique()
    if len(pred_pos) == 0:
        print(f"No predictions for {seq_ctxt}")
    if len(pred_pos) > 1:
        print(f"Multiple predictions for {seq_ctxt}")
        raise ValueError
    pred_pos = pred_pos[0]
    preds = ins.phi_psi_predictions[(ins.phi_psi_predictions.pos >= pred_pos-center_idx) & (ins.phi_psi_predictions.pos < pred_pos-center_idx+q.winsize)].copy()
    preds = preds[['protein_id', 'pos', 'phi', 'psi']].pivot(index='protein_id', columns='pos', values=['phi', 'psi'])
    preds.columns = [f'{c[0]}_{c[1]-pred_pos+center_idx}' for c in preds.columns.to_flat_index()]
    preds = preds.dropna(axis=0)
    return preds

def xray_sil_score(phi_psi_dist, xrays, q):
    cluster_aves = phi_psi_dist.groupby('cluster').mean()
    nearest_cluster_idx = np.linalg.norm(diff(cluster_aves.values, xrays), axis=1).argmin()
    nearest_cluster = cluster_aves.iloc[nearest_cluster_idx].name

    values = np.vstack([phi_psi_dist.iloc[:,:q.winsize*2], xrays[np.newaxis,:]])
    precomputed_dists = np.linalg.norm(diff(values[:,np.newaxis], values), axis=2)
    xray_sil = silhouette_samples(
        precomputed_dists, 
        np.append(phi_psi_dist.cluster.values, nearest_cluster),
        metric='precomputed'
    )[-1]
    return xray_sil, nearest_cluster

def calc_and_assign_sil_score(q, preds, phi_psi_dist):
    cluster_aves = phi_psi_dist.groupby('cluster').mean().iloc[:,:q.winsize*2]
    nearest_cluster_idxs = np.linalg.norm(diff(preds.iloc[:,:q.winsize*2].values[:,np.newaxis], cluster_aves.values), axis=2).argmin(axis=1)
    preds['cluster'] = cluster_aves.iloc[nearest_cluster_idxs].index.values

    preds['sil'] = np.nan
    distances = np.linalg.norm(diff(preds.iloc[:,:q.winsize*2].values[:, np.newaxis], phi_psi_dist.iloc[:,:q.winsize*2].values), axis=2)
    for ci in phi_psi_dist.cluster.unique():
        dists_ci = distances[preds.cluster.values == ci]
        # a = dists_ci[:, phi_psi_dist.cluster == ci].mean(axis=1)
        a = dists_ci[:, phi_psi_dist.cluster == ci].sum(axis=1) / (sum(phi_psi_dist.cluster == ci) - 1)

        bs = []
        for cj in phi_psi_dist[phi_psi_dist.cluster != ci].cluster.unique():
            # bs.append(dists_ci[:, phi_psi_dist.cluster == cj].mean(axis=1))
            bs.append(dists_ci[:, phi_psi_dist.cluster == cj].sum(axis=1) / sum(phi_psi_dist.cluster == cj))
        bs = np.vstack(bs)
        b = np.min(bs, axis=0)

        sils = (b - a) / np.maximum(a, b)
        preds.loc[preds.cluster == ci, 'sil'] = sils

def assign_clusters(phi_psi_dist, eps=75):
    def diff(x1, x2):
        d = np.abs(x1 - x2)
        return np.minimum(d, 360-d)
    precomputed_dists = np.linalg.norm(diff(phi_psi_dist.values[:,np.newaxis], phi_psi_dist.values), axis=2)
    phi_psi_dist['cluster'] = DBSCAN(eps=eps, min_samples=5, metric='precomputed').fit(precomputed_dists).labels_
    n_clusters = len(phi_psi_dist.cluster.unique())
    return n_clusters


ins = da
q = ins.queries[0]
ins.phi_psi_predictions['sil'] = np.nan
ins.xray_phi_psi['sil'] = np.nan

for seq_ctxt in tqdm(ins.seqs_for_window):
    phi_psi_dist = get_phi_psi_dist(q, seq_ctxt)
    xrays = get_xrays(ins, q, seq_ctxt)
    preds = get_preds(ins, q, seq_ctxt)

    if xrays.shape[0] != q.winsize*2:
        print(f"Xray data for {seq_ctxt} is incomplete")
        continue
    if preds.shape[0] == 0:
        print(f"No predictions for {seq_ctxt}")
        continue
    if phi_psi_dist.shape[0] == 0:
        print(f"No pdbmine data for {seq_ctxt}")
        continue
    
    n_clusters = assign_clusters(phi_psi_dist)
    phi_psi_dist = phi_psi_dist[phi_psi_dist.cluster != -1]

    sil_score = silhouette_score(phi_psi_dist.iloc[:,:-1], phi_psi_dist.cluster)
    xray_sil, _ = xray_sil_score(phi_psi_dist, xrays, q)
    calc_and_assign_sil_score(q, preds, phi_psi_dist)

    ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq_ctxt, 'sil'] = xray_sil

    view = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq_ctxt].reset_index().set_index('protein_id')
    view.loc[preds.index, 'sil'] = preds.sil
    ins.phi_psi_predictions.loc[view['index'], 'sil'] = view.set_index('index').sil

ins.phi_psi_predictions.to_csv(ins.outdir / 'phi_psi_predictions_da_window.csv', index=False)
ins.xray_phi_psi.to_csv(ins.outdir / 'xray_phi_psi_da_window.csv', index=False)