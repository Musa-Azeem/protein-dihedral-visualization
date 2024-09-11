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
PDBMINE_URL = os.getenv("PDBMINE_URL")
PROJECT_DIR = 'casp_da'

def find_k_nearest_neighbors(phi_psi_dist, preds, k=10):
    phi_psi_dist = phi_psi_dist[['match_id', 'window_pos', 'phi', 'psi']].pivot(index='match_id', columns='window_pos', values=['phi', 'psi'])
    phi_psi_dist.columns = [f'{c[0]}_{c[1]}' for c in phi_psi_dist.columns.to_flat_index()]
    phi_psi_dist = phi_psi_dist.values
    pred = np.concatenate([preds['phi'].values, preds['psi'].values])

    # Calculate the distances
    def diff(x1, x2):
        d = np.abs(x1 - x2)
        return np.minimum(d, 360-d)
    distances = np.sqrt(np.sum((diff(pred, phi_psi_dist))**2, axis=1))

    nearest = np.argsort(distances)[:k]
    return phi_psi_dist[nearest], distances[nearest]

def find_k_nearest_neighbors2(phi_psi_dist, preds, k=10):
    phi_psi_dist = phi_psi_dist[['match_id', 'window_pos', 'phi', 'psi']].pivot(index='match_id', columns='window_pos', values=['phi', 'psi'])
    phi_psi_dist.columns = [f'{c[0]}_{c[1]}' for c in phi_psi_dist.columns.to_flat_index()]

    preds = preds[['protein_id', 'pos', 'phi', 'psi']].pivot(index='protein_id', columns='pos', values=['phi', 'psi'])
    preds.columns = [f'{c[0]}_{c[1]}' for c in preds.columns.to_flat_index()]

    phi_psi_dist = phi_psi_dist.values
    preds = preds.values

    # Calculate the distance matrix
    def diff(x1, x2):
        d = np.abs(x1 - x2)
        return np.minimum(d, 360-d)
    
    distances = np.linalg.norm(diff(preds[:,np.newaxis], phi_psi_dist), axis=2)
    nearest = np.argsort(distances, axis=1)[:, :k]
    return distances[np.arange(distances.shape[0])[:, np.newaxis], nearest]


def calc_score(ins, ks):
    # ins is dihedral adherence object
    # q is the query object for a window size
    ins.phi_psi_predictions['new_score'] = np.nan
    ins.xray_phi_psi['new_score'] = np.nan

    center_idx_ctxt = ins.queries[-1].get_center_idx()
    winsize_ctxt = ins.queries[-1].winsize
    if center_idx_ctxt < 0:
        center_idx_ctxt = winsize_ctxt + center_idx_ctxt
    for i,seq_ctxt in enumerate(ins.xray_phi_psi.seq_ctxt[center_idx_ctxt:-(winsize_ctxt - center_idx_ctxt - 1)]):
        print(f'{i}/{ins.xray_phi_psi.shape[0] - winsize_ctxt}', seq_ctxt)

        pred_scores_qs = []
        xray_score_qs = []
        n_queries = 1
        for q,k in zip(ins.queries[:n_queries], ks[:n_queries]):
            seq = q.get_subseq(seq_ctxt)
            phi_psi_dist = q.results_window[q.results_window['seq'] == seq]
            if phi_psi_dist.shape[0] < k:
                print('\tSkipping', seq, 'Not enough data')
                continue

            center_idx = q.get_center_idx()
            if center_idx < 0:
                center_idx = q.winsize + center_idx
            
            xray_pos = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq_ctxt].pos.iloc[0]
            xrays = ins.xray_phi_psi[(ins.xray_phi_psi.pos >= xray_pos-center_idx) & (ins.xray_phi_psi.pos < xray_pos-center_idx+q.winsize)]

            pred_pos = ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq_ctxt].pos.unique()
            if len(pred_pos) == 0:
                print('\tSkipping', seq_ctxt, 'Prediction Positions are not unique')
                continue
            pred_pos = pred_pos[0]
            preds = ins.phi_psi_predictions[(ins.phi_psi_predictions.pos >= pred_pos-center_idx) & (ins.phi_psi_predictions.pos < pred_pos-center_idx+q.winsize)]

            if xrays.shape[0] < q.winsize or preds.shape[0] % q.winsize != 0:
                print('\tSkipping', seq_ctxt, 'Not enough xray or prediction data')
                continue

            distances = find_k_nearest_neighbors2(phi_psi_dist, preds, k)
            pred_scores = np.mean(distances, axis=1)
            pred_scores_qs.append(pred_scores)

            nearest, xray_distances = find_k_nearest_neighbors(phi_psi_dist, xrays, k)
            xray_score = np.mean(xray_distances)
            xray_score_qs.append(xray_score)

            print('\t', seq, q.winsize, phi_psi_dist.shape[0] // q.winsize, np.nanmean(pred_scores).round(2), xray_score.round(2))

        # For now, only use first two queries
        if len(pred_scores_qs) < n_queries:
            print('Skipping', seq_ctxt, 'Missing data for window sizes')
            continue
        pred_scores = np.stack(pred_scores_qs)
        xrays = np.array(xray_score_qs)
        
        pred_scores = np.mean(pred_scores, axis=0)
        xray_score = np.mean(xrays)

        ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq_ctxt, 'new_score'] = pred_scores
        ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq_ctxt, 'new_score'] = xray_score

proteins = [
  'T1024', 'T1030', 'T1030-D2', 'T1024-D1', 'T1032-D1', 'T1053-D1', 'T1027-D1', 'T1029-D1',
  'T1025-D1', 'T1028-D1', 'T1030-D1', 'T1053-D2', 'T1057-D1','T1058-D1', 'T1058-D2'
]
ks = [20, 5, 2, 2]
for protein in proteins:
    da = DihedralAdherence(protein, [4,5,6,7], PDBMINE_URL, PROJECT_DIR, kdews=[1,1,1,1], 
                        mode='ml', weights_file='ml_runs/best_model-kde_16-32_383.pt', device='cpu')
    da.load_results_da()
    
    ks = [20, 5, 2, 2]
    calc_score(da, ks)

    da.xray_phi_psi.to_csv(da.outdir / 'xray_phi_psi_da_window.csv', index=False)
    da.phi_psi_predictions.to_csv(da.outdir / 'phi_psi_predictions_da_window.csv', index=False)