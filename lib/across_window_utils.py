import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from scipy.linalg import inv
from lib.utils import get_subseq_func

def get_phi_psi_dist_window(q, seq_ctxt):
    seq = q.get_subseq(seq_ctxt)
    phi_psi_dist = q.results_window[q.results_window.seq == seq]
    phi_psi_dist = phi_psi_dist[['match_id', 'window_pos', 'phi', 'psi']].pivot(index='match_id', columns='window_pos', values=['phi', 'psi'])
    phi_psi_dist.columns = [f'{c[0]}_{c[1]}' for c in phi_psi_dist.columns.to_flat_index()]
    phi_psi_dist = phi_psi_dist.dropna(axis=0)
    return phi_psi_dist

def get_combined_phi_psi_dist(ins, seq_ctxt, winsizes=None):
    phi_psi_dist = []
    winsizes = winsizes if winsizes is not None else ins.winsizes
    smallest_winsize = min(winsizes)
    for q in ins.queries:
        if q.winsize not in winsizes:
            continue
        inner_seq = q.get_subseq(seq_ctxt)
        matches_q = q.results_window[q.results_window.seq == inner_seq]
        # pivot to combine matches into single row covering all residues in the current subsequence
        matches_q = matches_q[['match_id', 'window_pos', 'phi', 'psi']].pivot(index='match_id', columns='window_pos', values=['phi', 'psi'])
        matches_q = matches_q.dropna(axis=0)
        if matches_q.shape[0] == 0:
            continue
        if matches_q.shape[1] != q.winsize*2:
            print(f"\tSkipping {inner_seq} - incomplete data")
            continue
        # flatten column index to be phi_0, phi_1, ..., psi_0, psi_1, ...
        matches_q.columns = [f'{c[0]}_{c[1]}' for c in matches_q.columns.to_flat_index()]
        # keep only the columns that are in the smallest window size - choose the columns so that the residues match up
        columns = get_subseq_func(smallest_winsize, q.winsize)(list(range(q.winsize)))
        matches_q = matches_q[[f'phi_{i}' for i in columns]+[f'psi_{i}' for i in columns]]
        # reset column index to be phi_0, psi_0, phi_1, ..., psi_0, psi_1, ...
        matches_q.columns = [f'phi_{i}' for i in range(smallest_winsize)] + [f'psi_{i}' for i in range(smallest_winsize)]
        matches_q['weight'] = q.weight
        matches_q['winsize'] = q.winsize
        matches_q['seq'] = inner_seq
        phi_psi_dist.append(matches_q)
    if len(phi_psi_dist) == 0:
        return None, None
    phi_psi_dist = pd.concat(phi_psi_dist).reset_index()
    phi_psi_dist = phi_psi_dist.loc[phi_psi_dist.index.repeat(phi_psi_dist.weight)].reset_index(drop=True)
    phi_psi_dist_v = phi_psi_dist[[f'phi_{i}' for i in range(smallest_winsize)]+[f'psi_{i}' for i in range(smallest_winsize)]]
    return phi_psi_dist, phi_psi_dist_v

def get_xrays_window(ins, q, seq_ctxt, return_df=False):
    center_idx = q.get_center_idx_pos()
    xray_pos = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt == seq_ctxt].pos.iloc[0]
    xrays = ins.xray_phi_psi[(ins.xray_phi_psi.pos >= xray_pos-center_idx) & (ins.xray_phi_psi.pos < xray_pos-center_idx+q.winsize)].copy()
    xray_point = np.concatenate([xrays['phi'].values, xrays['psi'].values])
    if return_df:
        return xray_point, xrays
    return xray_point

def get_afs_window(ins, q, seq_ctxt, return_df=False):
    center_idx = q.get_center_idx_pos()
    af_pos = ins.af_phi_psi[ins.af_phi_psi.seq_ctxt == seq_ctxt].pos
    if len(af_pos) == 0:
        return None
    af_pos = af_pos.iloc[0]
    afs = ins.af_phi_psi[(ins.af_phi_psi.pos >= af_pos-center_idx) & (ins.af_phi_psi.pos < af_pos-center_idx+q.winsize)].copy()
    af_point = np.concatenate([afs['phi'].values, afs['psi'].values])
    if return_df:
        return af_point, afs
    return af_point

def get_preds_window(ins, q, seq_ctxt):
    center_idx = q.get_center_idx_pos()
    pred_pos = ins.phi_psi_predictions[ins.phi_psi_predictions.seq_ctxt == seq_ctxt].pos.unique()
    if len(pred_pos) == 0:
        return None
    if len(pred_pos) > 1:
        print(f"Multiple predictions for {seq_ctxt}")
        raise ValueError
    pred_pos = pred_pos[0]
    preds = ins.phi_psi_predictions[(ins.phi_psi_predictions.pos >= pred_pos-center_idx) & (ins.phi_psi_predictions.pos < pred_pos-center_idx+q.winsize)].copy()
    preds = preds[['protein_id', 'pos', 'phi', 'psi']].pivot(index='protein_id', columns='pos', values=['phi', 'psi'])
    preds.columns = [f'{c[0]}_{c[1]-pred_pos+center_idx}' for c in preds.columns.to_flat_index()]
    preds = preds.dropna(axis=0)
    return preds

def precompute_dists(phi_psi_dist):
    def diff(x1, x2):
            d = np.abs(x1 - x2)
            return np.minimum(d, 360-d)
    precomputed_dists = np.linalg.norm(diff(phi_psi_dist.values[:,np.newaxis], phi_psi_dist.values), axis=2)
    return precomputed_dists

def find_clusters(precomputed_dists, min_cluster_size=20, cluster_selection_epsilon=30):
    precomputed_dists = precomputed_dists.copy()
    # phi_psi_dist['cluster'] = HDBSCAN(min_cluster_size=20, min_samples=5, metric='precomputed').fit(precomputed_dists).labels_
    clusters = HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=min(min_cluster_size, precomputed_dists.shape[0]), 
        metric='precomputed', 
        allow_single_cluster=True,
        cluster_selection_epsilon=cluster_selection_epsilon
    ).fit(precomputed_dists).labels_
    n_clusters = len(np.unique(clusters[clusters != -1]))
    return n_clusters, clusters

def filter_precomputed_dists(precomputed_dists, phi_psi_dist, clusters):
    return(
        precomputed_dists[clusters != -1][:,clusters != -1],
        phi_psi_dist[clusters != -1],
        clusters[clusters != -1]
    )

def calc_da_for_one_window(xrays, target, icov):
    xray_diff = diff(xrays, target)
    xray_da = np.sqrt(xray_diff @ icov @ xray_diff)
    return xray_da

def calc_da_window(preds, target, icov):
    preds_diff = diff(preds.values, target)
    preds_da = np.sqrt((preds_diff @ icov @ preds_diff.T).diagonal())
    return preds_da

def get_target_cluster_icov(phi_psi_dist, precomputed_dists, clusters, af):
    target_cluster = get_target_cluster(phi_psi_dist, clusters, af)
    cluster_medoid = get_cluster_medoid(phi_psi_dist, precomputed_dists, clusters, target_cluster)
    icov = estimate_icov(phi_psi_dist[clusters == target_cluster], cluster_medoid)
    if icov is None:
        return None, None, None
    return target_cluster, cluster_medoid, icov


# Internally used

def diff(x1, x2):
    d = np.abs(x1 - x2)
    return np.minimum(d, 360-d)

def get_target_cluster(phi_psi_dist, clusters, point):
    d = np.linalg.norm(diff(point[np.newaxis,:], phi_psi_dist.values), axis=1)
    d = pd.DataFrame({'d': d, 'c': clusters})
    nearest_cluster = d.groupby('c').d.mean().idxmin()
    return nearest_cluster

def get_cluster_medoid(phi_psi_dist, precomputed_dists, clusters, c):
    d = precomputed_dists[clusters == c][:,clusters == c]
    return phi_psi_dist[clusters == c].iloc[d.sum(axis=1).argmin()].values

def estimate_icov(phi_psi_dist_c, cluster_medoid):
    # estimate covariance matrix
    cluster_points = phi_psi_dist_c.values
    diffs = diff(cluster_points, cluster_medoid)

    cov = (diffs[...,np.newaxis] @ diffs[:,np.newaxis]).sum(axis=0) / (diffs.shape[0] - 1)
    cov = cov + np.eye(cov.shape[0]) * 1e-6 # add small value to diagonal to avoid singular matrix
    if np.any(cov <= 0):
        print("Non-positive covariance matrix")
        return None
    if np.any(cov.diagonal() < 1):
        print("Covariance matrix less than 1")
        return None
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    if np.any(eigenvalues < 0):
        print("Negative eigenvalues - non-positive semi-definite covariance matrix")
        return None
    icov = inv(cov)
    return icov