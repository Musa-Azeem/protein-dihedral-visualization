from lib.across_window_utils import (
    get_combined_phi_psi_dist,
    get_xrays_window,
    get_afs_window,
    get_preds_window,
    precompute_dists,
    find_clusters,
    filter_precomputed_dists,
    calc_da_for_one_window,
    calc_da_window,
    get_target_cluster_icov,
)
from lib.utils import get_phi_psi_dist
import numpy as np
import pandas as pd
from pathlib import Path

MIN_SAMPLES = [100, 20, 1, 1]
MIN_CLUSTER_SIZES = [20, 5, 1, 1]

def get_da_for_all_predictions_window(ins, replace):
    if replace or not Path(ins.outdir / ins.pred_da_fn).exists():
        get_da_for_all_predictions_window_(ins)
    else:
        ins.phi_psi_predictions = pd.read_csv(ins.outdir / ins.pred_da_fn)
        ins.xray_phi_psi = pd.read_csv(ins.outdir / ins.xray_da_fn)
    
def get_da_for_all_predictions_window_(ins):
    ins.phi_psi_predictions['da'] = np.nan
    ins.phi_psi_predictions['n_samples'] = np.nan
    ins.phi_psi_predictions['n_samples_list'] = ''
    ins.xray_phi_psi['da'] = np.nan
    ins.xray_phi_psi['n_samples'] = np.nan
    ins.xray_phi_psi['n_samples_list'] = ''

    center_idx_ctxt = ins.queries[-1].get_center_idx_pos()
    winsize_ctxt = ins.queries[-1].winsize
    seqs_for_window = ins.seqs[center_idx_ctxt:-(winsize_ctxt - center_idx_ctxt - 1)]

    for i,seq_ctxt in enumerate(seqs_for_window):
        print(f'{i}/{len(ins.xray_phi_psi.seq_ctxt.unique())-1}: {seq_ctxt}')
        if 'X' in seq_ctxt:
            print(f'\tSkipping {seq_ctxt} - X in sequence')
            continue

        _, info = get_phi_psi_dist(ins.queries, seq_ctxt)
        for j in info:
            print(f'\tWin {j[0]}: {j[1]} - {j[2]} samples')

        # TODO n_samples

        q = ins.queries[0]
        xrays = get_xrays_window(ins, q, seq_ctxt)
        preds = get_preds_window(ins, q, seq_ctxt)
        afs = get_afs_window(ins, q, seq_ctxt)

        if xrays.shape[0] != q.winsize*2:
            print(f"Xray data for {seq_ctxt} is incomplete")
            continue
        if preds is None or preds.shape[0] == 0:
            print(f"No predictions for {seq_ctxt}")
            continue
        if afs is None or afs.shape[0] != q.winsize*2:
            print(f"AF data for {seq_ctxt} is incomplete")
            continue
        
        phi_psi_dist, phi_psi_dist_v = get_combined_phi_psi_dist(ins, seq_ctxt)
        if phi_psi_dist is None or phi_psi_dist.shape[0] == 0:
            print(f"No pdbmine data for {seq_ctxt}")
            continue
        if phi_psi_dist.shape[0] < MIN_SAMPLES[0]:
            print(f"Not enough pdbmine data for {seq_ctxt}")
            continue

        precomputed_dists = precompute_dists(phi_psi_dist_v)
        n_clusters, clusters = find_clusters(precomputed_dists, MIN_CLUSTER_SIZES[0])
        if n_clusters == 0:
            print(f"No clusters found for {seq_ctxt}")
            continue
        precomputed_dists, phi_psi_dist_v, clusters = filter_precomputed_dists(precomputed_dists, phi_psi_dist_v, clusters)
        target_cluster, target, icov = get_target_cluster_icov(phi_psi_dist_v, precomputed_dists, clusters, afs)
        if icov is None:
            print(f"Error calculating mahalanobis distance for {seq_ctxt}")
            continue

        xray_maha = calc_da_for_one_window(xrays, target, icov)
        preds_maha = calc_da_window(preds, target, icov)

        print(f'\t{i}: {xray_maha:.2f}, {np.nanmean(preds_maha):.2f}')

        # Distance from preds to xray
        col_name = f'da'
        ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq_ctxt, col_name] = xray_maha

        view = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq_ctxt].reset_index().set_index('protein_id')
        view.loc[preds.index, col_name] = preds_maha
        ins.phi_psi_predictions.loc[view['index'], col_name] = view.set_index('index')[col_name]

    ins.phi_psi_predictions.to_csv(ins.outdir / ins.pred_da_fn, index=False)
    ins.xray_phi_psi.to_csv(ins.outdir / ins.xray_da_fn, index=False)