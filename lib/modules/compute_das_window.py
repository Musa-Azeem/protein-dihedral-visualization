from lib.across_window_utils import (
    get_phi_psi_dist_window,
    get_xrays_window,
    get_afs_window,
    get_preds_window,
    precompute_dists,
    find_clusters,
    filter_precomputed_dists,
    calc_da_for_one_window,
    calc_da_window
)
from lib.utils import get_phi_psi_dist
import numpy as np
import pandas as pd

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
        for i in info:
            print(f'\tWin {i[0]}: {i[1]} - {i[2]} samples')

        # TODO n_samples

        # TODO handle all queries
        q = ins.queries[0]
        phi_psi_dist = get_phi_psi_dist_window(q, seq_ctxt)
        
        xrays = get_xrays_window(ins, q, seq_ctxt)
        preds = get_preds_window(ins, q, seq_ctxt)
        afs = get_afs_window(ins, q, seq_ctxt)

        if xrays.shape[0] != q.winsize*2:
            print(f"Xray data for {seq_ctxt} is incomplete")
            continue
        if preds.shape[0] == 0:
            print(f"No predictions for {seq_ctxt}")
            continue
        if phi_psi_dist.shape[0] == 0:
            print(f"No pdbmine data for {seq_ctxt}")
            continue
        if phi_psi_dist.shape[0] < 100:
            print(f"Not enough pdbmine data for {seq_ctxt}")
            continue
        if afs.shape[0] != q.winsize*2:
            print(f"AF data for {seq_ctxt} is incomplete")
            continue

        precomputed_dists = precompute_dists(phi_psi_dist.iloc[:,:q.winsize*2])
        n_clusters, clusters = find_clusters(phi_psi_dist, precomputed_dists)
        precomputed_dists, phi_psi_dist, clusters = filter_precomputed_dists(precomputed_dists, phi_psi_dist, clusters)

        xray_maha, c = calc_da_for_one_window(phi_psi_dist, xrays, precomputed_dists, clusters, afs)
        if xray_maha is None:
            print(f"Error calculating mahalanobis distance for {seq_ctxt}")
            print(f"Cluster size: {len(phi_psi_dist[phi_psi_dist.cluster == c])}")
            xray_maha = np.nan

        # Distance from preds to xray
        preds_maha = calc_da_window(phi_psi_dist, preds, precomputed_dists, clusters, afs)
        
        ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq_ctxt, 'da'] = xray_maha

        view = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq_ctxt].reset_index().set_index('protein_id')
        view.loc[preds.index, 'da'] = preds_maha
        ins.phi_psi_predictions.loc[view['index'], 'da'] = view.set_index('index')['da']

    ins.phi_psi_predictions.to_csv(ins.outdir / ins.pred_da_fn, index=False)
    ins.xray_phi_psi.to_csv(ins.outdir / ins.xray_da_fn, index=False)