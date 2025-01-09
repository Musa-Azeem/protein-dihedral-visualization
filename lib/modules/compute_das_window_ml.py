from lib.across_window_utils import (
    get_combined_phi_psi_dist,
    get_xrays_window,
    get_afs_window,
    get_preds_window,
    precompute_dists,
    find_clusters,
    filter_precomputed_dists,
    get_cluster_medoid
)
from lib.utils import get_phi_psi_dist
import numpy as np
import pandas as pd
from pathlib import Path
from lib.ml.models import MLPredictorWindow

MIN_SAMPLES = [100, 20, 1, 1]
MIN_CLUSTER_SIZES = [20, 5, 1, 1]

def get_da_for_all_predictions_window_ml(ins, replace):
    if replace or not Path(ins.outdir / ins.pred_da_fn).exists():
        get_da_for_all_predictions_window_ml_(ins)
    else:
        ins.phi_psi_predictions = pd.read_csv(ins.outdir / ins.pred_da_fn)
        ins.xray_phi_psi = pd.read_csv(ins.outdir / ins.xray_da_fn)
    
def get_da_for_all_predictions_window_ml_(ins):
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

        q = ins.queries[-1]
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
        
        phi_psi_dist, phi_psi_dist_v = get_combined_phi_psi_dist(ins, seq_ctxt, winsizes=[ins.winsizes[-1]])
        if phi_psi_dist is None or phi_psi_dist.shape[0] <= 1: # TODO seperate case for 1
            print(f"No pdbmine data for {seq_ctxt}")
            continue

        precomputed_dists = precompute_dists(phi_psi_dist_v)
        n_clusters, clusters = find_clusters(precomputed_dists, min_cluster_size=np.min([phi_psi_dist_v.shape[0], 20]))
        if n_clusters == 0:
            n_clusters, clusters = find_clusters(precomputed_dists, min_cluster_size=2, cluster_selection_epsilon=60)
        if n_clusters == 0:
            n_clusters, clusters = find_clusters(precomputed_dists, min_cluster_size=2, cluster_selection_epsilon=120)
        if n_clusters == 0:
            print(f"No clusters found for {seq_ctxt}")
            continue
        precomputed_dists, phi_psi_dist_v, clusters = filter_precomputed_dists(precomputed_dists, phi_psi_dist_v, clusters)

        # Get target phi psi for center residue with model
        cluster_counts = pd.Series(clusters).value_counts().sort_values(ascending=False)
        medoids = np.zeros([ins.ml_lengths[-1], ins.queries[-1].winsize*2])
        for k,cluster in zip(range(ins.ml_lengths[-1]), cluster_counts.index):
            medoid = get_cluster_medoid(phi_psi_dist_v, precomputed_dists, clusters, cluster)
            medoids[k] = medoid

        c_idx = q.get_center_idx_pos()
        target = ins.model.predict(medoids, seq_ctxt).numpy()[0,c_idx]

        def diff(x1, x2):
            d = np.abs(x1 - x2)
            return np.minimum(d, 360-d)
        preds_point = preds.values.reshape(-1, 2, ins.winsizes[-1]).transpose((0,2,1))[:,c_idx]
        xray_point = xrays.reshape(2, ins.winsizes[-1]).T[c_idx]
        xray_da = np.linalg.norm(diff(xray_point, target))
        preds_da = np.linalg.norm(diff(preds_point, target), axis=1)

        print(f'\t{i}: {xray_da:.2f}, {np.nanmean(preds_da):.2f}')

        # Distance from preds to xray
        col_name = f'da'
        ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq_ctxt, col_name] = xray_da

        view = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq_ctxt].reset_index().set_index('protein_id')
        view.loc[preds.index, col_name] = preds_da
        ins.phi_psi_predictions.loc[view['index'], col_name] = view.set_index('index')[col_name]

    ins.phi_psi_predictions.to_csv(ins.outdir / ins.pred_da_fn, index=False)
    ins.xray_phi_psi.to_csv(ins.outdir / ins.xray_da_fn, index=False)