from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from tqdm import tqdm
# For one window
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

WINDOW_SIZE = 3
WINDOW_SIZE_CONTEXT = 7
casp_protein_id = 'T1024'
outdir = Path(f'csvs/{casp_protein_id}_win{WINDOW_SIZE}-{WINDOW_SIZE_CONTEXT}')

phi_psi_mined = pd.read_csv(outdir / f'phi_psi_mined_win{WINDOW_SIZE}.csv')
phi_psi_mined_ctxt = pd.read_csv(outdir / f'phi_psi_mined_win{WINDOW_SIZE_CONTEXT}.csv')
phi_psi_predictions = pd.read_csv(outdir / 'phi_psi_predictions.csv')
xray_phi_psi = pd.read_csv(outdir / 'xray_phi_psi.csv')

def get_md_for_all_predictions(eps=10):
    mds = []
    phi_psi_predictions['md'] = np.nan
    xray_phi_psi['md'] = np.nan
    for i,seq in tqdm(enumerate(phi_psi_mined_ctxt.seq.unique())):
        inner_seq = seq[WINDOW_SIZE_CONTEXT // 2 - WINDOW_SIZE // 2:WINDOW_SIZE_CONTEXT // 2 + WINDOW_SIZE // 2 + 1]
        phi_psi_dist = phi_psi_mined.loc[phi_psi_mined.seq == inner_seq][['phi','psi']]
        phi_psi_ctxt_dist = phi_psi_mined_ctxt.loc[phi_psi_mined_ctxt.seq == seq][['phi','psi']]
        print(phi_psi_dist.shape)
        print(phi_psi_ctxt_dist.shape)

        xray = xray_phi_psi[xray_phi_psi.seq_ctxt == seq][['phi','psi']].values
        preds = phi_psi_predictions.loc[phi_psi_predictions.seq_ctxt == seq][['phi','psi']].values

        if phi_psi_dist.shape[0] < 2 or xray.shape[0] < 1 or preds.shape[0] < 1:
            print(f'Skipping {seq} - not enough data points')
            xray_phi_psi.loc[xray_phi_psi.seq_ctxt == seq, 'md'] = np.nan
            phi_psi_predictions.loc[phi_psi_predictions.seq_ctxt == seq, 'md'] = np.nan        
            continue
        # Find clusters
        clustering = DBSCAN(eps=eps, min_samples=3).fit(phi_psi_dist.values)
        phi_psi_dist['cluster'] = clustering.labels_

        # Find most probable data point from context dist and the cluster it belongs to
        if phi_psi_ctxt_dist.shape[0] == 0:
            print('No context data - Using smaller window size')
            kernel = gaussian_kde(phi_psi_dist[['phi','psi']].T)
        elif phi_psi_ctxt_dist.shape[0] < 3:
            print('Not enough context data for KDE - Using smaller window size')
            kernel = gaussian_kde(phi_psi_dist[['phi','psi']].T)
        else:
            kernel = gaussian_kde(phi_psi_ctxt_dist[['phi','psi']].T)
        phi_psi_most_likely_idx = kernel(phi_psi_dist[['phi','psi']].T).argmax()
        phi_psi_c = phi_psi_dist.loc[phi_psi_dist.cluster == phi_psi_dist.iloc[phi_psi_most_likely_idx].cluster, ['phi','psi']].values

        # Mahalanobis distance to most common cluster
        cov = np.cov(phi_psi_c.T)
        if np.linalg.det(cov) == 0:
            print(f'Skipping {seq} - singular matrix')
            xray_phi_psi.loc[xray_phi_psi.seq_ctxt == seq, 'md'] = np.nan
            phi_psi_predictions.loc[phi_psi_predictions.seq == seq, 'md'] = np.nan
            continue
        icov = np.linalg.inv(cov)

        # xray
        md_xray = (xray - phi_psi_c.mean(axis=0)) @ icov @ (xray - phi_psi_c.mean(axis=0)).T
        if np.any(md_xray < 0):
            md_xray = np.nan
        else:
            md_xray = np.sqrt(md_xray)[0,0]
        xray_phi_psi.loc[xray_phi_psi.seq_ctxt == seq, 'md'] = md_xray

        # All predictions
        mean = phi_psi_dist[['phi','psi']].mean(axis=0).values
        md = (np.expand_dims((preds - mean), 1) @ icov @ np.expand_dims((preds - mean), 2)).squeeze()
        if np.any(md < 0):
            md = np.nan
        else:
            md = np.sqrt(md)
        phi_psi_predictions.loc[phi_psi_predictions.seq_ctxt == seq, 'md'] = md

    phi_psi_predictions.to_csv(outdir / f'phi_psi_predictions_md-eps{eps}.csv', index=False)
    xray_phi_psi.to_csv(outdir / f'xray_phi_psi_md-eps{eps}.csv', index=False)
eps=1.5
get_md_for_all_predictions(eps)