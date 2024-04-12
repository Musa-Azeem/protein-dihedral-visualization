from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from tqdm import tqdm

phi_psi_mined_by_window = pd.read_csv('phi_psi_mined_by_window.csv')
phi_psi_predictions_by_window = pd.read_csv('phi_psi_predictions_by_window.csv')
xray_phi_psi = pd.read_csv('xray_phi_psi.csv')

mds = []
phi_psi_predictions_by_window['md'] = np.nan
xray_phi_psi['md'] = np.nan
for seq in tqdm(phi_psi_mined_by_window.seq.unique()):
    phi_psi_dist = phi_psi_mined_by_window.loc[phi_psi_mined_by_window.seq == seq][['phi','psi']]
    xray = xray_phi_psi[xray_phi_psi.seq == seq][['phi','psi']].values
    preds = phi_psi_predictions_by_window.loc[phi_psi_predictions_by_window.seq == seq][['phi','psi']].values

    if phi_psi_dist.shape[0] < 2 or xray.shape[0] < 1 or preds.shape[0] < 1:
        print(f'Skipping {seq} - not enough data points')
        xray_phi_psi.loc[xray_phi_psi.seq == seq, 'md'] = np.nan
        phi_psi_predictions_by_window.loc[phi_psi_predictions_by_window.seq == seq, 'md'] = np.nan        
        continue
    # Find clusters
    clustering = DBSCAN(eps=10, min_samples=3).fit(phi_psi_dist.values)
    phi_psi_dist['cluster'] = clustering.labels_

    # Find most probable data point and the cluster it belongs to
    kernel = gaussian_kde(phi_psi_dist[['phi','psi']].T)
    phi_psi_most_likely_idx = kernel(phi_psi_dist[['phi','psi']].T).argmax()
    phi_psi_c = phi_psi_dist.loc[phi_psi_dist.cluster == phi_psi_dist.iloc[phi_psi_most_likely_idx].cluster, ['phi','psi']].values

    # Mahalanobis distance to most common cluster
    cov = np.cov(phi_psi_c.T)
    if np.linalg.det(cov) == 0:
        print(f'Skipping {seq} - singular matrix')
        xray_phi_psi.loc[xray_phi_psi.seq == seq, 'md'] = np.nan
        phi_psi_predictions_by_window.loc[phi_psi_predictions_by_window.seq == seq, 'md'] = np.nan
        continue
    icov = np.linalg.inv(cov)

    # xray
    md_xray = (xray - phi_psi_c.mean(axis=0)) @ icov @ (xray - phi_psi_c.mean(axis=0)).T
    if np.any(md_xray < 0):
        md_xray = np.nan
    else:
        md_xray = np.sqrt(md_xray)[0,0]
    xray_phi_psi.loc[xray_phi_psi.seq == seq, 'md'] = md_xray

    # All predictions
    mean = phi_psi_dist[['phi','psi']].mean(axis=0).values
    md = (np.expand_dims((preds - mean), 1) @ icov @ np.expand_dims((preds - mean), 2)).squeeze()
    if np.any(md < 0):
        md = np.nan
    else:
        md = np.sqrt(md)
    phi_psi_predictions_by_window.loc[phi_psi_predictions_by_window.seq == seq, 'md'] = md

phi_psi_predictions_by_window.to_csv('phi_psi_predictions_by_window_md.csv', index=False)
xray_phi_psi.to_csv('xray_phi_psi_md.csv', index=False)