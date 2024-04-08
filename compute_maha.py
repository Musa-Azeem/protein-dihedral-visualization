import numpy as np

def get_mahalanobis_dist(x, dist):
    # x should be 1 x n_features
    # dist should be n_obs x n_features
    icov = np.linalg.inv(np.cov(dist.T))
    md = (x - dist.mean(axis=0)) @ icov @ (x - dist.mean(axis=0))
    if md < 0:
        print(x, dist.mean(axis=0), icov)
    return np.sqrt(md)

def get_maha_for_row(dist_df, row):
    res = row['res']
    if row.name % 10000 == 0:
        print(row.name)
    if res == 'X':
        return np.nan
    phi_psi = dist_df.loc[dist_df.res == res, ['phi','psi']].dropna().values
    return get_mahalanobis_dist(row[['phi','psi']].values, phi_psi)

import warnings
import pandas as pd
pdb_code = '6t1z'

phi_psi_mined = pd.read_csv('phi_psi_mined.csv')
phi_psi_predictions = pd.read_csv('phi_psi_predictions.csv')

phi_psi_mined_filtered = phi_psi_mined.copy()
phi_psi_mined_filtered = phi_psi_mined_filtered[phi_psi_mined_filtered.protein_id != pdb_code.upper()]
phi_psi_mined_filtered['source'] = 'Query (PDBMine)'
phi_psi_predictions['source'] = 'Prediction'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    phi_psi_predictions['mahalanobis_dist'] = phi_psi_predictions.apply(lambda x: get_maha_for_row(phi_psi_mined_filtered, x), axis=1)

phi_psi_predictions.to_csv('phi_psi_predictions_maha.csv', index=False)