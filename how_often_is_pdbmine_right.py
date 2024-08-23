from lib import DihedralAdherence
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lib.fit_model_on_multiple import fit_lr, predict_lr, fit_rf, plot_md_vs_rmsd, predict_rf
from pathlib import Path
from lib import DihedralAdherence
from lib import PDBMineQuery
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import linregress, pearsonr
PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")
PROJECT_DIR = 'casp_da'

from lib.utils import get_phi_psi_dist, find_kdepeak, calc_da
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

max_clusters = 7

proteins = [
	'T1024', 'T1030', 'T1030-D2', 'T1024-D1', 'T1032-D1', 'T1053-D1', 'T1027-D1', 'T1029-D1',
	'T1025-D1', 'T1028-D1', 'T1030-D1', 'T1053-D2', 'T1057-D1','T1058-D1', 'T1058-D2'
]

for protein in proteins:
    da = DihedralAdherence(protein, [4,5,6,7], PDBMINE_URL, PROJECT_DIR, kdews=[1,32,64,128], 
                           mode='ml', weights_file='ml_runs/best_model-kde_16-32_383.pt', device='cpu')
    da.load_results()
    rows = []
    for seq in tqdm(da.xray_phi_psi.seq_ctxt.unique()):
        if 'X' in seq:
            print('Skipping', protein, seq, '\t[Contains X]')
            continue
        phi_psi_dist, info = get_phi_psi_dist(da.queries, seq)
        xray = da.xray_phi_psi.loc[da.xray_phi_psi.seq_ctxt == seq][['phi','psi']]
        if xray.shape[0] == 0:
            print('Skipping', protein, seq, '\t[No Xray]')
            continue
        xray = xray.iloc[0].values

        max_sil_avg = -1
        one_cluster_wss = -1
        for k in range(1, min(phi_psi_dist.shape[0], max_clusters)):
            kmeans = KMeans(n_clusters=k, n_init=10)
            labels = kmeans.fit_predict(phi_psi_dist[['phi', 'psi']])
            wss = kmeans.inertia_
            if k == 1:
                one_cluster_wss = wss
                phi_psi_dist['cluster'] = labels
                continue

            elif k == 2:
                if wss >= one_cluster_wss:
                    break

            sil_avg = silhouette_score(phi_psi_dist[['phi', 'psi']], labels)
            if sil_avg > max_sil_avg:
                max_sil_avg = sil_avg
                phi_psi_dist['cluster'] = labels
        
        peaks = []
        cs = phi_psi_dist.cluster.unique()
        for c in cs:
            kdepeak = find_kdepeak(phi_psi_dist.loc[phi_psi_dist.cluster == c], 0.5)
            peaks.append(kdepeak)
        peaks = np.array(peaks)

        dists = calc_da(xray, peaks)
        min_dist = np.min(dists)
        chosen_peak = peaks[np.argmin(dists)]
        chosen_c = cs[np.argmin(dists)]
        cluster = phi_psi_dist.loc[phi_psi_dist.cluster == chosen_c, ['phi', 'psi']].values
        cluster_std = np.std(cluster, axis=0)

        n_samples = [infoi[2] for infoi in info]
        dists = list(np.pad(dists, (0, max_clusters - len(dists)), 'constant', constant_values=-1).round(3))
        rows.append([
            protein, seq, min_dist.round(3), 
            xray[0].round(3), xray[1].round(3), 
            chosen_peak[0].round(3), chosen_peak[1].round(3),
            max_sil_avg,
            *cluster_std, *n_samples, *dists
        ])

        # sns.scatterplot(x='phi', y='psi', hue='cluster', data=phi_psi_dist)
        # plt.scatter([p[0] for p in peaks], [p[1] for p in peaks], marker='x')
        # plt.scatter(xray[0], xray[1], marker='x')
        # plt.show()

    df = pd.DataFrame(rows, columns=[
        'protein', 'seq', 'min_dist', 'xray_phi', 'xray_psi', 
        'chosen_peak_phi', 'chosen_peak_psi', 
        'cluster_std_phi', 'cluster_std_psi',
        'kmeans_silhouette',
        'win4', 'win5', 'win6', 'win7', 
        'dist_1', 'dist_2', 'dist_3', 'dist_4', 'dist_5', 'dist_6', 'dist_7'
    ])
    df.to_csv(f'quantative_analysis.csv', index=False)