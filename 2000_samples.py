
from lib import DihedralAdherencePDB
from pathlib import Path
import os
from lib.across_window_utils import get_combined_phi_psi_dist, precompute_dists, find_clusters, filter_precomputed_dists
from lib.utils import get_phi_psi_dist
import numpy as np
import pandas as pd
PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")
PROJECT_DIR_OTHER = 'ml_data'

pdb_codes = [f.name.split('_')[0] for f in Path(PROJECT_DIR_OTHER).iterdir() if f.is_dir()]

from lib.across_window_utils import get_combined_phi_psi_dist, precompute_dists, find_clusters, filter_precomputed_dists
from lib.utils import get_phi_psi_dist
def get_n_clusters(ins, k):
    MIN_SAMPLES = [100, 20, 5, 5]
    MIN_CLUSTER_SIZES = [20, 5, 1, 1]

    center_idx_ctxt = ins.queries[-1].get_center_idx_pos()
    winsize_ctxt = ins.queries[-1].winsize
    seqs_for_window = ins.seqs[center_idx_ctxt:-(winsize_ctxt - center_idx_ctxt - 1)]

    seq_ctxt = None
    n_skipped = 0
    while seq_ctxt is None:
        if n_skipped > 30:
            print(f"Skipping {da.pdb_code}")
            return None, None, None, None
        n_skipped += 1
        seq_ctxt = np.random.choice(seqs_for_window, 1)[0]
        if 'X' in seq_ctxt:
            print(f'\tSkipping {seq_ctxt} - X in sequence')
            seq_ctxt = None
            continue
        phi_psi_dist, phi_psi_dist_v = get_combined_phi_psi_dist(ins, seq_ctxt)
        if phi_psi_dist is None or phi_psi_dist.shape[0] == 0:
            print(f"No pdbmine data for {seq_ctxt}")
            seq_ctxt = None
            continue
        if phi_psi_dist.shape[0] < MIN_SAMPLES[k]:
            print(f"Not enough pdbmine data for {seq_ctxt} ({phi_psi_dist.shape[0]})")
            seq_ctxt = None
            continue
    if phi_psi_dist.shape[0] > 2500:
        phi_psi_dist = phi_psi_dist.sample(2500)
    _, info = get_phi_psi_dist(ins.queries, seq_ctxt)
    print(f'\tWin {info[k][0]}: {info[k][1]} - {info[k][2]} samples')
    precomputed_dists = precompute_dists(phi_psi_dist_v)
    n_clusters, clusters = find_clusters(precomputed_dists, MIN_CLUSTER_SIZES[0])
    return da.pdb_code, seq_ctxt, info[k][2], n_clusters

winsizes = [4,5,6,7]
n_seqs = 1000
seqs_per_protein = 8
# for k in [0, 1, 2, 3]:
for k in [2]:
    past_proteins = pd.read_csv(f'win{winsizes[k]}-clusters-3000.csv')['protein'].unique()
    pdb_codes_2 = np.setdiff1d(np.array(pdb_codes), past_proteins)
    samples = np.random.choice(pdb_codes, n_seqs // seqs_per_protein, replace=False)

    results = []
    n_skipped = 0
    kdews = [0, 0, 0, 0]
    kdews[k] = 1
    for i,sample in enumerate(samples):
        print(f'{i}/{len(samples)-1}: {sample}')
        da = DihedralAdherencePDB(
            sample, [4,5,6,7], PDBMINE_URL, PROJECT_DIR_OTHER, mode='full_window',
            kdews=kdews
        )
        if (not da.has_af or
            len([d for d in da.outdir.iterdir()]) == 0 or 
            not (da.outdir / 'xray_phi_psi.csv').exists()
        ):
            n_skipped += 1
            continue
        if not da.load_results():
            n_skipped += 1
            continue
        if da.xray_phi_psi.shape[0] == 0 or len(da.seqs) <= 6:
            n_skipped += 1
            continue
        for i in range(seqs_per_protein):
            ret = get_n_clusters(da, k)
            if ret[0] is None:
                n_skipped += 1
                break
            results.append(ret)
        results_df = pd.DataFrame(results, columns = ['protein', 'seq_ctxt', 'n_samples', 'n_clusters'])
        results_df.to_csv(f'win{winsizes[k]}-clusters-500.csv', index=False)