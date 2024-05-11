from pathlib import Path
from Bio.PDB import PDBList, PDBParser
import os
from tqdm import tqdm
import warnings
import pandas as pd
from Bio import SeqIO
from dotenv import load_dotenv
import os
import requests
import time
import json
from Bio.PDB.ic_rebuild import structure_rebuild_test
import numpy as np
import sys
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from scipy.stats import linregress

load_dotenv()
amino_acid_codes = json.load(open('amino_acid_codes.json'))
PDBMINE_URL = os.getenv("PDBMINE_URL")

WINDOW_SIZE, WINDOW_SIZE_CONTEXT, eps, bw_method = sys.argv[1:]
WINDOW_SIZE = int(WINDOW_SIZE)
WINDOW_SIZE_CONTEXT = int(WINDOW_SIZE_CONTEXT)
eps = float(eps)
bw_method = float(bw_method) if bw_method.lower() != 'none' else None
print(WINDOW_SIZE, WINDOW_SIZE_CONTEXT, eps, bw_method)

# Protein info
casp_protein_id = 'T1030'
casp_protein_id2 = 'T1030'
pdb_code = '6poo'

# casp_protein_id = 'T1024'
# casp_protein_id2 = 'T1024'
# pdb_code = '6t1z'

outdir = Path(f'tests/{casp_protein_id}_win{WINDOW_SIZE}-{WINDOW_SIZE_CONTEXT}')
if outdir.exists():
    print('Results already exist')
else:
    outdir.mkdir(exist_ok=False, parents=True)
outfile = Path('results.txt')

# Helpers
def get_center(seq):
    if WINDOW_SIZE % 2 == 0:
        return seq[WINDOW_SIZE // 2 - 1]
    else:
        return seq[-WINDOW_SIZE // 2]
def get_seq(i):
    if WINDOW_SIZE % 2 == 0:
        if WINDOW_SIZE_CONTEXT % 2 == 0:
            return slice(i-WINDOW_SIZE//2+1,i+WINDOW_SIZE//2+1)
        return slice(i-WINDOW_SIZE//2,i+WINDOW_SIZE//2)
    else:
        return slice(i-WINDOW_SIZE//2,i+WINDOW_SIZE//2+1)
def get_seq_ctxt(i):
    if WINDOW_SIZE_CONTEXT % 2 == 0:
        return slice(i-WINDOW_SIZE_CONTEXT//2+1,i+WINDOW_SIZE_CONTEXT//2+1)
    return slice(i-WINDOW_SIZE_CONTEXT//2,i+WINDOW_SIZE_CONTEXT//2+1)
def get_subseq(seq):
    if WINDOW_SIZE % 2 == 0:
        return seq[WINDOW_SIZE_CONTEXT//2 - WINDOW_SIZE//2:WINDOW_SIZE_CONTEXT//2 + WINDOW_SIZE//2]
    else:
        if WINDOW_SIZE_CONTEXT % 2 == 0:
            return seq[WINDOW_SIZE_CONTEXT//2 - WINDOW_SIZE//2-1:WINDOW_SIZE_CONTEXT//2 + WINDOW_SIZE//2]
        return seq[WINDOW_SIZE_CONTEXT//2 - WINDOW_SIZE//2:WINDOW_SIZE_CONTEXT//2 + WINDOW_SIZE//2 + 1]

## --------------------------- Retrieve Data ---------------------------------
# Get X-ray pdb
pdbl = PDBList()
parser = PDBParser()
xray_fn = pdbl.retrieve_pdb_file(pdb_code, pdir='pdb', file_format='pdb', obsolete=False)

# Get CASP predictions
predictions_url = f'https://predictioncenter.org/download_area/CASP14/predictions/regular/{casp_protein_id}.tar.gz'
predictions_dir = Path(f'./casp-predictions/')
if not (predictions_dir / casp_protein_id).exists():
    predictions_dir.mkdir(exist_ok=True)
    os.system(f'wget -O {predictions_dir}/{casp_protein_id}.tar.gz {predictions_url}')
    os.system(f'tar -xvf {predictions_dir}/{casp_protein_id}.tar.gz -C {predictions_dir}')

# Get CASP results
results_url = 'https://predictioncenter.org/download_area/CASP14/results/tables/casp14.res_tables.T.tar.gz'
results_dir = Path('casp-results')
if not results_dir.exists():
    results_dir.mkdir(exist_ok=True)
    os.system(f'wget -O {results_dir / "casp14.res_tables.T.tar.gz"} {results_url}')
    os.system(f'tar -xvf {results_dir / "casp14.res_tables.T.tar.gz"} -C {results_dir}')
results_file = results_dir / f'{casp_protein_id2}.txt'
results = pd.read_csv(results_file, delim_whitespace=True)
results = results[results.columns[1:]]
results['Model'] = results['Model'].apply(lambda x: x.split('-')[0])

# ------------------ Collect Dihedrals ---------------------------------------
xray_structure = parser.get_structure(pdb_code, xray_fn)
xray_chain = list(xray_structure[0].get_chains())[0]

def get_phi_psi_for_structure(protein_structure, protein_id):
    protein_structure.atom_to_internal_coordinates(verbose=False)
    resultDict = structure_rebuild_test(protein_structure)
    if not resultDict['pass']:
        raise Exception('Failed to rebuild')
    residues = list(protein_structure.get_residues())
    phi_psi_ = []
    for i in range(WINDOW_SIZE_CONTEXT//2, len(residues) - WINDOW_SIZE_CONTEXT // 2):
        # Convert 3 char codes to 1 char codes
        seq = ''.join([amino_acid_codes.get(r.resname, 'X') for r in residues[get_seq(i)]])
        seq_ctxt = ''.join([amino_acid_codes.get(r.resname, 'X') for r in residues[get_seq_ctxt(i)]])
        # Get the center residue
        res = get_center(seq)
        if not residues[i].internal_coord:
            psi,phi = np.nan, np.nan
        else:
            psi = residues[i].internal_coord.get_angle("psi")
            phi = residues[i].internal_coord.get_angle("phi")
            psi = psi if psi else np.nan # if psi is None, set it to np.nan
            phi = phi if phi else np.nan # if phi is None, set it to np.nan
        phi_psi_.append([i, seq, seq_ctxt, res, phi, psi, xray_chain.id, protein_id])
    return phi_psi_

if not (outdir / 'xray_phi_psi.csv').exists():
    xray_phi_psi = get_phi_psi_for_structure(xray_chain, pdb_code)
    xray_phi_psi = pd.DataFrame(xray_phi_psi, columns=['pos', 'seq', 'seq_ctxt', 'res', 'phi', 'psi', 'chain', 'protein_id'])
    xray_phi_psi.to_csv(outdir / 'xray_phi_psi.csv', index=False)
else:
    xray_phi_psi = pd.read_csv(outdir / 'xray_phi_psi.csv')

# Get phi_psi's of each prediction
if not (outdir / 'phi_psi_predictions.csv').exists():

    phi_psi_predictions_ = []
    for prediction_pdb in tqdm((predictions_dir / casp_protein_id).iterdir()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = parser.get_structure(prediction_pdb.name, prediction_pdb)
            try:
                chain = list(prediction[0].get_chains())[0]
                phi_psi_predictions_ += get_phi_psi_for_structure(chain, prediction.id)
            except Exception as e:
                print(e)

    phi_psi_predictions = pd.DataFrame(phi_psi_predictions_, columns=['pos', 'seq', 'seq_ctxt', 'res', 'phi', 'psi', 'chain', 'protein_id'])
    phi_psi_predictions.to_csv(outdir / 'phi_psi_predictions.csv', index=False)
else:
    phi_psi_predictions = pd.read_csv(outdir / 'phi_psi_predictions.csv')

# ------------------------ PDBMine -------------------------------------------
# Get Phi-Psi distribution from PDBMine

def query_pdbmine(window_size):
    record = next(iter(SeqIO.parse(xray_fn, "pdb-seqres")))
    residue_chain = str(record.seq)

    code_length = 1
    broken_chains = []

    # break chain into sections of length 100 - for memory reasons
    # overlap by window_size-1
    for i in range(0, len(residue_chain), 100-window_size+1):
        broken_chains.append(residue_chain[i:i+100])

    match_outdir = Path(f'cache/{casp_protein_id}/matches-{window_size}')
    match_outdir.mkdir(exist_ok=False, parents=True)

    for i,chain in enumerate(tqdm(broken_chains)):
        if len(chain) < window_size: # in case the last chain is too short
            continue

        response = requests.post(
            PDBMINE_URL + '/v1/api/query',
            json={
                "residueChain": chain,
                "codeLength": code_length,
                "windowSize": window_size
            }
        )
        assert(response.ok)
        print(response.json())
        query_id = response.json().get('queryID')
        assert(query_id)

        time.sleep(60)
        while(True):
            response = requests.get(PDBMINE_URL + f'/v1/api/query/{query_id}')
            if response.ok:
                matches = response.json()['frames']
                break
            else:
                print('Waiting')
                time.sleep(15)
        print(f'Received matches - {i}')
        json.dump(matches, open(match_outdir / f'matches-win{window_size}_{i}.json', 'w'), indent=4)
if not Path(f'cache/{casp_protein_id}/matches-{WINDOW_SIZE}').exists():
    query_pdbmine(WINDOW_SIZE)
if not Path(f'cache/{casp_protein_id}/matches-{WINDOW_SIZE_CONTEXT}').exists():
    query_pdbmine(WINDOW_SIZE_CONTEXT)

# Get phi-psi from PDBMine matches
# If any sequence appears twice, only take the first one bc the distribution is the same
def get_phi_psi_mined(window_size):
    seqs = []
    phi_psi_mined = []
    for matches in Path(f'cache/{casp_protein_id}/matches-{window_size}').iterdir():
        matches = json.load(matches.open())
        for seq_win,v in matches.items():
            seq = seq_win[4:]
            if seq in seqs:
                continue
            seqs.append(seq)
            for protein,seq_matches in v.items():
                protein_id, chain = protein.split('_')
                if protein_id.lower() == pdb_code.lower(): # skip the protein we're looking at
                    continue
                for seq_match in seq_matches:
                    center_res = seq_match[window_size//2]
                    res, phi, psi = center_res.values()
                    phi_psi_mined.append([seq, res, phi, psi, chain, protein_id])
    phi_psi_mined = pd.DataFrame(phi_psi_mined, columns=['seq', 'res', 'phi', 'psi', 'chain', 'protein_id'])
    phi_psi_mined.to_csv(outdir / f'phi_psi_mined_win{window_size}.csv', index=False)
    return phi_psi_mined
phi_psi_mined = get_phi_psi_mined(WINDOW_SIZE)
phi_psi_mined_ctxt = get_phi_psi_mined(WINDOW_SIZE_CONTEXT)

# ------------------------- Compute Maha --------------------------------------

def find_phi_psi_c(phi_psi_dist, phi_psi_ctxt_dist, eps, bw_method, kdews):
    
    # combine with weights
    phi_psi_dist['weight'] = kdews[0]
    phi_psi_ctxt_dist['weight'] = kdews[1]
    phi_psi_dist = pd.concat([phi_psi_dist, phi_psi_ctxt_dist])

    # Cluster with DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=3).fit(phi_psi_dist.values)
    phi_psi_dist['cluster'] = clustering.labels_

    # Find most probable cluster
    kernel = gaussian_kde(phi_psi_dist[['phi','psi']].T, weights=phi_psi_dist['weight'], bw_method=bw_method)
    most_likely = phi_psi_dist.iloc[kernel(phi_psi_dist[['phi', 'psi']].values.T).argmax()]
    phi_psi_dist['prob'] = kernel(phi_psi_dist[['phi', 'psi']].values.T)
    c = phi_psi_dist[phi_psi_dist.cluster != -1].groupby('cluster').sum(numeric_only=True)
    if c.shape[0] == 0:
        print('No clusters found, using entire dist')
        phi_psi_dist_c = phi_psi_dist
    else:
        phi_psi_dist_c = phi_psi_dist[phi_psi_dist.cluster == c.prob.idxmax()]

    return phi_psi_dist, phi_psi_dist_c, most_likely

def get_md_for_all_predictions(eps=10, bw_method=None, kdews=None):
    kdews = kdews or [1,128]
    phi_psi_predictions['md'] = np.nan
    xray_phi_psi['md'] = np.nan
    for i,seq in enumerate(phi_psi_predictions.seq_ctxt.unique()):
        inner_seq = get_subseq(seq)
        phi_psi_dist = phi_psi_mined.loc[phi_psi_mined.seq == inner_seq][['phi','psi']]
        phi_psi_ctxt_dist = phi_psi_mined_ctxt.loc[phi_psi_mined_ctxt.seq == seq][['phi','psi']]
        print(f'{seq}: {phi_psi_dist.shape[0]} {phi_psi_ctxt_dist.shape[0]}')

        if phi_psi_ctxt_dist.shape[0] > 2:
            print('Enough context data for KDE - Using Full Context')
        if phi_psi_dist.shape[0] <= 2:
            print(f'Skipping {seq} - not enough data points')
            # leave as nan
            continue

        xray = xray_phi_psi[xray_phi_psi.seq_ctxt == seq][['phi','psi']].values
        preds = phi_psi_predictions.loc[phi_psi_predictions.seq_ctxt == seq][['phi','psi']].values

        phi_psi_dist, phi_psi_dist_c, most_likely = find_phi_psi_c(phi_psi_dist, phi_psi_ctxt_dist, eps, bw_method, kdews)
        phi_psi_c = phi_psi_dist_c[['phi', 'psi']].values
        print(xray.shape, preds.shape, phi_psi_dist.shape, phi_psi_ctxt_dist.shape, phi_psi_dist_c.shape)

        # Mahalanobis distance to most common cluster
        cov = np.cov(phi_psi_c.T)
        if np.linalg.det(cov) == 0:
            print(f'Skipping {seq} - singular matrix')
            # leave as nan
            continue
        icov = np.linalg.inv(cov)
        mean = phi_psi_c.mean(axis=0)

        md_xray = np.nan
        if xray.shape[0] > 0:
            # xray
            md_xray = (xray - mean) @ icov @ (xray - mean).T
            if np.any(md_xray < 0):
                md_xray = np.nan
            else:
                md_xray = np.sqrt(md_xray)[0,0]
            xray_phi_psi.loc[xray_phi_psi.seq_ctxt == seq, 'md'] = md_xray
        else:
            print(f'No xray seq {seq}')

        # All predictions
        if preds.shape[0] > 0:
            md = (np.expand_dims((preds - mean), 1) @ icov @ np.expand_dims((preds - mean), 2)).squeeze()
            if np.any(md < 0):
                md = np.nan
            else:
                md = np.sqrt(md)
            phi_psi_predictions.loc[phi_psi_predictions.seq_ctxt == seq, 'md'] = md
        else:
            print(f'No predictions seq {seq}')

    phi_psi_predictions.to_csv(outdir / f'phi_psi_predictions_md-eps{eps}.csv', index=False)
    xray_phi_psi.to_csv(outdir / f'xray_phi_psi_md-eps{eps}.csv', index=False)
        
kdews=[1,128]
get_md_for_all_predictions(eps, bw_method, kdews)

def filter_and_sum(series):
    series = series[series < series.quantile(0.80)]
    return series.sum()

def plot_md_vs_rmsd(rmsd_lim=np.inf, md_lim_low=0, md_lim=np.inf):
    group_maha = phi_psi_predictions.groupby('protein_id', as_index=False).agg({'md': filter_and_sum})
    group_maha = group_maha.merge(results[['Model', 'RMS_CA']], left_on='protein_id', right_on='Model', how='inner')
    group_maha = group_maha[group_maha.md_na <= group_maha.md_na.quantile(0.90)]
    group_maha = group_maha[(group_maha.RMS_CA < rmsd_lim) & (group_maha.md > md_lim_low) & (group_maha.md < md_lim)].dropna()

    regr = linregress(group_maha.md, group_maha.RMS_CA)
    print(f"R-squared: {regr.rvalue**2:.6f}")
    return regr.rvalue**2

rsquared = plot_md_vs_rmsd()
print(f'{WINDOW_SIZE},{WINDOW_SIZE_CONTEXT},{rsquared}')

with outfile.open('a') as f:
    print(f'{WINDOW_SIZE},{WINDOW_SIZE_CONTEXT},{rsquared}', file=f)