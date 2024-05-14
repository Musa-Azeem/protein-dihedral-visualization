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
import seaborn as sns
from Bio.Align import PairwiseAligner
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import statsmodels.api as sm
import re
import sys
load_dotenv()

PDBMINE_URL = os.getenv("PDBMINE_URL")

WINDOW_SIZE, WINDOW_SIZE_CONTEXT, casp_protein_id = sys.argv[1:]
WINDOW_SIZE = int(WINDOW_SIZE)
WINDOW_SIZE_CONTEXT = int(WINDOW_SIZE_CONTEXT)

## --------------------------- Get metadata ---------------------------
amino_acid_codes = json.load(open('amino_acid_codes.json'))

targetlist_url = 'https://predictioncenter.org/casp14/targetlist.cgi?type=csv'
targetlist_file = Path('targetlist.csv')
if not targetlist_file.exists():
    with open(targetlist_file, 'wb') as f:
        f.write(requests.get(targetlist_url).content)
targetlist = pd.read_csv(targetlist_file, sep=';').set_index('Target')
def get_pdb_code(x):
    m = re.search(r"\b\d[0-9a-z]{3}\b", x)
    return m.group() if m else ''
targetlist['pdb_code'] = targetlist['Description'].apply(get_pdb_code)

pdb_code = targetlist.loc[casp_protein_id].pdb_code
alphafold_id = f'{casp_protein_id}TS427_1'
outdir = Path(f'tests/{casp_protein_id}_win{WINDOW_SIZE}-{WINDOW_SIZE_CONTEXT}')
if outdir.exists():
    print('Results already exist')
else:
    outdir.mkdir(exist_ok=False, parents=True)
outfile = Path('results.txt')

print(WINDOW_SIZE, WINDOW_SIZE_CONTEXT, casp_protein_id, pdb_code)

## ------------------------------ Helpers ----------------------------
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
results_file = results_dir / f'{casp_protein_id}.txt'
if not results_file.exists():
    results_file = Path(results_dir / f'{casp_protein_id}-D1.txt')
results = pd.read_csv(results_file, sep='\s+')
results = results[results.columns[1:]]
results['Model'] = results['Model'].apply(lambda x: x.split('-')[0])


## ------------------ Collect Dihedrals ---------------------------------------
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

## ------------------------- Compute Maha --------------------------------------

def find_phi_psi_c(phi_psi_dist, phi_psi_ctxt_dist, bw_method, kdews):
    # combine with weights
    phi_psi_dist['weight'] = kdews[0]
    phi_psi_ctxt_dist['weight'] = kdews[1]
    phi_psi_dist = pd.concat([phi_psi_dist, phi_psi_ctxt_dist])

    # Find probability of each point
    kernel = gaussian_kde(phi_psi_dist[['phi','psi']].T, weights=phi_psi_dist['weight'], bw_method=bw_method)
    most_likely = phi_psi_dist.iloc[kernel(phi_psi_dist[['phi', 'psi']].values.T).argmax()]
    phi_psi_dist['prob'] = kernel(phi_psi_dist[['phi', 'psi']].values.T)

    # cluster with kmeans
    max_sil_avg = -1
    for k in range(2, min(phi_psi_dist.shape[0], 7)):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(phi_psi_dist[['phi', 'psi']])
        sil_avg = silhouette_score(phi_psi_dist[['phi', 'psi']], labels)
        if sil_avg > max_sil_avg:
            max_sil_avg = sil_avg
            phi_psi_dist['cluster'] = labels
    
    # Find most probable cluster
    c = phi_psi_dist.groupby('cluster').sum(numeric_only=True)
    phi_psi_dist_c = phi_psi_dist[phi_psi_dist.cluster == c.prob.idxmax()]
    if phi_psi_dist_c.shape[0] < 3:
        print('Too few points in cluster - using entire dist')
        phi_psi_dist_c = phi_psi_dist

    print('Chosen dist:', phi_psi_dist_c[['phi', 'psi']].shape, phi_psi_dist.phi.mean(), phi_psi_dist.psi.mean())    
    return phi_psi_dist, phi_psi_dist_c, most_likely

def calc_maha_for_one(phi_psi: np.ndarray, phi_psi_dist: np.ndarray, kdepeak):    
    cov = np.cov(phi_psi_dist.T)
    if np.diag(cov).min() < 1:
        print('No significant variance in distribution - using distance to kde peak')
        return np.sqrt((phi_psi[0] - kdepeak[0])**2 + (phi_psi[1] - kdepeak[1])**2)
    if np.linalg.det(cov) == 0:
        print('Singular covariance matrix - using distance to kde peak')
        return np.sqrt((phi_psi[0] - kdepeak[0])**2 + (phi_psi[1] - kdepeak[1])**2)
    
    icov = np.linalg.inv(cov)
    mean = phi_psi_dist.mean(axis=0)
    return np.sqrt((phi_psi - mean) @ icov @ (phi_psi - mean).T)

def calc_maha(phi_psi_preds, phi_psi_dist, kdepeak):
    cov = np.cov(phi_psi_dist.T)
    if np.diag(cov).min() < 1:
        print('No significant variance in distribution - using distance to kde peak')
        return np.sqrt(((phi_psi_preds[:,0] - kdepeak[0])**2) + ((phi_psi_preds[:,1] - kdepeak[1])**2))
    if np.linalg.det(cov) == 0:
        print('Singular covariance matrix - using distance to kde peak')
        return np.sqrt(((phi_psi_preds[:,0] - kdepeak[0])**2) + ((phi_psi_preds[:,1] - kdepeak[1])**2))
    
    icov = np.linalg.inv(cov)
    diff = phi_psi_preds - phi_psi_dist.mean(axis=0)
    return np.sqrt((np.expand_dims((diff), 1) @ icov @ np.expand_dims((diff), 2)).squeeze())

def pre_md_filter():
    # remove sequences missing phi and psi
    global xray_phi_psi, phi_psi_predictions
    xray_phi_psi = xray_phi_psi[~xray_phi_psi.phi.isna() & ~xray_phi_psi.psi.isna()]
    # remove all predictions with outlier overlapping sequences with xray
    xray_seqs_unique = set(xray_phi_psi.seq_ctxt.unique())
    grouped = phi_psi_predictions.groupby('protein_id').agg(
        overlapping_seqs=('seq_ctxt', lambda series: len(set(series.unique()) & xray_seqs_unique)),
        length=('seq_ctxt', 'count')
    )
    overlapping_seqs = grouped.overlapping_seqs[grouped.overlapping_seqs == grouped.overlapping_seqs.mode().values[0]]
    lengths = grouped.length[grouped.length == grouped.length.mode().values[0]]
    phi_psi_predictions = phi_psi_predictions[(phi_psi_predictions.protein_id.isin(overlapping_seqs.index)) & (phi_psi_predictions.protein_id.isin(lengths.index))]

def get_md_for_all_predictions(bw_method=None, kdews=None):
    kdews = kdews or [1,128]
    phi_psi_predictions['md'] = np.nan
    xray_phi_psi['md'] = np.nan
    for i,seq in enumerate(xray_phi_psi.seq_ctxt.unique()):
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

        phi_psi_dist, phi_psi_dist_c, most_likely = find_phi_psi_c(phi_psi_dist, phi_psi_ctxt_dist, bw_method, kdews)

        # Mahalanobis distance to most common cluster
        xray = xray_phi_psi[xray_phi_psi.seq_ctxt == seq][['phi','psi']]
        if xray.shape[0] == 0:
            print(f'No xray seq {seq}')
        else:
            md_xray = calc_maha_for_one(xray[['phi','psi']].values[0], phi_psi_dist_c[['phi','psi']].values, most_likely[['phi', 'psi']].values)
            xray_phi_psi.loc[xray_phi_psi.seq_ctxt == seq, 'md'] = md_xray
            
        preds = phi_psi_predictions.loc[phi_psi_predictions.seq_ctxt == seq][['phi','psi']]
        if preds.shape[0] == 0:
            print(f'No predictions seq {seq}')
        else:
            md = calc_maha(preds[['phi','psi']].values, phi_psi_dist_c[['phi','psi']].values, most_likely[['phi', 'psi']].values)
            phi_psi_predictions.loc[phi_psi_predictions.seq_ctxt == seq, 'md'] = md
        print(xray.shape, preds.shape, phi_psi_dist.shape, phi_psi_ctxt_dist.shape)

    phi_psi_predictions.to_csv(outdir / f'phi_psi_predictions_md-kmeans.csv', index=False)
    xray_phi_psi.to_csv(outdir / f'xray_phi_psi_md-kmeans.csv', index=False)

pre_md_filter()
get_md_for_all_predictions()

def filter_and_agg(series, agg='sum', quantile=0.8):
    series = series[series < series.quantile(quantile)]
    return series.agg(agg)

def calc_perc_na(series):
    return series.sum() / len(series)

def ols_md_vs_rmsd(rmsd_lim_low=0, rmsd_lim_high=np.inf, md_lim_low=0, md_lim_high=np.inf):
    phi_psi_predictions['md_na'] = phi_psi_predictions.md.isna()
    group_maha = phi_psi_predictions.groupby('protein_id', as_index=False).agg(
        md=('md',lambda x: filter_and_agg(x, agg='sum', quantile=1)), 
        std_md=('md',lambda x: filter_and_agg(x, agg='std', quantile=1)), 
        md_na=('md_na',calc_perc_na),
        mds=('md', list)
    )
    group_maha = group_maha.merge(results[['Model', 'RMS_CA']], left_on='protein_id', right_on='Model', how='inner')
    
    group_maha = group_maha[group_maha.md_na <= group_maha.md_na.quantile(.9)]
    group_maha = group_maha[
        (group_maha.RMS_CA > rmsd_lim_low) & \
        (group_maha.RMS_CA < rmsd_lim_high) & \
        (group_maha.md > md_lim_low) &\
        (group_maha.md < md_lim_high)
    ].dropna()

    X = np.array(group_maha.mds.values.tolist())
    y = group_maha.RMS_CA.values
    X = np.where(np.isnan(X), np.nanmean(X,axis=0), X)
    X[np.isnan(X)] = 0  # only nans left are where all values are nan

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    group_maha['rms_pred'] = model.predict(X)

    return model.rsquared, model.rsquared_adj, model.f_pvalue

rsquared, rsquared_adj, f_pvalue = ols_md_vs_rmsd()

print(f'{WINDOW_SIZE},{WINDOW_SIZE_CONTEXT},{rsquared},{rsquared_adj},{f_pvalue}')

with outfile.open('a') as f:
    f.write(f'{WINDOW_SIZE},{WINDOW_SIZE_CONTEXT},{rsquared},{rsquared_adj},{f_pvalue}\n')