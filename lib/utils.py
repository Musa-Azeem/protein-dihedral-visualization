from Bio import SeqIO
import warnings
from Bio.PDB import Superimposer, PDBParser
from Bio.Align import PairwiseAligner
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, linregress
import pandas as pd
import numpy as np
from lib.constants import AMINO_ACID_CODES
from lib.ml.utils import get_ml_pred
from pathlib import Path
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

def get_seq_funcs(winsize_ctxt):
    def get_center_idx():
        if winsize_ctxt % 2 == 0:
            return winsize_ctxt // 2 - 1
        return -winsize_ctxt // 2
    def get_center(seq):
        return seq[get_center_idx()]
    def get_seq_ctxt(residues, i):
        if winsize_ctxt % 2 == 0:
            return residues[i-winsize_ctxt//2+1:i+winsize_ctxt//2+1]
        return residues[i-winsize_ctxt//2:i+winsize_ctxt//2+1]
    return get_center_idx, get_center, get_seq_ctxt

def get_subseq_func(winsize, winsize_ctxt):
    # Given the context sequence, find the subsequence of the given window size
    def get_subseq(seq):
        if winsize % 2 == 0:
            if winsize_ctxt % 2 == 0:
                # even window size, even context size
                return seq[winsize_ctxt//2 - winsize//2:winsize_ctxt//2 + winsize//2]
            # even window size, odd context size
            return seq[winsize_ctxt//2 - winsize//2 + 1:winsize_ctxt//2 + winsize//2 + 1]
        else:
            if winsize_ctxt % 2 == 0:
                # odd window size, even context size
                return seq[winsize_ctxt//2 - winsize//2-1:winsize_ctxt//2 + winsize//2]
            # odd window size, odd context size
            return seq[winsize_ctxt//2 - winsize//2:winsize_ctxt//2 + winsize//2 + 1]
    return get_subseq

def get_phi_psi_dist(queries, seq):
    phi_psi_dist = []
    info = []
    for q in queries:
        inner_seq = q.get_subseq(seq)
        phi_psi_dist.append(q.results[q.results.seq == inner_seq][['phi', 'psi', 'weight']])
        phi_psi_dist[-1]['winsize'] = q.winsize
        info.append((q.winsize, inner_seq, phi_psi_dist[-1].shape[0], q.weight))
    phi_psi_dist = pd.concat(phi_psi_dist)
    phi_psi_dist['seq'] = seq
    return phi_psi_dist, info

def check_alignment(xray_fn, pred_fn):
    # Check alignment of casp prediction and x-ray structure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        record = next(iter(SeqIO.parse(xray_fn, "pdb-seqres")))
        residue_chain = str(record.seq)#[residue_range[0]-1:residue_range[1]]

        pred_seq = str(next(iter(SeqIO.parse(pred_fn, "pdb-atom"))).seq)

        aligner = PairwiseAligner()
        aligner.mode = 'global'
        alignments =  aligner.align(residue_chain, pred_seq)

        print(f'Xray Length: {len(residue_chain)}, Prediction Length: {len(pred_seq)}')
        print(alignments[0])
        print('Large matches:')
        for i,((t1,t2),(q1,q2)) in enumerate(zip(*alignments[0].aligned)):
            if t2-t1 > 5:
                print(f'Match of length: {t2-t1} residues at position t={t1}, q={q1}')

def find_kdepeak(phi_psi_dist, bw_method, return_prob=False):
    # Find probability of each point
    phi_psi_dist = phi_psi_dist.loc[~phi_psi_dist[['phi', 'psi']].isna().any(axis=1)]

    kernel = gaussian_kde(
        phi_psi_dist[['phi','psi']].T, 
        weights=phi_psi_dist['weight'], 
        bw_method=bw_method
    )
    phi_grid, psi_grid = np.meshgrid(np.linspace(-180, 180, 360), np.linspace(-180, 180, 360))
    grid = np.vstack([phi_grid.ravel(), psi_grid.ravel()])
    kde = kernel(grid).reshape(phi_grid.shape)
    kdepeak = grid[:,kde.argmax()]
    kdepeak = pd.Series({'phi': kdepeak[0], 'psi': kdepeak[1]})

    if return_prob:
        return kdepeak, kde.max()
    return kdepeak

def find_kdepeak_af(phi_psi_dist, bw_method, af, return_peaks=False, find_peak=find_kdepeak):
    # Find probability of each point
    if af.shape[0] == 0:
        print('\tNo AlphaFold prediction - Using ordinary KDE')
        if return_peaks:
            peak = find_peak(phi_psi_dist, bw_method)
            return peak, peak, None
        return find_peak(phi_psi_dist, bw_method)

    af = af[['phi', 'psi']].values[0]

    phi_psi_dist = phi_psi_dist.loc[~phi_psi_dist[['phi', 'psi']].isna().any(axis=1)]

    # Find clusters
    bandwidth = 100
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(phi_psi_dist[['phi','psi']])
    phi_psi_dist['cluster'] = ms.labels_

    cluster_counts = phi_psi_dist.groupby('cluster').size()
    phi_psi_dist['cluster'] = phi_psi_dist['cluster'].apply(lambda x: x if cluster_counts[x] > 4 else -1)

    # find kdepeak for each cluster and entire dist
    kdepeak = find_peak(phi_psi_dist, bw_method)
    cluster_peaks = []
    for i in phi_psi_dist.cluster.unique():
        if i == -1:
            continue
        kdepeak_c = find_peak(phi_psi_dist[phi_psi_dist.cluster == i], bw_method)
        cluster_peaks.append(kdepeak_c)
    print(f'\tFound {len(cluster_peaks)} Clusters')
    # Choose peak that is closest to AlphaFold prediction
    targets = np.array([kdepeak.values] + [k.values for k in cluster_peaks])
    dists = calc_da(af, targets)
    argmin = dists.argmin()
    if argmin == 0:
        print('\tKDEPEAK: Using kdepeak of entire distribution')
    else:
        print(f'\tKDEPEAK: Using kdepeak of cluster {argmin - 1}')
    target = targets[argmin]
    target = pd.Series({'phi': target[0], 'psi': target[1]})

    if return_peaks:
        return target, kdepeak, cluster_peaks
    return target

def get_ml_pred_wrapper(phi_psi_dist, winsizes, res, af, ml, bw_method):
    # Find probability of each point
    if af.shape[0] == 0:
        print('\tNo AlphaFold prediction - Using ordinary KDE')
        return find_kdepeak(phi_psi_dist, bw_method)
    af = af[['phi', 'psi']].values[0]
    phi_psi_dist = phi_psi_dist.loc[~phi_psi_dist[['phi', 'psi']].isna().any(axis=1)]
    peaks = []
    for w in winsizes:
        x = phi_psi_dist.loc[phi_psi_dist.winsize == w, ['phi', 'psi']].values.T
        if x.shape[1] < 3:
            if x.shape[1] == 0:
                peaks.append([0,0])
            else:
                peaks.append(x.mean(axis=1).tolist())
            continue
        kde = gaussian_kde(x, bw_method=0.5)
        # print(kde.cho_cov)
        phi_grid, psi_grid = np.meshgrid(np.linspace(-180, 180, 180), np.linspace(-180, 180, 180))
        grid = np.vstack([phi_grid.ravel(), psi_grid.ravel()])
        probs = kde(grid).reshape(phi_grid.shape)
        # print(probs)
        kdepeak = grid[:,probs.argmax()]
        peaks.append(kdepeak.tolist())
    peaks = np.array(peaks)
    pred = get_ml_pred(peaks, res, af, ml)
    return pd.Series({'phi': pred[0], 'psi': pred[1]})

def calc_da_for_one(kdepeak, phi_psi):
    diff = lambda x1, x2: min(abs(x1 - x2), 360 - abs(x1 - x2))
    return np.sqrt(diff(phi_psi[0], kdepeak[0])**2 + diff(phi_psi[1], kdepeak[1])**2)

def calc_da(kdepeak, phi_psi_preds):
    def diff(x1, x2):
        d = np.abs(x1 - x2)
        return np.minimum(d, 360-d)
    return np.sqrt(diff(phi_psi_preds[:,0], kdepeak[0])**2 + diff(phi_psi_preds[:,1], kdepeak[1])**2)


def get_aligned_atoms(fnA, fnB, startA=None, endA=None, startB=None, endB=None, print_alignment=True):
    # Compute RMSD between two structures
    pdb_parser = PDBParser()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structureA = pdb_parser.get_structure('', fnA)
        structureB = pdb_parser.get_structure('', fnB)
    
    chainA = next(iter(structureA[0].get_chains()))
    chainB = next(iter(structureB[0].get_chains()))

    startA = startA or 0
    endA = endA or len(chainA)
    startB = startB or 0
    endB = endB or len(chainB)

    residuesA = ''.join([AMINO_ACID_CODES.get(r.resname, 'X') for r in chainA.get_residues()])[startA:endA]
    residuesB = ''.join([AMINO_ACID_CODES.get(r.resname, 'X') for r in chainB.get_residues()])[startB:endB]
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    alignments =  aligner.align(residuesA, residuesB)
    if print_alignment:
        print(alignments[0])

    atomsA = []
    atomsB = []
    residuesA = list(chainA.get_residues())[startA:endA]
    residuesB = list(chainB.get_residues())[startB:endB]
    for i,((t1,t2),(q1,q2)) in enumerate(zip(*alignments[0].aligned)):
        for j, (residueA, residueB) in enumerate(zip(residuesA[t1:t2], residuesB[q1:q2])):
            if residueA.resname != residueB.resname:
                print(f'WARNING: Residues {residueA.resname} and {residueB.resname} don\'t match at position: {j}')
            try:
                atomA = residueA['CA']
                atomB = residueB['CA']
            except KeyError:
                print(f'WARNING: Atom "CA" missing at position: {i}')
                continue
            if atomB is None or atomA is None:
                print(f'WARNING: Atom "CA" missing at position: {i}')
                continue
            atomsA.append(atomA)
            atomsB.append(atomB)
    return atomsA, atomsB

def compute_rmsd(fnA, fnB, startA=None, endA=None, startB=None, endB=None, print_alignment=True, return_n=False):
    atomsA, atomsB = get_aligned_atoms(fnA, fnB, startA, endA, startB, endB, print_alignment)

    sup = Superimposer()
    sup.set_atoms(atomsA, atomsB)
    sup.apply(atomsB)
    if return_n:
        atomsA = np.array([a.coord for a in atomsA])
        atomsB = np.array([a.coord for a in atomsB])
        dist = np.sum((atomsA - atomsB)**2)
        return sup.rms, len(atomsA), dist
    return sup.rms

def compute_gdt(fnA, fnB, startA=None, endA=None, startB=None, endB=None, print_alignment=True, return_n=False, thresholds=[1,2,4,8]):
    # WARNING: This is not consistent with the GDT_TS calculation from CASP
    atomsA, atomsB = get_aligned_atoms(fnA, fnB, startA, endA, startB, endB, print_alignment)

    sup = Superimposer()
    sup.set_atoms(atomsA, atomsB)
    sup.apply(atomsB)
    atomsA = np.array([a.coord for a in atomsA])
    atomsB = np.array([a.coord for a in atomsB])
    n = atomsA.shape[0]

    dist = np.linalg.norm(atomsA - atomsB, axis=1)
    gdt = []
    for t in thresholds:
        gdt.append((dist <= t).sum() / n)
    return np.mean(gdt)

def test_correlation(ins):
    grouped_preds = ins.grouped_preds.dropna(subset=['log_da', 'GDT_TS'])
    regr = linregress(grouped_preds.log_da, grouped_preds.GDT_TS)
    print(f'LinRegr - Slope: {regr.slope}, Intercept: {regr.intercept}', 'R-squared:', regr.rvalue**2, 'p-value:', regr.pvalue)

    corr, pval = pearsonr(grouped_preds.log_da, grouped_preds.GDT_TS)
    print(f'Pearson Correlation: {corr}, p-value: {pval}')

    return (regr.rvalue**2, corr)

def get_find_target(ins):
    xray_da_fn = 'xray_phi_psi_da.csv'
    pred_da_fn = 'phi_psi_predictions_da.csv'
    def get_af(seq):
        if ins.af_phi_psi is not None:
            return ins.af_phi_psi[ins.af_phi_psi.seq_ctxt == seq]
        else:
            return ins.phi_psi_predictions[(ins.phi_psi_predictions.protein_id == ins.alphafold_id) & (ins.phi_psi_predictions.seq_ctxt == seq)]
    match(ins.mode):
        case 'kde':
            def find_target_wrapper(phi_psi_dist, bw_method):
                return find_kdepeak(phi_psi_dist, bw_method)
        case 'ml':
            xray_da_fn = 'xray_phi_psi_da_ml.csv'
            pred_da_fn = 'phi_psi_predictions_da_ml.csv'
            def find_target_wrapper(phi_psi_dist, bw_method):
                af = get_af(phi_psi_dist.seq.values[0])
                res = ins.get_center(phi_psi_dist.seq.values[0])
                return get_ml_pred_wrapper(phi_psi_dist, ins.winsizes, res, af, ins.model, bw_method)
        case 'kde_af':
            xray_da_fn = 'xray_phi_psi_da_af.csv'
            pred_da_fn = 'phi_psi_predictions_da_af.csv'
            def find_target_wrapper(phi_psi_dist, bw_method):
                af = get_af(phi_psi_dist.seq.values[0])
                return find_kdepeak_af(phi_psi_dist, bw_method, af)
        case 'af':
            xray_da_fn = 'xray_phi_psi_da_afonly.csv'
            pred_da_fn = 'phi_psi_predictions_da_afonly.csv'
            def find_target_wrapper(phi_psi_dist, bw_method):
                af = get_af(phi_psi_dist.seq.values[0])
                return af[['phi', 'psi']].iloc[0] if af.shape[0] > 0 else find_kdepeak(phi_psi_dist, bw_method)
        case 'full_window':
            xray_da_fn = 'xray_phi_psi_da_window.csv'
            pred_da_fn = 'phi_psi_predictions_da_window.csv'

            # In this case, use a different method, not "get_da_for_all_predictions" like for others
            # Therefore, find_target is not used
            def find_target_wrapper(phi_psi_dist, bw_method):
                raise NotImplementedError('Full Window Mode does implement find_target')
        case 'full_window_ml':
            xray_da_fn = 'xray_phi_psi_da_window_ml.csv'
            pred_da_fn = 'phi_psi_predictions_da_window_ml.csv'

            # In this case, use a different method, not "get_da_for_all_predictions" like for others
            # Therefore, find_target is not used
            def find_target_wrapper(phi_psi_dist, bw_method):
                raise NotImplementedError('Full Window ML Mode does implement find_target')
    return find_target_wrapper, Path(xray_da_fn), Path(pred_da_fn)