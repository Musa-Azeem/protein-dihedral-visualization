from Bio import SeqIO
import warnings
from Bio.Align import PairwiseAligner
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

def get_seq_funcs(winsize, winsize_ctxt):
    def get_center(seq):
        if winsize % 2 == 0:
            return seq[winsize // 2 - 1]
        else:
            return seq[-winsize // 2]
    def get_seq(i):
        if winsize % 2 == 0:
            if winsize_ctxt % 2 == 0:
                return slice(i-winsize//2+1,i+winsize//2+1)
            return slice(i-winsize//2,i+winsize//2)
        else:
            return slice(i-winsize//2,i+winsize//2+1)
    def get_seq_ctxt(i):
        if winsize_ctxt % 2 == 0:
            return slice(i-winsize_ctxt//2+1,i+winsize_ctxt//2+1)
        return slice(i-winsize_ctxt//2,i+winsize_ctxt//2+1)
    def get_subseq(seq):
        if winsize % 2 == 0:
            return seq[winsize_ctxt//2 - winsize//2:winsize_ctxt//2 + winsize//2]
        else:
            if winsize_ctxt % 2 == 0:
                return seq[winsize_ctxt//2 - winsize//2-1:winsize_ctxt//2 + winsize//2]
            return seq[winsize_ctxt//2 - winsize//2:winsize_ctxt//2 + winsize//2 + 1]
    return get_center, get_seq, get_seq_ctxt, get_subseq

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

def find_phi_psi_c(phi_psi_dist, phi_psi_ctxt_dist, bw_method):
    phi_psi_dist = pd.concat([phi_psi_dist, phi_psi_ctxt_dist])

    # Find probability of each point
    kernel = gaussian_kde(phi_psi_dist[['phi','psi']].T, weights=phi_psi_dist['weight'], bw_method=bw_method)
    most_likely = phi_psi_dist.iloc[kernel(phi_psi_dist[['phi', 'psi']].values.T).argmax()]
    phi_psi_dist['prob'] = kernel(phi_psi_dist[['phi', 'psi']].values.T)

    # cluster with kmeans
    max_sil_avg = -1
    for k in range(2, min(phi_psi_dist.shape[0], 7)):
        kmeans = KMeans(n_clusters=k, n_init=10) # TODO experiment with n_init
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

    print((
        f"Chosen dist has {phi_psi_dist_c[['phi', 'psi']].shape[0]} "
        f"samples and a mean of ({phi_psi_dist.phi.mean():.03f}, "
        f"{phi_psi_dist.psi.mean():.03f})"
    ))
    return phi_psi_dist, phi_psi_dist_c, most_likely

def find_kdepeak(phi_psi_dist, phi_psi_ctxt_dist, bw_method):
    phi_psi_dist = pd.concat([phi_psi_dist, phi_psi_ctxt_dist])
    # Find probability of each point
    kernel = gaussian_kde(phi_psi_dist[['phi','psi']].T, weights=phi_psi_dist['weight'], bw_method=bw_method)
    kdepeak = phi_psi_dist.iloc[kernel(phi_psi_dist[['phi', 'psi']].values.T).argmax()]
    phi_psi_dist['prob'] = kernel(phi_psi_dist[['phi', 'psi']].values.T)

    return kdepeak

def calc_da_for_one(kdepeak, phi_psi):
    return np.sqrt((phi_psi[0] - kdepeak[0])**2 + (phi_psi[1] - kdepeak[1])**2)

def calc_da(kdepeak, phi_psi_preds):
    return np.sqrt(((phi_psi_preds[:,0] - kdepeak[0])**2) + ((phi_psi_preds[:,1] - kdepeak[1])**2))

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

def calc_maha(phi_psi_preds: np.ndarray, phi_psi_dist: np.ndarray, kdepeak):
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