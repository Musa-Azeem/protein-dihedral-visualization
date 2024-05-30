from Bio import SeqIO
import warnings
from Bio.Align import PairwiseAligner
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

def get_seq_funcs(winsize_ctxt):
    def get_center_idx():
        if winsize_ctxt % 2 == 0:
            return winsize_ctxt // 2 - 1
        return -winsize_ctxt // 2
    def get_center(seq):
        return seq[get_center_idx()]
        # if winsize_ctxt % 2 == 0:
        #     return seq[winsize_ctxt // 2 - 1]
        # else:
        #     return seq[-winsize_ctxt // 2]
    def get_seq_ctxt(residues, i):
        if winsize_ctxt % 2 == 0:
            return residues[i-winsize_ctxt//2+1:i+winsize_ctxt//2+1]
        return residues[i-winsize_ctxt//2:i+winsize_ctxt//2+1]
    return get_center_idx, get_center, get_seq_ctxt

def get_subseq_func(winsize, winsize_ctxt):
    # Given the context sequence, find the subsequence of the given window size
    def get_subseq(seq):
        if winsize % 2 == 0:
            return seq[winsize_ctxt//2 - winsize//2:winsize_ctxt//2 + winsize//2]
        else:
            if winsize_ctxt % 2 == 0:
                return seq[winsize_ctxt//2 - winsize//2-1:winsize_ctxt//2 + winsize//2]
            return seq[winsize_ctxt//2 - winsize//2:winsize_ctxt//2 + winsize//2 + 1]
    return get_subseq

def get_phi_psi_dist(queries, seq):
    phi_psi_dist = []
    info = []
    for q in queries:
        inner_seq = q.get_subseq(seq)
        phi_psi_dist.append(q.results[q.results.seq == inner_seq][['phi', 'psi', 'weight']])
        info.append((q.winsize, inner_seq, phi_psi_dist[-1].shape[0], q.weight))
    return pd.concat(phi_psi_dist), info

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

def find_kdepeak(phi_psi_dist, bw_method):
    # Find probability of each point
    phi_psi_dist = phi_psi_dist.loc[~phi_psi_dist[['phi', 'psi']].isna().any(axis=1)]

    kernel = gaussian_kde(
        phi_psi_dist[['phi','psi']].T, 
        weights=phi_psi_dist['weight'], 
        bw_method=bw_method
    )
    kdepeak = phi_psi_dist.iloc[kernel(phi_psi_dist[['phi', 'psi']].values.T).argmax()]
    # phi_psi_dist['prob'] = kernel(phi_psi_dist[['phi', 'psi']].values.T)

    return kdepeak

def calc_da_for_one(kdepeak, phi_psi):
    return np.sqrt((phi_psi[0] - kdepeak[0])**2 + (phi_psi[1] - kdepeak[1])**2)

def calc_da(kdepeak, phi_psi_preds):
    return np.sqrt(((phi_psi_preds[:,0] - kdepeak[0])**2) + ((phi_psi_preds[:,1] - kdepeak[1])**2))