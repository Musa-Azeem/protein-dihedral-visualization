from Bio import SeqIO
import warnings
from Bio.Align import PairwiseAligner

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
