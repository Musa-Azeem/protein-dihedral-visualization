import numpy as np
from lib.utils import find_kdepeak, calc_da, calc_da_for_one
from pathlib import Path
import pandas as pd
from numpy.linalg import LinAlgError

def get_md_for_all_predictions(ins, replace, bw_method=None):
    if replace or not Path(ins.outdir / 'phi_psi_predictions_md.csv').exists():
        get_md_for_all_predictions_(ins, bw_method)
    else:
        ins.phi_psi_predictions = pd.read_csv(ins.outdir / 'phi_psi_predictions_md.csv')
        ins.xray_phi_psi = pd.read_csv(ins.outdir / 'xray_phi_psi_md.csv')

def get_md_for_all_predictions_(ins, bw_method=None):
    bw_method = bw_method or ins.bw_method
    ins.phi_psi_predictions['md'] = np.nan
    ins.xray_phi_psi['md'] = np.nan
    for i,seq in enumerate(ins.xray_phi_psi.seq_ctxt.unique()):
        inner_seq = ins.get_subseq(seq)
        phi_psi_dist = ins.phi_psi_mined.loc[ins.phi_psi_mined.seq == inner_seq][['phi','psi', 'weight']]
        phi_psi_ctxt_dist = ins.phi_psi_mined_ctxt.loc[ins.phi_psi_mined_ctxt.seq == seq][['phi','psi', 'weight']]
        print(f'{i}/{len(ins.xray_phi_psi.seq_ctxt.unique())}: {seq} - win{ins.winsize}: {phi_psi_dist.shape[0]}, win{ins.winsize_ctxt}: {phi_psi_ctxt_dist.shape[0]}')
        
        if phi_psi_ctxt_dist.shape[0] > 2:
            print('\tEnough context data for KDE - Using Full Context')
        if phi_psi_dist.shape[0] <= 2:
            print(f'\tSkipping {seq} - not enough data points')
            continue # leave as nan
        try:
            # phi_psi_dist, phi_psi_dist_c, most_likely = find_phi_psi_c(phi_psi_dist, phi_psi_ctxt_dist, bw_method)
            kdepeak = find_kdepeak(phi_psi_dist, phi_psi_ctxt_dist, bw_method)[['phi','psi']]
        except LinAlgError as e:
            print('\tSingular Matrix - skipping')
            continue # leave as nan
        # Distance to kde peak
        xray = ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq][['phi','psi']]
        if xray.shape[0] == 0:
            print(f'No xray seq {seq}')
        else:
            da_xray = calc_da(kdepeak.values, xray.values)
            ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq, 'md'] = da_xray
            
        preds = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq][['phi','psi']]
        if preds.shape[0] == 0:
            print(f'No predictions seq {seq}')
        else:
            da = calc_da(kdepeak.values, preds.values)
            ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq, 'md'] = da
        print(xray.shape, preds.shape, phi_psi_dist.shape, phi_psi_ctxt_dist.shape)

    ins.phi_psi_predictions.to_csv(ins.outdir / f'phi_psi_predictions_md.csv', index=False)
    ins.xray_phi_psi.to_csv(ins.outdir / f'xray_phi_psi_md.csv', index=False)