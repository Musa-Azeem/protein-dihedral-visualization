import numpy as np
from lib.utils import find_kdepeak, calc_da, get_phi_psi_dist
from pathlib import Path
import pandas as pd
from numpy.linalg import LinAlgError

def get_da_for_all_predictions(ins, replace, bw_method=None):
    if replace or not Path(ins.outdir / 'phi_psi_predictions_da.csv').exists():
        get_da_for_all_predictions_(ins, bw_method)
    else:
        ins.phi_psi_predictions = pd.read_csv(ins.outdir / 'phi_psi_predictions_da.csv')
        ins.xray_phi_psi = pd.read_csv(ins.outdir / 'xray_phi_psi_da.csv')

def get_da_for_all_predictions_(ins, bw_method=None):
    bw_method = bw_method or ins.bw_method
    ins.phi_psi_predictions['da'] = np.nan
    ins.xray_phi_psi['da'] = np.nan
    for i,seq in enumerate(ins.xray_phi_psi.seq_ctxt.unique()):
        # inner_seq = ins.get_subseq(seq)
        print(f'{i}/{len(ins.xray_phi_psi.seq_ctxt.unique())-1}: {seq}')
        phi_psi_dist, info = get_phi_psi_dist(ins.queries, seq)
        for i in info:
            print(f'\tWin {i[0]}: {i[1]} - {i[2]} samples')
        if phi_psi_dist.shape[0] < 2:
            print(f'\tSkipping {seq} - not enough samples')
            continue # leave as nan

        try:
            kdepeak = find_kdepeak(phi_psi_dist, bw_method)[['phi','psi']]
        except LinAlgError as e:
            print('\tSingular Matrix - skipping')
            continue # leave as nan
        # Distance to kde peak
        xray = ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq][['phi','psi']]
        if xray.shape[0] == 0:
            print(f'No xray seq {seq}')
        else:
            da_xray = calc_da(kdepeak.values, xray.values)
            ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq, 'da'] = da_xray
        
        preds = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq][['phi','psi']]
        print(f'\t{preds.shape[0]} predictions')
        if preds.shape[0] == 0:
            print(f'No predictions seq {seq}')
        else:
            da = calc_da(kdepeak.values, preds.values)
            ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq, 'da'] = da

    ins.phi_psi_predictions.to_csv(ins.outdir / f'phi_psi_predictions_da.csv', index=False)
    ins.xray_phi_psi.to_csv(ins.outdir / f'xray_phi_psi_da.csv', index=False)