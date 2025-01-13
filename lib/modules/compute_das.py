import numpy as np
from lib.utils import calc_da, get_phi_psi_dist
from pathlib import Path
import pandas as pd
from numpy.linalg import LinAlgError
from lib.ml.utils import get_ml_pred

def get_da_for_all_predictions(ins, replace, da_scale, bw_method=None):
    if replace or not Path(ins.outdir / ins.pred_da_fn).exists():
        get_da_for_all_predictions_(ins, da_scale, bw_method)
    else:
        ins.phi_psi_predictions = pd.read_csv(ins.outdir / ins.pred_da_fn)
        ins.xray_phi_psi = pd.read_csv(ins.outdir / ins.xray_da_fn)

def get_da_for_all_predictions_(ins, da_scale, scale_das=True, bw_method=None):
    bw_method = bw_method or ins.bw_method
    ins.phi_psi_predictions['da'] = np.nan
    ins.phi_psi_predictions['n_samples'] = np.nan
    ins.phi_psi_predictions['n_samples_list'] = ''
    ins.xray_phi_psi['da'] = np.nan
    ins.xray_phi_psi['n_samples'] = np.nan
    ins.xray_phi_psi['n_samples_list'] = ''
    for i,seq in enumerate(ins.xray_phi_psi.seq_ctxt.unique()):
        print(f'{i}/{len(ins.xray_phi_psi.seq_ctxt.unique())-1}: {seq}')
        if 'X' in seq:
            print(f'\tSkipping {seq} - X in sequence')
            continue

        if ins.af_phi_psi is not None:
            af = ins.af_phi_psi.loc[ins.af_phi_psi.seq_ctxt == seq]
            if len(af) > 0:
                if af['conf'].values[0] < 50:
                    print(f'\tSkipping {seq} - low confidence')

        phi_psi_dist, info = get_phi_psi_dist(ins.queries, seq)
        for j in info:
            print(f'\tWin {j[0]}: {j[1]} - {j[2]} samples')

        # Calculate number of samples weighted by kdeweight
        weighted_n_samples = sum([i[2]*w for i,w in zip(info, da_scale)])
        ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq, 'n_samples'] = weighted_n_samples
        ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq, 'n_samples_list'] = str([i[2] for i in info])
        ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq, 'n_samples'] = weighted_n_samples
        ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq, 'n_samples_list'] = str([i[2] for i in info])
        print(f'\tWeighted n samples: {weighted_n_samples}')

        if phi_psi_dist.shape[0] < 2:
            print(f'\tSkipping {seq} - not enough samples')
            continue # leave as nan
        
        try:
            target = ins.find_target(phi_psi_dist, bw_method=bw_method)[['phi','psi']]
        except LinAlgError as e:
            print('\tSingular Matrix - skipping')
            continue # leave as nan
        except ValueError as e:
            print('\tSample count error - skipping')
            continue

        # Distance to kde peak
        xray = ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq][['phi','psi']]
        if xray.shape[0] == 0:
            print(f'No xray seq {seq}')
        else:
            da_xray = calc_da(target.values, xray.values)
            ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq, 'da'] = da_xray
        
        preds = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq][['phi','psi']]
        print(f'\t{preds.shape[0]} predictions')
        if preds.shape[0] == 0:
            print(f'\tNo predictions seq {seq}')
        else:
            da = calc_da(target.values, preds.values)
            ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq, 'da'] = da
        
        print(f'\tXray DA: {da_xray}', f'Pred DA: {np.nanmean(da)}' if preds.shape[0] > 0 else '')
    
    # scale da by number of samples
    mean, std = ins.phi_psi_predictions['n_samples'].describe()[['mean', 'std']]
    # expected is mean-std/2, but at least 1
    expected = max(1,mean - std / 2)
    # if n_samples < expected, scale by n_samples / expected
    def scale(n_samples):
        return min(1, n_samples / expected)
    ins.xray_phi_psi['da_no_scale'] = ins.xray_phi_psi['da']
    ins.phi_psi_predictions['da_no_scale'] = ins.phi_psi_predictions['da']
    if scale_das:
        ins.xray_phi_psi['da'] = ins.xray_phi_psi['da'] * ins.xray_phi_psi['n_samples'].apply(scale)
        ins.phi_psi_predictions['da'] = ins.phi_psi_predictions['da'] * ins.phi_psi_predictions['n_samples'].apply(scale)

    ins.phi_psi_predictions.to_csv(ins.outdir / ins.pred_da_fn, index=False)
    ins.xray_phi_psi.to_csv(ins.outdir / ins.xray_da_fn, index=False)


############################## ML ######################################


# def get_da_for_all_predictions_ml(ins, replace, da_scale):
#     if replace or not Path(ins.outdir / 'phi_psi_predictions_da_ml.csv').exists():
#         get_da_for_all_predictions_ml_(ins, da_scale)
#     else:
#         ins.phi_psi_predictions = pd.read_csv(ins.outdir / 'phi_psi_predictions_da_ml.csv')
#         ins.xray_phi_psi = pd.read_csv(ins.outdir / 'xray_phi_psi_da_ml.csv')

# def get_da_for_all_predictions_ml_(ins, da_scale):
#     ins.phi_psi_predictions['da'] = np.nan
#     ins.phi_psi_predictions['n_samples'] = np.nan
#     ins.xray_phi_psi['da'] = np.nan
#     ins.xray_phi_psi['n_samples'] = np.nan
#     for i,seq in enumerate(ins.xray_phi_psi.seq_ctxt.unique()):
#         print(f'{i}/{len(ins.xray_phi_psi.seq_ctxt.unique())-1}: {seq}')
#         if 'X' in seq:
#             print(f'\tSkipping {seq} - X in sequence')
#             continue

#         phi_psi_dist, info = get_phi_psi_dist(ins.queries, seq)
#         for i in info:
#             print(f'\tWin {i[0]}: {i[1]} - {i[2]} samples')

#         # Calculate number of samples weighted by kdeweight
#         weighted_n_samples = sum([i[2]*w for i,w in zip(info, da_scale)])
#         ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq, 'n_samples'] = weighted_n_samples
#         ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq, 'n_samples'] = weighted_n_samples
#         print(f'\tWeighted n samples: {weighted_n_samples}')

#         if phi_psi_dist.shape[0] < 2:
#             print(f'\tSkipping {seq} - not enough samples')
#             continue # leave as nan

#         target = get_ml_pred(phi_psi_dist, ins.winsizes, ins.get_center(seq), ins.model)[['phi','psi']]

#         # Distance to kde peak
#         xray = ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq][['phi','psi']]
#         if xray.shape[0] == 0:
#             print(f'No xray seq {seq}')
#         else:
#             da_xray = calc_da(target.values, xray.values)
#             ins.xray_phi_psi.loc[ins.xray_phi_psi.seq_ctxt == seq, 'da'] = da_xray
        
#         preds = ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq][['phi','psi']]
#         print(f'\t{preds.shape[0]} predictions')
#         if preds.shape[0] == 0:
#             print(f'No predictions seq {seq}')
#         else:
#             da = calc_da(target.values, preds.values)
#             ins.phi_psi_predictions.loc[ins.phi_psi_predictions.seq_ctxt == seq, 'da'] = da
    
#     # scale da by number of samples
#     mean, std = ins.phi_psi_predictions['n_samples'].describe()[['mean', 'std']]
#     # expected is mean-std/2, but at least 1
#     expected = max(1,mean - std / 2)
#     # if n_samples < expected, scale by n_samples / expected
#     def scale(n_samples):
#         return min(1, n_samples / expected)
#     ins.xray_phi_psi['da'] = ins.xray_phi_psi['da'] * ins.xray_phi_psi['n_samples'].apply(scale)
#     ins.phi_psi_predictions['da'] = ins.phi_psi_predictions['da'] * ins.phi_psi_predictions['n_samples'].apply(scale)

#     ins.phi_psi_predictions.to_csv(ins.outdir / f'phi_psi_predictions_da_ml.csv', index=False)
#     ins.xray_phi_psi.to_csv(ins.outdir / f'xray_phi_psi_da_ml.csv', index=False)