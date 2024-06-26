from pathlib import Path
import numpy as np
from lib import PDBMineQuery
from lib.retrieve_data import (
    retrieve_target_list, 
    retrieve_pdb_file, 
    retrieve_casp_predictions, 
    retrieve_casp_results
)
from lib.utils import get_seq_funcs, check_alignment, compute_rmsd, get_find_target
from lib.modules import (
    get_phi_psi_xray,
    get_phi_psi_predictions,
    seq_filter,
    get_da_for_all_predictions,
    fit_linregr,
)
from lib.plotting import (
    plot_one_dist,
    plot_one_dist_3d,
    plot_da_for_seq,
    plot_res_vs_da,
    plot_da_vs_rmsd,
    plot_heatmap,
    plot_da_vs_rmsd_simple,
    plot_res_vs_da_1plot
)
from lib.constants import AMINO_ACID_CODES, AMINO_ACID_CODES_INV, AMINO_ACID_CODE_NAMES
import pandas as pd
import requests
import math
from Bio.PDB import PDBParser
import warnings
from scipy.stats import gmean, hmean
from lib.ml.models import MLPredictor

class DihedralAdherence():
    def __init__(
            self, casp_protein_id, winsizes, pdbmine_url, projects_dir='tests', kdews=None, mode='kde',
            model=None, ml_lengths=[4096, 512, 256, 256], weights_file='ml_data/best_model.pt', device='cpu'
        ):
        print(f'Initializing {casp_protein_id} ...')
        self.casp_protein_id = casp_protein_id
        self.winsizes = winsizes
        self.winsize_ctxt = winsizes[-1]
        self.pdbmine_url = pdbmine_url
        self.outdir = Path(f'{projects_dir}/{casp_protein_id}_win{"-".join([str(w) for w in winsizes])}')
        if self.outdir.exists():
            print('Results already exist')
        else:
            self.outdir.mkdir(exist_ok=False, parents=True)

        # Get targetlist and corresponding pdbcode
        targetlist = retrieve_target_list()
        self.pdb_code = targetlist.loc[casp_protein_id, 'pdb_code']
        if self.pdb_code == '':
            raise ValueError(f'No PDB code found for {casp_protein_id}')
        self.alphafold_id = f'{casp_protein_id}TS427_1'
        print('Casp ID:', casp_protein_id, '\tPDB:', self.pdb_code)

        # Retrieve results and pdb files for xray and predictions
        self.results = retrieve_casp_results(casp_protein_id)
        self.xray_fn, self.sequence = retrieve_pdb_file(self.pdb_code)
        self.predictions_dir = retrieve_casp_predictions(casp_protein_id)

        # Get sequence and sequence context functions
        _, self.get_center, self.get_seq_ctxt = get_seq_funcs(self.winsize_ctxt)
        
        self.xray_phi_psi = None
        self.phi_psi_predictions = None
        self.overlapping_seqs = None
        self.seqs = None
        self.protein_ids = None
        self.grouped_preds = None
        self.grouped_preds_da = None
        self.model = None

        self.bw_method = None
        self.quantile = 1
        self.kdews = [1] * len(winsizes) if kdews is None else kdews

        self.queries = []
        for i,winsize in enumerate(self.winsizes):
            self.queries.append(PDBMineQuery(
                self.casp_protein_id, self.pdb_code, winsize, self.pdbmine_url,
                self.sequence, self.kdews[i]
            ))
            self.queries[-1].set_get_subseq(self.winsize_ctxt)
        self.queried = False

        self.mode = mode
        if model is not None:
            self.model = model
        else:
            self.model = MLPredictor(ml_lengths, device, weights_file)
        if self.mode == 'ml':
            self.model.load_weights()
    
        self.find_target, self.xray_da_fn, self.pred_da_fn = \
            get_find_target(self)

    def get_sequence(self, start, end, code=1):
        if code == 1:
            return list(self.sequence[start:end])
        elif code == 3:
            return [AMINO_ACID_CODES_INV[aa] for aa in self.sequence[start:end]]
        elif code == 'name':
            return [AMINO_ACID_CODE_NAMES[aa] for aa in self.sequence[start:end]]

    def get_window(self, i, code=1): # of size winsize
        if code == 1:
            return self.get_seq_ctxt(self.sequence, i)
        elif code == 3:
            return [AMINO_ACID_CODES_INV[aa] for aa in self.get_seq_ctxt(self.sequence, i)]
        elif code == 'name':
            return [AMINO_ACID_CODE_NAMES[aa] for aa in self.get_seq_ctxt(self.sequence, i)]

    def check_alignment(self, i=None, pred_id=None):
        if i and pred_id:
            raise ValueError('Only one of i or pred_id must be provided')
        if pred_id:
            pred_fn = self.predictions_dir / pred_id
        else:
            i = i or 0
            pred_files = list(self.predictions_dir.iterdir())
            pred_fn = pred_files[i]
            print('id =', pred_fn.stem)
        check_alignment(self.xray_fn, pred_fn)

    def compute_structures(self, replace=False):
        # TODO: align pos column of predictions with xray_phi_psi using sequence alignment
        self.xray_phi_psi = get_phi_psi_xray(self, replace)
        self.phi_psi_predictions = get_phi_psi_predictions(self, replace)
        if self.queried:
            self.get_results_metadata()
        # filter
        seq_filter(self)
    
    def test_pdbmine_conn(self):
        response = requests.get(self.pdbmine_url + f'/v1/api/protein/{self.pdb_code}')
        print('PDBMine Connection:', response.status_code)
        return response.ok

    def query_pdbmine(self, replace=False):
        for query in self.queries:
            if replace or not (self.outdir / f'phi_psi_mined_win{query.winsize}.csv').exists():
                query.query_and_process_pdbmine()
                query.results.to_csv(self.outdir / f'phi_psi_mined_win{query.winsize}.csv', index=False)
            else:
                query.results = pd.read_csv(self.outdir / f'phi_psi_mined_win{query.winsize}.csv')
                query.results['weight'] = query.weight
        self.queried = True

        if self.xray_phi_psi is not None:
            self.get_results_metadata()
    
    def compute_das(self, replace=True, da_scale=None):
        if self.xray_phi_psi is None or self.phi_psi_predictions is None:
            print('Run compute_structures() or load_results() first')
            return
        if da_scale is None:
            da_scale = [math.log2(i)+1 for i in self.kdews]
        get_da_for_all_predictions(self, replace, da_scale)
        # get_da_for_all_predictions_ml(self, replace, da_scale)
        self._get_grouped_preds()

    def load_results(self):
        for query in self.queries:
            query.results = pd.read_csv(self.outdir / f'phi_psi_mined_win{query.winsize}.csv')
            query.results['weight'] = query.weight
        self.queried = True
        self.xray_phi_psi = pd.read_csv(self.outdir / 'xray_phi_psi.csv')
        self.phi_psi_predictions = pd.read_csv(self.outdir / 'phi_psi_predictions.csv')
        self.get_results_metadata()
        
    def load_results_da(self):
        for query in self.queries:
            query.results = pd.read_csv(self.outdir / f'phi_psi_mined_win{query.winsize}.csv')
            if query.results['weight'].values[0] != query.weight:
                print('WARNING: Weights used to calculate DA are different')
            query.results['weight'] = query.weight
        self.queried = True
        self.xray_phi_psi = pd.read_csv(self.outdir / self.xray_da_fn)
        self.phi_psi_predictions = pd.read_csv(self.outdir / self.pred_da_fn)
        self.get_results_metadata()
        self._get_grouped_preds()

    def get_results_metadata(self):
        self.overlapping_seqs = list(
            set(self.queries[-1].results.seq) & 
            set(self.phi_psi_predictions.seq_ctxt) & 
            set(self.xray_phi_psi.seq_ctxt)
        )
        self.seqs = self.xray_phi_psi.seq_ctxt.unique()
        self.protein_ids = self.phi_psi_predictions.protein_id.unique()
        return self.overlapping_seqs, self.seqs, self.protein_ids
    
    def fit_model(self):
        fit_linregr(self)

    def compute_rmsd(self, pred_id=None, xray_start=None, xray_end=None, pred_start=None, pred_end=None):
        if pred_id is None:
            pred_id = self.protein_ids[0]
        rmsd = compute_rmsd(
            self.xray_fn, self.predictions_dir / pred_id, 
            xray_start, xray_end, pred_start, pred_end)
        print(f'RMSD={rmsd:.3f}')
        return rmsd
    
    def split_and_compute_rmsd(self, pred_id=None, split=None, print_alignment=True):
        # split should be int, tuple, list of ints or list of tuples
        if pred_id is None:
            pred_id = self.protein_ids[0]
        
        if split is None:
            rmsd = compute_rmsd(self.xray_fn, self.predictions_dir / pred_id, print_alignment)
            print(f'RMSD={rmsd:.3f}')
            return rmsd
        else:
            if not isinstance(split, list):
                split = [split]
            split = sorted(split)
            rmsds = []
            rmsd_inner = []
            n = []
            prev = (0,0)
            for s in split:
                if not isinstance(s, tuple):
                    s = (s,s)
                if s[0] - prev[0] < 2:
                    prev = (s[0] + 1, s[1] + 1)
                    continue
                if s[1] - prev[1] < 2:
                    prev = (s[0] + 1, s[1] + 1)
                    continue
                rmsd, ni, dist = compute_rmsd(
                    self.xray_fn, self.predictions_dir / pred_id,
                    prev[0], s[0], prev[1], s[1],
                    print_alignment, return_n=True
                )
                print(f'\nRMSD({prev[0]}-{s[0]})={rmsd:.3f}\n')
                prev = (s[0] + 1, s[1] + 1)
                rmsds.append(rmsd)
                n.append(ni)
                rmsd_inner.append(dist)
            rmsd, ni, dist = compute_rmsd(
                self.xray_fn, self.predictions_dir / pred_id,
                prev[0], None, prev[1], None,
                print_alignment, return_n=True
            )
            print(f'\nRMSD({prev[0]}-end)={rmsd:.3f}\n')
            rmsds.append(rmsd)
            n.append(ni)
            rmsd_inner.append(dist)
            print(f'\nTotal RMSD = {"+".join([f"{r:.03f}" for r in rmsds])} = {sum(rmsds):.3f}')
            print(f'Original RMSD={compute_rmsd(self.xray_fn, self.predictions_dir / pred_id, print_alignment=False):.3f}')
            print(f'Computed Total RMSD: {np.sqrt((1/sum(n)) * sum(rmsd_inner))}')
            print(f'Mean RMSD: {np.mean(rmsds):.3f}')
            return rmsds, n, rmsd_inner
    
    def plot_one_dist(self, seq=None, pred_id=None, pred_name=None, axlims=None, bw_method=-1, fn=None):
        seq = seq or self.overlapping_seqs[0]
        pred_id = pred_id or self.protein_ids[0]
        plot_one_dist(self, seq, pred_id, pred_name, axlims, bw_method, fn)

    def plot_one_dist_3d(self, seq=None, i=None, bw_method=-1, fn=None):
        if i is None and seq is None:
            seq = seq or self.overlapping_seqs[0]
        elif i is not None:
            if seq is not None:
                raise ValueError('Only one of i or seq must be provided')
            seq = self.xray_phi_psi[self.xray_phi_psi.pos == i].seq_ctxt.values
            if len(seq) == 0:
                raise ValueError(f'No sequence found for position {i}')
            seq = seq[0]
        plot_one_dist_3d(self, seq, bw_method, fn)

    def plot_da_for_seq(self, seq=None, i=None, pred_id=None, pred_name=None, axlims=None, bw_method=None, fn=None, fill=False):
        if i is None and seq is None:
            seq = seq or self.overlapping_seqs[0]
        elif i is not None:
            if seq is not None:
                raise ValueError('Only one of i or seq must be provided')
            seq = self.xray_phi_psi[self.xray_phi_psi.pos == i].seq_ctxt.values
            if len(seq) == 0:
                raise ValueError(f'No sequence found for position {i}')
            seq = seq[0]
        pred_id = pred_id or self.protein_ids[0]
        plot_da_for_seq(self, seq, pred_id, pred_name, bw_method, axlims, fn, fill)
    
    def plot_res_vs_da(self, pred_id=None, pred_name=None, highlight_res=None, limit_quantile=None, legend_loc='upper right', fn=None, text_loc='right'):
        highlight_res = highlight_res or []
        if not 'da' in self.phi_psi_predictions.columns:
            print('No DA data available. Run compute_das() or load_results_da() first')
            return
        protein_id = pred_id or self.protein_ids[0]
        return plot_res_vs_da(self, protein_id, pred_name, highlight_res, limit_quantile, legend_loc, fn, text_loc)
    
    def plot_res_vs_da_1plot(self, pred_id=None, pred_name=None, highlight_res=None, limit_quantile=None, legend_loc='upper right', fn=None, text_loc='right', rmsds=None):
        highlight_res = highlight_res or []
        if not 'da' in self.phi_psi_predictions.columns:
            print('No DA data available. Run compute_das() or load_results_da() first')
            return
        protein_id = pred_id or self.protein_ids[0]
        return plot_res_vs_da_1plot(self, protein_id, pred_name, highlight_res, limit_quantile, legend_loc, fn, text_loc, rmsds)
    
    def plot_da_vs_rmsd(self, axlims=None, fn=None):
        if not 'da' in self.phi_psi_predictions.columns:
            print('No DA data available. Run compute_das() or load_results_da() first')
            return
        if not 'rms_pred' in self.grouped_preds.columns:
            self.fit_model()
        else:
            print(f'Model R-squared: {self.model.rsquared:.6f}, Adj R-squared: {self.model.rsquared_adj:.6f}, p-value: {self.model.f_pvalue}')
        plot_da_vs_rmsd(self, axlims, fn)
    
    def plot_da_vs_rmsd_simple(self, axlims=None, fn=None):
        if not 'da' in self.phi_psi_predictions.columns:
            print('No DA data available. Run compute_das() or load_results_da() first')
            return
        plot_da_vs_rmsd_simple(self, axlims, fn)
    
    def plot_heatmap(self, fillna=False, fn=None):
        if not 'da' in self.phi_psi_predictions.columns:
            print('No DA data available. Run compute_das() or load_results_da() first')
            return
        plot_heatmap(self, fillna, fn)
    
    def get_id(self, group_id):
        return f'{self.casp_protein_id}TS{group_id}'
    def _get_grouped_preds(self):
        self.phi_psi_predictions['da_na'] = self.phi_psi_predictions.da.isna()
        def agg_da(x):
            x = x[x < x.quantile(self.quantile)]
            return x.agg('mean')
        self.grouped_preds = self.phi_psi_predictions.groupby('protein_id', as_index=False).agg(
            # da=('da', lambda x: x[x < x.quantile(self.quantile)].agg('mean')), 
            da=('da', agg_da), 
            da_std=('da', lambda x: x[x < x.quantile(self.quantile)].agg('std')),
            da_na=('da_na', lambda x: x.sum() / len(x)),
        )
        self.grouped_preds = pd.merge(
            self.grouped_preds,
            self.results[['Model', 'RMS_CA']],
            left_on='protein_id',
            right_on='Model',
            how='inner'
        )
        self.grouped_preds['target'] = self.casp_protein_id
        self.grouped_preds_da = self.phi_psi_predictions.pivot(index='protein_id', columns='pos', values='da')

    def filter_nas(self, quantile=0.9):
        self.grouped_preds = self.grouped_preds[self.grouped_preds.da_na < self.grouped_preds.da_na.quantile(quantile)]
        self.grouped_preds_da = self.grouped_preds_da.loc[self.grouped_preds.protein_id]
