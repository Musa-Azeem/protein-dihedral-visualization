from pathlib import Path
import numpy as np
from lib.retrieve_data import (
    retrieve_target_list, 
    retrieve_pdb_file, 
    retrieve_casp_predictions, 
    retrieve_casp_results
)
from lib.utils import get_seq_funcs, check_alignment
from lib.modules import (
    get_phi_psi_xray,
    get_phi_psi_predictions,
    query_and_process_pdbmine,
    seq_filter,
    get_md_for_all_predictions,
    fit_linregr
)
from lib.plotting import (
    plot_one_dist,
    plot_one_dist_3d,
    plot_clusters_for_seq,
    plot_md_for_seq,
    plot_res_vs_md,
    plot_md_vs_rmsd,
    plot_heatmap
)
from lib.constants import AMINO_ACID_CODES
import pandas as pd
import requests

class DihedralAdherence():
    def __init__(self, casp_protein_id, winsize, winsize_ctxt, pdbmine_url, projects_dir='tests'):
        self.casp_protein_id = casp_protein_id
        self.winsize = winsize
        self.winsize_ctxt = winsize_ctxt
        self.pdbmine_url = pdbmine_url
        self.outdir = Path(f'{projects_dir}/{casp_protein_id}_win{winsize}-{winsize_ctxt}')
        if self.outdir.exists():
            print('Results already exist')
        else:
            self.outdir.mkdir(exist_ok=False, parents=True)

        # Get targetlist and corresponding pdbcode
        targetlist = retrieve_target_list()
        self.pdb_code = targetlist.loc[casp_protein_id, 'pdb_code']
        if self.pdb_code == '':
            raise ValueError('No PDB code found')
        self.alphafold_id = f'{casp_protein_id}TS427_1'
        print('PDB:', self.pdb_code)

        # Retrieve results and pdb files for xray and predictions
        self.results = retrieve_casp_results(casp_protein_id)
        self.xray_fn = retrieve_pdb_file(self.pdb_code)
        self.predictions_dir = retrieve_casp_predictions(casp_protein_id)

        # Get sequence and sequence context functions
        self.get_center, self.get_seq, self.get_seq_ctxt, self.get_subseq = \
            get_seq_funcs(winsize, winsize_ctxt)
        
        self.xray_phi_psi = None
        self.phi_psi_predictions = None
        self.phi_psi_mined = None
        self.phi_psi_mined_ctxt = None
        self.overlapping_seqs = None
        self.seqs = None
        self.protein_ids = None
        self.grouped_preds = None
        self.grouped_preds_md = None
        self.model = None

        self.kdews = [1, 128]
        self.bw_method = None
        self.quantile = 1

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

    def compute_structures(self):
        self.xray_phi_psi = get_phi_psi_xray(self)
        self.phi_psi_predictions = get_phi_psi_predictions(self)
        if self.phi_psi_mined is not None:
            self.get_results_metadata()
        # filter
        seq_filter(self)
    
    def test_pdbmine_conn(self):
        response = requests.get(self.pdbmine_url + f'/v1/api/protein/{self.pdb_code}')
        print('PDBMine Connection:', response.status_code)
        return response.ok

    def query_pdbmine(self):
        self.phi_psi_mined, self.phi_psi_mined_ctxt = query_and_process_pdbmine(self)
        self.phi_psi_mined['weight'] = self.kdews[0]
        self.phi_psi_mined_ctxt['weight'] = self.kdews[1]
        if self.xray_phi_psi is not None:
            self.get_results_metadata()
    
    def compute_mds(self, skip_existing=True):
        if self.xray_phi_psi is None or self.phi_psi_predictions is None:
            print('Run compute_structures() or load_results() first')
            return
        get_md_for_all_predictions(self, skip_existing)
        self._get_grouped_preds()

    def load_results(self):
        self.phi_psi_mined = pd.read_csv(self.outdir / f'phi_psi_mined_win{self.winsize}.csv')
        self.phi_psi_mined_ctxt = pd.read_csv(self.outdir / f'phi_psi_mined_win{self.winsize_ctxt}.csv')
        self.xray_phi_psi = pd.read_csv(self.outdir / 'xray_phi_psi.csv')
        self.phi_psi_predictions = pd.read_csv(self.outdir / 'phi_psi_predictions.csv')
        self.get_results_metadata()
        # Temporary:
        if not 'weight' in self.phi_psi_mined.columns:
            self.phi_psi_mined['weight'] = self.kdews[0]
            self.phi_psi_mined_ctxt['weight'] = self.kdews[1]
            self.phi_psi_mined.to_csv(self.outdir / f'phi_psi_mined_win{self.winsize}.csv', index=False)
            self.phi_psi_mined_ctxt.to_csv(self.outdir / f'phi_psi_mined_win{self.winsize_ctxt}.csv', index=False)

    def load_results_md(self):
        self.phi_psi_mined = pd.read_csv(self.outdir / f'phi_psi_mined_win{self.winsize}.csv')
        self.phi_psi_mined_ctxt = pd.read_csv(self.outdir / f'phi_psi_mined_win{self.winsize_ctxt}.csv')
        self.xray_phi_psi = pd.read_csv(self.outdir / 'xray_phi_psi_md.csv')
        self.phi_psi_predictions = pd.read_csv(self.outdir / 'phi_psi_predictions_md.csv')
        self.get_results_metadata()
        self._get_grouped_preds()
        # Temporary:
        if not 'weight' in self.phi_psi_mined.columns:
            self.phi_psi_mined['weight'] = self.kdews[0]
            self.phi_psi_mined_ctxt['weight'] = self.kdews[1]
            self.phi_psi_mined.to_csv(self.outdir / f'phi_psi_mined_win{self.winsize}.csv', index=False)
            self.phi_psi_mined_ctxt.to_csv(self.outdir / f'phi_psi_mined_win{self.winsize_ctxt}.csv', index=False)

    def get_results_metadata(self):
        self.overlapping_seqs = list(
            set(self.phi_psi_mined_ctxt.seq) & 
            set(self.phi_psi_predictions.seq_ctxt) & 
            set(self.xray_phi_psi.seq_ctxt)
        )
        self.seqs = self.xray_phi_psi.seq_ctxt.unique()
        self.protein_ids = self.phi_psi_predictions.protein_id.unique()
        return self.overlapping_seqs, self.seqs, self.protein_ids
    
    def fit_model(self):
        fit_linregr(self)
    
    def plot_one_dist(self, seq=None, pred_id=None, pred_name=None, axlims=None, bw_method=-1, fn=None):
        seq = seq or self.overlapping_seqs[0]
        pred_id = pred_id or self.protein_ids[0]
        plot_one_dist(self, seq, pred_id, pred_name, axlims, bw_method, fn)

    def plot_one_dist_3d(self, seq=None, bw_method=-1, fn=None):
        seq = seq or self.overlapping_seqs[0]
        plot_one_dist_3d(self, seq, bw_method, fn)

    def plot_clusters_for_seq(self, seq=None, bw_method=-1, fn=None):
        seq = seq or self.overlapping_seqs[0]
        plot_clusters_for_seq(self, seq, bw_method, fn)

    def plot_md_for_seq(self, seq=None, pred_id=None, pred_name=None, axlims=None, bw_method=None, fn=None):
        seq = seq or self.overlapping_seqs[0]
        pred_id = pred_id or self.protein_ids[0]
        plot_md_for_seq(self, seq, pred_id, pred_name, axlims, bw_method, fn)
    
    def plot_res_vs_md(self, pred_id=None, pred_name=None, highlight_res=None, limit_quantile=None, legend_loc='upper right', fn=None):
        highlight_res = highlight_res or []
        if not 'md' in self.phi_psi_predictions.columns:
            print('No MD data available. Run compute_mds() or load_results_md() first')
            return
        protein_id = pred_id or self.protein_ids[0]
        return plot_res_vs_md(self, protein_id, pred_name, highlight_res, limit_quantile, legend_loc, fn)
    
    def plot_md_vs_rmsd(self, axlims=None, fn=None):
        if not 'md' in self.phi_psi_predictions.columns:
            print('No MD data available. Run compute_mds() or load_results_md() first')
            return
        if not 'rms_pred' in self.grouped_preds.columns:
            self.fit_model()
        else:
            print(f'Model R-squared: {self.model.rsquared:.6f}, Adj R-squared: {self.model.rsquared_adj:.6f}, p-value: {self.model.f_pvalue}')
        plot_md_vs_rmsd(self, axlims, fn)
    
    def plot_heatmap(self, fillna=True, fn=None):
        if not 'md' in self.phi_psi_predictions.columns:
            print('No MD data available. Run compute_mds() or load_results_md() first')
            return
        plot_heatmap(self, fillna, fn)

    def _get_grouped_preds(self):
        self.phi_psi_predictions['md_na'] = self.phi_psi_predictions.md.isna()
        self.grouped_preds = self.phi_psi_predictions.groupby('protein_id', as_index=False).agg(
            md=('md', lambda x: x[x < x.quantile(self.quantile)].agg('mean')), 
            md_std=('md', lambda x: x[x < x.quantile(self.quantile)].agg('std')),
            md_na=('md_na', lambda x: x.sum() / len(x)),
        )
        self.grouped_preds = pd.merge(
            self.grouped_preds,
            self.results[['Model', 'RMS_CA']],
            left_on='protein_id',
            right_on='Model',
            how='inner'
        )
        self.grouped_preds['target'] = self.casp_protein_id
        self.grouped_preds_md = self.phi_psi_predictions.pivot(index='protein_id', columns='pos', values='md')

    def filter_nas(self, quantile=0.9):
        self.grouped_preds = self.grouped_preds[self.grouped_preds.md_na < self.grouped_preds.md_na.quantile(0.9)]
        self.grouped_preds_md = self.grouped_preds_md.loc[self.grouped_preds.protein_id]
