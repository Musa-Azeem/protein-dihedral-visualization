import numpy as np
import pandas as pd
from pathlib import Path
from lib import MultiWindowQuery
from lib.utils import get_find_target, compute_rmsd, compute_gdt
from lib.modules import (
    get_da_for_all_predictions, get_da_for_all_predictions_window, get_da_for_all_predictions_window_ml
)
from lib.plotting import (
    plot_res_vs_da,
    plot_across_window_clusters,
    plot_across_window_cluster_medoids
)
from lib.ml.models import MLPredictor, MLPredictorWindow
import math

class DihedralAdherencePDB(MultiWindowQuery):
    def __init__(
            self, pdb_code, winsizes, pdbmine_url, 
            projects_dir='tests',
            kdews=None, mode='kde', quantile=1,
            model=None, ml_lengths=[4096, 512, 256, 256], weights_file='ml_data/best_model.pt', device='cpu',
            pdbmine_cache_dir='casp_cache',
        ):
        super().__init__(pdb_code, winsizes, pdbmine_url, projects_dir, pdbmine_cache_dir, match_outdir=pdbmine_cache_dir)
        self.has_af = True
        if self.af_fn is None:
            self.has_af = False
            return
        self.pred_fn = self.af_fn
        self.phi_psi_predictions = None
        self.overlapping_seqs = None
        self.model = None
        
        self.bw_method = None
        self.quantile = quantile
        self.kdews = [1] * len(winsizes) if kdews is None else kdews
        
        self.mode = mode
        # if model is not None:
            # self.model = model
        if self.mode == 'ml':
            self.model = MLPredictor(ml_lengths, device, weights_file)
            self.model.load_weights()
        elif self.mode == 'full_window_ml':
            self.ml_lengths = [0,0,0,2]
            self.model = MLPredictorWindow(device, self.ml_lengths, self.winsizes, weights_file)
            self.model.load_weights()
    
        self.find_target, self.xray_da_fn, self.pred_da_fn = \
            get_find_target(self)
        
        for w,q in zip(self.kdews, self.queries):
            q.weight = w

        self.results=pd.DataFrame([[self.pdb_code, np.nan, np.nan, np.nan]], columns=['Model', 'GDT_TS', 'RMS_CA', 'DA'])
        
    def compute_das(self, replace=True, da_scale=None):
        if self.xray_phi_psi is None or self.phi_psi_predictions is None:
            print('Run compute_structures() or load_results() first')
            return
        if da_scale is None:
            # da_scale = [math.log2(i)+1 for i in self.kdews]
            da_scale = [1] * len(self.kdews)
        
        if self.mode == 'full_window':
            get_da_for_all_predictions_window(self, replace)
        elif self.mode == 'full_window_ml':
            get_da_for_all_predictions_window_ml(self, replace)
        else:
            # for all other modes
            get_da_for_all_predictions(self, replace, da_scale)
        self.get_total_da()
            
    def compute_structures(self, replace=False):
        super().compute_structure(replace)
        # For now, only prediction is alphafold
        if self.af_phi_psi is not None:
            self.phi_psi_predictions = self.af_phi_psi.drop('conf', axis=1).copy()
        self.phi_psi_predictions.to_csv(self.outdir / 'phi_psi_predictions.csv', index=False)
        if self.queried:
            self.get_results_metadata()
    
    def query_pdbmine(self, replace=False):
        super().query_pdbmine(replace)
        if self.xray_phi_psi is not None:
            self.get_results_metadata()

    def seq_filter(self):
        self.xray_phi_psi = self.xray_phi_psi[~self.xray_phi_psi.phi.isna() & ~self.xray_phi_psi.psi.isna()]
        self.xray_phi_psi = self.xray_phi_psi[~self.xray_phi_psi.seq_ctxt.str.contains('X')]

    def load_results(self):
        for query in self.queries:
            query.load_results(self.outdir)
            # query.results = pd.read_csv(self.outdir / f'phi_psi_mined_win{query.winsize}.csv')
            query.results['weight'] = query.weight
        self.queried = True
        self.xray_phi_psi = pd.read_csv(self.outdir / 'xray_phi_psi.csv')
        if (self.outdir / 'af_phi_psi.csv').exists():
            self.af_phi_psi = pd.read_csv(self.outdir / 'af_phi_psi.csv')
        else:
            print('No AlphaFold phi-psi data found')
            return False
        if (self.outdir / 'phi_psi_predictions.csv').exists():
            self.phi_psi_predictions = pd.read_csv(self.outdir / 'phi_psi_predictions.csv')
        else:
            self.phi_psi_predictions = self.af_phi_psi.drop('conf', axis=1).copy()
        self.seq_filter()
        self.get_results_metadata()
        return True
        
    def load_results_da(self):
        for query in self.queries:
            # query.results = pd.read_csv(self.outdir / f'phi_psi_mined_win{query.winsize}.csv')
            query.load_results(self.outdir)
            if len(query.results) > 0 and query.results['weight'].values[0] != query.weight:
                print('WARNING: Weights used to calculate DA are different')
            query.results['weight'] = query.weight
        self.queried = True
        self.xray_phi_psi = pd.read_csv(self.outdir / self.xray_da_fn)
        self.phi_psi_predictions = pd.read_csv(self.outdir / self.pred_da_fn)
        if (self.outdir / 'af_phi_psi.csv').exists():
            self.af_phi_psi = pd.read_csv(self.outdir / 'af_phi_psi.csv')
        else:
            print('No AlphaFold phi-psi data found')
        self.seq_filter()
        self.get_results_metadata()
        self.get_total_da()

    def get_results_metadata(self):
        self.overlapping_seqs = list(
            set(self.queries[-1].results.seq) & 
            set(self.phi_psi_predictions.seq_ctxt) & 
            set(self.xray_phi_psi.seq_ctxt)
        )
        self.seqs = self.xray_phi_psi.seq_ctxt.unique()
        self.protein_ids = self.phi_psi_predictions.protein_id.unique()
        if self.af_phi_psi is not None:
            if 'conf' in self.phi_psi_predictions.columns:
                self.phi_psi_predictions = self.phi_psi_predictions.drop(columns='conf')
            self.phi_psi_predictions = self.phi_psi_predictions.merge(self.af_phi_psi[['seq_ctxt', 'conf']], on='seq_ctxt', how='left')
        else:
            self.phi_psi_predictions['conf'] = np.nan

        rmsd = compute_rmsd(self.xray_fn, self.pred_fn, print_alignment=False)
        gdt = compute_gdt(self.xray_fn, self.pred_fn, print_alignment=False)
        self.results.loc[0, 'RMS_CA'] = rmsd
        self.results.loc[0, 'GDT_TS'] = gdt
        return self.overlapping_seqs, self.seqs, self.protein_ids

    def get_total_da(self):
        da = self.phi_psi_predictions.groupby('protein_id').da.mean()
        log_da = np.log10(da)
        self.results.loc[0, 'DA'] = log_da.values[0]

    def plot_res_vs_da(self,highlight_res=None, limit_quantile=None, legend_loc='upper right', fn=None, text_loc='right'):
        highlight_res = highlight_res or []
        if not 'da' in self.phi_psi_predictions.columns:
            print('No DA data available. Run compute_das() or load_results_da() first')
            return
        protein_id = self.protein_ids[0]
        pred_name = 'AlphaFold'
        return plot_res_vs_da(self, protein_id, pred_name, highlight_res, limit_quantile, legend_loc, fn, text_loc)
    
    def plot_across_window_clusters(self, seq=None, plot_xrays=True, plot_afs=True, n_cluster_lines=50):
        center_idx_ctxt = self.queries[-1].get_center_idx_pos()
        winsize_ctxt = self.queries[-1].winsize
        seqs_for_window = self.seqs[center_idx_ctxt:-(winsize_ctxt - center_idx_ctxt - 1)]
        seq = seq or seqs_for_window[0]
        plot_across_window_clusters(self, seq, plot_xrays, plot_afs, n_cluster_lines)
    
    def plot_across_window_cluster_medoids(self, seq=None, plot_xrays=False, plot_afs=False, verbose=False, mode_scatter=False, cse=30, fn=None):
        center_idx_ctxt = self.queries[-1].get_center_idx_pos()
        winsize_ctxt = self.queries[-1].winsize
        seqs_for_window = self.seqs[center_idx_ctxt:-(winsize_ctxt - center_idx_ctxt - 1)]
        seq = seq or seqs_for_window[0]
        plot_across_window_cluster_medoids(self, seq, plot_xrays, plot_afs, verbose, mode_scatter, cse, fn)