import numpy as np
import pandas as pd
from pathlib import Path
from lib import MultiWindowQuery
from lib.utils import get_find_target
from lib.modules import (
    get_da_for_all_predictions, get_da_for_all_predictions_window, 
)
from lib.ml.models import MLPredictor
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

        self.phi_psi_predictions = None
        self.overlapping_seqs = None
        self.model = None
        
        self.bw_method = None
        self.quantile = quantile
        self.kdews = [1] * len(winsizes) if kdews is None else kdews
        
        self.mode = mode
        if model is not None:
            self.model = model
        else:
            self.model = MLPredictor(ml_lengths, device, weights_file)
        if self.mode == 'ml':
            self.model.load_weights()
    
        self.find_target, self.xray_da_fn, self.pred_da_fn = \
            get_find_target(self)
        
    def compute_das(self, replace=True, da_scale=None):
        if self.xray_phi_psi is None or self.phi_psi_predictions is None:
            print('Run compute_structures() or load_results() first')
            return
        if da_scale is None:
            da_scale = [math.log2(i)+1 for i in self.kdews]
        
        if self.mode == 'full_window':
            get_da_for_all_predictions_window(self, replace)
        else:
            # for all other modes
            get_da_for_all_predictions(self, replace, da_scale)
            # get_da_for_all_predictions_ml(self, replace, da_scale)
    
    def compute_structures(self, replace=False):
        super().compute_structure(replace)
        self.phi_psi_predictions = self.xray_phi_psi.copy()

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
        self.seq_filter()
        self.phi_psi_predictions = self.xray_phi_psi.copy()
        self.get_results_metadata()
        
    def load_results_da(self):
        for query in self.queries:
            # query.results = pd.read_csv(self.outdir / f'phi_psi_mined_win{query.winsize}.csv')
            query.load_results(self.outdir)
            if query.results['weight'].values[0] != query.weight:
                print('WARNING: Weights used to calculate DA are different')
            query.results['weight'] = query.weight
        self.queried = True
        self.xray_phi_psi = pd.read_csv(self.outdir / self.xray_da_fn)
        if (self.outdir / 'af_phi_psi.csv').exists():
            self.af_phi_psi = pd.read_csv(self.outdir / 'af_phi_psi.csv')
        else:
            print('No AlphaFold phi-psi data found')
        self.seq_filter()
        self.phi_psi_predictions = self.xray_phi_psi.copy()
        self.get_results_metadata()

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
        return self.overlapping_seqs, self.seqs, self.protein_ids
