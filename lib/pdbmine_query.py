from lib.modules import query_and_process_pdbmine
from pathlib import Path
from lib.utils import get_seq_funcs, get_subseq_func
import pandas as pd

# Class to represent a PDBMine query for a certain sequence and window size
class PDBMineQuery():
    def __init__(self, casp_protein_id, pdb_code, winsize, pdbmine_url, sequence, weight=1, match_outdir='cache'):
        self.casp_protein_id = casp_protein_id
        self.pdb_code = pdb_code
        self.winsize = winsize
        self.pdbmine_url = pdbmine_url
        self.weight = weight
        self.sequence = sequence
        self.match_outdir = (Path(match_outdir) / casp_protein_id) / f'matches-{self.winsize}'
        self.get_center_idx, _, _ = get_seq_funcs(self.winsize)
        self.get_subseq = None
        
        self.results = None
        self.results_window = None
    
    def get_center_idx_pos(self):
        center_idx = self.get_center_idx()
        if center_idx < 0:
            center_idx = self.winsize + center_idx
        return center_idx
    
    def set_get_subseq(self, winsize_ctxt):
        self.get_subseq = get_subseq_func(self.winsize, winsize_ctxt)
    def query_and_process_pdbmine(self, outdir):
        self.results, self.results_window = query_and_process_pdbmine(self)
        self.results = self.results[(self.results.phi <= 180) & (self.results.psi <= 180)]
        self.results_window = self.results_window[(self.results_window.phi <= 180) & (self.results_window.psi <= 180)]
        self.results.to_csv(outdir / f'phi_psi_mined_win{self.winsize}.csv', index=False)
        self.results_window.to_csv(outdir / f'phi_psi_mined_window_win{self.winsize}.csv', index=False)
    
    def load_results(self, outdir):
        self.results = pd.read_csv(outdir / f'phi_psi_mined_win{self.winsize}.csv')
        self.results_window = pd.read_csv(outdir / f'phi_psi_mined_window_win{self.winsize}.csv')
        self.results_window = self.results_window[(self.results_window.phi <= 180) & (self.results_window.psi <= 180)]