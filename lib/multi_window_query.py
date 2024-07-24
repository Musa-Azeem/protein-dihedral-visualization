from pathlib import Path
from lib.retrieve_data import retrieve_pdb_file, retrieve_alphafold_prediction
from lib.utils import get_seq_funcs
from lib import PDBMineQuery
from lib.modules import get_phi_psi_xray, get_phi_psi_af
import requests
import pandas as pd
import time

class MultiWindowQuery:
    def __init__(self, pdb_code, winsizes, pdbmine_url, projects_dir='ml_data', casp_protein_id=None, match_outdir='cache'):
        self.pdb_code = pdb_code
        self.casp_protein_id = casp_protein_id if casp_protein_id else pdb_code
        self.winsizes = winsizes
        self.winsize_ctxt = winsizes[-1]
        self.pdbmine_url = pdbmine_url
        self.outdir = Path(f'{projects_dir}/{pdb_code}_win{"-".join([str(w) for w in winsizes])}')
        if self.outdir.exists():
            print('Results already exist')
        else:
            self.outdir.mkdir(exist_ok=False, parents=True)

        self.xray_fn, self.sequence = retrieve_pdb_file(self.pdb_code)
        self.af_fn = retrieve_alphafold_prediction(self.pdb_code)

        _, self.get_center, self.get_seq_ctxt = get_seq_funcs(self.winsize_ctxt)

        self.xray_phi_psi = None
        self.af_phi_psi = None
        self.queries = []

        for i,winsize in enumerate(self.winsizes):
            self.queries.append(PDBMineQuery(
                self.casp_protein_id, self.pdb_code, winsize, self.pdbmine_url,
                self.sequence, 1, match_outdir
            ))
            self.queries[-1].set_get_subseq(self.winsize_ctxt)
        self.queried = False

    def compute_structure(self, replace=False):
        self.xray_phi_psi = get_phi_psi_xray(self, replace)
        self.xray_phi_psi = self.xray_phi_psi[~self.xray_phi_psi.phi.isna() & ~self.xray_phi_psi.psi.isna()]
        if self.af_fn is not None:
            self.af_phi_psi = get_phi_psi_af(self, replace)
    def compute_af_structure(self, replace=False):
        if self.af_fn is not None:
            self.af_phi_psi = get_phi_psi_af(self, replace)
        else:
            print('No alphafold prediction found')
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

    def load_results(self):
        for query in self.queries:
            query.results = pd.read_csv(self.outdir / f'phi_psi_mined_win{query.winsize}.csv')
            query.results['weight'] = query.weight
        self.queried = True
        self.xray_phi_psi = pd.read_csv(self.outdir / 'xray_phi_psi.csv')
        if (self.outdir / 'af_phi_psi.csv').exists():
            self.af_phi_psi = pd.read_csv(self.outdir / 'af_phi_psi.csv')
        else:
            print('No alphafold phi-psi predictions found')