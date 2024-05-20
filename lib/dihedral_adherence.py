from pathlib import Path
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
    query_and_process_pdbmine
)
from lib.plotting import (
    plot_one_dist
)
from lib.constants import AMINO_ACID_CODES
import pandas as pd

class DihedralAdherence():
    def __init__(self, casp_protein_id, winsize, winsize_ctxt, pdbmine_url):
        self.casp_protein_id = casp_protein_id
        self.winsize = winsize
        self.winsize_ctxt = winsize_ctxt
        self.pdbmine_url = pdbmine_url
        self.outdir = Path(f'tests/{casp_protein_id}_win{winsize}-{winsize_ctxt}')
        if self.outdir.exists():
            print('Results already exist')
        else:
            self.outdir.mkdir(exist_ok=False, parents=True)

        # Get targetlist and corresponding pdbcode
        targetlist = retrieve_target_list()
        self.pdb_code = targetlist.loc[casp_protein_id, 'pdb_code']
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
    
    # def test_pdbmine_conn(self):
        # response = requests.get(self.pdbmine_url + '/v1/api')
        # assert(response.ok)

    def query_pdbmine(self):
        self.phi_psi_mined, self.phi_psi_mined_ctxt = query_and_process_pdbmine(self)
        if self.xray_phi_psi is not None:
            self.get_results_metadata()

    def load_results(self):
        self.phi_psi_mined = pd.read_csv(self.outdir / 'phi_psi_mined.csv')
        self.phi_psi_mined_ctxt = pd.read_csv(self.outdir / 'phi_psi_mined_ctxt.csv')
        self.xray_phi_psi = pd.read_csv(self.outdir / 'xray_phi_psi.csv')
        self.phi_psi_predictions = pd.read_csv(self.outdir / 'phi_psi_predictions.csv')

    def get_results_metadata(self):
        self.overlapping_seqs = list(set(self.phi_psi_mined_ctxt.seq) & set(self.phi_psi_predictions.seq_ctxt) & set(self.xray_phi_psi.seq_ctxt))
        self.seqs = self.xray_phi_psi.seq_ctxt.unique()
        self.protein_ids = self.phi_psi_predictions.protein_id.unique()
        return self.overlapping_seqs, self.seqs, self.protein_ids
    
    def plot_one_dist(self, seq=None, pred_id=None, pred_name=None, axlims=None, bw_method=None):
        seq = seq or self.overlapping_seqs[0]
        pred_id = pred_id or self.protein_ids[0]
        plot_one_dist(self, seq, pred_id, pred_name, axlims, bw_method)