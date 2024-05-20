from Bio.PDB.ic_rebuild import structure_rebuild_test
from Bio.PDB import PDBParser
import numpy as np
from lib.constants import AMINO_ACID_CODES
import warnings
from tqdm import tqdm
import pandas as pd

def get_phi_psi_xray(ins):
    parser = PDBParser()
    xray_structure = parser.get_structure(ins.pdb_code, ins.xray_fn)
    xray_chain = list(xray_structure[0].get_chains())[0]

    if not (ins.outdir / 'xray_phi_psi.csv').exists():
        xray_phi_psi = get_phi_psi_for_structure(xray_chain, ins.pdb_code)
        xray_phi_psi = pd.DataFrame(xray_phi_psi, columns=['pos', 'seq', 'seq_ctxt', 'res', 'phi', 'psi', 'protein_id'])
        xray_phi_psi.to_csv(ins.outdir / 'xray_phi_psi.csv', index=False)
    else:
        xray_phi_psi = pd.read_csv(ins.outdir / 'xray_phi_psi.csv')

    return xray_phi_psi

def get_phi_psi_predictions(ins):
    parser = PDBParser()
    if not (ins.outdir / 'phi_psi_predictions.csv').exists():
        phi_psi_predictions_ = []
        for prediction_pdb in tqdm((ins.predictions_dir).iterdir()):
                prediction = parser.get_structure(prediction_pdb.name, prediction_pdb)
                try:
                    chain = list(prediction[0].get_chains())[0]
                    phi_psi_predictions_ += get_phi_psi_for_structure(chain, prediction.id)
                except Exception as e:
                    print(e)

        phi_psi_predictions = pd.DataFrame(phi_psi_predictions_, columns=['pos', 'seq', 'seq_ctxt', 'res', 'phi', 'psi', 'protein_id'])
        phi_psi_predictions.to_csv(ins.outdir / 'phi_psi_predictions.csv', index=False)
    else:
        phi_psi_predictions = pd.read_csv(ins.outdir / 'phi_psi_predictions.csv')
    
    return phi_psi_predictions

def get_phi_psi_for_structure(ins, protein_structure, protein_id):
    protein_structure.atom_to_internal_coordinates(verbose=False)
    resultDict = structure_rebuild_test(protein_structure)
    if not resultDict['pass']:
        raise Exception('Failed to rebuild')
    residues = list(protein_structure.get_residues())
    phi_psi_ = []
    print([r.resname for r in residues])
    for i in range(ins.winsize_ctxt//2, len(residues) - ins.winsize_ctxt // 2):
        # Convert 3 char codes to 1 char codes
        seq = ''.join([AMINO_ACID_CODES.get(r.resname, 'X') for r in residues[ins.get_seq(i)]])
        seq_ctxt = ''.join([AMINO_ACID_CODES.get(r.resname, 'X') for r in residues[ins.get_seq_ctxt(i)]])
        # Get the center residue
        res = ins.get_center(seq)
        if not residues[i].internal_coord:
            psi,phi = np.nan, np.nan
        else:
            psi = residues[i].internal_coord.get_angle("psi")
            phi = residues[i].internal_coord.get_angle("phi")
            psi = psi if psi else np.nan # if psi is None, set it to np.nan
            phi = phi if phi else np.nan # if phi is None, set it to np.nan
        phi_psi_.append([i, seq, seq_ctxt, res, phi, psi, protein_id])
    return phi_psi_