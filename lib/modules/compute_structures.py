from Bio.PDB.ic_rebuild import structure_rebuild_test
from Bio.PDB import PDBParser
import numpy as np
from lib.constants import AMINO_ACID_CODES
import warnings
from tqdm import tqdm
import pandas as pd

def get_phi_psi_xray(ins, replace):
    if not (ins.outdir / 'xray_phi_psi.csv').exists() or replace:
        print('Computing phi-psi for xray')
        parser = PDBParser()
        xray_structure = parser.get_structure(ins.pdb_code, ins.xray_fn)
        xray_chain = list(xray_structure[0].get_chains())[0]
        xray_phi_psi = get_phi_psi_for_structure(ins, xray_structure, ins.pdb_code)
        xray_phi_psi = pd.DataFrame(xray_phi_psi, columns=['pos', 'seq_ctxt', 'res', 'phi', 'psi', 'protein_id'])
        xray_phi_psi.to_csv(ins.outdir / 'xray_phi_psi.csv', index=False)
    else:
        xray_phi_psi = pd.read_csv(ins.outdir / 'xray_phi_psi.csv')

    return xray_phi_psi

def get_phi_psi_predictions(ins, replace):
    if not (ins.outdir / 'phi_psi_predictions.csv').exists() or replace:
        print('Computing phi-psi for predictions')
        parser = PDBParser()
        phi_psi_predictions_ = []
        for prediction_pdb in tqdm(ins.predictions_dir.iterdir()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prediction = parser.get_structure(prediction_pdb.name, prediction_pdb)
                    try:
                        chain = list(prediction[0].get_chains())[0]
                        phi_psi_predictions_ += get_phi_psi_for_structure(ins, prediction, prediction.id)
                    except Exception as e:
                        print(prediction_pdb.name, e)

        phi_psi_predictions = pd.DataFrame(phi_psi_predictions_, columns=['pos', 'seq_ctxt', 'res', 'phi', 'psi', 'protein_id'])
        phi_psi_predictions.to_csv(ins.outdir / 'phi_psi_predictions.csv', index=False)
    else:
        phi_psi_predictions = pd.read_csv(ins.outdir / 'phi_psi_predictions.csv')
    
    return phi_psi_predictions

def get_phi_psi_for_structure(ins, protein_structure, protein_id, bfactor=False):
    protein_structure.atom_to_internal_coordinates(verbose=False)
    resultDict = structure_rebuild_test(protein_structure)
    if not resultDict['pass']:
        raise Exception('Failed to rebuild')
    # TODO if you index the chain object you get position in chain rather than index in list
    chain = next(iter((protein_structure[0].get_chains())))
    residues = list(chain.get_residues())
    phi_psi_ = []
    for i in range(ins.winsize_ctxt//2, len(residues) - ins.winsize_ctxt // 2):
        # Convert 3 char codes to 1 char codes
        seq_ctxt = ''.join([AMINO_ACID_CODES.get(r.resname, 'X') for r in ins.get_seq_ctxt(residues, i)])
        # Get the center residue
        res = ins.get_center(seq_ctxt)
        if not residues[i].internal_coord:
            psi,phi = np.nan, np.nan
        else:
            psi = residues[i].internal_coord.get_angle("psi")
            phi = residues[i].internal_coord.get_angle("phi")
            psi = psi if psi else np.nan # if psi is None, set it to np.nan
            phi = phi if phi else np.nan # if phi is None, set it to np.nan
        if bfactor:
            ave_bfactor = np.mean([atom.bfactor for atom in residues[i]])
            phi_psi_.append([i, seq_ctxt, res, phi, psi, protein_id, ave_bfactor])
        else:
            phi_psi_.append([i, seq_ctxt, res, phi, psi, protein_id])
    return phi_psi_

def seq_filter(ins):
    # Remove sequences with missing phi or psi
    ins.xray_phi_psi = ins.xray_phi_psi[~ins.xray_phi_psi.phi.isna() & ~ins.xray_phi_psi.psi.isna()]

    # remove sequences in xray that never appear in any predictions
    ins.xray_phi_psi = ins.xray_phi_psi[ins.xray_phi_psi.seq_ctxt.isin(ins.phi_psi_predictions.seq_ctxt)]

    # remove all predictions with outlier number of overlapping sequences with xray
    # or outlier length
    xray_seqs_unique = set(ins.xray_phi_psi.seq_ctxt.unique())
    grouped = ins.phi_psi_predictions.groupby('protein_id').agg(
        n_overlapping_seqs=('seq_ctxt', lambda series: len(set(series.unique()) & xray_seqs_unique)),
        length=('seq_ctxt', 'count')
    )
    # keep predictions with the mode length and number of overlapping sequences
    grouped = grouped[
        (grouped.n_overlapping_seqs == grouped.n_overlapping_seqs.mode().values[0]) & 
        (grouped.length == grouped.length.mode().values[0])]
    ins.phi_psi_predictions = ins.phi_psi_predictions[
        ins.phi_psi_predictions.protein_id.isin(grouped.index)
    ]

def get_phi_psi_af(ins, replace=False):
    if not (ins.outdir / 'af_phi_psi.csv').exists() or replace:
        print('Computing phi-psi for alphafold')
        parser = PDBParser()
        af_structure = parser.get_structure(ins.pdb_code, ins.af_fn)
        print(ins.af_fn)
        af_phi_psi = get_phi_psi_for_structure(ins, af_structure, ins.pdb_code, bfactor=True)
        af_phi_psi = pd.DataFrame(af_phi_psi, columns=['pos', 'seq_ctxt', 'res', 'phi', 'psi', 'protein_id', 'conf'])
        af_phi_psi.to_csv(ins.outdir / 'af_phi_psi.csv', index=False)
    else:
        af_phi_psi = pd.read_csv(ins.outdir / 'af_phi_psi.csv')

    return af_phi_psi