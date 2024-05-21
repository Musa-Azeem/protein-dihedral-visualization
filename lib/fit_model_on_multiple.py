from lib import DihedralAdherence
import pandas as pd

def fit_model_on_multiple(protein_ids: list, winsize, winsize_ctxt, pdbmine_url):
    grouped_preds = []
    grouped_preds_md = []
    for protein_id in protein_ids:
        da = DihedralAdherence(protein_id, winsize, winsize_ctxt, pdbmine_url)
        da.compute_structures()
        da.query_pdbmine()
        da.compute_mds()
        grouped_preds.append(da.grouped_preds.sort_values['protein_id'])

    grouped_preds = pd.concat(grouped_preds)