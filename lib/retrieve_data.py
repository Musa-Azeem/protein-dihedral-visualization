import pandas as pd
import requests
from pathlib import Path
import re
from Bio.PDB import PDBList
import os

TARGETLIST_URL = 'https://predictioncenter.org/casp14/targetlist.cgi?type=csv'
PREDICTIONS_URL = 'https://predictioncenter.org/download_area/CASP14/predictions/regular/{casp_protein_id}.tar.gz'
RESULTS_URL = 'https://predictioncenter.org/download_area/CASP14/results/tables/casp14.res_tables.T.tar.gz'

# def_retrieve_data():


def retrieve_target_list():
    targetlist_file = Path('targetlist.csv')
    if not targetlist_file.exists():
        with open(targetlist_file, 'wb') as f:
            f.write(requests.get(TARGETLIST_URL).content)
    targetlist = pd.read_csv(targetlist_file, sep=';').set_index('Target')

    def re_pdb_code(x):
        m = re.search(r"\b\d[0-9a-z]{3}\b", x)
        return m.group() if m else ''
    targetlist['pdb_code'] = targetlist['Description'].apply(re_pdb_code)

    return targetlist

def retrieve_pdb_file(pdb_code):
    pdbl = PDBList()
    xray_fn = pdbl.retrieve_pdb_file(pdb_code, pdir='pdb', file_format='pdb', obsolete=False)
    return xray_fn

def retrieve_casp_predictions(casp_protein_id):
    predictions_url = PREDICTIONS_URL.format(casp_protein_id=casp_protein_id)
    predictions_dir = Path(f'./casp-predictions/')
    if not (predictions_dir / casp_protein_id).exists():
        predictions_dir.mkdir(exist_ok=True)
        os.system(f'wget -O {predictions_dir}/{casp_protein_id}.tar.gz {predictions_url}')
        os.system(f'tar -xvf {predictions_dir}/{casp_protein_id}.tar.gz -C {predictions_dir}')
    # Return path to the extracted directory
    return predictions_dir / casp_protein_id

def retrieve_casp_results(casp_protein_id):
    results_dir = Path('casp-results')
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)
        os.system(f'wget -O {results_dir / "casp14.res_tables.T.tar.gz"} {RESULTS_URL}')
        os.system(f'tar -xvf {results_dir / "casp14.res_tables.T.tar.gz"} -C {results_dir}')
    results_file = results_dir / f'{casp_protein_id}.txt'
    # some files are named differently
    if not results_file.exists():
        results_file = Path(results_dir / f'{casp_protein_id}-D1.txt')
    results = pd.read_csv(results_file, sep='\s+')
    results = results[results.columns[1:]] # remove line number column
    results['Model'] = results['Model'].apply(lambda x: x.split('-')[0])
    return results