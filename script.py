from lib import DihedralAdherence, MultiWindowQuery
from lib.retrieve_data import retrieve_target_list
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import json

PDBMINE_URL = os.getenv("PDBMINE_URL")
winsizes = [4,5,6,7]
# ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
# winsizes = [4,5,6,7]
# PROJECT_DIR = 'tests'
# for id in ids:
#     da = DihedralAdherence(id, winsizes, PDBMINE_URL, PROJECT_DIR)
#     outdir = Path('ml_data') / (da.pdb_code.upper() + '_' + da.outdir.name.split('_')[1])
#     outdir.mkdir(exist_ok=True)
#     os.system(f'cp {da.outdir}/xray_phi_psi.csv {outdir}/xray_phi_psi.csv')
#     for winsize in winsizes:
#         os.system(f'cp {da.outdir}/phi_psi_mined_win{winsize}.csv {outdir}/phi_psi_mined_win{winsize}.csv')

ids = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
targetlist = retrieve_target_list()
skip = [targetlist.loc[id, 'pdb_code'].upper() for id in ids]

PROJECT_DIR = 'ml_data'
proteins = json.load(open('proteins.json'))
pdb_codes = proteins[:1000]
for pdb_code in pdb_codes:
    da = MultiWindowQuery(pdb_code, winsizes, PDBMINE_URL, PROJECT_DIR)
    try:
        da.compute_structure()
    except Exception as e:
        print(f'Error {pdb_code}: {e}')
        os.system(f'rm -r {da.outdir}')
        continue
    da.query_pdbmine()