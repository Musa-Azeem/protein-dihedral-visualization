from lib import DihedralAdherence, MultiWindowQuery
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import json

WINDOW_SIZE = 5
WINDOW_SIZE_CONTEXT = 6
PDBMINE_URL = os.getenv("PDBMINE_URL")
PROJECT_DIR = 'ml_data'
winsizes = [4,5,6,7]

# proteins = json.load(open('proteins.json'))
# pdb_codes = proteins[:1000]
pdb_codes = ['2lkf']
for pdb_code in pdb_codes:
    da = MultiWindowQuery(pdb_code, winsizes, PDBMINE_URL, PROJECT_DIR)
    try:
        da.compute_structure()
    except Exception as e:
        print(f'Error {pdb_code}: {e}')
        os.system(f'rm -r {da.outdir}')
        continue
    da.query_pdbmine()