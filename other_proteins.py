from lib import DihedralAdherencePDB
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns

PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")
PROJECT_DIR = 'ml_data'

pdb_codes = [f.name.split('_')[0] for f in Path(PROJECT_DIR).iterdir() if f.is_dir()]


# da = DihedralAdherencePDB(
#     pdb_codes[0], [4,5,6,7], PDBMINE_URL, PROJECT_DIR, mode='ml',
#     weights_file='ml_runs/best_model-kde_16-32_383.pt', device='cuda'
# )
for pdb_code in pdb_codes[50:150]:
    try:
        if Path(f'{PROJECT_DIR}/{pdb_code}_win4-5-6-7/xray_phi_psi_da.csv').exists():
            continue
        da = DihedralAdherencePDB(
            pdb_code, [4,5,6,7], PDBMINE_URL, PROJECT_DIR, mode='kde',
            kdews=[0, 0, 0, 1]
        )
        if not da.has_af:
            continue
        da.load_results()
        if da.phi_psi_predictions is None:
            continue
        da.compute_das()
    except ValueError as e:
        print(e)
        continue