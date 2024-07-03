from lib import DihedralAdherence
from load_dotenv import load_dotenv
import os
load_dotenv()

PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")
PROJECTS_DIR = 'casp_da'

proteins = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
winsizes = [4,5,6,7]
kdews = [1, 32, 64, 128]

for protein in proteins:
    da = DihedralAdherence(protein, winsizes, PDBMINE_URL, PROJECTS_DIR, kdews, mode='kde_af')
    da.compute_structures()
    da.query_pdbmine()
    da.compute_das()