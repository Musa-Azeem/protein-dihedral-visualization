from lib import DihedralAdherence
from dotenv import load_dotenv
import os
load_dotenv()

PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")
PROJECTS_DIR = 'casp_da'

# proteins = ['T1024', 'T1096', 'T1027', 'T1082', 'T1091', 'T1058', 'T1049', 'T1030', 'T1056', 'T1038', 'T1025', 'T1028']
# winsizes = [4,5,6,7]
# kdews = [1, 32, 64, 128]

# for protein in proteins:
#     da = DihedralAdherence(protein, winsizes, PDBMINE_URL, PROJECTS_DIR, kdews, mode='kde_af')
#     da.compute_structures()
#     da.query_pdbmine()
#     da.load_results()
#     da.compute_das(replace=True)

proteins = ['T1030-D2', 'T1024-D1', 'T1032-D1', 'T1041-D1', 'T1029-D1', 'T1053-D1', 'T1027-D1']
for protein in proteins[:1]:
    da = DihedralAdherence(
        protein, [4,5,6,7], PDBMINE_URL, PROJECTS_DIR, mode='ml',
        weights_file='ml_data/best_model_kde_64-64_390.pt', device='cuda')
    da.compute_structures()
    da.query_pdbmine()
    da.compute_das(replace=True)