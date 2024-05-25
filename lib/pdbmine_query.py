from lib.modules import query_and_process_pdbmine

# Class to represent a PDBMine query for a certain sequence and window size
class PDBMineQuery():
    def __init__(self, casp_protein_id, winsize, pdbmine_url, sequence, kdews=[1, 128]):
        self.casp_protein_id = casp_protein_id
        self.winsize = winsize
        self.pdbmine_url = pdbmine_url
        self.kdews = kdews
        self.sequence = sequence

    def query_and_process_pdbmine(self):
