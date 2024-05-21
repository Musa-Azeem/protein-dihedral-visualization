from Bio import SeqIO
import requests
from pathlib import Path
import json
import time
from tqdm import tqdm
import pandas as pd

def query_and_process_pdbmine(ins):
    if not Path(f'cache/{ins.casp_protein_id}/matches-{ins.winsize}').exists():
        query_pdbmine(ins.winsize)
    if not Path(f'cache/{ins.casp_protein_id}/matches-{ins.winsize_ctxt}').exists():
        query_pdbmine(ins.winsize_ctxt)

    phi_psi_mined = get_phi_psi_mined(ins, ins.winsize)
    phi_psi_mined_ctxt = get_phi_psi_mined(ins, ins.winsize_ctxt)

    return phi_psi_mined, phi_psi_mined_ctxt

# Get Phi-Psi distribution from PDBMine
def query_pdbmine(ins, window_size):
    print(f'Querying PDBMine - {window_size}')
    record = next(iter(SeqIO.parse(ins.xray_fn, "pdb-seqres")))
    residue_chain = str(record.seq)

    code_length = 1
    broken_chains = []

    # break chain into sections of length 100 - for memory reasons
    # overlap by window_size-1
    for i in range(0, len(residue_chain), 100-window_size+1):
        broken_chains.append(residue_chain[i:i+100])

    match_outdir = Path(f'cache/{ins.casp_protein_id}/matches-{window_size}')
    match_outdir.mkdir(exist_ok=False, parents=True)

    for i,chain in enumerate(tqdm(broken_chains)):
        if len(chain) < window_size: # in case the last chain is too short
            continue

        response = requests.post(
            ins.pdbmine_url + '/v1/api/query',
            json={
                "residueChain": chain,
                "codeLength": code_length,
                "windowSize": window_size
            }
        )
        assert(response.ok)
        print(response.json())
        query_id = response.json().get('queryID')
        assert(query_id)

        time.sleep(60)
        while(True):
            response = requests.get(ins.pdbmine_url + f'/v1/api/query/{query_id}')
            if response.ok:
                matches = response.json()['frames']
                break
            else:
                print('Waiting')
                time.sleep(15)
        print(f'Received matches - {i}')
        json.dump(matches, open(match_outdir / f'matches-win{window_size}_{i}.json', 'w'), indent=4)

def get_phi_psi_mined(ins, window_size):
    seqs = []
    phi_psi_mined = []
    for matches in Path(f'cache/{ins.casp_protein_id}/matches-{window_size}').iterdir():
        matches = json.load(matches.open())
        for seq_win,v in matches.items():
            seq = seq_win[4:]
            if seq in seqs:
                continue
            seqs.append(seq)
            for protein,seq_matches in v.items():
                protein_id, chain = protein.split('_')
                if protein_id.lower() == ins.pdb_code.lower(): # skip the protein we're looking at
                    continue
                for seq_match in seq_matches:
                    center_res = seq_match[window_size//2]
                    res, phi, psi = center_res.values()
                    phi_psi_mined.append([seq, res, phi, psi, chain, protein_id])
    phi_psi_mined = pd.DataFrame(phi_psi_mined, columns=['seq', 'res', 'phi', 'psi', 'chain', 'protein_id'])
    phi_psi_mined.to_csv(ins.outdir / f'phi_psi_mined_win{window_size}.csv', index=False)
    return phi_psi_mined