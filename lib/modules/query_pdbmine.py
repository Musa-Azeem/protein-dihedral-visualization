from Bio import SeqIO
import requests
from pathlib import Path
import json
import time
from tqdm import tqdm
import pandas as pd

def query_and_process_pdbmine(ins):
    if not ins.match_outdir.exists() or len(list(ins.match_outdir.iterdir())) == 0:
        ins.match_outdir.mkdir(exist_ok=True, parents=True)
        query_pdbmine(ins)
    phi_psi_mined = get_phi_psi_mined(ins)

    return phi_psi_mined

# Get Phi-Psi distribution from PDBMine
def query_pdbmine(ins):
    print(f'Querying PDBMine - {ins.winsize}')
    code_length = 1
    broken_chains = []

    # break chain into sections of length 100 - for memory reasons
    # overlap by window_size-1
    for i in range(0, len(ins.sequence), 100-ins.winsize+1):
        broken_chains.append(ins.sequence[i:i+100])

    for i,chain in enumerate(tqdm(broken_chains)):
        if len(chain) < ins.winsize: # in case the last chain is too short
            continue

        response = requests.post(
            ins.pdbmine_url + '/v1/api/query',
            json={
                "residueChain": chain,
                "codeLength": code_length,
                "windowSize": ins.winsize
            }
        )
        assert(response.ok)
        print(response.json())
        query_id = response.json().get('queryID')
        assert(query_id)

        time.sleep(60)
        while(True):
            response = requests.get(ins.pdbmine_url + f'/v1/api/query/{query_id}')
            if response.ok and response.json().get('frames'):
                matches = response.json()['frames']
                break
            else:
                print('Waiting')
                time.sleep(15)
        print(f'Received matches - {i}')
        json.dump(matches, open(ins.match_outdir / f'matches-win{ins.winsize}_{i}.json', 'w'), indent=4)

def get_phi_psi_mined(ins):
    seqs = []
    phi_psi_mined = []
    for matches in ins.match_outdir.iterdir():
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
                    center_res = seq_match[ins.get_center_idx()]
                    res, phi, psi = center_res.values()
                    phi_psi_mined.append([seq, res, phi, psi, chain, protein_id])
    phi_psi_mined = pd.DataFrame(phi_psi_mined, columns=['seq', 'res', 'phi', 'psi', 'chain', 'protein_id'])
    phi_psi_mined['weight'] = ins.weight
    return phi_psi_mined