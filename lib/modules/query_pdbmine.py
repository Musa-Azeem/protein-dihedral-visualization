from Bio import SeqIO
import requests
from pathlib import Path
import json
import time
from tqdm import tqdm
import pandas as pd

def query_and_process_pdbmine(ins):
    if not ins.match_outdir.exists() or len(list(ins.match_outdir.iterdir())) == 0:
        print(f'Querying PDBMine - {ins.winsize}')
        ins.match_outdir.mkdir(exist_ok=True, parents=True)
        query_pdbmine(ins)

    phi_psi_mined = get_phi_psi_mined(ins)
    phi_psi_mined_window = get_phi_psi_mined_window(ins)

    return phi_psi_mined, phi_psi_mined_window

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
                    # if phi > 180 or psi > 180:
                    #     continue
                    phi_psi_mined.append([seq, res, phi, psi, chain, protein_id])
    phi_psi_mined = pd.DataFrame(phi_psi_mined, columns=['seq', 'res', 'phi', 'psi', 'chain', 'protein_id'])
    phi_psi_mined['weight'] = ins.weight
    return phi_psi_mined

def get_phi_psi_mined_window(ins):
    seqs = []
    rows = []
    # iterate json files in the match_outdir
    for matches in ins.match_outdir.iterdir():
        matches = json.load(matches.open())
        # iterate over the matches for each sequence window
        for seq_win,v in matches.items():
            seq = seq_win[4:]   # remove numbering
            if seq in seqs:     # never need to process same sequence twice
                continue
            seqs.append(seq)
            match_id = 0
            # iterate over all the protein chains that contain matches
            for protein,seq_matches in v.items():
                protein_id, chain = protein.split('_')
                if protein_id.lower() == ins.pdb_code.lower(): # skip the protein we're looking at
                    continue
                # iterate over the matches in one chain (usually only one)
                for seq_match in seq_matches:
                    # iterate over the residues in the sequence window of this match
                    for window_pos,residue in enumerate(seq_match):
                        res, phi, psi = residue.values()
                        rows.append([seq, res, match_id, window_pos, phi, psi, chain, protein_id])
                    match_id += 1

    phi_psi_mined_window = pd.DataFrame(rows, columns=['seq', 'res', 'match_id', 'window_pos', 'phi', 'psi', 'chain', 'protein_id'])
    return phi_psi_mined_window