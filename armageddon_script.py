from dotenv import load_dotenv
import os
import requests
import time
import pandas as pd
load_dotenv()

PDBMINE_URL = os.getenv("GREEN_PDBMINE_URL")

amino_acids = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V"
]
n_aa = len(amino_acids)
def get_amino_acid_seq(length, i):
    seq = ""
    for j in range(length):
        seq += amino_acids[(i//(n_aa**j))%n_aa]
    return seq

def query_pdbmine(seq):
    response = requests.post(
    PDBMINE_URL + '/v1/api/query',
        json={
            "residueChain": seq,
            "codeLength": 1,
            "windowSize": len(seq)
        }
    )
    if not response.ok:
        return None
    
    query_id = response.json().get('queryID')
    if not query_id:
        return None
    
    time.sleep(2)
    while(True):
        response = requests.get(PDBMINE_URL + f'/v1/api/query/{query_id}')
        if response.ok:
            matches = response.json()['frames']
            break
        else:
            print('waiting')
            time.sleep(15)
    return matches

results = []
for length in [5]:
    result = []
    for i in range(n_aa**length):
        seq = get_amino_acid_seq(length,i)
        start = time.time()
        matches = query_pdbmine(seq)
        end = time.time()
        n_matches = 0
        for seq,match in matches.items():
            for protein, m in match.items():
                n_matches += len(m)
        result.append([seq, n_matches])
        print(seq, n_matches, (start-end))
        if i > 5:
            break
    result = pd.DataFrame(result, columns=['seq', 'n_matches'])
    result.to_csv(f'search/win{length}.csv', index=False)
    results.append(result)