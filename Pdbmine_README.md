# DiReDB

A database of 3D coordinates and dihedral angles mined from over 144,000 protein structures contained within PDB (Protein DataBank).

This program provides a RESTful API to query (GET) information from the database. The program loads Database.dat and listens on port 8077 by default but these are configuable using the DIREDB_PORT environment variable to set the port and the DIREDB_DB environment variable to set the database file name and path. It also allows a user to get the dihedral angles of any protein. Finally it allows a user to query all of the proteins for an chain of amino acid residues and see all of the dihdral angles in the protein where that chain has been observed.

## Build and deploy with Docker using the following commands

Note that right now, this image uses about 5.75 GB of memory when it runs because the entire DB is stored in memory for speed.

```shell

docker build -t yourhubusername/yourrepo:diredb .
docker run --rm -it -p 8077:8077 diredb

```

## Current API's include the following

### To Browse the Database

View the Header records from the database

- /v1/api/database/header

View the entire Manifest of the database

- /v1/api/database/manifest

View one Manifest record in the database

- /v1/api/database/manifest/{index}

View a summary of the jump table

- /v1/api/database/jumptable

View a list of the keys in the jump table

- /v1/api/database/jumptable/keys

View a jump table entry by residue and index

- /v1/api/database/jumptable/residue/{residue}/index/{index}

View a summary of the set member array

- /v1/api/database/setmember

View a specific entry in the set member array

- /v1/api/database/setmember/{index}

View an array of set members by starting index and length

- /v1/api/database/setmember/{index}/length/{length}

View a summary of the proteins loaded from the pdb database

- /v1/api/database/pdbdata

View a particular protein by protein ID (i.e. 1RWD)

- /v1/api/database/pdbdata/{proteinID}

View a summary of the chain array in the database

- /v1/api/database/chain

View a specific entry in the chain array

- /v1/api/database/chain/{index}

View a summary of the dihedral array in the database

- /v1/api/database/dihedral

View a specific entry in the dihedral database

- /v1/api/database/dihedral/{index}

View a summary of the residue array in the database

- /v1/api/database/residue

View a particular residue by index

- /v1/api/database/residue/{index}

View an array of residues by index and length

- /v1/api/database/residue/{index}/length/{length}

### For Proteins

View a list of all proteins

- /v1/api/protein/

View a particular protein

- /v1/api/protein/{proteinID}

View a particular chain of a protein

- /v1/api/protein/{proteinID}?chain={chainID}

### For Queries

Create a query (POST)

- /v1/api/query

```json

{
  "residueChain": "AKYVCKICGYIYDEDAGDPDNGVSPG",
  "codeLength": 1,
  "windowSize": 7
}

```
