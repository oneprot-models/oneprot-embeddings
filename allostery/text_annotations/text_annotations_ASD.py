"""
collect_uniprot_annotations_pdb.py  —  run on LOGIN NODE (internet required)

Input files:
    /p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/train_df_pdb.csv
    /p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/test_df_pdb.csv

    Columns: pdb_id, chains, Sequences, Labels
    chains is a Python list string e.g. "['A']" or "['A', 'B']"

Workflow:
    1. For each (pdb_id, chain) pair, query the RCSB PDB API to get the
       UniProt accession mapped to that chain
    2. Fetch UniProt annotations via the UniProt REST API (once per unique ID)
    3. Write train_df_pdb_text.csv and test_df_pdb_text.csv with columns:
           pdb_id | chain | uniprot_id | annotation_text
"""

import os
import re
import ast
import time
import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo'
SPLITS = {
    'train': 'train_df_pdb.csv',
    'test':  'test_df_pdb.csv',
}

# RCSB PDB GraphQL endpoint — maps chain -> UniProt accession
RCSB_URL = 'https://data.rcsb.org/graphql'

UNIPROT_URL   = 'https://rest.uniprot.org/uniprotkb/{accession}?format=json'
MAX_RETRIES   = 3
RETRY_DELAY   = 2
REQUEST_DELAY = 0.2


# ---------------------------------------------------------------------------
# Step 1: PDB -> UniProt mapping via RCSB GraphQL
# ---------------------------------------------------------------------------
RCSB_QUERY = """
query($id: String!) {
  entry(entry_id: $id) {
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        auth_asym_ids
        uniprot_ids
      }
    }
  }
}
"""

def fetch_uniprot_ids_for_pdb(pdb_id: str) -> dict[str, str]:
    """
    Query RCSB for a PDB entry and return a mapping:
        {chain_id -> uniprot_accession}
    for all chains that have a UniProt mapping.
    If a chain has multiple UniProt IDs, the first is used.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                RCSB_URL,
                json={'query': RCSB_QUERY, 'variables': {'id': pdb_id.upper()}},
                timeout=15
            )
            if resp.status_code != 200:
                print(f"  [{pdb_id}] RCSB HTTP {resp.status_code}, attempt {attempt + 1}")
                time.sleep(RETRY_DELAY)
                continue

            data = resp.json()
            entities = (data.get('data') or {}).get('entry') or {}
            entities = entities.get('polymer_entities') or []

            chain_to_uniprot = {}
            for entity in entities:
                ids = entity.get('rcsb_polymer_entity_container_identifiers', {})
                chains   = ids.get('auth_asym_ids') or []
                uniprots = ids.get('uniprot_ids') or []
                if uniprots:
                    for chain in chains:
                        chain_to_uniprot[chain] = uniprots[0]  # take first if multiple
            return chain_to_uniprot

        except requests.RequestException as e:
            print(f"  [{pdb_id}] Request error: {e}, attempt {attempt + 1}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    print(f"  [{pdb_id}] Failed to fetch from RCSB after {MAX_RETRIES} attempts")
    return {}


# ---------------------------------------------------------------------------
# Step 2: UniProt annotation fetch (identical to previous scripts)
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Strip PubMed IDs, ECO tags, GO terms, EC numbers and other reference noise."""
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'\([^)]*(?:PubMed|GO:|EC |MIM:|Reactome|EMBL|UniProtKB)[^)]*\)', '', text)
    text = re.sub(r'\bEC\s+\d+\.\d+\.\d+\.\d+\b', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([.,;])', r'\1', text)
    return text.strip()


def _parse_uniprot_json(data: dict) -> str:
    """Build a single annotation string from a UniProt JSON response."""
    parts = []

    # PROTEIN NAME
    try:
        name = data['proteinDescription']['recommendedName']['fullName']['value']
        parts.append(f"PROTEIN: {_clean_text(name)}")
    except (KeyError, IndexError):
        pass

    # GENE
    try:
        genes = [g['geneName']['value'] for g in data.get('genes', [])
                 if 'geneName' in g]
        if genes:
            parts.append(f"GENE: {', '.join(_clean_text(g) for g in genes)}")
    except (KeyError, TypeError):
        pass

    # ORGANISM
    try:
        parts.append(f"ORGANISM: {_clean_text(data['organism']['scientificName'])}")
    except KeyError:
        pass

    # FUNCTION
    try:
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'FUNCTION':
                texts = [_clean_text(t['value']) for t in comment.get('texts', [])]
                texts = [t for t in texts if t]
                if texts:
                    parts.append(f"FUNCTION: {' '.join(texts)}")
                break
    except (KeyError, TypeError):
        pass

    # SUBUNIT
    try:
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'SUBUNIT':
                texts = [_clean_text(t['value']) for t in comment.get('texts', [])]
                texts = [t for t in texts if t]
                if texts:
                    parts.append(f"SUBUNIT: {' '.join(texts)}")
                break
    except (KeyError, TypeError):
        pass

    # SUBCELLULAR LOCATION
    try:
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                locations = []
                for loc_block in comment.get('subcellularLocations', []):
                    loc = loc_block.get('location', {})
                    if isinstance(loc, dict):
                        val = loc.get('value')
                        if val:
                            locations.append(_clean_text(val))
                    elif isinstance(loc, list):
                        for item in loc:
                            val = item.get('value') if isinstance(item, dict) else None
                            if val:
                                locations.append(_clean_text(val))
                if locations:
                    parts.append(f"SUBCELLULAR LOCATION: {', '.join(locations)}")
                break
    except (KeyError, TypeError):
        pass

    # KEYWORDS
    try:
        keywords = [_clean_text(kw['name']) for kw in data.get('keywords', [])]
        keywords = [k for k in keywords if k]
        if keywords:
            parts.append(f"KEYWORDS: {', '.join(keywords)}")
    except (KeyError, TypeError):
        pass

    return '. '.join(parts)


def fetch_uniprot_annotation(accession: str) -> str | None:
    url = UNIPROT_URL.format(accession=accession)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return _parse_uniprot_json(resp.json())
            elif resp.status_code == 404:
                print(f"  [{accession}] Not found in UniProt (404)")
                return None
            else:
                print(f"  [{accession}] HTTP {resp.status_code}, attempt {attempt + 1}")
        except requests.RequestException as e:
            print(f"  [{accession}] Request error: {e}, attempt {attempt + 1}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    print(f"  [{accession}] Failed after {MAX_RETRIES} attempts")
    return None


# ---------------------------------------------------------------------------
# Step 3: process one split file
# ---------------------------------------------------------------------------
def parse_chains(chains_str: str) -> list[str]:
    """
    Parse the chains column string e.g. "['A']" or "['A', 'B']"
    into a Python list. Falls back to splitting on commas if ast fails.
    """
    try:
        result = ast.literal_eval(chains_str)
        if isinstance(result, list):
            return [str(c).strip() for c in result]
    except (ValueError, SyntaxError):
        pass
    # Fallback: strip brackets and split
    cleaned = chains_str.strip().strip("[]").replace("'", "").replace('"', '')
    return [c.strip() for c in cleaned.split(',') if c.strip()]


def process_split(split_name: str, csv_path: str,
                  pdb_cache: dict, annotations_cache: dict) -> pd.DataFrame:
    """
    Read a split CSV, resolve (pdb_id, chain) -> UniProt ID via RCSB,
    fetch UniProt annotations, return output DataFrame.
    """
    df = pd.read_csv(csv_path)
    print(f"  {os.path.basename(csv_path)}: {len(df)} rows")

    rows = []
    # Collect unique PDB IDs to batch RCSB lookups
    unique_pdbs = df['pdb_id'].dropna().str.lower().unique()
    new_pdbs    = [p for p in unique_pdbs if p not in pdb_cache]

    if new_pdbs:
        print(f"  Querying RCSB for {len(new_pdbs)} PDB entries "
              f"({len(unique_pdbs) - len(new_pdbs)} cached)...")
        for pdb in tqdm(new_pdbs, desc=f"RCSB [{split_name}]"):
            pdb_cache[pdb] = fetch_uniprot_ids_for_pdb(pdb)
            time.sleep(REQUEST_DELAY)
    else:
        print(f"  All PDB entries already cached.")

    # Resolve UniProt IDs for all (pdb, chain) pairs
    uniprot_ids_needed = set()
    resolved_rows = []   # (pdb_id, chain, uniprot_id) per input row
    for _, row in df.iterrows():
        pdb    = str(row['pdb_id']).strip().lower()
        chains = parse_chains(str(row['chains']))
        chain_map = pdb_cache.get(pdb, {})
        for chain in chains:
            uid = chain_map.get(chain)
            if uid is None:
                print(f"  No UniProt mapping for {pdb} chain {chain}")
            else:
                uniprot_ids_needed.add(uid)
            resolved_rows.append((pdb, chain, uid))

    # Fetch UniProt annotations for new IDs
    new_uids = [u for u in uniprot_ids_needed if u not in annotations_cache]
    if new_uids:
        print(f"  Fetching {len(new_uids)} UniProt annotations "
              f"({len(uniprot_ids_needed) - len(new_uids)} cached)...")
        for uid in tqdm(new_uids, desc=f"UniProt [{split_name}]"):
            annotations_cache[uid] = fetch_uniprot_annotation(uid)
            time.sleep(REQUEST_DELAY)
    else:
        print(f"  All UniProt IDs already cached.")

    # Build output rows
    for pdb, chain, uid in resolved_rows:
        annotation = annotations_cache.get(uid, '') if uid else ''
        rows.append({
            'pdb_id':          pdb,
            'chain':           chain,
            'uniprot_id':      uid or '',
            'annotation_text': annotation or '',
        })

    return pd.DataFrame(rows, columns=['pdb_id', 'chain', 'uniprot_id', 'annotation_text'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("Collecting UniProt annotations from PDB ID + chain (RCSB lookup)")
    print("=" * 70)

    # Shared caches so each PDB / UniProt ID is fetched at most once
    pdb_cache:         dict[str, dict] = {}   # pdb_id -> {chain -> uniprot_id}
    annotations_cache: dict[str, str]  = {}   # uniprot_id -> annotation_text

    for split_name, filename in SPLITS.items():
        csv_path = os.path.join(DATA_DIR, filename)
        print(f"\n[{split_name}]")

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping.")
            continue

        df_out = process_split(split_name, csv_path, pdb_cache, annotations_cache)

        out_filename = filename.replace('.csv', '_text.csv')
        out_path     = os.path.join(DATA_DIR, out_filename)
        df_out.to_csv(out_path, index=False)

        n_with = (df_out['annotation_text'] != '').sum()
        print(f"  Wrote {len(df_out)} rows -> {out_path} "
              f"({n_with} with annotation text, {len(df_out) - n_with} empty)")

    print(f"\n{'=' * 70}")
    print(f"Done!")
    print(f"  Unique PDB entries queried:       {len(pdb_cache)}")
    print(f"  Unique UniProt IDs fetched:       "
          f"{sum(1 for v in annotations_cache.values() if v is not None)}/"
          f"{len(annotations_cache)}")
    print(f"{'=' * 70}")