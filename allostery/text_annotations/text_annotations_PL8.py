"""
collect_uniprot_annotations_allosteric.py  —  run on LOGIN NODE (internet required)

Split files: /p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/
             splits/allosteric/train.txt, valid.txt, test.txt
             IDs of the form:  1ap8_A_M7G_214

CSV file:    PL_part8_20230317_matrix_liganded_allosteric.csv
             Cavity column format:  {pdb}-{chain}-{UniProtID}-{ligand}-{number}_CAVITY_...
             e.g.  5ayf-A-Q8WTS6-SAM-401_CAVITY_N1_liganded_allosteric

Workflow:
  1. Parse Cavity column -> extract UniProt ID and build a lookup key
     (pdb, chain, ligand, number) -> uniprot_id
  2. For each split ID  e.g. "1ap8_A_M7G_214"
     parse into          (pdb, chain, ligand, number)
     and look up the matching UniProt ID from the CSV
  3. Fetch UniProt annotations via REST API (once per unique UniProt ID)
  4. Write train_text.txt / valid_text.txt / test_text.txt with columns:
        split_id | uniprot_id | annotation_text
"""

import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPLIT_DIR  = '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/splits/competitive'
CSV_FILE   = '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/PL_part8_20230317_matrix_liganded_orthosteric_competitive.csv'
SPLITS     = ['train', 'valid', 'test']

UNIPROT_URL  = 'https://rest.uniprot.org/uniprotkb/{accession}?format=json'
MAX_RETRIES  = 3
RETRY_DELAY  = 2    # seconds between retries
REQUEST_DELAY = 0.2  # polite delay between API calls


# ---------------------------------------------------------------------------
# Step 1: parse the CSV Cavity column into a lookup dict
#         key: (pdb, chain, ligand, number)  ->  value: uniprot_id
# ---------------------------------------------------------------------------
def parse_cavity_id(cavity: str):
    """
    Cavity format: {pdb}-{chain}-{UniProtID}-{ligand}-{number}_CAVITY_...
    e.g.           5ayf-A-Q8WTS6-SAM-401_CAVITY_N1_liganded_allosteric

    Returns (pdb, chain, uniprot_id, ligand, number) or None if unparseable.

    UniProt accession IDs are 6 or 10 characters matching [OPQ][0-9][A-Z0-9]{3}[0-9]
    or [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}
    We identify them by position: field index 2 in the dash-split prefix.
    """
    # Drop suffix after '_CAVITY'
    prefix = cavity.split('_CAVITY')[0]      # e.g. 5ayf-A-Q8WTS6-SAM-401
    parts  = prefix.split('-')               # ['5ayf', 'A', 'Q8WTS6', 'SAM', '401']

    if len(parts) < 5:
        return None

    pdb       = parts[0].lower()
    chain     = parts[1]
    uniprot   = parts[2]
    # ligand and number are the remaining parts (ligand can contain dashes itself)
    # the number is always the last part; ligand is everything between uniprot and number
    number    = parts[-1]
    ligand    = '-'.join(parts[3:-1])        # handles multi-dash ligand names

    return pdb, chain, uniprot, ligand, number


def build_cavity_lookup(csv_file: str) -> dict:
    """
    Read the CSV and build:
      {(pdb, chain, ligand, number) -> uniprot_id}
    """
    print(f"Reading CSV: {csv_file}")
    df = pd.read_csv(csv_file, usecols=['Cavity'])
    lookup = {}
    skipped = 0
    for cavity in df['Cavity']:
        parsed = parse_cavity_id(str(cavity))
        if parsed is None:
            skipped += 1
            continue
        pdb, chain, uniprot, ligand, number = parsed
        key = (pdb, chain, ligand, number)
        lookup[key] = uniprot

    print(f"  Built lookup with {len(lookup)} entries ({skipped} unparseable rows skipped)")
    return lookup


# ---------------------------------------------------------------------------
# Step 2: parse split file IDs into the same key format
# ---------------------------------------------------------------------------
def parse_split_id(split_id: str):
    """
    Split ID format: {pdb}_{chain}_{ligand}_{number}
    e.g.             1ap8_A_M7G_214

    Returns (pdb, chain, ligand, number) or None.
    Note: ligand can contain underscores so we split on the first,
    second, and last underscore only.
    """
    parts = split_id.strip().split('_')
    if len(parts) < 4:
        return None

    pdb    = parts[0].lower()
    chain  = parts[1]
    number = parts[-1]
    ligand = '_'.join(parts[2:-1])   # handles multi-underscore ligand names

    return pdb, chain, ligand, number


# ---------------------------------------------------------------------------
# Step 3: UniProt REST API fetch
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Strip PubMed IDs, ECO tags, GO terms, EC numbers and other reference noise."""
    # ECO evidence tags  {ECO:0000269|PubMed:12345678}
    text = re.sub(r'\{[^}]*\}', '', text)
    # Parenthesised reference IDs
    text = re.sub(r'\([^)]*(?:PubMed|GO:|EC |MIM:|Reactome|EMBL|UniProtKB)[^)]*\)', '', text)
    # Bare EC numbers
    text = re.sub(r'\bEC\s+\d+\.\d+\.\d+\.\d+\b', '', text)
    # Cleanup
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
                print(f"  [{accession}] Not found (404)")
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
# Step 4: process one split
# ---------------------------------------------------------------------------
def process_split(split_name: str, cavity_lookup: dict,
                  annotations_cache: dict) -> pd.DataFrame:
    """
    Read {split_name}.txt, resolve each ID to a UniProt accession via
    cavity_lookup, fetch annotations, return a DataFrame with columns:
        split_id | uniprot_id | annotation_text
    """
    txt_path = os.path.join(SPLIT_DIR, f'{split_name}.txt')
    if not os.path.exists(txt_path):
        print(f"  Warning: {txt_path} not found, skipping.")
        return pd.DataFrame(columns=['split_id', 'uniprot_id', 'annotation_text'])

    with open(txt_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]
    print(f"  {split_name}.txt: {len(split_ids)} IDs")

    # Resolve split IDs -> UniProt IDs
    resolved = {}   # split_id -> uniprot_id
    unresolved = []
    for sid in split_ids:
        key = parse_split_id(sid)
        if key is None:
            print(f"  Cannot parse split ID: '{sid}'")
            unresolved.append(sid)
            continue
        uid = cavity_lookup.get(key)
        if uid is None:
            print(f"  No CSV match for: '{sid}' (key={key})")
            unresolved.append(sid)
            continue
        resolved[sid] = uid

    print(f"  Resolved: {len(resolved)}/{len(split_ids)}  "
          f"({len(unresolved)} unmatched)")

    # Fetch annotations for any UniProt IDs not yet cached
    new_ids = [uid for uid in set(resolved.values()) if uid not in annotations_cache]
    if new_ids:
        print(f"  Fetching {len(new_ids)} new UniProt IDs "
              f"({len(set(resolved.values())) - len(new_ids)} already cached)...")
        for uid in tqdm(new_ids, desc=f"UniProt API [{split_name}]"):
            annotations_cache[uid] = fetch_uniprot_annotation(uid)
            time.sleep(REQUEST_DELAY)
    else:
        print(f"  All UniProt IDs already cached.")

    # Build output rows
    rows = []
    for sid, uid in resolved.items():
        annotation = annotations_cache.get(uid) or ''
        rows.append({'split_id': sid, 'uniprot_id': uid, 'annotation_text': annotation})

    return pd.DataFrame(rows, columns=['split_id', 'uniprot_id', 'annotation_text'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("Collecting UniProt annotations for allosteric splits")
    print("=" * 70)

    # Build the lookup once from the CSV
    cavity_lookup = build_cavity_lookup(CSV_FILE)

    # Shared annotation cache so each UniProt ID is fetched at most once
    annotations_cache: dict[str, str | None] = {}

    for split_name in SPLITS:
        print(f"\n[{split_name}]")
        df = process_split(split_name, cavity_lookup, annotations_cache)

        if df.empty:
            continue

        out_path = os.path.join(SPLIT_DIR, f'{split_name}_text.csv')
        df.to_csv(out_path, index=False)
        n_with = (df['annotation_text'] != '').sum()
        print(f"  Wrote {len(df)} rows -> {out_path} "
              f"({n_with} with text, {len(df) - n_with} empty)")

    print(f"\n{'=' * 70}")
    print(f"Done! Unique UniProt IDs fetched: "
          f"{sum(1 for v in annotations_cache.values() if v is not None)}/"
          f"{len(annotations_cache)}")
    print(f"{'=' * 70}")