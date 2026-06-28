"""
collect_uniprot_annotations.py  —  run on LOGIN NODE (internet access required)

For each split CSV in 0.3/ (train.csv, valid.csv, test.csv):
  1. Collect unique UniProt IDs
  2. Fetch annotations from the UniProt API (function, protein name,
     organism, keywords, gene names)
  3. Combine into a single annotation text string per UniProt ID
  4. Write 0.3/train_text.csv, valid_text.csv, test_text.csv with columns:
       uniprot_id | h5_identifier | annotation_text
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPLIT_DIR   = '0.3'
SPLITS      = {
    'train': 'train.csv',
    'valid': 'valid.csv',
    'test':  'test.csv',
}
UNIPROT_COL = 'Uniprot ID'
H5_COL      = 'h5_identifier'

# UniProt REST API — returns JSON for a single accession
UNIPROT_URL = 'https://rest.uniprot.org/uniprotkb/{accession}?format=json'

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2   # seconds between retries
REQUEST_DELAY = 0.2  # polite delay between API calls


# ---------------------------------------------------------------------------
# UniProt fetching
# ---------------------------------------------------------------------------
def fetch_uniprot_annotation(accession: str) -> str | None:
    """
    Fetch a UniProt entry via the REST API and build a single annotation
    string combining:
        - Recommended protein name
        - Gene name(s)
        - Organism
        - Function (CC -FUNCTION comment)
        - Keywords

    Returns None if the fetch fails after retries.
    """
    url = UNIPROT_URL.format(accession=accession)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return _parse_uniprot_json(resp.json(), accession)
            elif resp.status_code == 404:
                print(f"  [{accession}] Not found in UniProt (404)")
                return None
            else:
                print(f"  [{accession}] HTTP {resp.status_code}, attempt {attempt+1}")
        except requests.RequestException as e:
            print(f"  [{accession}] Request error: {e}, attempt {attempt+1}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    print(f"  [{accession}] Failed after {MAX_RETRIES} attempts")
    return None


def _clean_text(text: str) -> str:
    """
    Remove reference/ID noise from free-text annotation fields:
      - PubMed IDs          e.g. (PubMed:12345678)
      - ECO evidence tags   e.g. {ECO:0000269|PubMed:12345678}
      - Gene Ontology refs  e.g. (GO:0005634)
      - Enzyme Commission   e.g. (EC 2.7.11.1) or EC 2.7.11.1
      - Sequence positions  e.g. (By similarity), (Probable)
      - Leftover empty parens / double spaces
    """
    import re

    # ECO evidence tags (curly-brace blocks), e.g. {ECO:0000269|PubMed:12345678}
    text = re.sub(r'\{[^}]*\}', '', text)
    # Parenthesised IDs: PubMed, GO, EC, MIM, Reactome, EMBL, etc.
    text = re.sub(r'\([^)]*(?:PubMed|GO:|EC |MIM:|Reactome|EMBL|UniProtKB)[^)]*\)', '', text)
    # Bare EC numbers, e.g.  EC 2.7.11.1
    text = re.sub(r'\bEC\s+\d+\.\d+\.\d+\.\d+\b', '', text)
    # Cleanup: empty parentheses, extra whitespace, leading/trailing commas
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([.,;])', r'\1', text)
    return text.strip()


def _parse_uniprot_json(data: dict, accession: str) -> str:
    """
    Extract relevant fields from a UniProt JSON response into one string.
    Labels are written in ALL CAPS.  Reference IDs (PubMed, GO, EC, etc.)
    are stripped from all free-text fields.
    """
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
        organism = data['organism']['scientificName']
        parts.append(f"ORGANISM: {_clean_text(organism)}")
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

    # SUBUNIT / INTERACTIONS
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
                    # 'location' is a dict {'value': 'Nucleus', ...}, not a list
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_annotations_for_split(csv_path: str, annotations_cache: dict) -> pd.DataFrame:
    """
    Read a split CSV, fetch annotations for any UniProt IDs not yet in cache,
    and return a DataFrame with columns: uniprot_id, h5_identifier, annotation_text.
    """
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in [UNIPROT_COL, H5_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in {csv_path}. "
                         f"Available: {df.columns.tolist()}")

    # Fetch annotations for UniProt IDs not yet cached
    unique_ids = df[UNIPROT_COL].dropna().astype(str).str.strip()
    unique_ids = unique_ids[unique_ids != 'nan'].unique()
    new_ids    = [uid for uid in unique_ids if uid not in annotations_cache]

    if new_ids:
        print(f"  Fetching {len(new_ids)} new UniProt IDs "
              f"({len(unique_ids) - len(new_ids)} already cached)...")
        for uid in tqdm(new_ids, desc="UniProt API"):
            annotations_cache[uid] = fetch_uniprot_annotation(uid)
            time.sleep(REQUEST_DELAY)
    else:
        print(f"  All {len(unique_ids)} UniProt IDs already cached, no API calls needed.")

    # Build output rows: one row per (h5_identifier, uniprot_id) pair
    rows = []
    for _, row in df.iterrows():
        uid = str(row[UNIPROT_COL]).strip()
        h5  = str(row[H5_COL]).strip()
        if uid == 'nan' or h5 == 'nan':
            continue
        annotation = annotations_cache.get(uid)
        rows.append({
            'uniprot_id':       uid,
            'h5_identifier':    h5,
            'annotation_text':  annotation if annotation else '',
        })

    return pd.DataFrame(rows, columns=['uniprot_id', 'h5_identifier', 'annotation_text'])


if __name__ == '__main__':
    print("=" * 70)
    print("Collecting UniProt annotations for all splits")
    print("=" * 70)

    # Shared cache so each UniProt ID is fetched at most once across all splits
    annotations_cache: dict[str, str | None] = {}

    for split_name, input_filename in SPLITS.items():
        input_path  = os.path.join(SPLIT_DIR, input_filename)
        output_name = input_filename.replace('.csv', '_text.csv')
        output_path = os.path.join(SPLIT_DIR, output_name)

        print(f"\n[{split_name}] Reading {input_path}...")
        if not os.path.exists(input_path):
            print(f"  Warning: {input_path} not found, skipping.")
            continue

        result_df = collect_annotations_for_split(input_path, annotations_cache)

        result_df.to_csv(output_path, index=False)
        n_with_text = (result_df['annotation_text'] != '').sum()
        print(f"  Wrote {len(result_df)} rows to {output_path} "
              f"({n_with_text} with annotation text, "
              f"{len(result_df) - n_with_text} empty)")

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"  Unique UniProt IDs fetched: "
          f"{sum(1 for v in annotations_cache.values() if v is not None)}/"
          f"{len(annotations_cache)}")
    print(f"{'=' * 70}")