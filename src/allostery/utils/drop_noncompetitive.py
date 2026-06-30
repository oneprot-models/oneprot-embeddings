"""
create_comp_embeddings.py  —  run on CPU (no GPU or model loading needed)

For every model subfolder under EMBEDDINGS_ROOT, creates *_comp versions of
all ASD-related embedding tasks by dropping the noncompetitive rows.

The noncompetitive rows always occupy fixed positions (0-indexed, exclusive end):
    train: rows 2767:3280  (513 rows)
    valid: rows  421:557   (136 rows)
    test:  rows  341:440   (99  rows)

These ranges are identical for every model and every task type because the
ASD_pockets block always comes first and the noncompetitive segment is always
last within it. For *_text variants the ranges will differ slightly (text
filtering dropped some rows within each segment), so for those we fall back
to identifier-level reconstruction.

Source tasks -> _comp counterparts (folder and filename *_comp appended):
  ASD_pockets_binary                       -> ASD_pockets_binary_comp
  ASD_pockets_sequence_binary              -> ASD_pockets_sequence_binary_comp
  ASD_pockets_binary_text                  -> ASD_pockets_binary_text_comp
  ASD_pockets_sequence_binary_text         -> ASD_pockets_sequence_binary_text_comp
  merged_pocket_binary                     -> merged_pocket_binary_comp
  merged_pocket_sequence_binary            -> merged_pocket_sequence_binary_comp
  merged_pocket_binary_text                -> merged_pocket_binary_text_comp
  merged_pocket_sequence_binary_text       -> merged_pocket_sequence_binary_text_comp
  ASD_merged_pocket_binary                 -> ASD_merged_pocket_binary_comp
  ASD_merged_pocket_sequence_binary        -> ASD_merged_pocket_sequence_binary_comp
  ASD_merged_pocket_binary_text            -> ASD_merged_pocket_binary_text_comp
  ASD_merged_pocket_sequence_binary_text   -> ASD_merged_pocket_sequence_binary_text_comp
"""

import os
import ast
import torch
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration — EDIT THESE
# ---------------------------------------------------------------------------
EMBEDDINGS_ROOT = 'embeddings'
SPLITS          = ['train', 'valid', 'test']

# Fixed drop ranges (0-indexed, exclusive end) — same for every model/task
# Derived from: ASD total = 3280/557/440, noncomp = 513/136/99
DROP_RANGES = {
    'train': (2767, 3280),
    'valid': (421,  557),
    'test':  (341,  440),
}

ASD_SPLIT_DIRS = {
    'allosteric':     'PPI-site_splits/allosteric',
    'competitive':    'PPI-site_splits/competitive',
    'noncompetitive': 'PPI-site_splits/noncompetitive',
}

KINASE_SPLIT_DIR   = '0.3'
PDB_TRAIN_CSV      = 'ASD_original_splits/train_df_pdb.csv'
PDB_TEST_CSV       = 'ASD_original_splits/test_df_pdb.csv'
PDB_TRAIN_TEXT_CSV = 'ASD_original_splits/train_df_pdb_text.csv'
PDB_TEST_TEXT_CSV  = 'ASD_original_splits/test_df_pdb_text.csv'

# Task definitions: name -> has_text
TASK_TYPES = {
    'ASD_pockets_binary':                     False,
    'ASD_pockets_sequence_binary':            False,
    'ASD_pockets_binary_text':                True,
    'ASD_pockets_sequence_binary_text':       True,
    'merged_pocket_binary':                   False,
    'merged_pocket_sequence_binary':          False,
    'merged_pocket_binary_text':              True,
    'merged_pocket_sequence_binary_text':     True,
    'ASD_merged_pocket_binary':               False,
    'ASD_merged_pocket_sequence_binary':      False,
    'ASD_merged_pocket_binary_text':          True,
    'ASD_merged_pocket_sequence_binary_text': True,
}


# ---------------------------------------------------------------------------
# Non-text: simple fixed-range drop
# ---------------------------------------------------------------------------

def get_keep_indices_no_text(n_actual: int, split: str) -> list[int]:
    drop_start, drop_end = DROP_RANGES[split]

    if drop_end > n_actual:
        print(f"    WARNING: drop_end={drop_end} > n_actual={n_actual}. "
              f"Clamping drop range.")
        drop_end = n_actual
    if drop_start > n_actual:
        print(f"    WARNING: drop_start={drop_start} > n_actual={n_actual}. "
              f"Nothing to drop.")
        return list(range(n_actual))

    return list(range(drop_start)) + list(range(drop_end, n_actual))


# ---------------------------------------------------------------------------
# Text variant: identifier-level reconstruction (text filtering shifted rows)
# ---------------------------------------------------------------------------

def parse_chains(s: str) -> list[str]:
    try:
        r = ast.literal_eval(s)
        if isinstance(r, list):
            return r
    except Exception:
        pass
    return [c.strip() for c in str(s).strip("[]").replace("'","").replace('"','').split(',') if c.strip()]


def _build_text_keep_sets(split: str):
    kinase_keep = set()
    csv_split   = 'val' if split == 'valid' else split
    for fname in [f'{csv_split}_text.csv', f'{split}_text.csv']:
        p = os.path.join(KINASE_SPLIT_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                h5id = str(row.get('h5_identifier', '')).strip()
                text = str(row.get('annotation_text', '')).strip()
                if h5id and h5id != 'nan' and text and text != 'nan':
                    kinase_keep.add(h5id)
            break

    asd_keep = set()
    for mech in ['allosteric', 'competitive', 'noncompetitive']:
        p = os.path.join(ASD_SPLIT_DIRS[mech], f'{split}_text.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                sid  = str(row.get('split_id', '')).strip()
                text = str(row.get('annotation_text', '')).strip()
                if sid and sid != 'nan' and text and text != 'nan':
                    asd_keep.add(sid)

    pdb_keep = set()
    for p in [PDB_TRAIN_TEXT_CSV, PDB_TEST_TEXT_CSV]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                pdb   = str(row.get('pdb_id', '')).strip().lower()
                chain = str(row.get('chain', '')).strip()
                text  = str(row.get('annotation_text', '')).strip()
                if pdb and chain and text and text != 'nan':
                    pdb_keep.add(f"{pdb}_{chain}")

    return kinase_keep, asd_keep, pdb_keep


def _get_pdb_ordered_ids(split: str) -> list[str]:
    train_df  = pd.read_csv(PDB_TRAIN_CSV)
    test_df   = pd.read_csv(PDB_TEST_CSV)
    test_pdbs = test_df['pdb_id'].unique().tolist()
    mid       = len(test_pdbs) // 2

    if split == 'train':
        src_df  = train_df
        pdb_set = set(p.lower() for p in train_df['pdb_id'].unique())
    elif split == 'valid':
        src_df  = test_df
        pdb_set = set(p.lower() for p in test_pdbs[:mid])
    else:
        src_df  = test_df
        pdb_set = set(p.lower() for p in test_pdbs[mid:])

    seen, ids = set(), []
    for _, row in src_df.iterrows():
        pdb = str(row['pdb_id']).strip().lower()
        if pdb not in pdb_set:
            continue
        for chain in parse_chains(str(row['chains'])):
            key = f"{pdb}_{chain}"
            if key not in seen:
                seen.add(key)
                ids.append(key)
    return ids


def _get_kinase_ordered_ids(split: str) -> list[str]:
    csv_split = 'val' if split == 'valid' else split
    csv_path  = os.path.join(KINASE_SPLIT_DIR, f'{csv_split}.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(KINASE_SPLIT_DIR, f'{split}.csv')
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    ids = []
    for mech in ['competitive', 'allosteric']:
        ids.extend(df[df['Mechanism'].str.lower() == mech]['h5_identifier'].dropna().astype(str).tolist())
    return ids


def get_keep_indices_text(n_actual: int, task_name: str, split: str) -> list[int]:
    """
    Reconstruct which rows survived text filtering, in order:
      ASD (allo, comp, noncomp) → Kinase (if merged/asd_merged) → PDB (if asd_merged)
    Then return indices of non-noncompetitive rows.
    """
    kinase_keep, asd_keep, pdb_keep = _build_text_keep_sets(split)

    # (identifier, is_noncompetitive)
    ordered = []

    for mech in ['allosteric', 'competitive', 'noncompetitive']:
        is_noncomp = (mech == 'noncompetitive')
        txt = os.path.join(ASD_SPLIT_DIRS[mech], f'{split}.txt')
        if os.path.exists(txt):
            with open(txt) as f:
                for line in f:
                    sid = line.strip()
                    if sid and sid in asd_keep:
                        ordered.append((sid, is_noncomp))

    if 'merged' in task_name or 'ASD_merged' in task_name:
        for h5id in _get_kinase_ordered_ids(split):
            if h5id in kinase_keep:
                ordered.append((h5id, False))

    if 'ASD_merged' in task_name:
        for pdb_id in _get_pdb_ordered_ids(split):
            if pdb_id in pdb_keep:
                ordered.append((pdb_id, False))

    n_reconstructed = len(ordered)
    if n_reconstructed != n_actual:
        print(f"    Note: reconstructed {n_reconstructed} rows, "
              f".pt has {n_actual} — aligning to min.")

    n_align           = min(n_reconstructed, n_actual)
    n_noncomp_dropped = sum(1 for i in range(n_align) if ordered[i][1])
    keep              = [i for i in range(n_align) if not ordered[i][1]]

    if n_actual > n_reconstructed:
        keep.extend(range(n_reconstructed, n_actual))

    print(f"    Noncompetitive rows dropped (text variant): {n_noncomp_dropped}")
    return keep


# ---------------------------------------------------------------------------
# Core: apply indices and save
# ---------------------------------------------------------------------------

def filter_and_save(source_pt: str, output_pt: str,
                    split: str, task_name: str, has_text: bool,
                    label_str: str) -> bool:
    data     = torch.load(source_pt, map_location='cpu')
    embs     = data['embeddings']
    lbls     = data['labels_fitness']
    n_actual = len(embs)

    if has_text:
        keep = get_keep_indices_text(n_actual, task_name, split)
    else:
        keep = get_keep_indices_no_text(n_actual, split)

    n_drop = n_actual - len(keep)
    if not keep:
        print(f"    WARNING [{label_str}]: nothing to keep after filtering")
        return False

    out_embs = embs[keep]
    out_lbls = lbls[keep]

    os.makedirs(os.path.dirname(output_pt), exist_ok=True)
    tmp = output_pt + f'.{os.getpid()}.tmp'
    torch.save({'embeddings': out_embs, 'labels_fitness': out_lbls}, tmp)
    os.replace(tmp, output_pt)

    unique, counts_lbl = torch.unique(out_lbls, return_counts=True)
    dist = {int(k): int(c) for k, c in zip(unique, counts_lbl)}
    print(f"    Saved shape={out_embs.shape}  (dropped {n_drop} rows)")
    print(f"    -> {output_pt}")
    print(f"    Label distribution: {dist}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    root = Path(EMBEDDINGS_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"EMBEDDINGS_ROOT not found: {EMBEDDINGS_ROOT}")

    print("Drop ranges (0-indexed, exclusive end):")
    for split, (s, e) in DROP_RANGES.items():
        print(f"  {split}: rows {s}:{e}  ({e - s} rows dropped)")
    print()

    model_dirs     = sorted([d for d in root.iterdir() if d.is_dir()])
    total_written  = 0
    total_skipped  = 0

    for model_dir in model_dirs:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_dir.name}")
        print(f"{'=' * 70}")

        for task_name, has_text in TASK_TYPES.items():
            src_base = model_dir / task_name
            if not src_base.is_dir():
                continue

            out_name = task_name + '_comp'
            print(f"\n  {task_name}  ->  {out_name}")

            for split in SPLITS:
                split_dir = src_base / split
                src_pts   = list(split_dir.glob('*.pt')) if split_dir.is_dir() else []
                if not src_pts:
                    print(f"    [{split}] No .pt found, skipping.")
                    total_skipped += 1
                    continue

                src_pt    = str(src_pts[0])
                out_fname = os.path.basename(src_pt).replace(task_name, out_name, 1)
                out_pt    = str(model_dir / out_name / split / out_fname)

                print(f"    [{split.upper()}]  drop={DROP_RANGES[split]}")
                success = filter_and_save(
                    src_pt, out_pt, split, task_name, has_text,
                    label_str=f"{model_dir.name}/{out_name}/{split}"
                )
                if success:
                    total_written += 1
                else:
                    total_skipped += 1

    print(f"\n{'=' * 70}")
    print(f"Done!  Written: {total_written}  Skipped/failed: {total_skipped}")
    print(f"{'=' * 70}")