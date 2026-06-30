"""
concatenate_text_embeddings_asd_merged.py  —  run as BATCH JOB (GPU)

Loads ASD_merged_pocket and ASD_merged_pocket_sequence .pt files,
encodes text annotations for ALL rows (both the head portion from merged_pocket
/ merged_pocket_sequence, and the tail portion from ASD_pocket_sequence100),
and writes new *_text versions.

Rows with no text annotation are DROPPED from the output.

File structure of the source merged files:
    HEAD: rows from merged_pocket / merged_pocket_sequence
          (Kinase + ASD allosteric/competitive/noncompetitive binary)
          Row order: Kinase rows first, then ASD rows
          Text join key for this portion: h5_identifier
              Kinase: via 0.3/{split}.csv  (h5_identifier column)
              ASD:    via splits/{mechanism}/{split}.txt  (split_id = h5_identifier)
    TAIL: rows appended from ASD_pocket_sequence100 (all label=1)
          Row order: H5 identifiers derived from CSV splits (train/valid/test)
          Text join key: pdb_id + chain, parsed from H5 identifier (e.g. '2z78_A')
          Text source:  train_df_pdb_text.csv / test_df_pdb_text.csv

Head row counts come from the already-saved merged_pocket /
merged_pocket_sequence files (before ASD_pocket_sequence100 was appended),
which are still on disk in their non-merged versions.

Processes ONE model subfolder (set MODEL_DIR below).
"""

import os
import sys
import glob
import json
import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.oneprot_module import OneProtLitModule


# ---------------------------------------------------------------------------
# Configuration — EDIT THESE
# ---------------------------------------------------------------------------
MODEL_DIR = 'embeddings/oneprot_md_combined_gpcr_32900'
SPLITS    = ['train', 'valid', 'test']

# Kinase split CSVs
KINASE_SPLIT_DIR = '0.3'   # contains train.csv, val.csv, test.csv

# ASD allosteric/competitive/noncompetitive split .txt dirs
ASD_SPLIT_DIRS = {
    'allosteric':     'PPI-site_splits/allosteric',
    'competitive':    'PPI-site_splits/competitive',
}
ASD_MECHANISM_ORDER = [('allosteric', 0), ('competitive', 1), ('noncompetitive', 2)]

# PDB text annotation CSVs (produced by collect_uniprot_annotations_pdb.py)
PDB_DATA_DIR = '.'
PDB_TRAIN_CSV = os.path.join(PDB_DATA_DIR, 'ASD_original_split/train_df_pdb.csv')
PDB_TEST_CSV  = os.path.join(PDB_DATA_DIR, 'ASD_original_split/test_df_pdb.csv')
PDB_TRAIN_TEXT_CSV = os.path.join(PDB_DATA_DIR, 'ASD_original_split/train_df_pdb_text.csv')
PDB_TEST_TEXT_CSV  = os.path.join(PDB_DATA_DIR, 'ASD_original_split/test_df_pdb_text.csv')

CONFIG_PATH     = 'config.yaml'
CHECKPOINT_PATH = 'epoch_012_01100-v1.ckpt'

TOKENIZER_SNAPSHOT = (
    'huggingface/hub/'
    'models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/'
    'snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062'
)

# Source merged folders -> output folder
SOURCE_TYPES = {
    'ASD_merged_pocket':          'ASD_merged_pocket_text',
    'ASD_merged_pocket_sequence': 'ASD_merged_pocket_sequence_text',
}

# The corresponding pre-append source (to measure head size)
# These are the files BEFORE ASD_pocket_sequence100 was appended
HEAD_SOURCE = {
    'ASD_merged_pocket':          'merged_pocket',
    'ASD_merged_pocket_sequence': 'merged_pocket_sequence',
}


# ---------------------------------------------------------------------------
# 1. Build text annotation lookups
# ---------------------------------------------------------------------------

def load_kinase_text_annotations(split_dir: str) -> dict[str, str]:
    """{ h5_identifier -> annotation_text } from Kinase *_text.csv files."""
    lookup = {}
    for split in SPLITS:
        text_csv = os.path.join(split_dir, f'{split}_text.csv')
        if not os.path.exists(text_csv):
            # try 'val' variant
            text_csv = os.path.join(split_dir, 'val_text.csv') if split == 'valid' else text_csv
        if not os.path.exists(text_csv):
            print(f"  Warning: {text_csv} not found")
            continue
        df = pd.read_csv(text_csv)
        for _, row in df.iterrows():
            h5id = str(row.get('h5_identifier', '')).strip()
            text = str(row.get('annotation_text', '')).strip()
            if h5id and h5id != 'nan' and text and text != 'nan':
                lookup[h5id] = text
    print(f"  Kinase text annotations: {len(lookup)} h5_identifiers")
    return lookup


def load_asd_text_annotations(split_dirs: dict) -> dict[str, str]:
    """{ split_id -> annotation_text } from ASD allosteric/competitive/noncompetitive *_text.csv."""
    lookup = {}
    for mechanism, split_dir in split_dirs.items():
        for split in SPLITS:
            text_csv = os.path.join(split_dir, f'{split}_text.csv')
            if not os.path.exists(text_csv):
                continue
            df = pd.read_csv(text_csv)
            for _, row in df.iterrows():
                sid  = str(row.get('split_id', '')).strip()
                text = str(row.get('annotation_text', '')).strip()
                if sid and sid != 'nan' and text and text != 'nan':
                    lookup[sid] = text
    print(f"  ASD text annotations: {len(lookup)} split_ids")
    return lookup


def load_pdb_text_annotations() -> dict[str, str]:
    """
    { (pdb_id, chain) -> annotation_text } from train_df_pdb_text.csv and test_df_pdb_text.csv.
    Key is lowercase pdb_id + '_' + chain, matching H5 identifier format.
    """
    lookup = {}
    for text_csv in [PDB_TRAIN_TEXT_CSV, PDB_TEST_TEXT_CSV]:
        if not os.path.exists(text_csv):
            print(f"  Warning: {text_csv} not found")
            continue
        df = pd.read_csv(text_csv)
        for _, row in df.iterrows():
            pdb   = str(row.get('pdb_id', '')).strip().lower()
            chain = str(row.get('chain', '')).strip()
            text  = str(row.get('annotation_text', '')).strip()
            if pdb and chain and text and text != 'nan':
                key = f"{pdb}_{chain}"
                lookup[key] = text
    print(f"  PDB text annotations: {len(lookup)} (pdb_chain) keys")
    return lookup


# ---------------------------------------------------------------------------
# 2. Reconstruct ordered identifier lists for each portion
# ---------------------------------------------------------------------------

def reconstruct_kinase_ids(split: str) -> list[str]:
    """
    Ordered h5_identifiers for the Kinase portion, mirroring the original
    extraction loop: competitive first, then allosteric.
    """
    csv_split = 'val' if split == 'valid' else split
    csv_path  = os.path.join(KINASE_SPLIT_DIR, f'{csv_split}.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(KINASE_SPLIT_DIR, f'{split}.csv')
    if not os.path.exists(csv_path):
        print(f"  Warning: Kinase CSV not found for split={split}")
        return []
    df = pd.read_csv(csv_path)
    ids = []
    for mechanism in ['competitive', 'allosteric']:
        df_mech = df[df['Mechanism'].str.lower() == mechanism]
        ids.extend(df_mech['h5_identifier'].dropna().astype(str).tolist())
    return ids


def reconstruct_asd_ids(split: str) -> list[str]:
    """
    Ordered split_ids for the ASD allosteric/competitive/noncompetitive portion.
    Order: allosteric, competitive, noncompetitive (matches extraction script).
    """
    ids = []
    for mechanism, _ in ASD_MECHANISM_ORDER:
        txt_path = os.path.join(ASD_SPLIT_DIRS[mechanism], f'{split}.txt')
        if not os.path.exists(txt_path):
            continue
        with open(txt_path) as f:
            for line in f:
                sid = line.strip()
                if sid:
                    ids.append(sid)
    return ids


def parse_chains(chains_str: str) -> list[str]:
    try:
        result = ast.literal_eval(chains_str)
        if isinstance(result, list):
            return [str(c).strip() for c in result]
    except (ValueError, SyntaxError):
        pass
    cleaned = chains_str.strip().strip("[]").replace("'", "").replace('"', '')
    return [c.strip() for c in cleaned.split(',') if c.strip()]


def reconstruct_pdb_ids(split: str) -> list[str]:
    """
    Ordered H5 identifiers for the ASD_pocket_sequence100 tail portion.
    Mirrors map_pdb_ids_to_h5_identifiers + load_split_from_csv logic.
    Returns list of '{pdb_id}_{chain}' strings (lowercase pdb_id).
    """
    train_df = pd.read_csv(PDB_TRAIN_CSV)
    test_df  = pd.read_csv(PDB_TEST_CSV)

    test_pdbs_all = test_df['pdb_id'].unique().tolist()
    mid           = len(test_pdbs_all) // 2
    split_pdbs = {
        'train': train_df['pdb_id'].unique().tolist(),
        'valid': test_pdbs_all[:mid],
        'test':  test_pdbs_all[mid:],
    }[split]

    # Build ordered (pdb_id, chain) pairs as they appear in the CSV
    # (same order map_pdb_ids_to_h5_identifiers would match them)
    split_pdbs_lower = set(p.lower() for p in split_pdbs)

    src_df = train_df if split == 'train' else test_df
    ids = []
    seen = set()
    for _, row in src_df.iterrows():
        pdb = str(row['pdb_id']).strip().lower()
        if pdb not in split_pdbs_lower:
            continue
        for chain in parse_chains(str(row['chains'])):
            key = f"{pdb}_{chain}"
            if key not in seen:
                ids.append(key)
                seen.add(key)
    return ids


# ---------------------------------------------------------------------------
# 3. Build the full ordered identifier list for a merged file
# ---------------------------------------------------------------------------

def build_full_id_list(split: str, src_folder: str) -> tuple[list[tuple[str, str]], int, int, int]:
    """
    Returns (ordered_ids, n_kinase, n_asd, n_pdb) where ordered_ids is a list of
    (identifier, source) tuples, source in {'kinase', 'asd', 'pdb'}.
    """
    kinase_ids = [(sid, 'kinase') for sid in reconstruct_kinase_ids(split)]
    asd_ids    = [(sid, 'asd')    for sid in reconstruct_asd_ids(split)]
    pdb_ids    = [(sid, 'pdb')    for sid in reconstruct_pdb_ids(split)]
    all_ids    = kinase_ids + asd_ids + pdb_ids
    return all_ids, len(kinase_ids), len(asd_ids), len(pdb_ids)


# ---------------------------------------------------------------------------
# 4. Text encoding
# ---------------------------------------------------------------------------

def encode_texts(texts_needed: list[tuple[str, str]],
                 kinase_lookup: dict, asd_lookup: dict, pdb_lookup: dict,
                 model, text_tokenizer, device: str) -> dict[tuple[str,str], np.ndarray]:
    """
    texts_needed: list of (identifier, source) where source in {'kinase','asd','pdb'}
    Returns { (identifier, source) -> np.ndarray }
    Deduplicates by (identifier, source).
    """
    def get_text(identifier, source):
        if source == 'kinase':
            return kinase_lookup.get(identifier)
        elif source == 'asd':
            return asd_lookup.get(identifier)
        else:  # pdb
            return pdb_lookup.get(identifier)

    seen, unique = set(), []
    for item in texts_needed:
        if item not in seen and get_text(*item) is not None:
            unique.append(item)
            seen.add(item)

    print(f"  Encoding {len(unique)} unique texts "
          f"({len(texts_needed) - len(unique)} duplicates/missing)...")

    embs = {}
    for identifier, source in tqdm(unique, desc="  Text encoding"):
        text = get_text(identifier, source)
        try:
            input_ids = text_tokenizer(
                text, return_tensors='pt', padding=True,
                truncation=True, max_length=512,
            )['input_ids'].to(device)
            with torch.no_grad():
                emb = model.network['text'](input_ids)
            emb_np = emb.cpu().numpy().flatten()
            if not (np.isnan(emb_np).any() or np.isinf(emb_np).any()):
                embs[(identifier, source)] = emb_np
        except Exception as e:
            print(f"    Error encoding {identifier} ({source}): {e}")

    print(f"  Encoded: {len(embs)}/{len(unique)}")
    return embs


# ---------------------------------------------------------------------------
# 5. Core concatenation
# ---------------------------------------------------------------------------

def concatenate_and_save(source_pt: str, split: str, src_folder: str,
                          text_embs: dict,
                          kinase_lookup: dict, asd_lookup: dict, pdb_lookup: dict,
                          output_pt: str, label_str: str) -> bool:
    src      = torch.load(source_pt, map_location='cpu')
    src_embs = src['embeddings'].numpy()
    src_lbls = src['labels_fitness'].numpy()
    N_src    = len(src_embs)

    all_ids, n_kinase, n_asd, n_pdb = build_full_id_list(split, src_folder)
    N_csv = len(all_ids)

    if N_csv != N_src:
        print(f"  WARNING [{label_str}]: reconstructed {N_csv} ids but .pt has "
              f"{N_src} rows — aligning up to min({N_csv},{N_src}).")

    out_embs, out_lbls = [], []
    missing_text = 0
    n_align = min(N_csv, N_src)

    for i in range(n_align):
        identifier, source = all_ids[i]
        text_np = text_embs.get((identifier, source))
        if text_np is None:
            missing_text += 1
            continue   # drop rows with no text
        out_embs.append(np.concatenate([src_embs[i], text_np]))
        out_lbls.append(int(src_lbls[i]))

    if not out_embs:
        print(f"  WARNING: No output embeddings for {label_str}")
        return False

    print(f"  Kept {len(out_embs)}/{n_align} rows  "
          f"({missing_text} dropped — no text annotation)")

    os.makedirs(os.path.dirname(output_pt), exist_ok=True)
    out_tensor = torch.tensor(np.stack(out_embs), dtype=torch.float32)
    lbl_tensor = torch.tensor(out_lbls, dtype=torch.int64)

    tmp = output_pt + f'.{os.getpid()}.tmp'
    torch.save({'embeddings': out_tensor, 'labels_fitness': lbl_tensor}, tmp)
    os.replace(tmp, output_pt)

    unique, counts = torch.unique(lbl_tensor, return_counts=True)
    dist = {int(k): int(c) for k, c in zip(unique, counts)}
    print(f"  Saved shape={out_tensor.shape} -> {output_pt}")
    print(f"  Label distribution: {dist}")
    return True


# ---------------------------------------------------------------------------
# 6. Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config_path: str, checkpoint_path: str, device: str):
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)

    print("Initializing model components...")
    components = {
        'sequence':     hydra.utils.instantiate(cfg.model.components.sequence),
        #'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'pocket':       hydra.utils.instantiate(cfg.model.components.pocket),
        'text':         hydra.utils.instantiate(cfg.model.components.text),
        #'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
        #'md':           hydra.utils.instantiate(cfg.model.components.md)
    }
    model = OneProtLitModule(
        components=components,
        optimizer=None,
        loss_fn=cfg.model.loss_fn,
        local_loss=cfg.model.local_loss,
        gather_with_grad=cfg.model.gather_with_grad,
        use_l1_regularization=cfg.model.use_l1_regularization,
        train_on_all_modalities_after_step=cfg.model.train_on_all_modalities_after_step,
        use_seqsim=cfg.model.use_seqsim
    )
    print(f"Loading checkpoint: {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)
    print("✓ Model loaded")

    print("Loading text tokenizer...")
    config_real = os.path.realpath(os.path.join(TOKENIZER_SNAPSHOT, 'config.json'))
    if os.path.isfile(config_real):
        with open(config_real, 'r') as f:
            cfg_tok = json.load(f)
        if 'model_type' not in cfg_tok:
            cfg_tok['model_type'] = 'bert'
            with open(config_real, 'w') as f:
                json.dump(cfg_tok, f, indent=2)
            print("  Patched config.json: added model_type='bert'")

    try:
        text_tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_SNAPSHOT, local_files_only=True
        )
        print("  Loaded via AutoTokenizer")
    except Exception:
        from transformers import BertTokenizer
        vocab_real = os.path.realpath(os.path.join(TOKENIZER_SNAPSHOT, 'vocab.txt'))
        text_tokenizer = BertTokenizer(vocab_file=vocab_real)
        print(f"  Loaded via BertTokenizer fallback")
    print("✓ Text tokenizer loaded")
    return model, text_tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model dir:    {MODEL_DIR}")

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")

    # Load all text annotation lookups
    print("\n" + "=" * 70)
    print("LOADING TEXT ANNOTATIONS")
    print("=" * 70)
    kinase_lookup = load_kinase_text_annotations(KINASE_SPLIT_DIR)
    asd_lookup    = load_asd_text_annotations(ASD_SPLIT_DIRS)
    pdb_lookup    = load_pdb_text_annotations()

    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    model, text_tokenizer = load_model_and_tokenizer(
        CONFIG_PATH, CHECKPOINT_PATH, device
    )

    total_written = 0
    total_skipped = 0

    for src_folder, out_folder in SOURCE_TYPES.items():
        src_base = os.path.join(MODEL_DIR, src_folder)
        if not os.path.isdir(src_base):
            print(f"\n{src_folder}/ not found, skipping.")
            continue

        print(f"\n{'=' * 70}")
        print(f"{src_folder}  ->  {out_folder}")
        print(f"{'=' * 70}")

        for split in SPLITS:
            print(f"\n[{split.upper()}]")

            src_pt = os.path.join(src_base, split,
                                  f"{src_folder}_{split}_embeddings_labels.pt")
            out_pt = os.path.join(MODEL_DIR, out_folder, split,
                                  f"{out_folder}_{split}_embeddings_labels.pt")

            if not os.path.exists(src_pt):
                print(f"  Source not found: {src_pt}, skipping.")
                total_skipped += 1
                continue

            # Build full ordered id list and encode all needed texts
            all_ids, n_k, n_a, n_p = build_full_id_list(split, src_folder)
            print(f"  Reconstructed order: {n_k} Kinase + {n_a} ASD + {n_p} PDB = {len(all_ids)} rows")

            split_text_embs = encode_texts(
                all_ids, kinase_lookup, asd_lookup, pdb_lookup,
                model, text_tokenizer, device
            )

            success = concatenate_and_save(
                src_pt, split, src_folder, split_text_embs,
                kinase_lookup, asd_lookup, pdb_lookup,
                out_pt, label_str=f"{src_folder}/{split}"
            )
            if success:
                total_written += 1
            else:
                total_skipped += 1

    print(f"\n{'=' * 70}")
    print(f"Done!  Written: {total_written}  Skipped/failed: {total_skipped}")
    print(f"{'=' * 70}")

    print("\nOutput files:")
    for _, out_folder in SOURCE_TYPES.items():
        for split in SPLITS:
            path = os.path.join(MODEL_DIR, out_folder, split,
                                f"{out_folder}_{split}_embeddings_labels.pt")
            status = "✓" if os.path.exists(path) else "✗"
            print(f"  {status} {path}")