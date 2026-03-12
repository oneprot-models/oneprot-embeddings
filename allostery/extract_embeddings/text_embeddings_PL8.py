"""
concatenate_text_embeddings_asd.py  —  run as BATCH JOB (GPU)

Loads existing ASD_pockets_binary and ASD_pockets_sequence_binary .pt files,
encodes text annotations from the *_text.csv files using model.network['text'],
and writes new .pt files into ASD_pockets_binary_text/ and
ASD_pockets_sequence_binary_text/ inside the same model subfolder.

Join key: split_id  (e.g. '1ap8_A_M7G_214')
    - split .txt files (allosteric/competitive/noncompetitive)  -> row order
    - {split}_text.csv (per mechanism split dir)                -> split_id -> annotation_text

Row order in existing .pt files:
    allosteric (label=0) first, competitive (label=1) second,
    noncompetitive (label=2) third — within each group, the .txt file order.
    The binary transform (label 2→1, then 0↔1) was applied afterwards,
    so the row ORDER is unchanged; only label values differ.

Processes ONE model subfolder (set MODEL_DIR below).
"""

import os
import sys
import glob
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.oneprot_module import OneProtLitModule


# ---------------------------------------------------------------------------
# Configuration — EDIT THESE
# ---------------------------------------------------------------------------
MODEL_DIR = 'embeddings/oneprot_md_combined_gpcr_no_struct_graph_32900'
SPLITS    = ['train', 'valid', 'test']

# Split .txt directories (used to reconstruct row order)
SPLIT_DIRS = {
    'allosteric':    '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/splits/allosteric',
    'competitive':   '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/splits/competitive',
    'noncompetitive':'/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/splits/noncompetitive',
}
# Original label assigned to each mechanism in the extraction script
MECHANISM_ORDER = [('allosteric', 0), ('competitive', 1), ('noncompetitive', 2)]

CONFIG_PATH     = '/p/project1/hai_oneprot/bazarova1/oneprot-panda/logs/train/runs/2025-07-27_06-21-43/config.yaml'
CHECKPOINT_PATH = '/p/data1/profound_data/checkpoints_oneprot_md/2025-07-25__18:31:15/epoch_029_04300-v2.ckpt'

# Hardcoded BiomedBERT tokenizer snapshot (confirmed to contain vocab.txt)
TOKENIZER_SNAPSHOT = (
    '/p/scratch/hai_oneprot/huggingface/hub/'
    'models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/'
    'snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062'
)

SOURCE_TYPES = {
    'ASD_pockets_binary':          'ASD_pockets_binary_text',
    'ASD_pockets_sequence_binary': 'ASD_pockets_sequence_binary_text',
}


# ---------------------------------------------------------------------------
# Load text annotations from all mechanism *_text.csv files
# Returns { split_id -> annotation_text } across all mechanisms and splits
# ---------------------------------------------------------------------------
def load_text_annotations(split_dirs: dict, splits: list) -> dict[str, str]:
    lookup = {}
    for mechanism, split_dir in split_dirs.items():
        for split in splits:
            text_csv = os.path.join(split_dir, f'{split}_text.csv')
            if not os.path.exists(text_csv):
                print(f"  Warning: {text_csv} not found, skipping.")
                continue
            df = pd.read_csv(text_csv)
            if 'split_id' not in df.columns or 'annotation_text' not in df.columns:
                print(f"  Warning: expected columns missing in {text_csv}, skipping.")
                continue
            for _, row in df.iterrows():
                sid  = str(row['split_id']).strip()
                text = str(row['annotation_text']).strip()
                if sid and sid != 'nan' and text and text != 'nan':
                    lookup[sid] = text
    print(f"  Loaded annotation text for {len(lookup)} unique split_ids")
    return lookup


# ---------------------------------------------------------------------------
# Reconstruct ordered (split_id, original_label) list for a split —
# mirrors the extraction loop: allosteric→competitive→noncompetitive,
# .txt file row order within each group.
# ---------------------------------------------------------------------------
def reconstruct_ordered_ids(split: str) -> list[tuple[str, int]]:
    rows = []
    for mechanism, label in MECHANISM_ORDER:
        txt_path = os.path.join(SPLIT_DIRS[mechanism], f'{split}.txt')
        if not os.path.exists(txt_path):
            print(f"  Warning: {txt_path} not found")
            continue
        with open(txt_path) as f:
            for line in f:
                sid = line.strip()
                if sid:
                    rows.append((sid, label))
    return rows


# ---------------------------------------------------------------------------
# Encode annotation texts with model.network['text']
# Deduplicates so each unique text is encoded only once.
# Returns { split_id -> np.ndarray }
# ---------------------------------------------------------------------------
def encode_texts(annotation_lookup: dict[str, str],
                 split_ids_needed: list[str],
                 model, text_tokenizer, device: str) -> dict[str, np.ndarray]:
    seen, unique_ids = set(), []
    for sid in split_ids_needed:
        if sid in annotation_lookup and sid not in seen:
            unique_ids.append(sid)
            seen.add(sid)

    n_missing = sum(1 for sid in split_ids_needed if sid not in annotation_lookup)
    print(f"  Encoding {len(unique_ids)} unique texts "
          f"({len(split_ids_needed) - len(unique_ids) - n_missing} duplicates reused, "
          f"{n_missing} missing from text CSVs)...")

    text_embs = {}
    for sid in tqdm(unique_ids, desc="  Text encoding"):
        try:
            input_ids = text_tokenizer(
                annotation_lookup[sid],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )['input_ids'].to(device)
            with torch.no_grad():
                emb = model.network['text'](input_ids)
            emb_np = emb.cpu().numpy().flatten()
            if not (np.isnan(emb_np).any() or np.isinf(emb_np).any()):
                text_embs[sid] = emb_np
            else:
                print(f"    Warning: NaN/Inf in text embedding for {sid}")
        except Exception as e:
            print(f"    Error encoding {sid}: {e}")

    print(f"  Encoded: {len(text_embs)}/{len(unique_ids)}")
    return text_embs


# ---------------------------------------------------------------------------
# Concatenate text embeddings onto a source .pt file and save
# ---------------------------------------------------------------------------
def concatenate_and_save(source_pt: str, split: str,
                          text_embs: dict[str, np.ndarray],
                          output_pt: str, label_str: str) -> bool:
    src        = torch.load(source_pt, map_location='cpu')
    src_embs   = src['embeddings'].numpy()
    src_labels = src['labels_fitness'].numpy()
    N_src      = len(src_embs)

    ordered_ids = reconstruct_ordered_ids(split)
    N_csv       = len(ordered_ids)

    if N_csv != N_src:
        print(f"  WARNING [{label_str}]: split .txt files have {N_csv} rows but "
              f".pt has {N_src} embeddings — aligning up to min({N_csv},{N_src}).")

    out_embs, out_labels = [], []
    missing_text = 0
    n_align = min(N_csv, N_src)

    for i in range(n_align):
        sid, _ = ordered_ids[i]
        text_np = text_embs.get(sid)
        if text_np is None:
            missing_text += 1
            continue
        out_embs.append(np.concatenate([src_embs[i], text_np]))
        out_labels.append(int(src_labels[i]))   # keep binary-transformed label as-is

    if not out_embs:
        print(f"  WARNING: No output embeddings produced for {label_str}")
        return False
    if missing_text:
        print(f"  Note: {missing_text}/{n_align} rows dropped (no text embedding)")

    os.makedirs(os.path.dirname(output_pt), exist_ok=True)
    out_tensor = torch.tensor(np.stack(out_embs), dtype=torch.float32)
    lbl_tensor = torch.tensor(out_labels, dtype=torch.int64)

    tmp = output_pt + f'.{os.getpid()}.tmp'
    torch.save({'embeddings': out_tensor, 'labels_fitness': lbl_tensor}, tmp)
    os.replace(tmp, output_pt)

    unique, counts = torch.unique(lbl_tensor, return_counts=True)
    dist = {int(k): int(c) for k, c in zip(unique, counts)}
    print(f"  Saved shape={out_tensor.shape} -> {output_pt}")
    print(f"  Label distribution: {dist}")
    return True


# ---------------------------------------------------------------------------
# Load model and text tokenizer
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
        'md':           hydra.utils.instantiate(cfg.model.components.md),
        'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
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
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)
    print("✓ Model loaded")

    # Patch config.json in the tokenizer snapshot (adds model_type='bert' if missing)
    print("Loading text tokenizer...")
    config_real = os.path.realpath(os.path.join(TOKENIZER_SNAPSHOT, 'config.json'))
    if os.path.isfile(config_real):
        with open(config_real, 'r') as f:
            cfg_tok = json.load(f)
        if 'model_type' not in cfg_tok:
            cfg_tok['model_type'] = 'bert'
            with open(config_real, 'w') as f:
                json.dump(cfg_tok, f, indent=2)
            print(f"  Patched config.json: added model_type='bert'")

    try:
        text_tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_SNAPSHOT, local_files_only=True
        )
        print(f"  Loaded via AutoTokenizer")
    except Exception as e:
        from transformers import BertTokenizer
        vocab_real = os.path.realpath(os.path.join(TOKENIZER_SNAPSHOT, 'vocab.txt'))
        text_tokenizer = BertTokenizer(vocab_file=vocab_real)
        print(f"  Loaded via BertTokenizer fallback from {vocab_real}")
    print("✓ Text tokenizer loaded")
    return model, text_tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model dir:    {MODEL_DIR}")

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")

    # Load text annotations from all mechanism split dirs
    print("\n" + "=" * 70)
    print("LOADING TEXT ANNOTATIONS")
    print("=" * 70)
    annotation_lookup = load_text_annotations(SPLIT_DIRS, SPLITS)

    # Load model and text tokenizer
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
            print(f"\n{src_folder}/ not found in {MODEL_DIR}, skipping.")
            continue

        print(f"\n{'=' * 70}")
        print(f"{src_folder}  ->  {out_folder}")
        print(f"{'=' * 70}")

        for split in SPLITS:
            print(f"\n[{split.upper()}]")

            src_pt  = os.path.join(src_base, split,
                                   f"{src_folder}_{split}_embeddings_labels.pt")
            out_pt  = os.path.join(MODEL_DIR, out_folder, split,
                                   f"{out_folder}_{split}_embeddings_labels.pt")

            if not os.path.exists(src_pt):
                print(f"  Source not found: {src_pt}, skipping.")
                total_skipped += 1
                continue

            # Collect ordered split_ids and encode text for this split
            ordered_ids  = reconstruct_ordered_ids(split)
            split_ids    = [sid for sid, _ in ordered_ids]
            split_text_embs = encode_texts(
                annotation_lookup, split_ids, model, text_tokenizer, device
            )

            success = concatenate_and_save(
                src_pt, split, split_text_embs, out_pt,
                label_str=f"{src_folder}/{split}"
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