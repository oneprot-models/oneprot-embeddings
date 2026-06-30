"""
concatenate_text_embeddings.py  —  run as BATCH JOB (GPU)

Loads existing Kinase_pocket and Kinase_combined .pt files for ONE model
subfolder, encodes text annotations from the *_text.csv files using
model.network['text'], and writes new .pt files into Kinase_pocket_text/
and Kinase_combined_text/ inside the same subfolder.

Usage:
    Edit the Configuration block below (MODEL_DIR, CONFIG_PATH,
    CHECKPOINT_PATH) and run as a batch job.

Join key: h5_identifier
    - split CSV (train.csv etc.)  -> row order of existing embeddings
    - train_text.csv etc.         -> h5_identifier -> annotation_text
"""

import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import hydra
from omegaconf import OmegaConf
import json
from transformers import AutoTokenizer, PreTrainedTokenizerFast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.oneprot_module import OneProtLitModule


# ---------------------------------------------------------------------------
# Configuration — EDIT THESE
# ---------------------------------------------------------------------------
MODEL_DIR   = 'embeddings/oneprot_struct_token_pocket_text_32900'
SPLIT_DIR   = 'KinSite_splits'
SPLITS      = ['train', 'valid', 'test']

CONFIG_PATH     = 'config.yaml'
CHECKPOINT_PATH = 'epoch_008_04200-v3.ckpt'

H5_ID_COL     = 'h5_identifier'
MECHANISM_COL = 'Mechanism'

SOURCE_TYPES = {
    'Kinase_pocket':   'Kinase_pocket_text',
    'Kinase_combined': 'Kinase_combined_text',
}


# ---------------------------------------------------------------------------
# Load text annotations: { h5_identifier -> annotation_text }
# ---------------------------------------------------------------------------
def load_text_annotations(split_dir: str) -> dict[str, str]:
    lookup = {}
    for split in SPLITS:
        text_csv = os.path.join(split_dir, f'{split}_text.csv')
        if not os.path.exists(text_csv):
            print(f"  Warning: {text_csv} not found, skipping.")
            continue
        df = pd.read_csv(text_csv)
        if 'h5_identifier' not in df.columns or 'annotation_text' not in df.columns:
            print(f"  Warning: expected columns missing in {text_csv}, skipping.")
            continue
        for _, row in df.iterrows():
            h5id = str(row['h5_identifier']).strip()
            text = str(row['annotation_text']).strip()
            if h5id and h5id != 'nan' and text and text != 'nan':
                lookup[h5id] = text
    print(f"  Loaded annotation text for {len(lookup)} unique h5_identifiers")
    return lookup


# ---------------------------------------------------------------------------
# Reconstruct ordered (h5_identifier, label) list from split CSV —
# mirrors the original extraction loop order exactly:
# competitive (label=0) first, allosteric (label=1) second.
# ---------------------------------------------------------------------------
def reconstruct_ordered_ids(csv_path: str) -> list[tuple[str, int]]:
    df = pd.read_csv(csv_path)
    rows = []
    for mechanism, label in [('competitive', 0), ('allosteric', 1)]:
        df_mech = df[df[MECHANISM_COL].str.lower() == mechanism]
        for _, row in df_mech.iterrows():
            h5id = str(row[H5_ID_COL]).strip()
            if h5id and h5id != 'nan':
                rows.append((h5id, label))
    return rows


# ---------------------------------------------------------------------------
# Encode annotation texts with model.network['text']
# Returns { h5_identifier -> np.ndarray }
# Deduplicates so each unique text is encoded only once.
# ---------------------------------------------------------------------------
def encode_texts(annotation_lookup: dict[str, str],
                 h5ids_needed: list[str],
                 model, text_tokenizer, device: str) -> dict[str, np.ndarray]:
    seen, unique_ids = set(), []
    for h5 in h5ids_needed:
        if h5 in annotation_lookup and h5 not in seen:
            unique_ids.append(h5)
            seen.add(h5)

    n_missing = sum(1 for h5 in h5ids_needed if h5 not in annotation_lookup)
    print(f"  Encoding {len(unique_ids)} unique texts "
          f"({len(h5ids_needed) - len(unique_ids) - n_missing} duplicates reused, "
          f"{n_missing} missing from text CSV)...")

    text_embs = {}
    for h5id in tqdm(unique_ids, desc="  Text encoding"):
        try:
            input_ids = text_tokenizer(
                annotation_lookup[h5id],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )['input_ids'].to(device)
            with torch.no_grad():
                emb = model.network['text'](input_ids)
            emb_np = emb.cpu().numpy().flatten()
            if not (np.isnan(emb_np).any() or np.isinf(emb_np).any()):
                text_embs[h5id] = emb_np
            else:
                print(f"    Warning: NaN/Inf in text embedding for {h5id}")
        except Exception as e:
            print(f"    Error encoding {h5id}: {e}")

    print(f"  Encoded: {len(text_embs)}/{len(unique_ids)}")
    return text_embs


# ---------------------------------------------------------------------------
# Concatenate text embeddings onto a source .pt file and save
# ---------------------------------------------------------------------------
def concatenate_and_save(source_pt: str, csv_path: str,
                          text_embs: dict[str, np.ndarray],
                          output_pt: str, label_str: str) -> bool:
    src        = torch.load(source_pt, map_location='cpu')
    src_embs   = src['embeddings'].numpy()
    src_labels = src['labels_fitness'].numpy()
    N_src      = len(src_embs)

    ordered_ids = reconstruct_ordered_ids(csv_path)
    N_csv       = len(ordered_ids)

    if N_csv != N_src:
        print(f"  WARNING [{label_str}]: CSV has {N_csv} rows but .pt has {N_src} "
              f"embeddings — aligning up to min({N_csv}, {N_src}) by position.")

    out_embs, out_labels = [], []
    missing_text = 0
    n_align = min(N_csv, N_src)

    for i in range(n_align):
        h5id, _ = ordered_ids[i]
        text_np  = text_embs.get(h5id)
        if text_np is None:
            missing_text += 1
            continue
        out_embs.append(np.concatenate([src_embs[i], text_np]))
        out_labels.append(int(src_labels[i]))

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
# Load model and text tokenizer from config + checkpoint
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
        'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
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
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)
    print("✓ Model loaded")

    # Resolve text tokenizer path from config
    print("Loading text tokenizer...")
    text_cfg = OmegaConf.to_container(cfg.model.components.text, resolve=True)
    candidate_keys = [
        'tokenizer_name_or_path', 'pretrained_model_name_or_path',
        'model_name_or_path', 'pretrained_model_name', 'tokenizer_path', 'model_name',
    ]
    tok_path = None
    for key in candidate_keys:
        val = text_cfg.get(key)
        if val:
            tok_path = str(val)
            print(f"  Found text tokenizer under '{key}': {tok_path}")
            break

    if tok_path and not os.path.isdir(tok_path):
        hf_home    = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        cache_name = 'models--' + tok_path.replace('/', '--')
        snaps      = glob.glob(os.path.join(hf_home, 'hub', cache_name, 'snapshots', '*'))
        if snaps:
            tok_path = max(snaps, key=os.path.getmtime)
            print(f"  Resolved to HF cache: {tok_path}")

    if not tok_path or not os.path.isdir(tok_path):
        raise FileNotFoundError(f"Could not locate text tokenizer. Resolved: {tok_path}")

    # The BiomedBERT tokenizer and config live in a different snapshot than
    # the model weights. Hardcode the correct snapshot path directly.
    TOKENIZER_SNAPSHOT = (
        'huggingface/hub/'
        'models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/'
        'snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062'
    )

    # Patch config.json to add missing model_type field (resolves symlinks to blob)
    config_real = os.path.realpath(os.path.join(TOKENIZER_SNAPSHOT, 'config.json'))
    if os.path.isfile(config_real):
        with open(config_real, 'r') as f:
            cfg = json.load(f)
        if 'model_type' not in cfg:
            cfg['model_type'] = 'bert'
            with open(config_real, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"  Patched config.json: added model_type='bert'")
        else:
            print(f"  config.json already has model_type='{cfg['model_type']}'")

    # Load tokenizer — snapshot has vocab.txt so BertTokenizer works,
    # and after patching AutoTokenizer works too
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_SNAPSHOT, local_files_only=True
        )
        print(f"  Loaded text tokenizer via AutoTokenizer")
    except Exception as e:
        # Ultimate fallback: BertTokenizer directly from vocab.txt
        from transformers import BertTokenizer
        vocab_real = os.path.realpath(
            os.path.join(TOKENIZER_SNAPSHOT, 'vocab.txt')
        )
        text_tokenizer = BertTokenizer(vocab_file=vocab_real)
        print(f"  Loaded text tokenizer via BertTokenizer from {vocab_real}")
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

    # Load text annotations
    print("\n" + "=" * 70)
    print("LOADING TEXT ANNOTATIONS")
    print("=" * 70)
    annotation_lookup = load_text_annotations(SPLIT_DIR)

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

            src_pt   = os.path.join(src_base, split,
                                    f"{src_folder}_{split}_embeddings_labels.pt")
            csv_path = os.path.join(SPLIT_DIR, f'{split}.csv')
            out_pt   = os.path.join(MODEL_DIR, out_folder, split,
                                    f"{out_folder}_{split}_embeddings_labels.pt")

            if not os.path.exists(src_pt):
                print(f"  Source not found: {src_pt}, skipping.")
                total_skipped += 1
                continue
            if not os.path.exists(csv_path):
                print(f"  Split CSV not found: {csv_path}, skipping.")
                total_skipped += 1
                continue

            ordered_ids     = reconstruct_ordered_ids(csv_path)
            h5ids_needed    = [h5 for h5, _ in ordered_ids]
            split_text_embs = encode_texts(
                annotation_lookup, h5ids_needed, model, text_tokenizer, device
            )

            success = concatenate_and_save(
                src_pt, csv_path, split_text_embs, out_pt,
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