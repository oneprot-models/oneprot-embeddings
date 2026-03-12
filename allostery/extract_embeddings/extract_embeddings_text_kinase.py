import h5py
import torch
import numpy as np
from torch_geometric.data import Batch
import os
import sys
import glob
import time
import pandas as pd
from tqdm import tqdm
from collections import Counter
import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.oneprot_module import OneProtLitModule
from src.data.utils.struct_graph_utils import protein_to_graph

# ---------------------------------------------------------------------------
# Sequence loading from H5
# ---------------------------------------------------------------------------
def load_sequences_from_h5(h5_file: str) -> dict[str, str]:
    """
    Read every 'full_sequence' dataset stored at the identifier root level.
    Returns: {identifier -> full chain sequence string}
    """
    sequences = {}
    with h5py.File(h5_file, 'r') as f:
        for identifier in f.keys():
            if 'full_sequence' in f[identifier]:
                raw = f[identifier]['full_sequence'][()]
                sequences[identifier] = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
    return sequences


# ---------------------------------------------------------------------------
# Split loading from CSV
# ---------------------------------------------------------------------------
def load_split_identifiers_from_csv(csv_file, h5_identifier_col='h5_identifier'):
    """
    Load identifiers from a CSV file.
    """
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found")
        return []
    df = pd.read_csv(csv_file)
    if h5_identifier_col not in df.columns:
        print(f"Error: Column '{h5_identifier_col}' not found in {csv_file}")
        print(f"Available columns: {df.columns.tolist()}")
        return []
    identifiers = df[h5_identifier_col].dropna().tolist()
    return identifiers


# ---------------------------------------------------------------------------
# UniProt annotation loading from XLS
# ---------------------------------------------------------------------------
def load_uniprot_annotations(xls_file: str, uniprot_col: str = 'Uniprot ID') -> dict[str, str]:
    """
    Load UniProt annotations from the XLS/CSV file and build a lookup dict.
    Combines all available annotation columns into a single text string per entry.

    The XLS has columns like: PDB ID, Ligand ID, CHEMBL ID, Uniprot ID,
    Protein kinase, Kinase group, SMILES, Mechanism, Privileged substrate, ...

    We use all non-SMILES text columns to form a descriptive annotation.

    Returns: {uniprot_id -> annotation_text}
    """
    if xls_file.endswith('.csv'):
        df = pd.read_csv(xls_file)
    else:
        df = pd.read_excel(xls_file)

    # Columns to include in the text annotation (skip SMILES - it's not natural language)
    text_columns = [
        'Uniprot ID', 'Protein kinase', 'Kinase group', 'Mechanism',
        # Add any other descriptive columns present in your XLS here
    ]
    # Only keep columns that actually exist in the dataframe
    text_columns = [c for c in text_columns if c in df.columns]

    if uniprot_col not in df.columns:
        raise ValueError(f"Column '{uniprot_col}' not found in {xls_file}. "
                         f"Available: {df.columns.tolist()}")

    annotations = {}
    for _, row in df.iterrows():
        uid = str(row[uniprot_col]).strip()
        if not uid or uid == 'nan':
            continue
        # Build annotation text from all relevant columns
        parts = []
        for col in text_columns:
            val = str(row[col]).strip()
            if val and val != 'nan':
                parts.append(f"{col}: {val}")
        annotation_text = ". ".join(parts)
        # If the same UniProt ID appears multiple times, keep the first non-empty annotation
        if uid not in annotations and annotation_text:
            annotations[uid] = annotation_text

    print(f"  Loaded {len(annotations)} UniProt annotations from {os.path.basename(xls_file)}")
    return annotations


def build_identifier_to_uniprot(split_csv: str,
                                 h5_identifier_col: str = 'h5_identifier',
                                 uniprot_col: str = 'Uniprot ID') -> dict[str, str]:
    """
    Build a mapping from h5_identifier -> Uniprot ID using the split CSV,
    which must contain both columns.

    Returns: {h5_identifier -> uniprot_id}
    """
    df = pd.read_csv(split_csv)
    if h5_identifier_col not in df.columns or uniprot_col not in df.columns:
        print(f"Warning: '{h5_identifier_col}' or '{uniprot_col}' not in {split_csv}. "
              f"Text embeddings will be unavailable for this split.")
        return {}
    mapping = {}
    for _, row in df.iterrows():
        uid = str(row[uniprot_col]).strip()
        hid = str(row[h5_identifier_col]).strip()
        if uid and hid and uid != 'nan' and hid != 'nan':
            mapping[hid] = uid
    return mapping


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_sequence_embedding(sequence: str, model, tokenizer, device='cuda'):
    """Tokenize a protein sequence and run it through model.network['sequence']."""
    encoded = tokenizer(
        sequence,
        return_tensors='pt',
        padding=False,
        truncation=True,
        add_special_tokens=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        seq_embedding = model.network['sequence'](encoded['input_ids'])
    return seq_embedding.cpu().numpy().flatten()


def extract_pocket_embedding(identifier, h5_file, model, chain='A',
                              modality='pocket', device='cuda'):
    """Extract pocket embedding via protein_to_graph + model.network['pocket']."""
    input_pocket = [protein_to_graph(identifier, h5_file, 'non_pdb', chain, pockets=True)]
    input_pocket = Batch.from_data_list(input_pocket).to(device)
    with torch.no_grad():
        pocket_emb = model.network[modality](input_pocket)
    return pocket_emb.cpu().numpy().flatten()


def extract_text_embedding(annotation_text: str, model, text_tokenizer, device='cuda'):
    """
    Tokenize a UniProt annotation string and run it through model.network['text'].
    Uses the same pattern as the OneProt HuggingFace example, but with the
    tokenizer and model loaded from checkpoint/config.
    """
    input_tensor = text_tokenizer(
        annotation_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    )['input_ids'].to(device)
    with torch.no_grad():
        text_emb = model.network['text'](input_tensor)
    return text_emb.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Combined embedding extraction
# ---------------------------------------------------------------------------
def extract_combined_embeddings(h5_file, identifiers, model, tokenizer, text_tokenizer,
                                 seq_dict, uniprot_map, uniprot_annotations,
                                 modality='pocket', label=1, device='cuda'):
    """
    For each identifier:
        1. Extract pocket embedding
        2. Extract sequence embedding
        3. Extract text embedding from UniProt annotation (if available)
        4. Build three concatenation variants:
              - pocket + sequence + text  (combined_pst)
              - pocket + text             (combined_pt)
              - pocket + sequence         (combined_ps, original behaviour)

    Returns dict of lists keyed by embedding type, plus labels_list.
    """
    results = {
        'pocket': [],
        'sequence': [],
        'text': [],
        'pocket_seq': [],        # pocket + sequence
        'pocket_text': [],       # pocket + text
        'pocket_seq_text': [],   # pocket + sequence + text
    }
    labels_list = []
    failed_identifiers = []

    print(f"Processing {len(identifiers)} pockets from {os.path.basename(h5_file)}...")
    model = model.to(device)

    try:
        with h5py.File(h5_file, 'r') as hf:
            h5_available = set(hf.keys())
    except Exception as e:
        print(f"  Error opening H5 file {h5_file}: {e}")
        h5_available = set()

    for identifier in tqdm(identifiers, desc=f"Label {label}"):
        try:
            parts = identifier.split('_')
            chain = parts[1] if len(parts) > 1 else 'A'

            if identifier not in h5_available:
                print(f"  Skipping {identifier}: not found in {os.path.basename(h5_file)}")
                failed_identifiers.append(identifier)
                continue

            # ----------------------------------------------------------
            # 1. Pocket embedding
            # ----------------------------------------------------------
            pocket_np = extract_pocket_embedding(identifier, h5_file, model, chain, modality, device)
            if np.isnan(pocket_np).any() or np.isinf(pocket_np).any():
                print(f"  Warning: NaN/Inf in pocket embedding for {identifier}")
                failed_identifiers.append(identifier)
                continue

            # ----------------------------------------------------------
            # 2. Sequence embedding
            # ----------------------------------------------------------
            sequence = seq_dict.get(identifier)
            if sequence is None:
                print(f"  Warning: full_sequence not found for {identifier}")
                failed_identifiers.append(identifier)
                continue
            seq_np = extract_sequence_embedding(sequence, model, tokenizer, device)
            if np.isnan(seq_np).any() or np.isinf(seq_np).any():
                print(f"  Warning: NaN/Inf in sequence embedding for {identifier}")
                failed_identifiers.append(identifier)
                continue

            # ----------------------------------------------------------
            # 3. Text embedding (from UniProt annotation)
            # ----------------------------------------------------------
            text_np = None
            uniprot_id = uniprot_map.get(identifier)
            if uniprot_id is not None:
                annotation_text = uniprot_annotations.get(uniprot_id)
                if annotation_text:
                    text_np = extract_text_embedding(annotation_text, model, text_tokenizer, device)
                    if np.isnan(text_np).any() or np.isinf(text_np).any():
                        print(f"  Warning: NaN/Inf in text embedding for {identifier}, skipping text")
                        text_np = None
                else:
                    print(f"  Warning: No annotation text for UniProt ID '{uniprot_id}' ({identifier})")
            else:
                print(f"  Warning: No UniProt ID mapping for {identifier}")

            # ----------------------------------------------------------
            # 4. Store results
            # ----------------------------------------------------------
            results['pocket'].append(pocket_np)
            results['sequence'].append(seq_np)
            results['pocket_seq'].append(np.concatenate([pocket_np, seq_np]))

            if text_np is not None:
                results['text'].append(text_np)
                results['pocket_text'].append(np.concatenate([pocket_np, text_np]))
                results['pocket_seq_text'].append(np.concatenate([pocket_np, seq_np, text_np]))
            else:
                # Append None placeholders so list lengths stay consistent
                # (they are filtered before saving)
                results['text'].append(None)
                results['pocket_text'].append(None)
                results['pocket_seq_text'].append(None)

            labels_list.append(label)

        except Exception as e:
            print(f"  Error processing {identifier}: {e}")
            import traceback
            traceback.print_exc()
            failed_identifiers.append(identifier)
            continue

    # Remove None entries (where text was unavailable) from text-dependent lists
    text_mask = [x is not None for x in results['text']]
    n_with_text = sum(text_mask)
    n_total = len(labels_list)
    print(f"  Successfully processed: {n_total}/{len(identifiers)}")
    print(f"  Entries with text embeddings: {n_with_text}/{n_total}")
    if failed_identifiers:
        print(f"  Failed ({len(failed_identifiers)}): {failed_identifiers[:5]}")

    # Filtered versions for text-dependent embedding types
    text_labels = [l for l, m in zip(labels_list, text_mask) if m]
    results['text'] = [x for x in results['text'] if x is not None]
    results['pocket_text'] = [x for x in results['pocket_text'] if x is not None]
    results['pocket_seq_text'] = [x for x in results['pocket_seq_text'] if x is not None]

    return results, labels_list, text_labels


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_embeddings_to_pt(embeddings_list, labels_list, output_path, embedding_type='combined'):
    """Save embeddings and labels to .pt file."""
    # Filter out None entries
    valid = [(e, l) for e, l in zip(embeddings_list, labels_list) if e is not None]
    if not valid:
        print(f"Warning: No valid embeddings to save for {embedding_type}!")
        return
    embeddings_list, labels_list = zip(*valid)

    print(f"\nConverting {len(embeddings_list)} {embedding_type} embeddings to tensors...")
    parent_dir = os.path.dirname(output_path) or '.'
    os.makedirs(parent_dir, exist_ok=True)

    embeddings_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
    data_dict = {'embeddings': embeddings_tensor, 'labels_fitness': labels_tensor}

    print(f"  Embeddings shape: {embeddings_tensor.shape}")
    print(f"  Labels shape:     {labels_tensor.shape}")

    tmp_path = os.path.join(parent_dir, f".{os.path.basename(output_path)}.{os.getpid()}.tmp")
    with open(tmp_path, 'wb') as f:
        torch.save(data_dict, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    print(f"  Saved to {output_path} (size={os.path.getsize(output_path)} bytes)")

    verify_data = torch.load(output_path)
    unique_labels, counts = torch.unique(verify_data['labels_fitness'], return_counts=True)
    print(f"  Label distribution:")
    for lv, cnt in zip(unique_labels.tolist(), counts.tolist()):
        print(f"    Label {lv}: {cnt} samples")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def process_all_splits(competitive_h5, allosteric_h5, split_dir,
                        model, tokenizer, text_tokenizer,
                        uniprot_annotations,
                        modality='pocket', output_dir='embeddings', device='cuda'):
    """
    Process competitive and allosteric H5 files for all splits, producing
    embeddings in four flavours:
        - Kinase_pocket            (pocket only)
        - Kinase_combined          (pocket + sequence)
        - Kinase_pocket_text       (pocket + text)
        - Kinase_pocket_seq_text   (pocket + sequence + text)
    """
    os.makedirs(output_dir, exist_ok=True)
    h5_files = {'competitive': competitive_h5, 'allosteric': allosteric_h5}
    labels   = {'competitive': 0, 'allosteric': 1}

    # ------------------------------------------------------------------
    # Load full-chain sequences from each H5
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING SEQUENCES FROM H5 FILES")
    print("=" * 70)
    h5_seq_dicts = {}
    for name, h5_file in h5_files.items():
        if not os.path.exists(h5_file):
            print(f"  Error: {h5_file} not found!")
            continue
        seqs = load_sequences_from_h5(h5_file)
        h5_seq_dicts[name] = seqs
        print(f"  {name}: loaded {len(seqs)} sequences from {os.path.basename(h5_file)}")

    if not h5_seq_dicts:
        print("Error: No H5 files found!")
        return

    # ------------------------------------------------------------------
    # Process each split
    # ------------------------------------------------------------------
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'=' * 70}")

        csv_file = os.path.join(split_dir, f'{split_name}.csv')
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping {split_name}")
            continue

        df_split = pd.read_csv(csv_file)
        if 'Mechanism' not in df_split.columns or 'h5_identifier' not in df_split.columns:
            print(f"Error: Required columns not found in {csv_file}")
            print(f"Available columns: {df_split.columns.tolist()}")
            continue

        # Build h5_identifier -> UniProt ID mapping for this split
        id_to_uniprot = build_identifier_to_uniprot(csv_file)

        # Accumulators for all mechanisms in this split
        all_results = {k: [] for k in
                       ['pocket', 'sequence', 'text', 'pocket_seq', 'pocket_text', 'pocket_seq_text']}
        all_labels = []
        all_text_labels = []

        for mechanism_name, mechanism_label in labels.items():
            df_mech = df_split[df_split['Mechanism'].str.lower() == mechanism_name.lower()]
            if len(df_mech) == 0:
                print(f"\nNo {mechanism_name} entries in {split_name}, skipping...")
                continue

            identifiers = df_mech['h5_identifier'].dropna().tolist()
            print(f"\n{'-' * 70}")
            print(f"Processing {mechanism_name} (label={mechanism_label})")
            print(f"{'-' * 70}")
            print(f"Loaded {len(identifiers)} identifiers")

            h5_file  = h5_files[mechanism_name]
            seq_dict = h5_seq_dicts[mechanism_name]

            results, lbl_list, txt_lbl_list = extract_combined_embeddings(
                h5_file, identifiers, model, tokenizer, text_tokenizer,
                seq_dict, id_to_uniprot, uniprot_annotations,
                modality, mechanism_label, device
            )

            for k in all_results:
                all_results[k].extend(results[k])
            all_labels.extend(lbl_list)
            all_text_labels.extend(txt_lbl_list)

        # ------------------------------------------------------------------
        # Save all four embedding variants
        # ------------------------------------------------------------------
        print(f"\n{'=' * 70}")
        print(f"SAVING {split_name.upper()} EMBEDDINGS")
        print(f"{'=' * 70}")

        variants = {
            'Kinase_pocket':          ('pocket',          all_labels),
            'Kinase_combined':        ('pocket_seq',       all_labels),
            'Kinase_pocket_text':     ('pocket_text',      all_text_labels),
            'Kinase_pocket_seq_text': ('pocket_seq_text',  all_text_labels),
        }
        for folder, (key, lbls) in variants.items():
            path = os.path.join(
                output_dir, folder, split_name,
                f"{folder}_{split_name}_embeddings_labels.pt"
            )
            save_embeddings_to_pt(all_results[key], lbls, path, embedding_type=key)

        # Label distribution summary
        label_counts = Counter(all_labels)
        print(f"\nLabel distribution in {split_name} (all embeddings):")
        for lv in sorted(label_counts):
            mechanism = 'competitive' if lv == 0 else 'allosteric'
            print(f"  Label {lv} ({mechanism}): {label_counts[lv]} samples")

    print(f"\n{'=' * 70}")
    print("ALL SPLITS PROCESSED!")
    print(f"{'=' * 70}")


# ============================================================================
# USAGE
# ============================================================================
if __name__ == "__main__":
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Configuration - EDIT THESE PATHS
    # -------------------------------------------------------------------------
    config_path     = '/p/project1/hai_oneprot/bazarova1/oneprot-panda/logs/train/runs/2025-03-22_11-50-19/config_1.yaml'
    checkpoint_path = '/p/scratch/hai_oneprot/checkpoints_refined_111024/2024-11-05_19-20-45/epoch_043_28400.ckpt'
    model_name      = 'oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity'

    competitive_h5 = 'competitive_pockets_csv.h5'
    allosteric_h5  = 'allosteric_pockets_csv.h5'
    split_dir      = '0.3/'
    output_dir     = 'embeddings/' + model_name

    # Path to the XLS file described in the prompt (used to extract UniProt annotations)
    xls_file       = 'kinase_data.xlsx'   # <-- EDIT: path to your XLS/CSV file

    # -------------------------------------------------------------------------
    # Load config and create model
    # -------------------------------------------------------------------------
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)

    print("Initializing model components...")
    components = {
        'sequence':     hydra.utils.instantiate(cfg.model.components.sequence),
        'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'pocket':       hydra.utils.instantiate(cfg.model.components.pocket),
        'text':         hydra.utils.instantiate(cfg.model.components.text),
        'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
    }

    print("Creating model...")
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
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✓ Model loaded")

    # -------------------------------------------------------------------------
    # Load sequence tokenizer (offline-safe, same as before)
    # -------------------------------------------------------------------------
    print("\nLoading sequence tokenizer...")
    seq_cfg = OmegaConf.to_container(cfg.model.components.sequence, resolve=True)
    _candidate_keys = [
        'tokenizer_name_or_path', 'pretrained_model_name_or_path',
        'model_name_or_path', 'pretrained_model_name', 'tokenizer_path', 'model_name',
    ]
    tokenizer_name_or_path = None
    for key in _candidate_keys:
        val = seq_cfg.get(key)
        if val:
            tokenizer_name_or_path = str(val)
            print(f"  Found tokenizer ref under config key '{key}': {tokenizer_name_or_path}")
            break

    def find_in_hf_cache(model_name: str):
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        alt_cache = os.environ.get('TRANSFORMERS_CACHE', '')
        search_dirs = [os.path.join(hf_home, 'hub')]
        if alt_cache:
            search_dirs.append(alt_cache)
        cache_folder_name = 'models--' + model_name.replace('/', '--')
        for base in search_dirs:
            candidate = os.path.join(base, cache_folder_name)
            if os.path.isdir(candidate):
                snapshots = glob.glob(os.path.join(candidate, 'snapshots', '*'))
                if snapshots:
                    latest = max(snapshots, key=os.path.getmtime)
                    print(f"  Found cached model at: {latest}")
                    return latest
        for base in search_dirs:
            flat = glob.glob(os.path.join(base, '**', 'tokenizer_config.json'), recursive=True)
            for f in flat:
                if model_name.split('/')[-1] in f:
                    return os.path.dirname(f)
        return None

    if tokenizer_name_or_path and not os.path.isdir(tokenizer_name_or_path):
        cached = find_in_hf_cache(tokenizer_name_or_path)
        if cached:
            tokenizer_name_or_path = cached

    if tokenizer_name_or_path is None or not os.path.isdir(tokenizer_name_or_path):
        _common_cache_roots = [
            os.path.expanduser('~/.cache/huggingface/hub'),
            '/p/project1/hai_oneprot',
            '/p/scratch/hai_oneprot',
        ]
        for root in _common_cache_roots:
            if not os.path.isdir(root):
                continue
            hits = glob.glob(os.path.join(root, '**', 'tokenizer_config.json'), recursive=True)
            if hits:
                tokenizer_name_or_path = os.path.dirname(hits[0])
                print(f"  Using: {tokenizer_name_or_path}")
                break

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
    print("✓ Sequence tokenizer loaded")

    # -------------------------------------------------------------------------
    # Load text tokenizer from config  (mirrors how sequence tokenizer is found)
    # -------------------------------------------------------------------------
    print("\nLoading text tokenizer...")
    text_cfg = OmegaConf.to_container(cfg.model.components.text, resolve=True)
    text_tokenizer_path = None
    for key in _candidate_keys:
        val = text_cfg.get(key)
        if val:
            text_tokenizer_path = str(val)
            print(f"  Found text tokenizer ref under config key '{key}': {text_tokenizer_path}")
            break

    # Apply the same offline cache resolution as for the sequence tokenizer
    if text_tokenizer_path and not os.path.isdir(text_tokenizer_path):
        cached = find_in_hf_cache(text_tokenizer_path)
        if cached:
            text_tokenizer_path = cached

    if text_tokenizer_path is None or not os.path.isdir(text_tokenizer_path):
        raise FileNotFoundError(
            "Could not locate text tokenizer from config or HF cache. "
            f"Resolved path: {text_tokenizer_path}\n"
            "Set HF_HOME or add the tokenizer path under cfg.model.components.text."
        )

    text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, local_files_only=True)
    print("✓ Text tokenizer loaded")

    # -------------------------------------------------------------------------
    # Load UniProt annotations from XLS
    # -------------------------------------------------------------------------
    print("\nLoading UniProt annotations...")
    uniprot_annotations = load_uniprot_annotations(xls_file)

    # -------------------------------------------------------------------------
    # Process all splits
    # -------------------------------------------------------------------------
    process_all_splits(
        competitive_h5=competitive_h5,
        allosteric_h5=allosteric_h5,
        split_dir=split_dir,
        model=model,
        tokenizer=tokenizer,
        text_tokenizer=text_tokenizer,
        uniprot_annotations=uniprot_annotations,
        modality='pocket',
        output_dir=output_dir,
        device=device
    )

    print("\n✓ Done!")