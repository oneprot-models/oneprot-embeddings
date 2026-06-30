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
    
    Args:
        csv_file: Path to CSV file (train.csv, val.csv, test.csv)
        h5_identifier_col: Column name containing H5 identifiers
    
    Returns:
        List of H5 identifiers
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
    """
    Extract pocket embedding via protein_to_graph + model.network['pocket'].
    
    Args:
        identifier: H5 identifier
        h5_file: Path to H5 file
        model: Model with network dict
        chain: Chain ID (default 'A')
        modality: Modality key in model.network (default 'pocket')
        device: Device to use
    
    Returns:
        Pocket embedding as numpy array
    """
    #print(identifier," identifier!!!!!!!!3333",flush=True)
    input_pocket = [protein_to_graph(identifier, h5_file, 'non_pdb', 'A', pockets=True)]
    input_pocket = Batch.from_data_list(input_pocket).to(device)
    #print(identifier," identifier!!!!!!!!4444",flush=True)

    with torch.no_grad():
        pocket_emb = model.network[modality](input_pocket)

    return pocket_emb.cpu().numpy().flatten()


def extract_combined_embeddings(h5_file, identifiers, model, tokenizer,
                                seq_dict, modality='pocket', label=1, device='cuda'):
    """
    For each identifier:
        1. Extract pocket embedding via protein_to_graph + model.network['pocket']
        2. Look up full_sequence from seq_dict, extract sequence embedding
        3. Concatenate [pocket | sequence] into a single vector

    Returns (pocket_embeddings_list, sequence_embeddings_list, combined_embeddings_list, labels_list).
    """
    pocket_embeddings_list = []
    sequence_embeddings_list = []
    combined_embeddings_list = []
    labels_list = []
    failed_identifiers = []

    print(f"Processing {len(identifiers)} pockets from {os.path.basename(h5_file)}...")
    model = model.to(device)

    # Precompute identifiers present in the H5 file so we only attempt
    # extraction for entries that actually exist there (pocket data).
    try:
        with h5py.File(h5_file, 'r') as hf:
            h5_available = set(hf.keys())
    except Exception as e:
        print(f"  Error opening H5 file {h5_file}: {e}")
        h5_available = set()

    for identifier in tqdm(identifiers, desc=f"Label {label}"):
        #print(identifier," identifier!!!!!!!!1111")
        try:
            # Extract chain from identifier if present
            parts = identifier.split('_')
            chain = parts[1] if len(parts) > 1 else 'A'

            # Skip identifiers not present in the H5 file (pocket not available)
            if identifier not in h5_available:
                print(f"  Skipping {identifier}: not found in {os.path.basename(h5_file)}")
                failed_identifiers.append(identifier)
                continue

            # --------------------------------------------------------------
            # 1. Pocket embedding
            # --------------------------------------------------------------
            #print(identifier," identifier!!!!!!!!2222")
            pocket_np = extract_pocket_embedding(
                identifier, h5_file, model, 'A', modality, device
            )

            if np.isnan(pocket_np).any() or np.isinf(pocket_np).any():
                print(f"  Warning: NaN/Inf in pocket embedding for {identifier}")
                failed_identifiers.append(identifier)
                continue

            # --------------------------------------------------------------
            # 2. Sequence embedding
            # --------------------------------------------------------------
            sequence = seq_dict.get(identifier)
            if sequence is None:
                print(f"  Warning: full_sequence not found in H5 for {identifier}")
                failed_identifiers.append(identifier)
                continue

            seq_np = extract_sequence_embedding(sequence, model, tokenizer, device)

            if np.isnan(seq_np).any() or np.isinf(seq_np).any():
                print(f"  Warning: NaN/Inf in sequence embedding for {identifier}")
                failed_identifiers.append(identifier)
                continue

            # --------------------------------------------------------------
            # 3. Concatenate [pocket | sequence]
            # --------------------------------------------------------------
            combined_np = np.concatenate([pocket_np, seq_np])

            pocket_embeddings_list.append(pocket_np)
            sequence_embeddings_list.append(seq_np)
            combined_embeddings_list.append(combined_np)
            labels_list.append(label)

        except Exception as e:
            print(f"  Error processing {identifier}: {e}")
            import traceback
            traceback.print_exc()
            failed_identifiers.append(identifier)
            continue

    print(f"  Successfully processed: {len(combined_embeddings_list)}/{len(identifiers)}")
    if failed_identifiers:
        print(f"  Failed ({len(failed_identifiers)}): {failed_identifiers[:5]}")

    return pocket_embeddings_list, sequence_embeddings_list, combined_embeddings_list, labels_list


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_embeddings_to_pt(embeddings_list, labels_list, output_path, embedding_type='combined'):
    """Save embeddings and labels to .pt file."""
    if not embeddings_list:
        print(f"Warning: No embeddings to save for {embedding_type}!")
        return

    print(f"\nConverting {len(embeddings_list)} {embedding_type} embeddings to tensors...")

    parent_dir = os.path.dirname(output_path) or '.'
    os.makedirs(parent_dir, exist_ok=True)

    embeddings_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

    data_dict = {
        'embeddings': embeddings_tensor,
        'labels_fitness': labels_tensor
    }

    print(f"  Embeddings shape: {embeddings_tensor.shape}")
    print(f"  Labels shape: {labels_tensor.shape}")
    print(f"  Embeddings dtype: {embeddings_tensor.dtype}")
    print(f"  Labels dtype: {labels_tensor.dtype}")

    # Atomic write
    tmp_path = os.path.join(parent_dir, f".{os.path.basename(output_path)}.{os.getpid()}.tmp")
    with open(tmp_path, 'wb') as f:
        torch.save(data_dict, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    print(f"  ✓ Saved to {output_path} (size={os.path.getsize(output_path)} bytes)")

    # Verify
    verify_data = None
    for attempt in range(3):
        try:
            verify_data = torch.load(output_path)
            break
        except EOFError:
            if attempt < 2:
                time.sleep(0.5)
            else:
                raise

    print(f"  Verification:")
    print(f"    Embeddings: {verify_data['embeddings'].shape}, dtype={verify_data['embeddings'].dtype}")
    print(f"    Labels: {verify_data['labels_fitness'].shape}, dtype={verify_data['labels_fitness'].dtype}")

    unique_labels, counts = torch.unique(verify_data['labels_fitness'], return_counts=True)
    print(f"    Label distribution:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"      Label {label}: {count} samples")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_all_splits(competitive_h5, allosteric_h5, split_dir, model, tokenizer, 
                      modality='pocket', output_dir='embeddings', device='cuda'):
    """
    Process competitive and allosteric H5 files with their splits to create embedding .pt files.
    
    Args:
        competitive_h5: Path to competitive_pockets.h5
        allosteric_h5: Path to allosteric_pockets.h5
        split_dir: Directory containing train.csv, val.csv, test.csv
        model: OneProt model
        tokenizer: Tokenizer for sequence encoding
        modality: Modality key for pocket encoding (default 'pocket')
        output_dir: Base output directory
        device: Device to use
    """
    os.makedirs(output_dir, exist_ok=True)

    h5_files = {
        'competitive': competitive_h5,
        'allosteric': allosteric_h5
    }

    labels = {
        'competitive': 0,  # Label 0 for competitive
        'allosteric': 1    # Label 1 for allosteric
    }

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

        all_pocket_embeddings = []
        all_sequence_embeddings = []
        all_combined_embeddings = []
        all_labels = []

        # Load split CSV
        csv_file = os.path.join(split_dir, f'{split_name}.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping {split_name}")
            continue

        print(f"\nLoading identifiers from: {csv_file}")
        df_split = pd.read_csv(csv_file)
        
        # Check for mechanism and h5_identifier columns
        if 'Mechanism' not in df_split.columns or 'h5_identifier' not in df_split.columns:
            print(f"Error: Required columns not found in {csv_file}")
            print(f"Available columns: {df_split.columns.tolist()}")
            continue

        # Process each mechanism
        for mechanism_name, mechanism_label in labels.items():
            # Filter by mechanism
            df_mechanism = df_split[df_split['Mechanism'].str.lower() == mechanism_name.lower()]
            
            if len(df_mechanism) == 0:
                print(f"\nNo {mechanism_name} entries in {split_name}, skipping...")
                continue

            identifiers = df_mechanism['h5_identifier'].dropna().tolist()
            
            print(f"\n{'-' * 70}")
            print(f"Processing {mechanism_name} (label={mechanism_label})")
            print(f"{'-' * 70}")
            print(f"Loaded {len(identifiers)} identifiers")

            h5_file = h5_files[mechanism_name]
            seq_dict = h5_seq_dicts[mechanism_name]

            # Extract embeddings
            pocket_embs, seq_embs, combined_embs, labels_list = extract_combined_embeddings(
                h5_file, identifiers, model, tokenizer,
                seq_dict, modality, mechanism_label, device
            )

            print(f"Extracted {len(combined_embs)} embeddings")

            all_pocket_embeddings.extend(pocket_embs)
            all_sequence_embeddings.extend(seq_embs)
            all_combined_embeddings.extend(combined_embs)
            all_labels.extend(labels_list)

            print(f"Total embeddings so far: {len(all_combined_embeddings)}")

        # Save combined embeddings
        print(f"\n{'=' * 70}")
        print(f"SAVING {split_name.upper()} EMBEDDINGS")
        print(f"{'=' * 70}")

        # Save pocket+sequence combined
        combined_output_path = os.path.join(
            output_dir+'/Kinase_combined/' , 
            split_name,
            f"Kinase_combined_{split_name}_embeddings_labels.pt"
        )
        save_embeddings_to_pt(all_combined_embeddings, all_labels, 
                            combined_output_path, embedding_type='combined')

        # Save pocket-only embeddings
        pocket_output_path = os.path.join(
            output_dir+'/Kinase_pocket/' ,
            split_name,
            f"Kinase_pocket_{split_name}_embeddings_labels.pt"
        )
        save_embeddings_to_pt(all_pocket_embeddings, all_labels,
                            pocket_output_path, embedding_type='pocket')

        # Print label distribution
        label_counts = Counter(all_labels)
        print(f"\nLabel distribution in {split_name}:")
        for label_val in sorted(label_counts.keys()):
            mechanism = 'competitive' if label_val == 0 else 'allosteric'
            print(f"  Label {label_val} ({mechanism}): {label_counts[label_val]} samples")

    print(f"\n{'=' * 70}")
    print("ALL SPLITS PROCESSED!")
    print(f"{'=' * 70}")
    print(f"\nOutput files:")
    print(f"  Combined (pocket+sequence):")
    print(f"    {output_dir}/Kinase_combined/train/Kinase_combined_train_embeddings_labels.pt")
    print(f"    {output_dir}/Kinase_combined/val/Kinase_combined_val_embeddings_labels.pt")
    print(f"    {output_dir}/Kinase_combined/test/Kinase_combined_test_embeddings_labels.pt")
    print(f"\n  Pocket-only:")
    print(f"    {output_dir}/Kinase_pocket/train/Kinase_pocket_train_embeddings_labels.pt")
    print(f"    {output_dir}/Kinase_pocket/val/Kinase_pocket_val_embeddings_labels.pt")
    print(f"    {output_dir}/Kinase_pocket/test/Kinase_pocket_test_embeddings_labels.pt")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":

    # Ensure single-threaded execution
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Configuration - EDIT THESE PATHS
    # -------------------------------------------------------------------------
    
    # Model configuration
    config_path = 'config_1.yaml'
    checkpoint_path = 'epoch_043_28400.ckpt'
    model_name = 'oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity'  # Used for output naming, e.g. 'oneprot_md' or 'oneprot_no_md'

    # H5 files (output from process_pockets_from_excel.py)
    competitive_h5 = 'competitive_pockets_csv.h5'
    allosteric_h5 = 'allosteric_pockets_csv.h5'
    
    # Split directory (output from step2_create_splits.py)
    split_dir = 'KinSite_splits'  # or 'splits' if using unverified splits
    
    # Output directory for embeddings
    output_dir = 'embeddings/' + model_name
    
    # -------------------------------------------------------------------------
    # Load config and create model
    # -------------------------------------------------------------------------
    
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)

    # Prepare components
    print("Initializing model components...")
    components = {
        'sequence': hydra.utils.instantiate(cfg.model.components.sequence),
        #'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'pocket': hydra.utils.instantiate(cfg.model.components.pocket),
        'text': hydra.utils.instantiate(cfg.model.components.text),
        #'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
        #'md': hydra.utils.instantiate(cfg.model.components.md)
    }

    # Create model
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
    # Load tokenizer (offline-safe)
    # -------------------------------------------------------------------------
    
    print("\nLoading tokenizer...")
    
    seq_cfg = OmegaConf.to_container(cfg.model.components.sequence, resolve=True)
    
    _candidate_keys = [
        'tokenizer_name_or_path',
        'pretrained_model_name_or_path',
        'model_name_or_path',
        'pretrained_model_name',
        'tokenizer_path',
        'model_name',
    ]

    tokenizer_name_or_path = None
    for key in _candidate_keys:
        val = seq_cfg.get(key)
        if val:
            tokenizer_name_or_path = str(val)
            print(f"  Found tokenizer ref under config key '{key}': {tokenizer_name_or_path}")
            break

    if tokenizer_name_or_path is None:
        print("  WARNING: No tokenizer key found in sequence config. Falling back to cache scan.")

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
                    print(f"  Found tokenizer_config.json at: {f}")
                    return os.path.dirname(f)
        return None

    if tokenizer_name_or_path and not os.path.isdir(tokenizer_name_or_path):
        cached = find_in_hf_cache(tokenizer_name_or_path)
        if cached:
            tokenizer_name_or_path = cached

    if tokenizer_name_or_path is None or not os.path.isdir(tokenizer_name_or_path):
        _common_cache_roots = [
            os.path.expanduser('~/.cache/huggingface/hub'),
            '..',
            '..',
        ]
        print("  Scanning common cache locations for any ESM2 tokenizer...")
        for root in _common_cache_roots:
            if not os.path.isdir(root):
                continue
            hits = glob.glob(os.path.join(root, '**', 'tokenizer_config.json'), recursive=True)
            if hits:
                tokenizer_name_or_path = os.path.dirname(hits[0])
                print(f"  Using: {tokenizer_name_or_path}")
                break

    if tokenizer_name_or_path is None or not os.path.isdir(tokenizer_name_or_path):
        raise FileNotFoundError(
            f"Could not locate a local tokenizer. Resolved path: {tokenizer_name_or_path}\n"
            "Please download the tokenizer first or set HF_HOME env var."
        )

    print(f"Loading tokenizer from: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
    print("✓ Tokenizer loaded")

    # -------------------------------------------------------------------------
    # Process all splits
    # -------------------------------------------------------------------------
    
    process_all_splits(
        competitive_h5=competitive_h5,
        allosteric_h5=allosteric_h5,
        split_dir=split_dir,
        model=model,
        tokenizer=tokenizer,
        modality='pocket',
        output_dir=output_dir,
        device=device
    )

    print("\n✓ Done!")