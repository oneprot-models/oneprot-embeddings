import h5py
import torch
import numpy as np
from torch_geometric.data import Batch
import os
import sys
import glob
import time
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
    These were written by create_h5.py before count_cut2 clipped the chain
    down to the pocket.

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
# Split loading
# ---------------------------------------------------------------------------

def load_split_identifiers(split_file):
    """Load identifiers from a split file."""
    with open(split_file, 'r') as f:
        identifiers = [line.strip() for line in f if line.strip()]
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


def extract_combined_embeddings(h5_file, identifiers, model, tokenizer,
                                seq_dict, modality='pocket', label=1, device='cuda'):
    """
    For each identifier:
        1. Extract pocket embedding via protein_to_graph + model.network['pocket']
        2. Look up full_sequence from seq_dict, extract sequence embedding
        3. Concatenate [pocket | sequence] into a single vector

    Returns (embeddings_list, labels_list).
    """
    embeddings_list = []
    labels_list = []
    failed_identifiers = []

    print(f"Processing {len(identifiers)} pockets from {os.path.basename(h5_file)}...")
    model = model.to(device)

    for identifier in tqdm(identifiers, desc=f"Label {label}"):
        try:
            parts = identifier.split('_')
            chain = parts[1] if len(parts) > 1 else 'A'

            # --------------------------------------------------------------
            # 1. Pocket embedding
            # --------------------------------------------------------------
            input_pocket = [protein_to_graph(identifier, h5_file, 'non_pdb', chain, pockets=True)]
            input_pocket = Batch.from_data_list(input_pocket).to(device)

            with torch.no_grad():
                pocket_emb = model.network[modality](input_pocket)

            pocket_np = pocket_emb.cpu().numpy().flatten()

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

            embeddings_list.append(combined_np)
            labels_list.append(label)

        except Exception as e:
            print(f"  Error processing {identifier}: {e}")
            import traceback
            traceback.print_exc()
            failed_identifiers.append(identifier)
            continue

    print(f"  Successfully processed: {len(embeddings_list)}/{len(identifiers)}")
    if failed_identifiers:
        print(f"  Failed ({len(failed_identifiers)}): {failed_identifiers[:5]}")

    return embeddings_list, labels_list


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_embeddings_to_pt(embeddings_list, labels_list, output_path):
    """Save embeddings and labels to .pt file in the required format."""
    if not embeddings_list:
        print(f"Warning: No embeddings to save!")
        return

    print(f"\nConverting {len(embeddings_list)} embeddings to tensors...")

    parent_dir = os.path.dirname(output_path) or '.'
    os.makedirs(parent_dir, exist_ok=True)

    embeddings_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

    data_dict = {
        'embeddings': embeddings_tensor,
        'labels_fitness': labels_tensor
    }

    print(f"Embeddings shape: {embeddings_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Embeddings dtype: {embeddings_tensor.dtype}")
    print(f"Labels dtype: {labels_tensor.dtype}")

    # Atomic write
    tmp_path = os.path.join(parent_dir, f".{os.path.basename(output_path)}.{os.getpid()}.tmp")
    with open(tmp_path, 'wb') as f:
        torch.save(data_dict, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    print(f"✓ Saved to {output_path} (size={os.path.getsize(output_path)} bytes)")

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

    print(f"Verification:")
    print(f"  Embeddings: {verify_data['embeddings'].shape}, dtype={verify_data['embeddings'].dtype}")
    print(f"  Labels: {verify_data['labels_fitness'].shape}, dtype={verify_data['labels_fitness'].dtype}")

    unique_labels, counts = torch.unique(verify_data['labels_fitness'], return_counts=True)
    print(f"  Label distribution:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"    Label {label}: {count} samples")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_all_splits(h5_files, split_dirs, model, tokenizer, modality='pocket',
                       output_dir='embeddings', device='cuda'):
    """Process all H5 files and their splits to create embedding .pt files."""
    os.makedirs(output_dir, exist_ok=True)

    labels = {
        'allosteric': 0,
        'competitive': 1,
        'noncompetitive': 2
    }

    # ------------------------------------------------------------------
    # Load full-chain sequences from each H5 (no network needed)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING SEQUENCES FROM H5 FILES")
    print("=" * 70)

    h5_seq_dicts = {}
    for name, h5_file in h5_files.items():
        seqs = load_sequences_from_h5(h5_file)
        h5_seq_dicts[name] = seqs
        print(f"  {name}: loaded {len(seqs)} sequences from {os.path.basename(h5_file)}")

    # ------------------------------------------------------------------
    # Process each split
    # ------------------------------------------------------------------
    for split_name in ['train', 'valid', 'test']:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'=' * 70}")

        all_embeddings = []
        all_labels = []

        for name, h5_file in h5_files.items():
            label = labels[name]
            split_file = os.path.join(split_dirs[name], f"{split_name}.txt")

            print(f"\n{'-' * 70}")
            print(f"Processing {name} (label={label})")
            print(f"{'-' * 70}")

            identifiers = load_split_identifiers(split_file)
            print(f"Loaded {len(identifiers)} identifiers from {split_file}")

            embeddings, labels_list = extract_combined_embeddings(
                h5_file, identifiers, model, tokenizer,
                h5_seq_dicts[name], modality, label, device
            )

            print(f"Extracted {len(embeddings)} embeddings")

            all_embeddings.extend(embeddings)
            all_labels.extend(labels_list)

            print(f"Total embeddings so far: {len(all_embeddings)}")

        # Save
        print(f"\n{'=' * 70}")
        print(f"SAVING {split_name.upper()} EMBEDDINGS")
        print(f"{'=' * 70}")

        output_path = os.path.join(output_dir, split_name,
                                   f"ASD_pockets_sequence_{split_name}_embeddings_labels.pt")
        save_embeddings_to_pt(all_embeddings, all_labels, output_path)

        label_counts = Counter(all_labels)
        print(f"\nLabel distribution in {split_name}:")
        for label_val in sorted(label_counts.keys()):
            print(f"  Label {label_val}: {label_counts[label_val]} samples")

    print(f"\n{'=' * 70}")
    print("ALL SPLITS PROCESSED!")
    print(f"{'=' * 70}")
    print(f"\nOutput files:")
    print(f"  {output_dir}/train/ASD_pockets_sequence_train_embeddings_labels.pt")
    print(f"  {output_dir}/valid/ASD_pockets_sequence_valid_embeddings_labels.pt")
    print(f"  {output_dir}/test/ASD_pockets_sequence_test_embeddings_labels.pt")


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

    # Load config
    model_name = 'oneprot_md_combined_gpcr_no_struct_token_32900'
    config_path = '/p/project1/hai_oneprot/bazarova1/oneprot-panda/logs/train/runs/2025-07-26_20-02-45/config.yaml'
    checkpoint_path = '/p/data1/profound_data/checkpoints_oneprot_md/2025-07-25__17:42:35/epoch_015_01100-v4.ckpt'

    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)

    # Prepare components
    components = {
        'sequence': hydra.utils.instantiate(cfg.model.components.sequence),
        ##'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
        'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'pocket': hydra.utils.instantiate(cfg.model.components.pocket),
        'text': hydra.utils.instantiate(cfg.model.components.text),
        'md': hydra.utils.instantiate(cfg.model.components.md)
    }

    # Create model
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

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ------------------------------------------------------------------
    # Tokenizer – offline-safe resolution.
    # Searches: config keys → HF cache on disk → common cluster paths.
    # ------------------------------------------------------------------
    import json as _json

    seq_cfg = OmegaConf.to_container(cfg.model.components.sequence, resolve=True)
    print(f"\n[DEBUG] Full sequence component config:\n{_json.dumps(seq_cfg, indent=2, default=str)}\n")

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

    def find_in_hf_cache(model_name: str) -> str | None:
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
        else:
            print(f"  WARNING: '{tokenizer_name_or_path}' is not a local dir and not found in HF cache.")

    if tokenizer_name_or_path is None or not os.path.isdir(tokenizer_name_or_path):
        _common_cache_roots = [
            os.path.expanduser('~/.cache/huggingface/hub'),
            '/p/project1/hai_oneprot',
            '/p/scratch/hai_oneprot',
        ]
        print("  Scanning common cache locations for any ESM2 tokenizer...")
        for root in _common_cache_roots:
            if not os.path.isdir(root):
                continue
            hits = glob.glob(os.path.join(root, '**', 'tokenizer_config.json'), recursive=True)
            for h in hits:
                print(f"    Found: {h}")
            if hits:
                tokenizer_name_or_path = os.path.dirname(hits[0])
                print(f"  Using: {tokenizer_name_or_path}")
                break

    if tokenizer_name_or_path is None or not os.path.isdir(tokenizer_name_or_path):
        raise FileNotFoundError(
            f"Could not locate a local tokenizer. Resolved path: {tokenizer_name_or_path}\n"
            "Options:\n"
            "  1. Pre-download the tokenizer on a login node with internet access:\n"
            "       python -c \"from transformers import AutoTokenizer; "
            "AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')"
            ".save_pretrained('/path/on/cluster/esm2_tokenizer')\"\n"
            "  2. Hard-code the local path to that saved directory here.\n"
            "  3. Set HF_HOME env var to point at a shared cache directory."
        )

    print(f"Loading tokenizer from: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)

    # Define H5 files
    h5_files = {
        'allosteric': '/p/project1/hai_oneprot/bazarova1/oneprot-panda/binding_pockets_seq_allosteric_competitive.h5',
        'competitive': '/p/project1/hai_oneprot/bazarova1/oneprot-panda/binding_pockets_seq_orthosteric_competitive.h5',
        'noncompetitive': '/p/project1/hai_oneprot/bazarova1/oneprot-panda/binding_pockets_seq_orthosteric_noncompetitive.h5'
    }

    # Define split directories
    split_dirs = {
        'allosteric': '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/splits/allosteric',
        'competitive': '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/splits/competitive',
        'noncompetitive': '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/splits/noncompetitive'
    }

    # Process all splits
    process_all_splits(
        h5_files=h5_files,
        split_dirs=split_dirs,
        model=model,
        tokenizer=tokenizer,
        modality='pocket',
        output_dir='embeddings/' + model_name + '/ASD_pockets_sequence/',
        device=device
    )

    print("\n✓ Done!")