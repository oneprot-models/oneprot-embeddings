import h5py
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Batch
import os
from tqdm import tqdm

import torch
import hydra
from omegaconf import OmegaConf
from huggingface_hub import  HfApi, hf_hub_download
import sys
import os
import h5py
from torch_geometric.data import Batch
from transformers import AutoTokenizer
import time



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.oneprot_module import OneProtLitModule
from src.data.utils.struct_graph_utils import protein_to_graph


def load_split_identifiers(split_file):
    """
    Load identifiers from a split file.
    """
    with open(split_file, 'r') as f:
        identifiers = [line.strip() for line in f if line.strip()]
    return identifiers


def extract_pocket_embeddings(h5_file, identifiers, model, modality='pocket', label=1, device='cuda'):
    """
    Extract pocket embeddings for given identifiers.
    Returns list of (embedding, label) tuples.
    """
    embeddings_list = []
    labels_list = []
    failed_identifiers = []
    
    print(f"Processing {len(identifiers)} pockets from {os.path.basename(h5_file)}...")
    
    # Move model to device
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
        try:
            # Convert H5 pocket to graph
            parts = identifier.split('_')
            pdb_id = parts[0] if len(parts) > 0 else identifier
            chain = parts[1] if len(parts) > 1 else 'A'
            
            # Skip identifiers not present in the H5 file
            if identifier not in h5_available:
                print(f"  Skipping {identifier}: not found in {os.path.basename(h5_file)}")
                failed_identifiers.append(identifier)
                continue

            # Create graph from H5 file
            input_pocket = [protein_to_graph(identifier, h5_file, 'non_pdb', chain, pockets=True)]
            input_pocket = Batch.from_data_list(input_pocket).to(device)
            
            # Get embedding
            with torch.no_grad():
                embedding = model.network[modality](input_pocket)
            
            # Convert to numpy and ensure 1D
            embedding_np = embedding.cpu().numpy()
            
            # Flatten to 1D array
            if len(embedding_np.shape) > 1:
                embedding_np = embedding_np.flatten()
            
            # Validate
            if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():
                print(f"  Warning: NaN or Inf in embedding for {identifier}")
                failed_identifiers.append(identifier)
                continue
            
            embeddings_list.append(embedding_np)
            labels_list.append(label)
            
        except Exception as e:
            print(f"  Error processing {identifier}: {e}")
            import traceback
            traceback.print_exc()
            failed_identifiers.append(identifier)
            continue
    
    print(f"  Successfully processed: {len(embeddings_list)}/{len(identifiers)}")
    if failed_identifiers:
        print(f"  Failed: {failed_identifiers[:5]}")
    
    return embeddings_list, labels_list


def save_embeddings_to_pt(embeddings_list, labels_list, output_path):
    """
    Save embeddings and labels to .pt file in the required format.
    """
    if not embeddings_list:
        print(f"Warning: No embeddings to save!")
        return

    print(f"\nConverting {len(embeddings_list)} embeddings to tensors...")

    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path) or '.'
    os.makedirs(parent_dir, exist_ok=True)

    # Stack embeddings into a single tensor [N, embedding_dim]
    embeddings_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)

    # Convert labels to tensor [N] (use int64/long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

    # Create dictionary
    data_dict = {
        'embeddings': embeddings_tensor,
        'labels_fitness': labels_tensor
    }

    print(f"Embeddings shape: {embeddings_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Embeddings dtype: {embeddings_tensor.dtype}")
    print(f"Labels dtype: {labels_tensor.dtype}")

    # Atomic write: write to temp file in same directory, fsync, then atomically replace
    tmp_path = os.path.join(parent_dir, f".{os.path.basename(output_path)}.{os.getpid()}.tmp")
    with open(tmp_path, 'wb') as f:
        torch.save(data_dict, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    print(f"✓ Saved to {output_path} (size={os.path.getsize(output_path)} bytes)")

    # Verify file with a small retry loop to avoid transient EOFError
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

    # Check label distribution
    unique_labels, counts = torch.unique(verify_data['labels_fitness'], return_counts=True)
    print(f"  Label distribution:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"    Label {label}: {count} samples")


def process_all_splits(h5_files, split_dirs, model, modality='pocket', 
                      output_dir='embeddings',
                      device='cuda'):
    """
    Process all H5 files and their splits to create embedding .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define labels (0-indexed for compatibility with your multi-class setup)
    labels = {
        'allosteric': 0,
        'competitive': 1,
        'noncompetitive': 2
    }
    
    # Process each split (train, valid, test)
    for split_name in ['train', 'valid', 'test']:
        print(f"\n{'='*70}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'='*70}")
        
        all_embeddings = []
        all_labels = []
        
        # Process each H5 file SEQUENTIALLY
        for name, h5_file in h5_files.items():
            label = labels[name]
            split_file = os.path.join(split_dirs[name], f"{split_name}.txt")
            
            print(f"\n{'-'*70}")
            print(f"Processing {name} (label={label})")
            print(f"{'-'*70}")
            
            # Load identifiers for this split
            identifiers = load_split_identifiers(split_file)
            print(f"Loaded {len(identifiers)} identifiers from {split_file}")
            
            # Extract embeddings for this source
            embeddings, labels_list = extract_pocket_embeddings(
                h5_file, identifiers, model, modality, label, device
            )
            
            print(f"Extracted {len(embeddings)} embeddings")
            
            # Extend the lists
            all_embeddings.extend(embeddings)
            all_labels.extend(labels_list)
            
            print(f"Total embeddings so far: {len(all_embeddings)}")
        
        # Save ALL embeddings at once (after all sources processed)
        print(f"\n{'='*70}")
        print(f"SAVING {split_name.upper()} EMBEDDINGS")
        print(f"{'='*70}")
        
        output_path = os.path.join(output_dir,split_name,f"ASD_pockets_{split_name}_embeddings_labels.pt")
        save_embeddings_to_pt(all_embeddings, all_labels, output_path)
        
        # Print statistics
        from collections import Counter
        label_counts = Counter(all_labels)
        
        print(f"\nLabel distribution in {split_name}:")
        for label_val in sorted(label_counts.keys()):
            print(f"  Label {label_val}: {label_counts[label_val]} samples")
    
    print(f"\n{'='*70}")
    print("ALL SPLITS PROCESSED!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  {output_dir}/ASD_pockets_train_labels.pt")
    print(f"  {output_dir}/ASD_pockets_valid_labels.pt")
    print(f"  {output_dir}/ASD_pockets_test_labels.pt")


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
    model_name='oneprot_sanity'
    config_path='/p/project1/hai_oneprot/bazarova1/oneprot-panda/logs/train/runs/2025-03-22_11-50-19/config_1.yaml'
    checkpoint_path='/p/scratch/hai_oneprot/checkpoints_refined_111024/2024-11-05_19-20-45/epoch_043_28400.ckpt'

    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)

    # Prepare components
    components = {
        'sequence': hydra.utils.instantiate(cfg.model.components.sequence),
        'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
        'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'pocket': hydra.utils.instantiate(cfg.model.components.pocket),
        'text': hydra.utils.instantiate(cfg.model.components.text),
        #'md': hydra.utils.instantiate(cfg.model.components.md)
    }

    # Load checkpoint

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
    
    # Define H5 files
    h5_files = {
        'allosteric': '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/binding_pockets_allosteric.h5',
        'competitive': '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/binding_pockets_orthosteric_competitive.h5',
        'noncompetitive': '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/binding_pockets_orthosteric_noncompetitive.h5'
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
        modality='pocket',
        output_dir='embeddings/'+ model_name+'/ASD_pockets/',
        device=device
    )
    
    print("\n✓ Done!")