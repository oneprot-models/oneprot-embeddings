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
from huggingface_hub import HfApi, hf_hub_download
import sys
import os
import h5py
from torch_geometric.data import Batch
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.oneprot_module import OneProtLitModule
from src.data.utils.struct_graph_utils import protein_to_graph


def load_identifiers_from_h5(h5_file):
    """
    Load all identifiers from H5 file.
    """
    with h5py.File(h5_file, 'r') as f:
        identifiers = list(f.keys())
    return identifiers


def load_split_from_csv(train_csv, test_csv):
    """
    Load train/valid/test splits from CSV files.
    
    Args:
        train_csv: Path to training CSV
        test_csv: Path to test CSV (will be split 50/50 into valid/test)
    
    Returns:
        train_pdb_ids, valid_pdb_ids, test_pdb_ids
    """
    print(f"\n{'='*70}")
    print(f"LOADING SPLITS FROM CSV FILES")
    print(f"{'='*70}")
    
    # Load train CSV
    train_df = pd.read_csv(train_csv)
    print(f"Loaded train CSV: {train_csv}")
    print(f"  Columns: {train_df.columns.tolist()}")
    print(f"  Rows: {len(train_df)}")
    
    # Load test CSV
    test_df = pd.read_csv(test_csv)
    print(f"\nLoaded test CSV: {test_csv}")
    print(f"  Columns: {test_df.columns.tolist()}")
    print(f"  Rows: {len(test_df)}")
    
    # Extract PDB IDs for train
    train_pdb_ids = train_df['pdb_id'].unique().tolist()
    print(f"\nTrain unique PDB IDs: {len(train_pdb_ids)}")
    
    # Split test CSV 50/50 into valid and test
    test_pdb_ids_all = test_df['pdb_id'].unique().tolist()
    mid_point = len(test_pdb_ids_all) // 2
    
    valid_pdb_ids = test_pdb_ids_all[:mid_point]
    test_pdb_ids = test_pdb_ids_all[mid_point:]
    
    print(f"Valid unique PDB IDs (first 50% of test): {len(valid_pdb_ids)}")
    print(f"Test unique PDB IDs (second 50% of test): {len(test_pdb_ids)}")
    
    return train_pdb_ids, valid_pdb_ids, test_pdb_ids


def map_pdb_ids_to_h5_identifiers(pdb_ids, all_h5_identifiers):
    """
    Map PDB IDs to H5 identifiers.
    H5 identifiers are in format: {pdb_id}_{chain}
    
    Args:
        pdb_ids: List of PDB IDs (e.g., ['1ABC', '2DEF'])
        all_h5_identifiers: List of all identifiers in H5 file (e.g., ['1ABC_A', '2DEF_B'])
    
    Returns:
        List of matching H5 identifiers
    """
    # Convert PDB IDs to lowercase for case-insensitive matching
    pdb_ids_lower = [pdb_id.lower() for pdb_id in pdb_ids]
    
    matched_identifiers = []
    for h5_id in all_h5_identifiers:
        # Extract PDB ID from H5 identifier (format: pdb_chain)
        pdb_part = h5_id.split('_')[0].lower()
        
        if pdb_part in pdb_ids_lower:
            matched_identifiers.append(h5_id)
    
    return matched_identifiers


def extract_labels_from_h5(h5_file, identifier, chain='A'):
    """
    Extract labels from H5 file for a given identifier.
    Returns the labels array (binary labels for each residue in the pocket).
    """
    with h5py.File(h5_file, 'r') as f:
        try:
            # Try to extract chain from identifier
            if '_' in identifier:
                parts = identifier.split('_')
                if len(parts) >= 2:
                    chain = parts[1]
            
            labels = f[identifier]['structure']['0'][chain]['residues']['labels'][()]
            return labels
        except KeyError as e:
            print(f"  Warning: Labels not found for {identifier}, chain {chain}: {e}")
            return None


def extract_pocket_embeddings_with_labels(h5_file, identifiers, model, modality='pocket', device='cuda'):
    """
    Extract pocket embeddings and their corresponding per-residue labels.
    
    Returns:
        embeddings_list: List of 1D embedding arrays
        labels_list: List of 1D label arrays (length = number of residues in pocket, typically 100)
    """
    embeddings_list = []
    labels_list = []
    failed_identifiers = []
    
    print(f"Processing {len(identifiers)} pockets from {os.path.basename(h5_file)}...")
    
    # Move model to device
    model = model.to(device)
    
    for identifier in tqdm(identifiers, desc="Extracting embeddings"):
        try:
            # Extract chain from identifier
            chain = 'A'  # default
            if '_' in identifier:
                parts = identifier.split('_')
                if len(parts) >= 2:
                    chain = parts[1]
            
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
            
            # Validate embedding
            if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():
                print(f"  Warning: NaN or Inf in embedding for {identifier}")
                failed_identifiers.append(identifier)
                continue
            
            # Extract per-residue labels from H5 file
            pocket_labels = extract_labels_from_h5(h5_file, identifier, chain)
            
            if pocket_labels is None:
                print(f"  Warning: Could not extract labels for {identifier}")
                failed_identifiers.append(identifier)
                continue
            
            # Ensure labels are the correct type and shape
            pocket_labels = np.array(pocket_labels, dtype=np.int32)
            
            embeddings_list.append(embedding_np)
            labels_list.append(pocket_labels)
            
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


def save_embeddings_to_pt(embeddings_list, labels_list, output_path):
    """
    Save embeddings and per-residue labels to .pt file.
    
    Args:
        embeddings_list: List of numpy arrays, each of shape [embedding_dim]
        labels_list: List of numpy arrays, each of shape [num_residues] (typically 100)
        output_path: Path to save the .pt file
    
    Output format:
        {
            'embeddings': torch.Tensor([N, embedding_dim], dtype=torch.float32),
            'labels_fitness': torch.Tensor([N, num_residues], dtype=torch.long)
        }
    """
    if not embeddings_list:
        print(f"Warning: No embeddings to save!")
        return
    
    print(f"\nConverting {len(embeddings_list)} embeddings to tensors...")
    
    # Stack embeddings into a single tensor [N, embedding_dim]
    embeddings_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)

    print(embeddings_tensor.shape," shape tensors!!!")
    
    # Stack labels into a single tensor [N, num_residues]

    for i in range(len(labels_list)):
        if labels_list[i].shape[0]!=100:
            labels_new=np.zeros((100,),dtype=np.int32)-1
            labels_new[:labels_list[i].shape[0]]=labels_list[i]
            labels_list[i]=labels_new
    labels_tensor = torch.tensor(np.stack(labels_list), dtype=torch.long)
    
    # Create dictionary
    data_dict = {
        'embeddings': embeddings_tensor,
        'labels_fitness': labels_tensor
    }
    
    print(f"Embeddings shape: {embeddings_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Embeddings dtype: {embeddings_tensor.dtype}")
    print(f"Labels dtype: {labels_tensor.dtype}")
    
    # Save to file
    torch.save(data_dict, output_path)
    print(f"✓ Saved to {output_path}")
    
    # Verify file
    # verify_data = torch.load(output_path, weights_only=False)
    # print(f"Verification:")
    # print(f"  Embeddings: {verify_data['embeddings'].shape}, dtype={verify_data['embeddings'].dtype}")
    # print(f"  Labels: {verify_data['labels_fitness'].shape}, dtype={verify_data['labels_fitness'].dtype}")
    
    # # Check label distribution across all residues
    # all_labels_flat = verify_data['labels_fitness'].flatten()
    # unique_labels, counts = torch.unique(all_labels_flat, return_counts=True)
    # print(f"  Label distribution (across all residues):")
    # for label, count in zip(unique_labels.tolist(), counts.tolist()):
    #     print(f"    Label {label}: {count} residues ({count/len(all_labels_flat)*100:.2f}%)")
    
    # # Show per-pocket statistics
    # labels_per_pocket = verify_data['labels_fitness'].sum(dim=1)
    # print(f"  Per-pocket allosteric residues:")
    # print(f"    Mean: {labels_per_pocket.float().mean():.2f}")
    # print(f"    Min: {labels_per_pocket.min()}")
    # print(f"    Max: {labels_per_pocket.max()}")


def process_h5_to_embeddings(h5_file, train_csv, test_csv, model, model_name, modality='pocket', 
                              output_dir='embeddings', device='cuda'):
    """
    Process H5 file and create embedding .pt files for train/valid/test splits
    based on CSV files.
    
    Args:
        h5_file: Path to H5 file containing pocket data
        train_csv: Path to training CSV with 'pdb_id' column
        test_csv: Path to test CSV with 'pdb_id' column (will be split 50/50)
        model: Trained model for embedding extraction
        modality: Model modality to use
        output_dir: Directory to save output files
        device: Device for computation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all identifiers from H5
    all_h5_identifiers = load_identifiers_from_h5(h5_file)
    print(f"\nTotal identifiers in H5: {len(all_h5_identifiers)}")
    print(f"Example identifiers: {all_h5_identifiers[:5]}")
    
    # Load splits from CSV
    train_pdb_ids, valid_pdb_ids, test_pdb_ids = load_split_from_csv(train_csv, test_csv)
    
    # Map PDB IDs to H5 identifiers
    print(f"\n{'='*70}")
    print(f"MAPPING PDB IDs TO H5 IDENTIFIERS")
    print(f"{'='*70}")
    
    train_identifiers = map_pdb_ids_to_h5_identifiers(train_pdb_ids, all_h5_identifiers)
    valid_identifiers = map_pdb_ids_to_h5_identifiers(valid_pdb_ids, all_h5_identifiers)
    test_identifiers = map_pdb_ids_to_h5_identifiers(test_pdb_ids, all_h5_identifiers)
    
    print(f"\nMatched H5 identifiers:")
    print(f"  Train: {len(train_identifiers)} (from {len(train_pdb_ids)} PDB IDs)")
    print(f"  Valid: {len(valid_identifiers)} (from {len(valid_pdb_ids)} PDB IDs)")
    print(f"  Test: {len(test_identifiers)} (from {len(test_pdb_ids)} PDB IDs)")
    
    # Check for unmatched PDB IDs
    matched_train = len(train_identifiers)
    matched_valid = len(valid_identifiers)
    matched_test = len(test_identifiers)
    
    if matched_train < len(train_pdb_ids):
        print(f"\n⚠ Warning: {len(train_pdb_ids) - matched_train} train PDB IDs not found in H5")
    if matched_valid < len(valid_pdb_ids):
        print(f"⚠ Warning: {len(valid_pdb_ids) - matched_valid} valid PDB IDs not found in H5")
    if matched_test < len(test_pdb_ids):
        print(f"⚠ Warning: {len(test_pdb_ids) - matched_test} test PDB IDs not found in H5")
    
    splits = {
        'train': train_identifiers,
        'valid': valid_identifiers,
        'test': test_identifiers
    }
    
    # Process each split
    for split_name, identifier_list in splits.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'='*70}")
        
        if len(identifier_list) == 0:
            print(f"⚠ Warning: No identifiers for {split_name} split, skipping...")
            continue
        
        # Extract embeddings and labels
        embeddings, labels = extract_pocket_embeddings_with_labels(
            h5_file, identifier_list, model, modality, device
        )
        
        # Save to .pt file
        dirpath = os.path.join(str(output_dir), str(model_name), "ASD_pockets100", str(split_name))
        os.makedirs(dirpath, exist_ok=True)
        output_path = os.path.join(dirpath, f"ASD_pockets100_{split_name}_embeddings_labels.pt")
        save_embeddings_to_pt(embeddings, labels, output_path)
    
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
    
    # # SET SEED FIRST - BEFORE ANYTHING ELSE
    # import random
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Ensure single-threaded execution
    # torch.set_num_threads(1)
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load config
    model_name = "oneprot_md_combined_gpcr_32900"
    config_path = 'config.yaml'
    checkpoint_path = 'epoch_012_01100-v1.ckpt'

    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)

    # Prepare components
    components = {
        'sequence': hydra.utils.instantiate(cfg.model.components.sequence),
        #'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
        #'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'pocket': hydra.utils.instantiate(cfg.model.components.pocket),
        'text': hydra.utils.instantiate(cfg.model.components.text),
        #'md': hydra.utils.instantiate(cfg.model.components.md)
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

    # Load checkpoint - FIXED VERSION
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict from Lightning checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✓ Loaded from Lightning checkpoint")
    else:
        state_dict = checkpoint
        print("✓ Loaded from raw state dict")
    
    # Load with checking
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠ WARNING: Missing keys in checkpoint:")
        for key in missing_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"⚠ WARNING: Unexpected keys in checkpoint:")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")
    
    # Set to eval mode
    model.eval()
    
    # Explicitly disable dropout/batch norm
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
    
    # Define paths
    h5_file = 'ASD_binding_pockets.h5'
    train_csv = 'train_df_pdb.csv' #original files from the ASD paper
    test_csv = 'test_df_pdb.csv'
    
    # Process H5 file and create embeddings
    process_h5_to_embeddings(
        h5_file=h5_file,
        train_csv=train_csv,
        test_csv=test_csv,
        model=model,
        model_name=model_name,
        modality='pocket',
        output_dir='embeddings',
        device=device
    )
    
    print("\n✓ Done!")