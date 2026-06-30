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


def load_sequences_from_csv(train_csv, test_csv):

    seq_dict = {}
    
    def extract_chains(chain_value):


        chain_str = str(chain_value).strip()

        # Remove outer brackets and quotes
        chain_str = chain_str.replace('[', '').replace(']', '').replace('"', '').replace("'", '').strip()
        
        # Split by comma if multiple chains
        if ',' in chain_str:
            chains = [c.strip() for c in chain_str.split(',')]
        else:
            chains = [chain_str]
        
        return chains
    
    def add_sequence_keys(pdb_id, chains, sequence, seq_dict):
        """Add sequence for all chain variations."""
        for chain in chains:
            # Create multiple key formats to maximize matching
            keys_to_add = [
                f"{pdb_id}_{chain}",           # e.g., '2z78_A'
                f"{pdb_id}_{chain.lower()}",   # e.g., '2z78_a'
                f"{pdb_id.upper()}_{chain}",   # e.g., '2Z78_A'
                f"{pdb_id.upper()}_{chain.lower()}"  # e.g., '2Z78_a'
            ]
            
            for key in keys_to_add:
                seq_dict[key] = sequence
    
    # Load train CSV
    train_df = pd.read_csv(train_csv)
    print(f"Train CSV columns: {train_df.columns.tolist()}")
    print(f"Train CSV first row sample: pdb_id={train_df.iloc[0]['pdb_id']}, chains={train_df.iloc[0]['chains']}")
    
    for _, row in train_df.iterrows():
        pdb_id = str(row['pdb_id']).strip().lower()
        chains = extract_chains(row['chains'])
        sequence = str(row['Sequences']).strip()
        
        add_sequence_keys(pdb_id, chains, sequence, seq_dict)
    
    # Load test CSV
    test_df = pd.read_csv(test_csv)
    for _, row in test_df.iterrows():
        pdb_id = str(row['pdb_id']).strip().lower()
        chains = extract_chains(row['chains'])
        sequence = str(row['Sequences']).strip()
        
        add_sequence_keys(pdb_id, chains, sequence, seq_dict)
    
    # Get unique sequences (since we added multiple keys per sequence)
    unique_sequences = len(set(seq_dict.values()))
    
    print(f"\nLoaded {unique_sequences} unique sequences from CSV files")
    print(f"Created {len(seq_dict)} total key mappings (multiple formats per sequence)")
    print(f"Example keys: {list(seq_dict.keys())[:10]}")
    
    return seq_dict


def extract_pocket_sequence_embeddings_with_labels(h5_file, identifiers, model, tokenizer, 
                                                    seq_dict, device='cuda'):
    """
    Extract both pocket and sequence embeddings and concatenate them, along with labels.
    
    Args:
        h5_file: Path to H5 file containing pocket data
        identifiers: List of identifiers to process
        model: Trained model for embedding extraction
        tokenizer: Tokenizer for sequence processing
        seq_dict: Dictionary mapping identifier to sequence
        device: Device for computation
    
    Returns:
        embeddings_list: List of concatenated embedding arrays [pocket_dim + sequence_dim]
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
            
            # ========================================
            # Extract POCKET embedding
            # ========================================
            input_pocket = [protein_to_graph(identifier, h5_file, 'non_pdb', chain, pockets=True)]
            input_pocket = Batch.from_data_list(input_pocket).to(device)
            
            with torch.no_grad():
                pocket_embedding = model.network['pocket'](input_pocket)
            
            # Convert to numpy and flatten
            pocket_embedding_np = pocket_embedding.cpu().numpy()
            if len(pocket_embedding_np.shape) > 1:
                pocket_embedding_np = pocket_embedding_np.flatten()
            
            # ========================================
            # Extract SEQUENCE embedding
            # ========================================
            # Look up sequence from dictionary
            sequence = seq_dict.get(identifier)
            
            if sequence is None:
                # Try lowercase version
                sequence = seq_dict.get(identifier.lower())
            
            if sequence is None:
                print(f"  Warning: Sequence not found for {identifier}")
                print(f"    Available keys sample: {list(seq_dict.keys())[:10]}")
                failed_identifiers.append(identifier)
                continue
            
            # Tokenize sequence
            sequence_tokens = tokenizer(
                sequence,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024  # Adjust as needed
            )
            
            # Move tokens to device
            sequence_tokens = {k: v.to(device) for k, v in sequence_tokens.items()}
            
            with torch.no_grad():
                sequence_embedding = model.network['sequence'](sequence_tokens['input_ids'])
            
            # Convert to numpy and flatten
            sequence_embedding_np = sequence_embedding.cpu().numpy()
            if len(sequence_embedding_np.shape) > 1:
                sequence_embedding_np = sequence_embedding_np.flatten()
            
            # ========================================
            # Concatenate embeddings
            # ========================================
            combined_embedding = np.concatenate([pocket_embedding_np, sequence_embedding_np])
            
            # Validate embedding
            if np.isnan(combined_embedding).any() or np.isinf(combined_embedding).any():
                print(f"  Warning: NaN or Inf in embedding for {identifier}")
                failed_identifiers.append(identifier)
                continue
            
            # ========================================
            # Extract labels
            # ========================================
            pocket_labels = extract_labels_from_h5(h5_file, identifier, chain)
            
            if pocket_labels is None:
                print(f"  Warning: Could not extract labels for {identifier}")
                failed_identifiers.append(identifier)
                continue
            
            # Ensure labels are the correct type and shape
            pocket_labels = np.array(pocket_labels, dtype=np.int32)
            
            embeddings_list.append(combined_embedding)
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
        embeddings_list: List of numpy arrays, each of shape [pocket_dim + sequence_dim]
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

    print(f"Embeddings shape: {embeddings_tensor.shape}")
    
    # Pad labels to consistent length (100)
    for i in range(len(labels_list)):
        if labels_list[i].shape[0] != 100:
            labels_new = np.zeros((100,), dtype=np.int32) - 1
            labels_new[:labels_list[i].shape[0]] = labels_list[i]
            labels_list[i] = labels_new
    
    labels_tensor = torch.tensor(np.stack(labels_list), dtype=torch.long)
    
    # Create dictionary
    data_dict = {
        'embeddings': embeddings_tensor,
        'labels_fitness': labels_tensor
    }
    
    print(f"Final embeddings shape: {embeddings_tensor.shape}")
    print(f"Final labels shape: {labels_tensor.shape}")
    print(f"Embeddings dtype: {embeddings_tensor.dtype}")
    print(f"Labels dtype: {labels_tensor.dtype}")
    
    # Save to file
    torch.save(data_dict, output_path)
    print(f"✓ Saved to {output_path}")


def process_h5_to_embeddings(h5_file, train_csv, test_csv, model, tokenizer, model_name, 
                              output_dir='embeddings', device='cuda'):
    """
    Process H5 file and create embedding .pt files for train/valid/test splits
    with concatenated pocket and sequence embeddings.
    
    Args:
        h5_file: Path to H5 file containing pocket data
        train_csv: Path to training CSV with 'pdb_id', 'chains', and 'Sequences' columns
        test_csv: Path to test CSV with 'pdb_id', 'chains', and 'Sequences' columns (will be split 50/50)
        model: Trained model for embedding extraction
        tokenizer: Tokenizer for sequence processing
        model_name: Name for output directory
        output_dir: Directory to save output files
        device: Device for computation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sequences from CSV files
    print(f"\n{'='*70}")
    print(f"LOADING SEQUENCES FROM CSV FILES")
    print(f"{'='*70}")
    seq_dict = load_sequences_from_csv(train_csv, test_csv)
    
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
        embeddings, labels = extract_pocket_sequence_embeddings_with_labels(
            h5_file, identifier_list, model, tokenizer, seq_dict, device
        )
        
        # Save to .pt file
        dirpath = os.path.join(str(output_dir), str(model_name), "ASD_pocket_sequence100", str(split_name))
        os.makedirs(dirpath, exist_ok=True)
        output_path = os.path.join(dirpath, f"ASD_pocket_sequence100_{split_name}_embeddings_labels.pt")
        save_embeddings_to_pt(embeddings, labels, output_path)
    
    print(f"\n{'='*70}")
    print("ALL SPLITS PROCESSED!")
    print(f"{'='*70}")
    print(f"\nOutput files saved in: {output_dir}/{model_name}/ASD_pocket_sequence100/")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    
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
        'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
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

    # Load checkpoint
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
        for key in missing_keys[:10]:
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
    
    # Initialize tokenizer (adjust based on your model's tokenizer)
    # You may need to change this to match your actual tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        print("✓ Loaded ESM2 tokenizer")
    except:
        print("⚠ Failed to load ESM2 tokenizer, trying alternative...")
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        print("✓ Loaded ProtBERT tokenizer")
    
    # Define paths
    h5_file = '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/ASD_binding_pockets.h5'
    train_csv = '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/train_df_pdb.csv'
    test_csv = '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/test_df_pdb.csv'
    
    # Process H5 file and create embeddings
    process_h5_to_embeddings(
        h5_file=h5_file,
        train_csv=train_csv,
        test_csv=test_csv,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        output_dir='embeddings',
        device=device
    )
    
    print("\n✓ Done!")