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
    """
    results = []
    failed_identifiers = []
    
    print(f"Processing {len(identifiers)} pockets from {os.path.basename(h5_file)}...")
    
    # Move model to device
    model = model.to(device)
    
    for identifier in tqdm(identifiers, desc=f"Label {label}"):
        try:
            # Convert H5 pocket to graph
            parts = identifier.split('_')
            pdb_id = parts[0] if len(parts) > 0 else identifier
            chain = parts[1] if len(parts) > 1 else 'A'
            
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
            
            results.append({
                'identifier': identifier,
                'embedding': embedding_np.tolist(),  # Convert to list for safer serialization
                'label': label
            })
            
        except Exception as e:
            print(f"  Error processing {identifier}: {e}")
            import traceback
            traceback.print_exc()
            failed_identifiers.append(identifier)
            continue
    
    print(f"  Successfully processed: {len(results)}/{len(identifiers)}")
    if failed_identifiers:
        print(f"  Failed: {failed_identifiers[:5]}")
    
    return results


def embeddings_to_dataframe(embeddings_list):
    """
    Convert embeddings list to DataFrame.
    This is a separate function to ensure atomic conversion.
    """
    if not embeddings_list:
        return pd.DataFrame()
    
    data = []
    expected_dim = len(embeddings_list[0]['embedding'])
    
    for item in embeddings_list:
        embedding = item['embedding']
        
        # Validate dimension
        if len(embedding) != expected_dim:
            print(f"  Warning: Skipping {item['identifier']} - dimension mismatch")
            continue
        
        row = {
            'identifier': item['identifier'],
            'label': item['label']
        }
        
        # Add embedding values
        for i, val in enumerate(embedding):
            row[f'emb_{i}'] = float(val)
        
        data.append(row)
    
    return pd.DataFrame(data)


def save_embeddings_to_csv_safe(embeddings_list, output_csv):
    """
    Safely save embeddings to CSV file.
    Uses a temporary file and atomic rename to prevent corruption.
    """
    import tempfile
    import shutil
    
    if not embeddings_list:
        print(f"Warning: No embeddings to save!")
        return
    
    print(f"\nConverting {len(embeddings_list)} embeddings to DataFrame...")
    df = embeddings_to_dataframe(embeddings_list)
    
    if df.empty:
        print("Warning: DataFrame is empty after conversion!")
        return
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {len(df.columns)} (identifier + label + {len(df.columns)-2} embedding dims)")
    
    # Write to temporary file first
    temp_dir = os.path.dirname(output_csv) or '.'
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir, suffix='.csv') as tmp_file:
        temp_path = tmp_file.name
        df.to_csv(temp_path, index=False)
    
    # Atomic rename
    shutil.move(temp_path, output_csv)
    
    print(f"✓ Saved to {output_csv}")
    
    # Verify file
    verify_df = pd.read_csv(output_csv)
    print(f"Verification: {len(verify_df)} rows, {len(verify_df.columns)} columns")
    
    # Check for any malformed rows
    if len(verify_df) != len(df):
        print(f"⚠ WARNING: Row count mismatch! Expected {len(df)}, got {len(verify_df)}")


def process_all_splits(h5_files, split_dirs, model, model_name, modality='pocket', 
                      output_dir='embeddings/',
                      device='cuda'):
    """
    Process all H5 files and their splits to create embedding CSVs.
    FIXED: Sequential processing with proper file handling.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define labels
    labels = {
        'allosteric': 1,
        'competitive': 2,
        'noncompetitive': 3
    }
    
    # Process each split (train, valid, test)
    for split_name in ['train', 'valid', 'test']:
        print(f"\n{'='*70}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'='*70}")
        
        all_embeddings = []
        
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
            embeddings = extract_pocket_embeddings(h5_file, identifiers, model, modality, label, device)
            
            print(f"Extracted {len(embeddings)} embeddings")
            
            # IMPORTANT: Extend the list (not append)
            # This ensures all embeddings are in a single flat list
            all_embeddings.extend(embeddings)
            
            print(f"Total embeddings so far: {len(all_embeddings)}")
        
        # Save ALL embeddings at once (after all sources processed)
        print(f"\n{'='*70}")
        print(f"SAVING {split_name.upper()} EMBEDDINGS")
        print(f"{'='*70}")
        

        base_dir = os.path.join(output_dir, model_name, "ASD_pockets")
        os.makedirs(base_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, model_name, "ASD_pockets", f"{split_name}.csv")
        save_embeddings_to_csv_safe(all_embeddings, output_csv)
        
        # Print statistics
        labels_only = [e['label'] for e in all_embeddings]
        from collections import Counter
        label_counts = Counter(labels_only)
        
        print(f"\nLabel distribution in {split_name}:")
        for label_val in sorted(label_counts.keys()):
            print(f"  Label {label_val}: {label_counts[label_val]} samples")
    
    print(f"\n{'='*70}")
    print("ALL SPLITS PROCESSED!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  {output_dir}/{model_name}/ASD_pockets/train.csv")
    print(f"  {output_dir}/{model_name}/ASD_pockets/valid.csv")
    print(f"  {output_dir}/{model_name}/ASD_pockets/test.csv")

    data_train=pd.read_csv(f"{output_dir}/{model_name}/ASD_pockets/train.csv")
    data_valid=pd.read_csv(f"{output_dir}/{model_name}/ASD_pockets/valid.csv")
    data_test=pd.read_csv(f"{output_dir}/{model_name}/ASD_pockets/test.csv")
    cols=data_train.columns[2:]

    embs_train=torch.tensor(data_train[cols].values,dtype=torch.float32)
    embs_valid=torch.tensor(data_valid[cols].values,dtype=torch.float32)
    embs_test=torch.tensor(data_test[cols].values,dtype=torch.float32)

    lab_train=torch.tensor(data_train['label'].values)
    lab_valid=torch.tensor(data_valid['label'].values)
    lab_test=torch.tensor(data_test['label'].values)

    train={}
    test={}
    valid={}

    train['embeddings']=embs_train
    train['labels_fitness']=lab_train-1

    valid['embeddings']=embs_valid
    valid['labels_fitness']=lab_valid-1

    test['embeddings']=embs_test
    test['labels_fitness']=lab_test-1


    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)

    torch.save(train,f"{output_dir}/{model_name}/ASD_pockets/train/ASD_pockets_train_embeddings_labels.pt")
    torch.save(test,f"{output_dir}/{model_name}/ASD_pockets/test/ASD_pockets_test_embeddings_labels.pt")
    torch.save(valid,f"{output_dir}/{model_name}/ASD_pockets/valid/ASD_pockets_valid_embeddings_labels.pt")



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
    # config_path = hf_hub_download(
    #     repo_id="HelmholtzAI-FZJ/oneprot",
    #     filename="config.yaml",
    # )
    model_name='oneprot_struct_graph_pocket_text_32900'
    config_path='config.yaml'
    checkpoint_path='epoch_010_05100-v3.ckpt'

    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)

    # Prepare components
    components = {
        'sequence': hydra.utils.instantiate(cfg.model.components.sequence),
        #'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
        'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'pocket': hydra.utils.instantiate(cfg.model.components.pocket),
        'text': hydra.utils.instantiate(cfg.model.components.text),
        #'md': hydra.utils.instantiate(cfg.model.components.md)
    }

    # Load checkpoint
    # checkpoint_path = hf_hub_download(
    #     repo_id="HelmholtzAI-FZJ/oneprot",
    #     filename="pytorch_model.bin",
    #     repo_type="model"
    # )
    

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)


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

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✓ Loaded from Lightning checkpoint")
    else:
        state_dict = checkpoint
        print("✓ Loaded from raw state dict")

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

    # Define H5 files
    h5_files = {
        'allosteric': 'binding_pockets_allosteric.h5',
        'competitive': 'binding_pockets_orthosteric_competitive.h5',
    }
    
    # Define split directories
    split_dirs = {
        'allosteric': 'PPI-site_splits/allosteric',
        'competitive': 'PPI-site_splits/competitive',
    }
    
    # Process all splits
    process_all_splits(
        h5_files=h5_files,
        split_dirs=split_dirs,
        model=model,
        model_name=model_name,
        modality='pocket',
        output_dir='embeddings',
        device=device
    )
    
    print("\n✓ Done!")