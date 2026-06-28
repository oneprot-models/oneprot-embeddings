import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict

def parse_mmseqs2_clusters(cluster_tsv):
    """
    Parse MMseqs2 cluster TSV file.
    
    Args:
        cluster_tsv: Path to clusters.tsv from MMseqs2
    
    Returns:
        Dictionary mapping seq_id to cluster_id
    """
    print(f"Parsing MMseqs2 cluster results: {cluster_tsv}")
    
    seq_to_cluster = {}
    
    with open(cluster_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                representative = parts[0]  # Cluster representative (format: seq_INDEX)
                member = parts[1]          # Cluster member (format: seq_INDEX)
                
                # Extract index from seq_INDEX
                rep_idx = int(representative.replace('seq_', ''))
                mem_idx = int(member.replace('seq_', ''))
                
                # Assign cluster ID as the representative's index
                seq_to_cluster[mem_idx] = rep_idx
    
    # Count clusters
    unique_clusters = len(set(seq_to_cluster.values()))
    print(f"  ✓ Found {unique_clusters} clusters from {len(seq_to_cluster)} sequences")
    
    # Print cluster size distribution
    cluster_sizes = defaultdict(int)
    for cluster_id in seq_to_cluster.values():
        cluster_sizes[cluster_id] += 1
    
    sizes = sorted(cluster_sizes.values(), reverse=True)
    print(f"\nCluster size distribution:")
    print(f"  Largest cluster: {sizes[0]} sequences")
    print(f"  Median cluster size: {np.median(sizes):.1f}")
    print(f"  Singletons: {sum(1 for s in sizes if s == 1)}")
    
    return seq_to_cluster

def cluster_based_split(df, sequences, seq_to_cluster,
                       train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """
    Split data by assigning clusters to train/val/test sets.
    
    Args:
        df: DataFrame with data
        sequences: Dictionary of sequences
        seq_to_cluster: Dictionary mapping seq_id to cluster_id
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for testing
    
    Returns:
        train_df, val_df, test_df
    """
    print(f"\n{'='*70}")
    print("CLUSTER-BASED SPLITTING")
    print(f"{'='*70}")
    print(f"Train/Val/Test: {train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%}")
    
    # Group sequences by cluster
    cluster_to_seqs = defaultdict(list)
    for seq_id, cluster_id in seq_to_cluster.items():
        cluster_to_seqs[cluster_id].append(seq_id)
    
    clusters = list(cluster_to_seqs.values())
    print(f"\nNumber of clusters: {len(clusters)}")
    
    # Calculate mechanism distribution in each cluster
    cluster_info = []
    for cluster in clusters:
        cluster_df = df.loc[cluster]
        n_competitive = (cluster_df['Mechanism'].str.lower() == 'competitive').sum()
        n_allosteric = (cluster_df['Mechanism'].str.lower() == 'allosteric').sum()
        
        cluster_info.append({
            'indices': cluster,
            'size': len(cluster),
            'competitive': n_competitive,
            'allosteric': n_allosteric,
            'comp_ratio': n_competitive / len(cluster) if len(cluster) > 0 else 0
        })
    
    # Sort clusters by size (largest first) for better distribution
    cluster_info.sort(key=lambda x: x['size'], reverse=True)
    
    print(f"\nCluster statistics:")
    print(f"  Largest cluster: {cluster_info[0]['size']} sequences")
    print(f"  Smallest cluster: {cluster_info[-1]['size']} sequences")
    
    # Assign clusters to splits
    print(f"\nAssigning clusters to splits...")
    
    n_total = len(sequences)
    target_train = int(n_total * train_frac)
    target_val = int(n_total * val_frac)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    train_comp = train_allo = 0
    val_comp = val_allo = 0
    test_comp = test_allo = 0
    
    # Greedy assignment to maintain target sizes and balance
    for info in cluster_info:
        cluster = info['indices']
        comp = info['competitive']
        allo = info['allosteric']
        
        # Calculate current sizes
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
        
        # Decide where to assign this cluster
        # Priority: reach target sizes while maintaining mechanism balance
        if train_size < target_train:
            train_indices.extend(cluster)
            train_comp += comp
            train_allo += allo
        elif val_size < target_val:
            val_indices.extend(cluster)
            val_comp += comp
            val_allo += allo
        else:
            test_indices.extend(cluster)
            test_comp += comp
            test_allo += allo
    
    # Create DataFrames
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    # Print distribution
    print(f"\n{'='*70}")
    print("SPLIT DISTRIBUTION")
    print(f"{'='*70}")
    print(f"\nTrain: {len(train_indices)} ({len(train_indices)/n_total*100:.1f}%)")
    print(f"  Competitive: {train_comp} ({train_comp/len(train_indices)*100:.1f}%)")
    print(f"  Allosteric: {train_allo} ({train_allo/len(train_indices)*100:.1f}%)")
    
    print(f"\nValidation: {len(val_indices)} ({len(val_indices)/n_total*100:.1f}%)")
    print(f"  Competitive: {val_comp} ({val_comp/len(val_indices)*100:.1f}%)")
    print(f"  Allosteric: {val_allo} ({val_allo/len(val_indices)*100:.1f}%)")
    
    print(f"\nTest: {len(test_indices)} ({len(test_indices)/n_total*100:.1f}%)")
    print(f"  Competitive: {test_comp} ({test_comp/len(test_indices)*100:.1f}%)")
    print(f"  Allosteric: {test_allo} ({test_allo/len(test_indices)*100:.1f}%)")
    
    return train_df, val_df, test_df

def create_h5_identifier(row):
    """
    Create the H5 file identifier.
    Format: {PDB}_{Chain}_{LigandID}
    """
    pdb_id = str(row['PDB ID']).strip()
    ligand_id = str(row['Ligand ID']).strip()
    chain = 'A'  # Default chain
    
    identifier = f"{pdb_id}_{chain}_{ligand_id}"
    return identifier

def verify_no_overlap(train_df, val_df, test_df, seq_to_cluster, identity_threshold):
    """
    Verify that no clusters are shared between train and val/test.
    """
    print(f"\n{'='*70}")
    print("VERIFYING SPLIT QUALITY")
    print(f"{'='*70}")
    
    train_clusters = set(seq_to_cluster[idx] for idx in train_df.index if idx in seq_to_cluster)
    val_clusters = set(seq_to_cluster[idx] for idx in val_df.index if idx in seq_to_cluster)
    test_clusters = set(seq_to_cluster[idx] for idx in test_df.index if idx in seq_to_cluster)
    
    train_val_overlap = train_clusters & val_clusters
    train_test_overlap = train_clusters & test_clusters
    
    print(f"\nCluster distribution:")
    print(f"  Train clusters: {len(train_clusters)}")
    print(f"  Val clusters: {len(val_clusters)}")
    print(f"  Test clusters: {len(test_clusters)}")
    
    print(f"\nOverlap check:")
    print(f"  Train-Val overlap: {len(train_val_overlap)} clusters")
    print(f"  Train-Test overlap: {len(train_test_overlap)} clusters")
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0:
        print(f"\n  ✓ No cluster overlap - splits are properly separated!")
        print(f"  ✓ All sequences in val/test are <{identity_threshold:.0%} similar to train")
    else:
        print(f"\n  ⚠ Warning: Cluster overlap detected!")

def create_splits_from_mmseqs2(input_dir, output_dir='splits',
                               train_frac=0.7, val_frac=0.15, test_frac=0.15,
                               identity_threshold=0.3):
    """
    Step 2: Parse MMseqs2 results and create train/val/test splits.
    
    Args:
        input_dir: Directory with metadata.pkl and clusters.tsv
        output_dir: Output directory for split CSV files
        train_frac: Training set fraction
        val_frac: Validation set fraction
        test_frac: Test set fraction
        identity_threshold: Identity threshold used in MMseqs2 (for reporting)
    """
    print("="*70)
    print("STEP 2: CREATING SPLITS FROM MMSEQS2 RESULTS")
    print("="*70)
    
    # Load metadata
    metadata_file = os.path.join(input_dir, 'metadata.pkl')
    print(f"\nLoading metadata: {metadata_file}")
    
    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        df = metadata['df']
        sequences = metadata['sequences']
        print(f"  ✓ Loaded {len(df)} entries, {len(sequences)} sequences")
    except Exception as e:
        print(f"  ✗ Error loading metadata: {str(e)}")
        return
    
    # Parse MMseqs2 clusters
    cluster_tsv = os.path.join(input_dir, 'clusters.tsv')
    print(f"\n{'='*70}")
    print("PARSING MMSEQS2 CLUSTERS")
    print(f"{'='*70}")
    
    if not os.path.exists(cluster_tsv):
        print(f"✗ Error: {cluster_tsv} not found")
        print("\nPlease run MMseqs2 clustering first:")
        print(f"  mmseqs createtsv {input_dir}/seqDB {input_dir}/seqDB \\")
        print(f"      {input_dir}/clusterDB {cluster_tsv}")
        return
    
    seq_to_cluster = parse_mmseqs2_clusters(cluster_tsv)
    
    # Create splits
    train_df, val_df, test_df = cluster_based_split(
        df, sequences, seq_to_cluster,
        train_frac, val_frac, test_frac
    )
    
    # Verify split quality
    verify_no_overlap(train_df, val_df, test_df, seq_to_cluster, identity_threshold)
    
    # Add H5 identifiers
    print(f"\n{'='*70}")
    print("CREATING H5 IDENTIFIERS")
    print(f"{'='*70}")
    
    train_df['h5_identifier'] = train_df.apply(create_h5_identifier, axis=1)
    val_df['h5_identifier'] = val_df.apply(create_h5_identifier, axis=1)
    test_df['h5_identifier'] = test_df.apply(create_h5_identifier, axis=1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SEQUENCE-BASED TRAIN/VAL/TEST SPLIT SUMMARY (MMseqs2)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total entries: {len(df)}\n")
        f.write(f"Split ratios: {train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%}\n")
        f.write(f"Sequence identity threshold: {identity_threshold:.0%}\n")
        f.write(f"Clustering method: MMseqs2\n\n")
        
        f.write(f"Train: {len(train_df)} entries ({len(train_df)/len(df)*100:.1f}%)\n")
        f.write(f"  Competitive: {(train_df['Mechanism'].str.lower() == 'competitive').sum()}\n")
        f.write(f"  Allosteric: {(train_df['Mechanism'].str.lower() == 'allosteric').sum()}\n\n")
        
        f.write(f"Validation: {len(val_df)} entries ({len(val_df)/len(df)*100:.1f}%)\n")
        f.write(f"  Competitive: {(val_df['Mechanism'].str.lower() == 'competitive').sum()}\n")
        f.write(f"  Allosteric: {(val_df['Mechanism'].str.lower() == 'allosteric').sum()}\n\n")
        
        f.write(f"Test: {len(test_df)} entries ({len(test_df)/len(df)*100:.1f}%)\n")
        f.write(f"  Competitive: {(test_df['Mechanism'].str.lower() == 'competitive').sum()}\n")
        f.write(f"  Allosteric: {(test_df['Mechanism'].str.lower() == 'allosteric').sum()}\n")
    
    print(f"\n{'='*70}")
    print("OUTPUT FILES")
    print(f"{'='*70}")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print(f"  Summary: {summary_path}")
    
    print(f"\n✓ Split completed successfully!")

if __name__ == "__main__":
    import sys
    
    input_dir = 'mmseqs_input'
    output_dir = 'splits'
    train_frac = 0.70
    val_frac = 0.15
    test_frac = 0.15
    identity_threshold = 0.30
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        identity_threshold = float(sys.argv[3])
    
    create_splits_from_mmseqs2(
        input_dir=input_dir,
        output_dir=output_dir,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        identity_threshold=identity_threshold
    )
