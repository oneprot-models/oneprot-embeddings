import os
from collections import defaultdict
import random
import numpy as np


def parse_mmseqs2_clusters(cluster_tsv):
    """
    Parse MMseqs2 cluster output file.
    
    Args:
        cluster_tsv: Path to *_cluster.tsv file from MMseqs2
    
    Returns:
        dict: {representative: [member1, member2, ...]}
    """
    clusters = defaultdict(list)
    
    with open(cluster_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                representative = parts[0]
                member = parts[1]
                clusters[representative].append(member)
    
    return dict(clusters)


def split_clusters(clusters, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split clusters into train/valid/test sets.
    
    Args:
        clusters: dict {representative: [members]}
        train_ratio: Fraction for training
        valid_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
    
    Returns:
        dict: {'train': [ids], 'valid': [ids], 'test': [ids]}
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all cluster representatives and shuffle
    cluster_reps = list(clusters.keys())
    random.shuffle(cluster_reps)
    
    # Calculate split points
    n_clusters = len(cluster_reps)
    n_train = int(n_clusters * train_ratio)
    n_valid = int(n_clusters * valid_ratio)
    
    # Split cluster representatives
    train_reps = cluster_reps[:n_train]
    valid_reps = cluster_reps[n_train:n_train + n_valid]
    test_reps = cluster_reps[n_train + n_valid:]
    
    # Collect all members from each split
    train_ids = []
    valid_ids = []
    test_ids = []
    
    for rep in train_reps:
        train_ids.extend(clusters[rep])
    
    for rep in valid_reps:
        valid_ids.extend(clusters[rep])
    
    for rep in test_reps:
        test_ids.extend(clusters[rep])
    
    return {
        'train': train_ids,
        'valid': valid_ids,
        'test': test_ids
    }


def verify_splits(splits, clusters):
    """
    Verify that splits don't overlap.
    """
    train_set = set(splits['train'])
    valid_set = set(splits['valid'])
    test_set = set(splits['test'])
    
    print(f"  Train: {len(train_set)} sequences")
    print(f"  Valid: {len(valid_set)} sequences")
    print(f"  Test: {len(test_set)} sequences")
    print(f"  Total: {len(train_set) + len(valid_set) + len(test_set)}")
    
    # Check for overlaps
    train_valid_overlap = train_set & valid_set
    train_test_overlap = train_set & test_set
    valid_test_overlap = valid_set & test_set
    
    if train_valid_overlap or train_test_overlap or valid_test_overlap:
        print(f"  ⚠ WARNING: Found overlaps!")
        if train_valid_overlap:
            print(f"    Train-Valid: {len(train_valid_overlap)}")
        if train_test_overlap:
            print(f"    Train-Test: {len(train_test_overlap)}")
        if valid_test_overlap:
            print(f"    Valid-Test: {len(valid_test_overlap)}")
    else:
        print(f"  ✓ No overlaps!")


def write_splits_to_files(splits, output_dir, prefix=""):
    """
    Write train/valid/test splits to separate text files.
    
    Args:
        splits: dict {'train': [ids], 'valid': [ids], 'test': [ids]}
        output_dir: Directory to save split files
        prefix: Optional prefix for filenames (e.g., "allosteric_")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, identifiers in splits.items():
        if prefix:
            output_file = os.path.join(output_dir, f"{prefix}{split_name}.txt")
        else:
            output_file = os.path.join(output_dir, f"{split_name}.txt")
        
        with open(output_file, 'w') as f:
            for identifier in sorted(identifiers):
                f.write(f"{identifier}\n")
        
        print(f"  {os.path.basename(output_file)}: {len(identifiers)} identifiers")


def process_single_cluster_file(cluster_tsv, output_dir, prefix="", 
                                train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Process a single MMseqs2 cluster file and create splits.
    
    Args:
        cluster_tsv: Path to MMseqs2 cluster.tsv file
        output_dir: Directory to save split files
        prefix: Prefix for output files (e.g., "allosteric_")
        train_ratio: Fraction for training
        valid_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
    """
    print(f"\n{'='*70}")
    print(f"Processing: {cluster_tsv}")
    print(f"{'='*70}")
    
    # Parse clusters
    print("Reading clusters...")
    clusters = parse_mmseqs2_clusters(cluster_tsv)
    
    # Statistics
    total_seqs = sum(len(members) for members in clusters.values())
    cluster_sizes = [len(members) for members in clusters.values()]
    
    print(f"  Total clusters: {len(clusters)}")
    print(f"  Total sequences: {total_seqs}")
    print(f"  Average cluster size: {np.mean(cluster_sizes):.2f}")
    print(f"  Median cluster size: {np.median(cluster_sizes):.0f}")
    print(f"  Max cluster size: {np.max(cluster_sizes)}")
    print(f"  Singletons: {sum(1 for s in cluster_sizes if s == 1)}")
    
    # Split clusters
    print("\nSplitting clusters...")
    splits = split_clusters(clusters, train_ratio, valid_ratio, test_ratio, seed)
    
    # Verify
    print("\nVerification:")
    verify_splits(splits, clusters)
    
    # Write to files
    print("\nWriting split files...")
    write_splits_to_files(splits, output_dir, prefix)
    
    print(f"\n✓ Completed: {prefix}train/valid/test.txt")


def process_all_cluster_files(cluster_files, output_base_dir='splits',
                              train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Process multiple MMseqs2 cluster files and create separate splits for each.
    
    Args:
        cluster_files: dict {name: cluster_tsv_path}
            Example: {
                'allosteric': 'fasta_sequences/allosteric_clusters_cluster.tsv',
                'competitive': 'fasta_sequences/competitive_clusters_cluster.tsv',
                'noncompetitive': 'fasta_sequences/noncompetitive_clusters_cluster.tsv'
            }
        output_base_dir: Base directory for all splits
        train_ratio: Fraction for training
        valid_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
    """
    print("="*70)
    print("CREATING SEPARATE SPLITS FOR EACH H5 FILE")
    print("="*70)
    
    for name, cluster_file in cluster_files.items():
        # Create separate output directory for each source
        output_dir = os.path.join(output_base_dir, name)
        
        # Process this cluster file
        process_single_cluster_file(
            cluster_tsv=cluster_file,
            output_dir=output_dir,
            prefix="",  # No prefix needed since they're in separate directories
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
    
    print("\n" + "="*70)
    print("ALL SPLITS CREATED!")
    print("="*70)
    
    # Print output structure
    print("\nOutput structure:")
    for name in cluster_files.keys():
        print(f"\n{output_base_dir}/{name}/")
        print(f"  ├── train.txt")
        print(f"  ├── valid.txt")
        print(f"  └── test.txt")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Define cluster files from MMseqs2 output
    cluster_files = {
        'allosteric': 'fasta_sequences/allosteric_clusters_cluster.tsv',
        'competitive': 'fasta_sequences/competitive_clusters_cluster.tsv',
        'noncompetitive': 'fasta_sequences/noncompetitive_clusters_cluster.tsv'
    }
    
    # Create separate splits for each
    process_all_cluster_files(
        cluster_files=cluster_files,
        output_base_dir='splits',
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print("\n✓ Done! You can now merge these splits as needed.")
# ```

# **Output structure:**
# ```
# splits/
# ├── allosteric/
# │   ├── train.txt
# │   ├── valid.txt
# │   └── test.txt
# ├── competitive/
# │   ├── train.txt
# │   ├── valid.txt
# │   └── test.txt
# └── noncompetitive/
#     ├── train.txt
#     ├── valid.txt
#     └── test.txt