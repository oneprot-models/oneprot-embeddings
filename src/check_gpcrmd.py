import h5py
import pandas as pd
import os

# File paths
val_ids_file = "/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/val_seqstruc.csv"
h5_file_seq = "/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/seqstruc.h5"
gpcrmd_csv = "/p/data1/profound_data/GPCRmd/extracted_with_related_values_2.csv"

def load_val_sequences():
    """Load sequences from validation dataset"""
    # Read sequence IDs from validation file
    with open(val_ids_file, 'r') as f:
        seq_ids = [line.strip() for line in f]
    
    # Extract sequences from h5 file
    val_sequences = {}
    with h5py.File(h5_file_seq, 'r') as h5f:
        for seq_id in seq_ids:
            print(seq_id)
            try:
                # Extract sequence from h5 file
                sequence = h5f[seq_id]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')
                val_sequences[seq_id] = sequence
            except KeyError:
                print(f"Warning: Sequence ID {seq_id} not found in h5 file")
                continue
    
    return val_sequences

def load_gpcrmd_sequences():
    """Load sequences from GPCRmd CSV"""
    # Read CSV file
    df = pd.read_csv(gpcrmd_csv)
    
    # Create dictionary of sequences from Column8
    gpcrmd_sequences = {}
    for _, row in df.iterrows():
        if pd.notna(row['seqres']):
            gpcrmd_sequences[row['name']] = row['seqres']
    
    return gpcrmd_sequences

def find_intersections():
    """Find sequence intersections between datasets"""
    print("Loading validation sequences...")
    val_sequences = load_val_sequences()
    print(f"Loaded {len(val_sequences)} validation sequences")
    
    print("Loading GPCRmd sequences...")
    gpcrmd_sequences = load_gpcrmd_sequences()
    print(f"Loaded {len(gpcrmd_sequences)} GPCRmd sequences")
    
    # Create sets of sequences for efficient intersection checking
    val_seq_set = set(val_sequences.values())
    gpcrmd_seq_set = set(gpcrmd_sequences.values())
    
    # Find intersection
    intersection = val_seq_set.intersection(gpcrmd_seq_set)
    print(f"Found {len(intersection)} sequences that appear in both datasets")
    
    # Find which IDs match
    matching_pairs = []
    for val_id, val_seq in val_sequences.items():
        if val_seq in gpcrmd_seq_set:
            # Find all GPCRmd IDs with this sequence
            matching_gpcrmd_ids = [gpcrmd_id for gpcrmd_id, gpcrmd_seq in gpcrmd_sequences.items() 
                                 if gpcrmd_seq == val_seq]
            matching_pairs.append((val_id, matching_gpcrmd_ids))
    
    # Output results
    if matching_pairs:
        print("\nMatching sequences found:")
        for val_id, gpcrmd_ids in matching_pairs:
            print(f"Validation ID: {val_id} matches GPCRmd IDs: {', '.join(gpcrmd_ids)}")
        
        # Save results to file
        output_file = "sequence_intersection_results.csv"
        with open(output_file, 'w') as f:
            f.write("validation_id,gpcrmd_ids,sequence\n")
            for val_id, gpcrmd_ids in matching_pairs:
                sequence = val_sequences[val_id]
                f.write(f"{val_id},{','.join(gpcrmd_ids)},{sequence}\n")
        print(f"\nResults saved to {output_file}")
    else:
        print("\nNo matching sequences found between the datasets")

if __name__ == "__main__":
    find_intersections()