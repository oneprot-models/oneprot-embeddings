import pandas as pd
import os
import numpy as np

# Define paths
input_file = '/p/scratch/hai_oneprot/merdivan1/csv_files/TopEnzyme/Comb826_train_seq.csv'
output_dir = '/p/scratch/hai_oneprot/bazarova1/csv_files/TopEnzyme/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

# Add sequence length column
df['sequence_length'] = df['sequence'].apply(len)

# Identify long sequences (length > 3000)
long_seq_mask = df['sequence_length'] > 3000
long_seq_df = df[long_seq_mask]
regular_seq_df = df[~long_seq_mask]

print(f"Found {len(long_seq_df)} sequences longer than 3000 characters")
print(f"Found {len(regular_seq_df)} regular sequences")

# If there are very few long sequences, we can't create 1000 files with exactly one long sequence each
num_files = min(1000, len(long_seq_df))
print(f"Will create {num_files} files")

# Split the regular sequences evenly
regular_chunks = np.array_split(regular_seq_df, num_files)

# Create each output file with one long sequence and a portion of regular sequences
for i in range(num_files):
    # Get one long sequence
    long_seq = long_seq_df.iloc[i:i+1]
    
    # Get a chunk of regular sequences
    regular_chunk = regular_chunks[i]
    
    # Combine into one dataframe
    combined_df = pd.concat([long_seq, regular_chunk])
    
    # Shuffle the combined dataframe to randomize the position of the long sequence
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    # Save to file
    output_file = os.path.join(output_dir, f'Comb826_train_seq_{i}.csv')
    combined_df.to_csv(output_file, index=False)
    
    # Print progress
    if (i + 1) % 100 == 0 or i == 0 or i == num_files - 1:
        print(f"Created file {i+1}/{num_files}: {output_file}")
        print(f"  Total sequences: {len(combined_df)}")
        print(f"  Max sequence length: {combined_df['sequence_length'].max()}")

print("Done!")