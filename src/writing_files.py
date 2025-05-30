#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd
import glob
import re

# Directory containing trajectory files
data_dir = '/p/scratch/hai_profound/data/mdCATH_data/'

# Output CSV file path
output_csv = '/p/project1/hai_oneprot/bazarova1/oneprot-panda/long_trajectories.csv'

# Regular expression to extract PDB ID from filename
# Matches patterns like 4hs1A00_R1_i40.npy -> 4hs1A00
pdb_id_pattern = re.compile(r'([A-Za-z0-9]+)_R\d+_i\d+\.npy')

# Lists to store results
pdb_ids = []
filenames = []
num_frames = []

# Find all .npy files
npy_files = glob.glob(os.path.join(data_dir, '*.npy'))
print(f"Found {len(npy_files)} .npy files. Checking frame counts...")

count = 0
# Check each file
for npy_file in npy_files:
    try:
        # Try to open the file as a memory-mapped array to avoid loading it completely
        arr = np.lib.format.open_memmap(npy_file, mode='r')
        
        # Check if first dimension is ≥ 500
        if arr.shape[0] >= 500:
            filename = os.path.basename(npy_file)
            match = pdb_id_pattern.match(filename)
            if match:
                pdb_id = match.group(1)
                pdb_ids.append(pdb_id)
                filenames.append(filename)
                num_frames.append(arr.shape[0])
                print(f"Found long trajectory: {filename} with {arr.shape[0]} frames")
        
        # Progress counter
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} files...")
    
    except Exception as e:
        print(f"Error processing {npy_file}: {e}")

# Create a DataFrame and save to CSV
if pdb_ids:
    df = pd.DataFrame({
        'pdb_id': pdb_ids,
        'filename': filenames,
        'num_frames': num_frames
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} entries to {output_csv}")
else:
    print("No trajectories with 500+ frames found.")