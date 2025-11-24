import os
import shutil
import pandas as pd
import re

# Path to your CSV file
csv_file = "/p/data1/profound_data/GPCRmd/extracted_with_related_values_2.csv"

# Read the CSV file
df = pd.read_csv(csv_file)

# Function to create directory if it doesn't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Process each row in the CSV file
for index, row in df.iterrows():
    # Extract the MDR ID from the first column
    mdr_id = row['Column1']
    
    # Get the related MDR IDs from the last column
    if pd.isna(row['Related_Column1_Values']):
        print(f"Skipping {mdr_id} - no related values")
        continue
        
    related_ids_str = row['Related_Column1_Values']
    related_ids = [id.strip() for id in related_ids_str.split(',')]
    
    # Create the destination directory
    dest_dir = f"/p/data1/profound_data/GPCRmd_sorted/{mdr_id}"
    ensure_dir(dest_dir)
    
    # Copy the main files
    try:
        # Copy the structure file
        mdr_id1=mdr_id.replace("MDR", "MDR_")
        src_pdb = f"/p/data1/profound_data/GPCRmd/{mdr_id1}/processed/structure.pdb"
        dst_pdb = f"{dest_dir}/{mdr_id}.pdb"
        if os.path.exists(src_pdb):
            shutil.copy2(src_pdb, dst_pdb)
            print(f"Copied {src_pdb} to {dst_pdb}")
        else:
            print(f"Warning: Source PDB file not found: {src_pdb}")
        
        # Copy the main trajectory file
        src_xtc = f"/p/data1/profound_data/GPCRmd/processed/{mdr_id1}/trajectory.xtc"
        dst_xtc = f"{dest_dir}/{mdr_id}_1.xtc"
        if os.path.exists(src_xtc):
            shutil.copy2(src_xtc, dst_xtc)
            print(f"Copied {src_xtc} to {dst_xtc}")
        else:
            print(f"Warning: Source XTC file not found: {src_xtc}")
            
        # Process related IDs, excluding the main ID
        other_ids = [id for id in related_ids if id.strip() != mdr_id]
        
        # Copy up to two additional trajectory files
        for idx, related_id in enumerate(other_ids[:2]):  # Limit to 2 additional trajectories
            related_id = related_id.strip()
            related_id1 = related_id.replace("MDR", "MDR_")
            src_xtc = f"/p/data1/profound_data/GPCRmd/processed/{related_id}/trajectory.xtc"
            dst_xtc = f"{dest_dir}/{mdr_id}_{idx+2}.xtc"  # Use _2 and _3 for the additional trajectories
            
            if os.path.exists(src_xtc):
                shutil.copy2(src_xtc, dst_xtc)
                print(f"Copied {src_xtc} to {dst_xtc}")
            else:
                print(f"Warning: Related XTC file not found: {src_xtc}")
                
    except Exception as e:
        print(f"Error processing {mdr_id}: {str(e)}")

print("Processing complete!")