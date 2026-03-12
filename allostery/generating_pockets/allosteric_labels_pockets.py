import h5py
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import requests
import os
from scipy.spatial import distance
import ast
from sklearn.cluster import DBSCAN


def download_pdb_file(pdb_id, output_dir='pdb_files'):
    """Download PDB file from RCSB."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(output_path):
        return output_path
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            return output_path
    except Exception as e:
        print(f"Error downloading {pdb_id}: {str(e)}")
    
    return None


def extract_structure_data_from_pdb(pdb_file, chain_id):
    """
    Extract structure data from PDB file.
    Returns: atom_amino_id, atom_names, atom_pos, amino_types, residue_mapping
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    atom_amino_ids = []
    atom_names_list = []
    atom_positions = []
    amino_types = []
    
    # Standard amino acid three-letter to one-letter code
    aa_3to1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    residue_counter = 1
    residue_to_id = {}
    residue_mapping = {}
    
    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            
            for residue in chain:
                # Only process standard amino acids
                if residue.id[0] == ' ' and residue.resname in aa_3to1:
                    res_key = (residue.id[1], residue.id[2])
                    
                    if res_key not in residue_to_id:
                        residue_to_id[res_key] = residue_counter
                        residue_mapping[residue_counter] = residue
                        amino_types.append(aa_3to1[residue.resname])
                        residue_counter += 1
                    
                    res_id = residue_to_id[res_key]
                    
                    for atom in residue:
                        atom_amino_ids.append(res_id)
                        atom_names_list.append(atom.name)
                        atom_positions.append(atom.coord)
    
    atom_amino_id = np.array(atom_amino_ids, dtype=np.int32)
    atom_names = np.array(atom_names_list, dtype='S4')
    atom_pos = np.array(atom_positions, dtype=np.float32)
    amino_types = ''.join(amino_types)
    
    return atom_amino_id, atom_names, atom_pos, amino_types, residue_mapping


def get_allosteric_pocket_center(pdb_file, chain_id, labels, cluster_distance=10.0):
    """
    Get allosteric pocket center from labels using 3D clustering.
    Returns the center of the largest pocket.
    """
    parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('protein', pdb_file)
        chain = structure[0][chain_id]
    except:
        return None
    
    # Extract residues
    residues = [res for res in chain if res.id[0] == ' ']
    
    if len(residues) != len(labels):
        min_len = min(len(residues), len(labels))
        residues = residues[:min_len]
        labels = labels[:min_len]
    
    # Collect labeled residues with CA coordinates
    labeled_residues = []
    for i, (residue, label) in enumerate(zip(residues, labels)):
        if label == 1 and 'CA' in residue:
            labeled_residues.append({
                'ca_coord': residue['CA'].coord,
                'residue': residue
            })
    
    if len(labeled_residues) == 0:
        return None
    
    # Perform 3D clustering
    ca_coords = np.array([res['ca_coord'] for res in labeled_residues])
    
    if len(ca_coords) == 1:
        # Single residue - use its position
        all_atoms = []
        for atom in labeled_residues[0]['residue']:
            all_atoms.append(atom.coord)
        return np.mean(all_atoms, axis=0)
    
    clustering = DBSCAN(eps=cluster_distance, min_samples=1).fit(ca_coords)
    cluster_labels = clustering.labels_
    
    # Find largest cluster
    unique_clusters = [c for c in set(cluster_labels) if c != -1]
    if not unique_clusters:
        cluster_residues = labeled_residues
    else:
        cluster_counts = [(c, list(cluster_labels).count(c)) for c in unique_clusters]
        largest_cluster = max(cluster_counts, key=lambda x: x[1])[0]
        cluster_residues = [labeled_residues[i] for i, c in enumerate(cluster_labels) if c == largest_cluster]
    
    # Calculate center from all atoms in cluster
    all_atoms = []
    for res_info in cluster_residues:
        for atom in res_info['residue']:
            all_atoms.append(atom.coord)
    
    return np.mean(all_atoms, axis=0)


def count_cut2_with_labels(center, count, atom_amino_id, atom_names, atom_pos, amino_types, 
                            residue_mapping, allosteric_labels):
    """
    Extract the closest 'count' residues to the binding pocket center with labels.
    
    Returns:
        atom_amino_id, atom_names, atom_pos, amino_types, pocket_labels
        
    pocket_labels: array of length 'count' with values:
        0 = no allosteric site
        1 = allosteric site
    """
    data = np.column_stack((atom_amino_id, atom_names, atom_pos))
    indices = data[:, 0].astype(int)
    center = np.array(center).reshape(1, -1)
    amino_types_array = np.array(list(amino_types))
    
    # Calculate residue centers
    x = []
    y = []
    z = []
    
    for i in np.unique(indices):
        mask = np.where(indices == i)
        x.append(np.average(data[:, 2][mask].astype(float)))
        y.append(np.average(data[:, 3][mask].astype(float)))
        z.append(np.average(data[:, 4][mask].astype(float)))
    
    average_xyz = np.column_stack((x, y, z))
    distance_mask = distance.cdist(center, average_xyz, "euclidean")[0]
    binding_data = np.column_stack((np.unique(indices), distance_mask))
    binding_data = binding_data[binding_data[:, 1].argsort()]
    
    # Get the top 'count' closest residues
    new_indices = binding_data[0:count]
    selected_residue_ids = new_indices[:, 0].astype(int)
    
    # Extract pocket labels for selected residues
    pocket_labels = []
    for res_id in selected_residue_ids:
        # res_id is 1-indexed sequential ID
        # Map to sequence position (0-indexed for label arrays)
        seq_pos = res_id - 1
        
        if seq_pos < len(allosteric_labels):
            pocket_labels.append(allosteric_labels[seq_pos])
        else:
            # Out of bounds, assign 0
            pocket_labels.append(0)
    
    pocket_labels = np.array(pocket_labels, dtype=np.int32)
    
    # Extract atoms for selected residues
    amino_types_selected = amino_types_array[np.isin(np.arange(1, len(amino_types_array)+1), selected_residue_ids)]
    binding_site = data[np.isin(data[:, 0].astype(int), selected_residue_ids)]
    
    atom_amino_id = binding_site[:, 0].astype(int)
    atom_names = binding_site[:, 1]
    atom_pos = binding_site[:, 2:5].astype(float)
    
    return atom_amino_id, atom_names, atom_pos, amino_types_selected, pocket_labels


def write_to_h5_with_labels(output_h5, identifier, atom_amino_id, atom_names, atom_pos, 
                             amino_types, pocket_labels, chain='A', pad_length=100, pad_token='X'):
    """
    Write binding pocket data with labels to H5 file.
    """
    with h5py.File(output_h5, 'a') as f:
        # Create the hierarchy
        if identifier in f:
            del f[identifier]
        
        grp = f.create_group(identifier)
        struct_grp = grp.create_group('structure')
        model_grp = struct_grp.create_group('0')
        chain_grp = model_grp.create_group(chain)
        
        # Residues group
        residues_grp = chain_grp.create_group('residues')
        
        # Store sequence as bytes
        if isinstance(amino_types, np.ndarray):
            seq_str = ''.join(amino_types)
        else:
            seq_str = amino_types

        # Pad or truncate sequence to pad_length using pad_token
        if len(seq_str) < pad_length:
            seq_str = seq_str + (pad_token * (pad_length - len(seq_str)))
        elif len(seq_str) > pad_length:
            seq_str = seq_str[:pad_length]

        residues_grp.create_dataset('seq1', data=seq_str.encode('utf-8'))

        # Ensure pocket_labels is numpy array and pad/truncate to pad_length with 0s
        pocket_labels = np.array(pocket_labels, dtype=np.int32)
        if len(pocket_labels) < pad_length:
            pad_size = pad_length - len(pocket_labels)
            pocket_labels = np.concatenate([pocket_labels, np.zeros(pad_size, dtype=np.int32)])
        elif len(pocket_labels) > pad_length:
            pocket_labels = pocket_labels[:pad_length]

        residues_grp.create_dataset('labels', data=pocket_labels)
        
        # Polypeptide group
        poly_grp = chain_grp.create_group('polypeptide')
        
        # Renumber atom_amino_id to start from 1 and be consecutive
        unique_ids = np.unique(atom_amino_id)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
        atom_amino_id_renumbered = np.array([id_mapping[old_id] for old_id in atom_amino_id], dtype=np.int32)
        
        poly_grp.create_dataset('atom_amino_id', data=atom_amino_id_renumbered)
        poly_grp.create_dataset('type', data=atom_names)
        poly_grp.create_dataset('xyz', data=atom_pos)
    
    print(f"  Saved to H5: {identifier}/{chain}")
    print(f"    Sequence length (stored): {len(seq_str)}")
    print(f"    Number of atoms: {len(atom_amino_id)}")
    print(f"    Label distribution: 0={list(pocket_labels).count(0)}, 1={list(pocket_labels).count(1)}")


def process_allosteric_annotation_to_h5(pdb_id, chains, allosteric_labels, 
                                         output_h5, count=100, cluster_distance=10.0, output_dir='pdb_files'):
    """
    Process allosteric annotation and write binding pocket to H5 file.
    
    Args:
        pdb_id: PDB identifier
        chains: List of chain identifiers
        allosteric_labels: Binary labels indicating allosteric residues (0 or 1)
        output_h5: Output H5 file path
        count: Number of closest residues to extract (default: 100)
        cluster_distance: Distance for clustering labeled regions
        output_dir: Directory for downloaded PDB files
    """
    print(f"\n{'='*70}")
    print(f"Processing: {pdb_id}")
    print(f"{'='*70}")
    
    chain_id = chains[0]  # Use first chain
    print(f"PDB: {pdb_id}, Chain: {chain_id}")
    
    # Download PDB file
    pdb_file = download_pdb_file(pdb_id, output_dir)
    if not pdb_file:
        print("Error: Could not download PDB file")
        return False
    
    # Get allosteric pocket center
    if sum(allosteric_labels) == 0:
        print("Error: No labeled residues found")
        return False
    
    center = get_allosteric_pocket_center(pdb_file, chain_id, allosteric_labels, cluster_distance)
    
    if center is None:
        print(f"Error: Could not find pocket center in chain {chain_id}")
        return False
    
    print(f"Pocket center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    
    # Extract structure data from PDB
    atom_amino_id, atom_names, atom_pos, amino_types, residue_mapping = extract_structure_data_from_pdb(pdb_file, chain_id)
    
    print(f"Extracted structure data:")
    print(f"  Total residues: {len(amino_types)}")
    print(f"  Total atoms: {len(atom_amino_id)}")
    print(f"  Labeled residues in full sequence: {sum(allosteric_labels)}")
    
    # Apply count_cut2 to get binding pocket with labels
    pocket_atom_amino_id, pocket_atom_names, pocket_atom_pos, pocket_amino_types, pocket_labels = count_cut2_with_labels(
        center, count, atom_amino_id, atom_names, atom_pos, amino_types, 
        residue_mapping, allosteric_labels
    )
    
    print(f"Binding pocket (top {count} residues):")
    print(f"  Pocket residues: {len(pocket_amino_types)}")
    print(f"  Pocket atoms: {len(pocket_atom_amino_id)}")
    print(f"  Pocket labels length: {len(pocket_labels)}")
    
    # Create identifier for H5 file
    identifier = f"{pdb_id}_{chain_id}"
    
    # Write to H5 file
    write_to_h5_with_labels(output_h5, identifier, pocket_atom_amino_id, pocket_atom_names, 
                            pocket_atom_pos, pocket_amino_types, pocket_labels, chain=chain_id, pad_length=count)
    
    print(f"✓ Successfully processed {pdb_id}")
    return True


def process_multiple_csv_files(csv_files, output_h5, count=100, cluster_distance=10.0, output_dir='pdb_files'):
    """
    Process multiple CSV files and combine into a single H5 file.
    
    Args:
        csv_files: List of CSV file paths
        output_h5: Output H5 file path
        count: Number of closest residues to extract (default: 100)
        cluster_distance: Distance threshold for clustering (default: 10.0Å)
        output_dir: Directory for PDB files
    """
    print("="*70)
    print("BATCH PROCESSING FROM MULTIPLE CSV FILES")
    print("="*70)
    print(f"\nInput CSV files: {csv_files}")
    print(f"Output H5 file: {output_h5}")
    print(f"Residues per pocket: {count}")
    print(f"Cluster distance: {cluster_distance}Å\n")
    
    # Combine all CSV files
    all_dfs = []
    for csv_file in csv_files:
        print(f"Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_cols = ['pdb_id', 'chains', 'Labels']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: '{col}' column not found in {csv_file}, skipping...")
                continue
        
        all_dfs.append(df)
        print(f"  Loaded {len(df)} entries from {csv_file}")
    
    if not all_dfs:
        print("Error: No valid CSV files loaded")
        return
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined entries: {len(combined_df)}")
    
    # Remove duplicates based on pdb_id (optional)
    print(f"Checking for duplicate PDB IDs...")
    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['pdb_id'], keep='first')
    if len(combined_df) < original_count:
        print(f"  Removed {original_count - len(combined_df)} duplicates")
    
    print(f"Final entries to process: {len(combined_df)}\n")
    
    # Process each entry
    success_count = 0
    failed_count = 0
    
    for idx, row in combined_df.iterrows():
        print(f"\n[{idx + 1}/{len(combined_df)}]")
        
        try:
            pdb_id = row['pdb_id']
            chains = ast.literal_eval(row['chains']) if isinstance(row['chains'], str) else row['chains']
            allosteric_labels = ast.literal_eval(row['Labels']) if isinstance(row['Labels'], str) else row['Labels']
            
            success = process_allosteric_annotation_to_h5(
                pdb_id, chains, allosteric_labels,
                output_h5, count, cluster_distance, output_dir
            )
            
            if success:
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {success_count}/{len(combined_df)}")
    print(f"Failed: {failed_count}/{len(combined_df)}")
    print(f"\nOutput saved to: {output_h5}")


def process_allosteric_batch(annotations_file, output_h5, count=100, cluster_distance=10.0, output_dir='pdb_files'):
    """
    Process single CSV file (wrapper for backward compatibility).
    """
    process_multiple_csv_files([annotations_file], output_h5, count, cluster_distance, output_dir)


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Process a single CSV file
    # print("Example 1: Single CSV file")
    # annotations_file = 'allosteric_annotations.csv'
    # output_h5 = 'binding_pockets_allosteric.h5'
    
    # process_allosteric_batch(
    #     annotations_file=annotations_file,
    #     output_h5=output_h5,
    #     count=100,
    #     cluster_distance=10.0,
    #     output_dir='pdb_files'
    # )
    
    # Example 2: Process multiple CSV files into one H5 file
    print("\n\nExample 2: Multiple CSV files")
    csv_files = [
        '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/train_df_pdb.csv',
        '/p/data1/profound_data/CDPPILBP/ippidb-pdb-analyses-042023-zenodo/test_df_pdb.csv'
    ]
    output_h5 = 'ASD_binding_pockets.h5'
    
    process_multiple_csv_files(
        csv_files=csv_files,
        output_h5=output_h5,
        count=100,
        cluster_distance=10.0,
        output_dir='pdb_files'
    )
    
    # Verify H5 file structure
    print("\n\n")
    print("="*70)
    print("H5 FILE STRUCTURE")
    print("="*70)
    
    with h5py.File(output_h5, 'r') as f:
        identifiers = list(f.keys())
        print(f"\nTotal pockets in H5: {len(identifiers)}")
        
        if identifiers:
            identifier = identifiers[0]
            print(f"\nExample: '{identifier}'")
            
            seq = f[identifier]['structure']['0']['A']['residues']['seq1'][()]
            labels = f[identifier]['structure']['0']['A']['residues']['labels'][()]
            
            print(f"  Sequence: {seq.decode('utf-8')}")
            print(f"  Sequence length: {len(seq.decode('utf-8'))}")
            print(f"  Labels length: {len(labels)}")
            print(f"  Labels: {labels}")
            print(f"  Label distribution:")
            print(f"    0 (no allosteric site): {list(labels).count(0)}")
            print(f"    1 (allosteric site): {list(labels).count(1)}")
            
            atom_amino_id = f[identifier]['structure']['0']['A']['polypeptide']['atom_amino_id'][()]
            print(f"  Number of atoms: {len(atom_amino_id)}")