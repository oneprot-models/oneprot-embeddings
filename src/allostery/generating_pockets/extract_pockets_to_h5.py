import h5py
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import requests
import os
from scipy.spatial import distance

def parse_pocket_annotation(annotation):
    """
    Parse pocket annotation.
    Format: pdb-chain-uniprot-ligand-ligandID_CAVITY_info
    Example: 5ayf-A-Q8WTS6-SAM-401_CAVITY_N1_liganded_allosteric
    """
    parts = annotation.split('-')
    
    if len(parts) >= 5:
        last_part = '-'.join(parts[4:])
        underscore_parts = last_part.split('_')
        
        return {
            'pdb_id': parts[0],
            'chain': parts[1],
            'uniprot': parts[2],
            'ligand_name': parts[3],
            'ligand_id': underscore_parts[0] if underscore_parts else '',
            'cavity_info': '_'.join(underscore_parts[1:]) if len(underscore_parts) > 1 else '',
            'full_annotation': annotation
        }
    return None

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

def get_ligand_center(pdb_file, chain_id, ligand_name, ligand_id):
    """Get the center coordinates of a ligand."""
    parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('protein', pdb_file)
        
        try:
            lig_id = int(ligand_id)
        except:
            lig_id = ligand_id
        
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                
                for residue in chain:
                    res_name = residue.resname.strip()
                    res_id = residue.id[1]
                    
                    if res_name == ligand_name and res_id == lig_id:
                        coords = []
                        for atom in residue:
                            coords.append(atom.coord)
                        
                        if coords:
                            center = np.mean(coords, axis=0)
                            return center
        
        return None
        
    except Exception as e:
        print(f"Error parsing PDB file: {str(e)}")
        return None

def extract_structure_data_from_pdb(pdb_file, chain_id):
    """
    Extract structure data from PDB file in the format needed for count_cut2.
    Returns: atom_amino_id, atom_names, atom_pos, amino_types
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
    
    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            
            for residue in chain:
                # Only process standard amino acids
                if residue.id[0] == ' ' and residue.resname in aa_3to1:
                    res_key = (residue.id[1], residue.id[2])
                    
                    if res_key not in residue_to_id:
                        residue_to_id[res_key] = residue_counter
                        amino_types.append(aa_3to1[residue.resname])
                        residue_counter += 1
                    
                    res_id = residue_to_id[res_key]
                    
                    for atom in residue:
                        atom_amino_ids.append(res_id)
                        atom_names_list.append(atom.name)
                        atom_positions.append(atom.coord)
    
    atom_amino_id = np.array(atom_amino_ids, dtype=np.int32)
    atom_names = np.array(atom_names_list, dtype='S4')  # 4-byte strings
    atom_pos = np.array(atom_positions, dtype=np.float32)
    amino_types = ''.join(amino_types)
    
    return atom_amino_id, atom_names, atom_pos, amino_types

def count_cut2(center, count, atom_amino_id, atom_names, atom_pos, amino_types):
    """
    Extract the closest 'count' residues to the binding pocket center.
    """
    data = np.column_stack((atom_amino_id, atom_names, atom_pos))
    indices = data[:, 0].astype(int)
    center = np.array(center).reshape(1, -1)
    amino_types = np.array(list(amino_types))
    
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
    new_indices = binding_data[0:count]
    
    amino_types = amino_types[np.isin(np.arange(1, len(amino_types)+1), new_indices[:, 0].astype(int))]
    binding_site = data[np.isin(data[:, 0].astype(int), new_indices[:, 0].astype(int))]
    
    atom_amino_id = binding_site[:, 0].astype(int)
    atom_names = binding_site[:, 1]
    atom_pos = binding_site[:, 2:5].astype(float)
    
    return atom_amino_id, atom_names, atom_pos, amino_types

def write_to_h5(output_h5, identifier, atom_amino_id, atom_names, atom_pos, amino_types, chain='A'):
    """
    Write binding pocket data to H5 file in the format expected by protein_to_graph.
    
    Structure:
    identifier/
        structure/
            0/
                chain/
                    residues/
                        seq1: amino acid sequence
                    polypeptide/
                        atom_amino_id: residue IDs for each atom
                        type: atom names
                        xyz: atom coordinates
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
        
        # Store sequence as bytes (HDF5 standard)
        if isinstance(amino_types, np.ndarray):
            seq_str = ''.join(amino_types)
        else:
            seq_str = amino_types
        
        residues_grp.create_dataset('seq1', data=seq_str.encode('utf-8'))
        
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
    print(f"    Sequence length: {len(amino_types)}")
    print(f"    Number of atoms: {len(atom_amino_id)}")

def process_annotation_to_h5(annotation, output_h5, count=100):
    """
    Process a single annotation and write binding pocket to H5 file.
    
    Args:
        annotation: Pocket annotation string
        output_h5: Output H5 file path
        count: Number of closest residues to extract
    """
    print(f"\n{'='*70}")
    print(f"Processing: {annotation}")
    print(f"{'='*70}")
    
    # Parse annotation
    parsed = parse_pocket_annotation(annotation)
    if not parsed:
        print("Error: Could not parse annotation")
        return False
    
    pdb_id = parsed['pdb_id']
    chain_id = parsed['chain']
    ligand_name = parsed['ligand_name']
    ligand_id = parsed['ligand_id']
    
    print(f"PDB: {pdb_id}, Chain: {chain_id}, Ligand: {ligand_name}-{ligand_id}")
    
    # Download PDB file
    pdb_file = download_pdb_file(pdb_id)
    if not pdb_file:
        print("Error: Could not download PDB file")
        return False
    
    # Get ligand center (binding pocket center)
    center = get_ligand_center(pdb_file, chain_id, ligand_name, ligand_id)
    if center is None:
        print(f"Error: Could not find ligand {ligand_name}-{ligand_id} in chain {chain_id}")
        return False
    
    print(f"Pocket center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    
    # Extract structure data from PDB
    atom_amino_id, atom_names, atom_pos, amino_types = extract_structure_data_from_pdb(pdb_file, chain_id)
    
    print(f"Extracted structure data:")
    print(f"  Total residues: {len(amino_types)}")
    print(f"  Total atoms: {len(atom_amino_id)}")
    
    # Apply count_cut2 to get binding pocket
    pocket_atom_amino_id, pocket_atom_names, pocket_atom_pos, pocket_amino_types = count_cut2(
        center, count, atom_amino_id, atom_names, atom_pos, amino_types
    )
    
    print(f"Binding pocket (top {count} residues):")
    print(f"  Pocket residues: {len(pocket_amino_types)}")
    print(f"  Pocket atoms: {len(pocket_atom_amino_id)}")
    
    # Create identifier for H5 file
    identifier = f"{pdb_id}_{chain_id}_{ligand_name}_{ligand_id}"
    
    # Write to H5 file
    write_to_h5(output_h5, identifier, pocket_atom_amino_id, pocket_atom_names, 
                pocket_atom_pos, pocket_amino_types, chain=chain_id)
    
    print(f"✓ Successfully processed {annotation}")
    return True

def process_annotations_batch(annotations_file, output_h5, count=100):
    """
    Process multiple annotations from a CSV file.
    
    Args:
        annotations_file: CSV file with 'Cavity' column
        output_h5: Output H5 file path
        count: Number of closest residues to extract
    """
    print("="*70)
    print("BATCH PROCESSING BINDING POCKETS")
    print("="*70)
    
    # Read annotations
    try:
        df = pd.read_csv(annotations_file)
    except:
        df = pd.read_csv(annotations_file)
    
    if 'Cavity' not in df.columns:
        print("Error: 'Cavity' column not found in CSV file")
        return
    
    annotations = df['Cavity'].unique().tolist()
    print(f"\nTotal unique annotations: {len(annotations)}")
    print(f"Output H5 file: {output_h5}")
    print(f"Residues per pocket: {count}\n")
    
    # Process each annotation
    success_count = 0
    failed_count = 0
    
    for i, annotation in enumerate(annotations, 1):
        print(f"\n[{i}/{len(annotations)}]")
        
        try:
            success = process_annotation_to_h5(annotation, output_h5, count)
            if success:
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"Error processing {annotation}: {str(e)}")
            failed_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {success_count}/{len(annotations)}")
    print(f"Failed: {failed_count}/{len(annotations)}")
    print(f"\nOutput saved to: {output_h5}")

# Example usage

# Single annotation
# annotation = '5ayf-A-Q8WTS6-SAM-401_CAVITY_N1_liganded_allosteric'
# output_h5 = 'binding_pockets_allosteric.h5'

# print("SINGLE ANNOTATION EXAMPLE")
# process_annotation_to_h5(annotation, output_h5, count=100)

# Batch processing
print("\n\n")
print("BATCH PROCESSING EXAMPLE")
# Uncomment to process a full CSV file:
annotations_file = 'PL_part8_20230317_matrix_liganded_orthosteric_noncompetitive.csv'
process_annotations_batch(annotations_file, 'binding_pockets_orthosteric_noncompetitive.h5', count=100)

# Test reading the H5 file with protein_to_graph
print("\n\n")
print("="*70)
print("TESTING H5 FILE COMPATIBILITY")
print("="*70)

# Show H5 structure
# with h5py.File(output_h5, 'r') as f:
#     print(f"\nH5 file structure:")
#     def print_structure(name, obj):
#         print(f"  {name}")
#     f.visititems(print_structure)
    
#     # Get the first identifier
#     identifiers = list(f.keys())
#     if identifiers:
#         identifier = identifiers[0]
#         print(f"\nExample data for '{identifier}':")
#         seq = f[identifier]['structure']['0']['A']['residues']['seq1'][()]
#         print(f"  Sequence: {seq.decode('utf-8')}")
#         print(f"  Sequence length: {len(seq.decode('utf-8'))}")
        
#         atom_amino_id = f[identifier]['structure']['0']['A']['polypeptide']['atom_amino_id'][()]
#         print(f"  Number of atoms: {len(atom_amino_id)}")
#         print(f"  Atom amino IDs range: {atom_amino_id.min()} to {atom_amino_id.max()}")
#```

# This script will:

# 1. **Parse the annotation** to extract PDB ID, chain, ligand info
# 2. **Download the PDB file** from RCSB
# 3. **Extract ligand center** coordinates (binding pocket center)
# 4. **Extract full protein structure** data (atom_amino_id, atom_names, atom_pos, amino_types)
# 5. **Apply count_cut2** to get the 100 closest residues to the pocket center
# 6. **Write to H5 file** in the exact format expected by `protein_to_graph`

# **H5 file structure created:**
# ```
# {pdb_id}_{chain}_{ligand_name}_{ligand_id}/
#     structure/
#         0/
#             {chain}/
#                 residues/
#                     seq1: "ACDEFG..."
#                 polypeptide/
#                     atom_amino_id: [1, 1, 1, 2, 2, 2, ...]
#                     type: [b'N', b'CA', b'C', b'O', ...]
#                     xyz: [[x1, y1, z1], [x2, y2, z2], ...]