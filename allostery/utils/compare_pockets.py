import h5py
import numpy as np
from collections import defaultdict


def extract_pocket_info(h5_file):
    """
    Extract pocket information from H5 file.
    Returns dict: {pdb_chain: {'sequence': str, 'num_atoms': int, 'xyz_hash': str}}
    """
    pockets = {}
    
    with h5py.File(h5_file, 'r') as f:
        for identifier in f.keys():
            try:
                # Extract sequence
                seq = f[identifier]['structure']['0']['A']['residues']['seq1'][()]
                seq_str = seq.decode('utf-8') if isinstance(seq, bytes) else seq
                
                # Extract atom positions
                xyz = f[identifier]['structure']['0']['A']['polypeptide']['xyz'][()]
                
                # Create a hash of coordinates (to detect identical structures)
                xyz_hash = hash(xyz.tobytes())
                
                # Extract PDB ID and chain from identifier
                # Format could be: pdb_chain or pdb_chain_ligand_ligandid
                parts = identifier.split('_')
                pdb_id = parts[0]
                chain = parts[1] if len(parts) > 1 else 'A'
                pdb_chain = f"{pdb_id}_{chain}"
                
                pockets[identifier] = {
                    'pdb_chain': pdb_chain,
                    'sequence': seq_str,
                    'num_atoms': len(xyz),
                    'xyz_hash': xyz_hash,
                    'xyz_mean': tuple(np.mean(xyz, axis=0))
                }
                
            except Exception as e:
                print(f"Error processing {identifier}: {e}")
    
    return pockets


def compare_h5_files(h5_files):
    """
    Compare multiple H5 files to find identical pockets.
    
    Args:
        h5_files: Dict with format {'file_label': 'file_path'}
    """
    print("="*80)
    print("COMPARING H5 FILES FOR IDENTICAL POCKETS")
    print("="*80)
    
    # Extract pocket info from all files
    all_pockets = {}
    file_stats = {}
    
    for label, filepath in h5_files.items():
        print(f"\nLoading {label}: {filepath}")
        pockets = extract_pocket_info(filepath)
        all_pockets[label] = pockets
        file_stats[label] = len(pockets)
        print(f"  Found {len(pockets)} pockets")
    
    print("\n" + "="*80)
    print("FILE STATISTICS")
    print("="*80)
    for label, count in file_stats.items():
        print(f"{label}: {count} pockets")
    
    # Find overlaps based on PDB_chain
    print("\n" + "="*80)
    print("OVERLAP ANALYSIS (by PDB ID + Chain)")
    print("="*80)
    
    # Group by pdb_chain
    pdb_chain_to_files = defaultdict(set)
    
    for label, pockets in all_pockets.items():
        for identifier, info in pockets.items():
            pdb_chain = info['pdb_chain']
            pdb_chain_to_files[pdb_chain].add(label)
    
    # Find pockets in multiple files
    overlaps = defaultdict(list)
    for pdb_chain, files in pdb_chain_to_files.items():
        if len(files) > 1:
            overlaps[tuple(sorted(files))].append(pdb_chain)
    
    if not overlaps:
        print("\n✓ No overlapping pockets found between files!")
    else:
        print(f"\n⚠ Found {sum(len(v) for v in overlaps.values())} overlapping pockets:")
        
        for files_combo, pdb_chains in sorted(overlaps.items()):
            print(f"\n  Present in: {' + '.join(files_combo)}")
            print(f"  Number of overlaps: {len(pdb_chains)}")
            print(f"  Examples: {pdb_chains[:5]}")
            if len(pdb_chains) > 5:
                print(f"  ... and {len(pdb_chains) - 5} more")
    
    # Detailed sequence comparison for overlaps
    print("\n" + "="*80)
    print("DETAILED COMPARISON OF OVERLAPPING POCKETS")
    print("="*80)
    
    if overlaps:
        for files_combo, pdb_chains in sorted(overlaps.items()):
            print(f"\n{' vs '.join(files_combo)}:")
            
            for pdb_chain in pdb_chains[:3]:  # Show first 3 detailed
                print(f"\n  PDB_Chain: {pdb_chain}")
                
                # Find identifiers with this pdb_chain in each file
                sequences = {}
                num_atoms = {}
                xyz_hashes = {}
                
                for label in files_combo:
                    for identifier, info in all_pockets[label].items():
                        if info['pdb_chain'] == pdb_chain:
                            sequences[label] = info['sequence']
                            num_atoms[label] = info['num_atoms']
                            xyz_hashes[label] = info['xyz_hash']
                            print(f"    {label}: {identifier}")
                            break
                
                # Compare sequences
                if len(set(sequences.values())) == 1:
                    print(f"    ✓ Sequences are IDENTICAL (length {len(list(sequences.values())[0])})")
                else:
                    print(f"    ✗ Sequences are DIFFERENT:")
                    for label, seq in sequences.items():
                        print(f"      {label}: length {len(seq)}")
                
                # Compare number of atoms
                if len(set(num_atoms.values())) == 1:
                    print(f"    ✓ Number of atoms is IDENTICAL ({list(num_atoms.values())[0]})")
                else:
                    print(f"    ✗ Number of atoms is DIFFERENT:")
                    for label, count in num_atoms.items():
                        print(f"      {label}: {count} atoms")
                
                # Compare 3D structures
                if len(set(xyz_hashes.values())) == 1:
                    print(f"    ✓ 3D structures are IDENTICAL")
                else:
                    print(f"    ✗ 3D structures are DIFFERENT")
            
            if len(pdb_chains) > 3:
                print(f"\n  ... and {len(pdb_chains) - 3} more overlapping pockets")
    
    # Summary matrix
    print("\n" + "="*80)
    print("OVERLAP MATRIX")
    print("="*80)
    
    labels = list(h5_files.keys())
    print(f"\n{'':30s}", end='')
    for label in labels:
        print(f"{label[:20]:>20s}", end='')
    print()
    
    for label1 in labels:
        print(f"{label1[:30]:30s}", end='')
        for label2 in labels:
            if label1 == label2:
                print(f"{'---':>20s}", end='')
            else:
                # Count overlaps between label1 and label2
                count = 0
                for files_combo, pdb_chains in overlaps.items():
                    if label1 in files_combo and label2 in files_combo:
                        count += len(pdb_chains)
                print(f"{count:>20d}", end='')
        print()
    
    print("\n" + "="*80)
    
    return overlaps, all_pockets


def save_overlap_report(overlaps, all_pockets, output_file='overlap_report.txt'):
    """
    Save detailed overlap report to a text file.
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED OVERLAP REPORT\n")
        f.write("="*80 + "\n\n")
        
        for files_combo, pdb_chains in sorted(overlaps.items()):
            f.write(f"\nFiles: {' + '.join(files_combo)}\n")
            f.write(f"Number of overlapping pockets: {len(pdb_chains)}\n")
            f.write("-"*80 + "\n")
            
            for pdb_chain in pdb_chains:
                f.write(f"\nPDB_Chain: {pdb_chain}\n")
                
                for label in files_combo:
                    for identifier, info in all_pockets[label].items():
                        if info['pdb_chain'] == pdb_chain:
                            f.write(f"  {label}: {identifier}\n")
                            f.write(f"    Sequence length: {len(info['sequence'])}\n")
                            f.write(f"    Number of atoms: {info['num_atoms']}\n")
                            break
    
    print(f"\nDetailed report saved to: {output_file}")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Define the three H5 files to compare
    h5_files = {
        'allosteric': 'binding_pockets_allosteric.h5',
        'orthosteric_competitive': 'binding_pockets_orthosteric_competitive.h5',
        'orthosteric_noncompetitive': 'binding_pockets_orthosteric_noncompetitive.h5'
    }
    
    # Run comparison
    overlaps, all_pockets = compare_h5_files(h5_files)
    
    # Save detailed report if overlaps found
    if overlaps:
        save_overlap_report(overlaps, all_pockets, 'overlap_report.txt')
    
    print("\n✓ Comparison complete!")