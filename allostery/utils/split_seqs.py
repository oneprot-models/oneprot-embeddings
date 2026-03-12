import pandas as pd
import numpy as np
import requests
import os
import tempfile
import pickle

def fetch_uniprot_sequence(uniprot_id):
    """
    Fetch protein sequence from UniProt.
    
    Args:
        uniprot_id: UniProt ID (e.g., 'P38398' or 'P38398_HUMAN')
    
    Returns:
        Protein sequence string or None
    """
    # Clean up the ID (remove _HUMAN suffix if present)
    clean_id = uniprot_id.split('_')[0].strip()
    
    url = f"https://www.uniprot.org/uniprot/{clean_id}.fasta"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            # Skip header, join sequence lines
            sequence = ''.join(lines[1:])
            print(f"  ✓ Fetched sequence for {clean_id} ({len(sequence)} aa)")
            return sequence
        else:
            print(f"  Warning: Could not fetch sequence for {clean_id} (HTTP {response.status_code})")
    except Exception as e:
        print(f"  Error fetching {clean_id}: {str(e)}")
    
    return None

def fetch_pdb_sequence(pdb_id, chain_id='A'):
    """
    Fetch protein sequence from PDB file.
    
    Args:
        pdb_id: PDB ID
        chain_id: Chain identifier (default 'A')
    
    Returns:
        Protein sequence string or None
    """
    from Bio.PDB import PDBParser
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                tmp.write(response.text)
                tmp_path = tmp.name
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', tmp_path)
            
            aa_3to1 = {
                'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
            }
            
            sequence = []
            for model in structure:
                for chain in model:
                    # Try to find the chain
                    if chain.id == chain_id or chain_id == 'A':
                        for residue in chain:
                            if residue.id[0] == ' ' and residue.resname in aa_3to1:
                                sequence.append(aa_3to1[residue.resname])
                        if sequence:
                            break
                if sequence:
                    break
            
            os.unlink(tmp_path)
            
            if sequence:
                seq_str = ''.join(sequence)
                print(f"  ✓ Fetched PDB sequence for {pdb_id} chain {chain_id} ({len(seq_str)} aa)")
                return seq_str
    except Exception as e:
        print(f"  Error fetching PDB sequence for {pdb_id}: {str(e)}")
    
    return None

def get_protein_sequence(row):
    """
    Get protein sequence from row, trying UniProt first, then PDB.
    """
    # Try UniProt first
    if 'Uniprot ID' in row and pd.notna(row['Uniprot ID']):
        seq = fetch_uniprot_sequence(str(row['Uniprot ID']))
        if seq:
            return seq
    
    # Fall back to PDB
    if 'PDB ID' in row and pd.notna(row['PDB ID']):
        seq = fetch_pdb_sequence(str(row['PDB ID']))
        if seq:
            return seq
    
    return None

def prepare_sequences_for_mmseqs2(excel_file, output_dir='mmseqs_input'):
    """
    Step 1: Fetch sequences and prepare FASTA file for MMseqs2.
    
    Args:
        excel_file: Path to Excel file
        output_dir: Output directory for FASTA and metadata
    
    Outputs:
        - sequences.fasta: FASTA file for MMseqs2
        - metadata.pkl: Pickle file with DataFrame and sequence info
    """
    print("="*70)
    print("STEP 1: PREPARING SEQUENCES FOR MMSEQS2")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read Excel file
    print(f"\nReading Excel file: {excel_file}")
    try:
        df = pd.read_excel(excel_file)
        print(f"  ✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"  ✗ Error reading Excel: {str(e)}")
        return
    
    # Verify required columns
    required_cols = ['PDB ID', 'Ligand ID', 'Mechanism']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        return
    
    # Fetch sequences
    print(f"\n{'='*70}")
    print("FETCHING PROTEIN SEQUENCES")
    print(f"{'='*70}")
    
    sequences = {}
    failed_indices = []
    
    for idx, row in df.iterrows():
        print(f"\n[{idx+1}/{len(df)}] PDB: {row['PDB ID']}")
        seq = get_protein_sequence(row)
        if seq:
            sequences[idx] = seq
        else:
            print(f"  ✗ Failed to fetch sequence")
            failed_indices.append(idx)
    
    print(f"\n✓ Successfully fetched {len(sequences)}/{len(df)} sequences")
    
    if failed_indices:
        print(f"⚠ Failed to fetch sequences for {len(failed_indices)} entries")
        print(f"  Removing these entries from analysis")
        df = df.drop(failed_indices)
    
    if len(sequences) == 0:
        print("✗ No sequences fetched. Cannot proceed.")
        return
    
    # Write FASTA file
    fasta_file = os.path.join(output_dir, 'sequences.fasta')
    print(f"\n{'='*70}")
    print(f"WRITING FASTA FILE")
    print(f"{'='*70}")
    print(f"Output: {fasta_file}")
    
    with open(fasta_file, 'w') as f:
        for idx in sequences.keys():
            f.write(f">seq_{idx}\n")
            f.write(f"{sequences[idx]}\n")
    
    print(f"  ✓ Wrote {len(sequences)} sequences")
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.pkl')
    metadata = {
        'df': df,
        'sequences': sequences,
        'seq_ids': list(sequences.keys())
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nSaved metadata: {metadata_file}")
    
    # Print instructions for MMseqs2
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("\n1. Create MMseqs2 database:")
    print(f"   mmseqs createdb {fasta_file} {output_dir}/seqDB")
    
    print("\n2. Cluster sequences (adjust --min-seq-id as needed):")
    print(f"   mmseqs cluster {output_dir}/seqDB {output_dir}/clusterDB {output_dir}/tmp \\")
    print(f"       --min-seq-id 0.3 -c 0.8 --cov-mode 0")
    
    print("\n3. Export clusters to TSV:")
    print(f"   mmseqs createtsv {output_dir}/seqDB {output_dir}/seqDB \\")
    print(f"       {output_dir}/clusterDB {output_dir}/clusters.tsv")
    
    print("\n4. Run step 2 to create splits:")
    print(f"   python step2_create_splits.py {output_dir}")
    
    print(f"\n✓ Preparation complete!")

if __name__ == "__main__":
    import sys
    
    excel_file = '/p/project1/hai_oneprot/bazarova1/oneprot-panda/Allosteric_and_competitive_inhibitors.csv-1.xls'  # Change this
    output_dir = 'mmseqs_input'
    
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    prepare_sequences_for_mmseqs2(excel_file, output_dir)
