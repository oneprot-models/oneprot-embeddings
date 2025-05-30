#!/usr/bin/env python3
import h5py
import sys
import os

def print_usage():
    print("Usage: python split_h5.py input.h5 output1.h5 output2.h5 [key1 key2 ...]")
    print("  - If no keys are provided, the file will be split evenly")
    print("  - If keys are provided, they will go to output1.h5, rest to output2.h5")
    sys.exit(1)

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 4:
        print_usage()
    
    input_file = sys.argv[1]
    output_file1 = sys.argv[2]
    output_file2 = sys.argv[3]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"Opening input file: {input_file}")
    
    # Open the input file
    with h5py.File(input_file, 'r') as infile:
        # Get top-level keys
        top_keys = list(infile.keys())
        
        if not top_keys:
            print("Error: Input file has no top-level groups/datasets")
            sys.exit(1)
            
        # Default: Split the keys in half
        middle = len(top_keys) // 2
        keys_file1 = top_keys[:middle]
        keys_file2 = top_keys[middle:]
        
        # If specific keys were provided, use those instead
        if len(sys.argv) > 4:
            requested_keys = sys.argv[4:]
            keys_file1 = [k for k in requested_keys if k in top_keys]
            missing_keys = [k for k in requested_keys if k not in top_keys]
            keys_file2 = [k for k in top_keys if k not in keys_file1]
            
            if missing_keys:
                print(f"Warning: These keys were not found: {', '.join(missing_keys)}")
        
        print(f"Keys for {output_file1}: {', '.join(keys_file1)}")
        print(f"Keys for {output_file2}: {', '.join(keys_file2)}")
        
        # Create and fill the first output file
        with h5py.File(output_file1, 'w') as outfile1:
            for key in keys_file1:
                print(f"Copying {key} to {output_file1}")
                infile.copy(key, outfile1)
        
        # Create and fill the second output file
        with h5py.File(output_file2, 'w') as outfile2:
            for key in keys_file2:
                print(f"Copying {key} to {output_file2}")
                infile.copy(key, outfile2)
                
    print("Split complete!")