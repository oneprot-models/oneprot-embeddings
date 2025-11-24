import pandas as pd

def combine_csv_files(file1_path, file2_path, output_path):
    """
    Combine two CSV files where the first has 2 columns and the second has 3 columns.
    The third column for entries from the first file will be empty strings.
    
    Args:
        file1_path: Path to the first CSV file (2 columns)
        file2_path: Path to the second CSV file (3 columns)
        output_path: Path where the combined CSV will be saved
    """
    # Read the first CSV (with 2 columns)
    df1 = pd.read_csv(file1_path)
    
    # Read the second CSV (with 3 columns)
    df2 = pd.read_csv(file2_path)
    
    # Add the missing column to the first dataframe with empty strings
    if len(df1.columns) == 2 and len(df2.columns) == 3:
        # Get the name of the third column from df2
        third_column_name = df2.columns[2]
        
        # Add this column to df1 with empty strings
        df1[third_column_name] = ""
        
        # Make sure columns are in the same order
        df1 = df1[df2.columns]
    else:
        print(f"Warning: Expected 2 and 3 columns but got {len(df1.columns)} and {len(df2.columns)}")
    
    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save the combined dataframe
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to {output_path}")
    
    return combined_df