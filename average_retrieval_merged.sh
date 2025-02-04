#!/bin/bash

# Check if the model name is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

# Get the model name from the command line argument
model_name="$1"

# Define the list of indices
indices=(6600 13200 19800 26400 32900)

# Define the output files for the merged results
merged_output_file_seq="results/mean_4000_retrieval_results_seq_allatom.csv"
merged_output_file_em="results/mean_4000_retrieval_results_em_allatom.csv"

# Initialize the headers and row content
header_seq="model"
row_seq="$model_name"
header_em="model"
row_em="$model_name"

# Arrays to store column names and values
declare -A column_names_seq
declare -A column_values_seq
declare -A column_names_em
declare -A column_values_em

# Loop through each index
for i in "${indices[@]}"; do
    # Define the input file name
    input_file="results/retrieval_${model_name}_${i}.csv"

    # Check if the input file exists
    if [[ ! -f "$input_file" ]]; then
        echo "File $input_file does not exist."
        continue
    fi

    # Extract the header from the input file and remove spaces
    file_header=$(head -n 1 "$input_file" | tr -d ' ')

    # Extract rows containing 'sequence' and compute column means
    sequence_means=$(awk -F, '/sequence/ {for(i=2;i<=NF;i++) sum[i]+=$i; count++} END {for(i=2;i<=NF;i++) printf "%s%s", sum[i]/count, (i==NF?ORS:OFS)}' OFS=, "$input_file")
    
    # Extract rows not containing 'sequence' and compute column means
    non_sequence_means=$(awk -F, '!/sequence/ {for(i=2;i<=NF;i++) sum[i]+=$i; count++} END {for(i=2;i<=NF;i++) printf "%s%s", sum[i]/count, (i==NF?ORS:OFS)}' OFS=, "$input_file")

    # Split the means into arrays
    IFS=',' read -r -a means_array_seq <<< "$sequence_means"
    IFS=',' read -r -a means_array_em <<< "$non_sequence_means"

    # Append the column names and means to the column_names and column_values arrays
    IFS=',' read -r -a file_header_array <<< "$file_header"
    for ((j=1; j<${#file_header_array[@]}; j++)); do
        base_name="${file_header_array[$j]}"
        full_name="${base_name}_${i}"
        column_names_seq["$base_name"]+="$full_name,"
        column_values_seq["$full_name"]="${means_array_seq[$((j-1))]}"
        column_names_em["$base_name"]+="$full_name,"
        column_values_em["$full_name"]="${means_array_em[$((j-1))]}"
    done

    echo "Processed $input_file."
done

# Function to remove trailing comma
remove_trailing_comma() {
    echo "${1%,}"
}

# Reorder columns for sequence
for base_name in "${!column_names_seq[@]}"; do
    header_seq+=",$(remove_trailing_comma "${column_names_seq[$base_name]}")"
done

# Reorder columns for non-sequence
for base_name in "${!column_names_em[@]}"; do
    header_em+=",$(remove_trailing_comma "${column_names_em[$base_name]}")"
done

# Build the row in the new order for sequence
for full_name in ${header_seq//,/ }; do
    if [[ "$full_name" != "model" ]]; then
        row_seq+=",${column_values_seq[$full_name]}"
    fi
done

# Build the row in the new order for non-sequence
for full_name in ${header_em//,/ }; do
    if [[ "$full_name" != "model" ]]; then
        row_em+=",${column_values_em[$full_name]}"
    fi
done

# Remove any remaining spaces from the headers
header_seq=$(echo "$header_seq" | tr -d ' ')
header_em=$(echo "$header_em" | tr -d ' ')

# Check if the merged output files already exist
if [[ ! -f "$merged_output_file_seq" ]]; then
    # Write the header and row to the merged output file if it doesn't exist
    echo "$header_seq" > "$merged_output_file_seq"
    echo "$row_seq" >> "$merged_output_file_seq"
else
    # Append only the row to the merged output file if it exists
    echo "$row_seq" >> "$merged_output_file_seq"
fi

if [[ ! -f "$merged_output_file_em" ]]; then
    # Write the header and row to the merged output file if it doesn't exist
    echo "$header_em" > "$merged_output_file_em"
    echo "$row_em" >> "$merged_output_file_em"
else
    # Append only the row to the merged output file if it exists
    echo "$row_em" >> "$merged_output_file_em"
fi

echo "Merged results written to $merged_output_file_seq and $merged_output_file_em."