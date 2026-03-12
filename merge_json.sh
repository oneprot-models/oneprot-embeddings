#!/bin/bash

# Define the base directory
BASE_DIR="/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/OPI/OPI_data"

# Define output files
TEST_OUTPUT="merged_test_files.json"
TRAIN_OUTPUT="merged_train_files.json"

# Define subfolder to exclude (just the folder name)
EXCLUDE_FOLDER="KM"

# Remove output files if they already exist
rm -f "$TEST_OUTPUT" "$TRAIN_OUTPUT"

# Create empty JSON arrays in the output files
echo "[]" > "$TEST_OUTPUT"
echo "[]" > "$TRAIN_OUTPUT"

# Function to merge a JSON/JSONL file with an existing JSON array file
merge_json_file() {
    local input_file=$1
    local output_file=$2
    
    # Check if file path contains the excluded folder name
    if [[ "$input_file" == *"/$EXCLUDE_FOLDER/"* ]]; then
        echo "Skipping excluded file: $input_file"
        return
    fi
    
    # Check if the input file is empty
    if [ ! -s "$input_file" ]; then
        echo "Skipping empty file: $input_file"
        return
    fi
    
    # Check if file is JSONL (one JSON object per line)
    if grep -q "^{" "$input_file"; then
        # Convert JSONL to JSON array if needed
        local temp_file=$(mktemp)
        echo "[" > "$temp_file"
        sed -e 's/$/,/' -e '$s/,$//' "$input_file" >> "$temp_file"
        echo "]" >> "$temp_file"
        input_file=$temp_file
    fi
    
    # Combine with existing output
    local temp_output=$(mktemp)
    jq -s '.[0] + .[1]' "$output_file" "$input_file" > "$temp_output" 2>/dev/null
    
    # Check if jq succeeded
    if [ $? -eq 0 ]; then
        mv "$temp_output" "$output_file"
        echo "Successfully merged $input_file"
    else
        echo "Error merging $input_file - skipping"
        rm -f "$temp_output"
    fi
    
    # Clean up temp file if created
    if [ "$input_file" != "$1" ]; then
        rm -f "$input_file"
    fi
}

echo "Finding and merging test files (excluding KM subfolder)..."
# Find all JSON and JSONL files containing "test" in their name
find "$BASE_DIR" -type f \( -name "*test*.json" -o -name "*test*.jsonl" \) | while read file; do
    # Skip files in the KM directory
    if [[ "$file" != *"/$EXCLUDE_FOLDER/"* ]]; then
        echo "Processing test file: $file"
        merge_json_file "$file" "$TEST_OUTPUT"
    else
        echo "Skipping excluded file: $file"
    fi
done

echo "Finding and merging train files (excluding KM subfolder)..."
# Find all JSON and JSONL files containing "train" in their name
find "$BASE_DIR" -type f \( -name "*train*.json" -o -name "*train*.jsonl" \) | while read file; do
    # Skip files in the KM directory
    if [[ "$file" != *"/$EXCLUDE_FOLDER/"* ]]; then
        echo "Processing train file: $file"
        merge_json_file "$file" "$TRAIN_OUTPUT"
    else
        echo "Skipping excluded file: $file"
    fi
done

# Final report
echo "Merge complete!"
echo "Test files merged into: $TEST_OUTPUT"
echo "Train files merged into: $TRAIN_OUTPUT"
echo "Test file size: $(du -h $TEST_OUTPUT | cut -f1)"
echo "Train file size: $(du -h $TRAIN_OUTPUT | cut -f1)"