#!/bin/bash

# Define the task name and file path
#!/bin/bash

# Define the task name and file path
taskname=$1
num=$2
file="results_checkpoints_no_sweep_${num}/downstream_results/${taskname}_results.csv"

# First pass: Find the maximum value in column 4 for each unique value in the first column
awk -F',' 'NR > 1 && $7 != "nan                 " && $2 != "nan                 " {if (!($1 in max) || $7 > max[$1]) max[$1] = $7} END {for (key in max) print key "," max[key]}' "$file" > max_values.txt
# Second pass: Print the header and all lines where the value in the first column matches and column 4 is the maximum for that group
{
    head -n 1 "$file"
    awk -F',' 'NR == FNR {max[$1] = $2; next} NR > 1 && $7 == max[$1]' max_values.txt "$file"
} > "results_checkpoints_no_sweep_${num}/downstream_results/${taskname}_results_final_valid.csv"

# Clean up temporary file
rm max_values.txt



    #     #(head -n 1 "$file" && tail -n +2 "$file" | sort -t',' -k1,1 -k7,7nr -k4,4nr | awk -F',' '!seen[$1]++ {print $0}' | sort -t',' -k4,4nr) > "results/downstream_results/${taskname}_results_final_valid.csv"
    # elif [ "$taskname" = "DeepLoc10_MLP" ] || [ "$taskname" = "TopEnzyme_MLP" ]; then
    #     #(head -n 1 "$file" && tail -n +2 "$file" | sort -t',' -k1,1 -k4,4nr -k2,2nr | awk -F',' '!seen[$1]++ {print $0}' | sort -t',' -k2,2nr) > "results/downstream_results/${taskname}_results_final_valid.csv"
    # elif [ "$taskname" = "DeepLoc2_MLP" ] || [ "$taskname" = "HumanPPI_MLP" ] || [ "$taskname" = "MetalIonBinding_MLP" ]; then
    #     #(head -n 1 "$file" && tail -n +2 "$file" | sort -t',' -k1,1 -k5,5nr -k2,2nr | awk -F',' '!seen[$1]++ {print $0}' | sort -t',' -k2,2nr) > "results/downstream_results/${taskname}_results_final_valid.csv"
    # else
    #     #(head -n 1 "$file" && tail -n +2 "$file" | awk -F, '$1 !~ /[nN][aA][nN]/ && $2 !~ /[nN][aA][nN]/ && $1 != "" && $2 != "" && $1 != "NA" && $2 != "NA"' | sort -t',' -k1,1 -k3,3nr -k2,2nr | awk -F, '!seen[$1]++ {print $0}' | sort -t',' -k2,2nr) > "results/downstream_results/${taskname}_results_final_valid.csv"
    #fi
#done
echo "Processing complete."
