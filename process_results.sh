#!/bin/bash

for file in results_checkpoints/downstream_results/*_results.csv; do
    taskname=$(basename "$file" _results.csv)
    if [ "$taskname" = "ThermoStability_MLP" ]; then
        
        (head -n 1 "$file" && tail -n +2 "$file" | sort -t',' -k1,1 -k4,4nr | awk -F',' '!seen[$1]++ {print $0}' | sort -t',' -k4,4nr) > "results_checkpoints/downstream_results/${taskname}_results_final.csv"
        #awk -F, '{OFS="\t"; $1=$1; print}' ThermoStability_MLP_results_final.csv | column -t -s $'\t' > ThermoStability_MLP_results_aligned.csv
    else
     
        (head -n 1 "$file" && tail -n +2 "$file" | sort -t',' -k1,1 -k2,2nr | awk -F',' '!seen[$1]++ {print $0}' | sort -t',' -k2,2nr) > "results_checkpoints/downstream_results/${taskname}_results_final.csv"
        #awk -F, '{OFS="\t"; $1=$1; print}' ${taskname}_MLP_results_final.csv | column -t -s $'\t' > ${taskname}_MLP_results_aligned.csv
    fi

done

echo "Processing complete."