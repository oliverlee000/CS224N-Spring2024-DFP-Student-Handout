#!/bin/bash

# Define an array of commands
commands=(
    "python3 multitask_classifier.py --ensembling y --use_gpu --fine-tune-mode full-model --cos_sim_loss y --balance_sampling 6 --sst_layers 2 --para_layers 2 --sts_layers 0 --epochs 6"
    "python3 multitask_classifier.py --ensembling y --use_gpu --fine-tune-mode full-model --neg_rankings_loss y --balance_sampling 6 --sst_layers 2 --para_layers 2 --sts_layers 0 --epochs 6"
    "python3 multitask_classifier.py --ensembling y --use_gpu --fine-tune-mode full-model --neg_rankings_loss y --balance_sampling 6 --sst_layers 2 --para_layers 2 --sts_layers 0 --cos_sim_loss y --epochs 6"
    "python3 multitask_classifier.py --ensembling y --use_gpu --fine-tune-mode full-model --balance_sampling 6 --sst_layers 2 --para_layers 2 --sts_layers 0 --pearson_loss y --epochs 6"
    "python3 multitask_classifier.py --ensembling y --use_gpu --fine-tune-mode full-model --neg_rankings_loss y --balance_sampling 6 --sst_layers 2 --para_layers 2 --sts_layers 0 --cos_sim_loss y --pearson_loss y --epochs 6"
)

# Define an array of corresponding output file names
output_files=(
    "cosineLoss.txt"
    "NRLAlone.txt"
    "CosNRL.txt"
    "Pearson.txt"
    "All.txt"
)

# Loop through commands and execute them
for i in "${!commands[@]}"; do
    command="${commands[$i]}"
    output_file="${output_files[$i]}"
    
    echo "Running command: $command"
    $command > "$output_file" 2>&1
    echo "Output saved to $output_file"
done
