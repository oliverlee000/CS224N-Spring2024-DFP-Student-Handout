#!/bin/bash

# Define the array of lora_size values
lora_sizes=(20 100 140 200 300)

# Loop through each lora_size value
for size in "${lora_sizes[@]}"
do
  # Construct the command
  command="python3 multitask_classifier.py --fine-tune-mode full-model --task sst --lora y --lora_size $size --epochs 5 --use_gpu"
  
  # Construct the output file name
  output_file="output_lora_size_${size}.txt"
  
  # Execute the command and save the output to the file
  $command > $output_file
  
  # Optional: print a message indicating the command has completed
  echo "Completed: $command"
done

