# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:43:53 2025

@author: Mert
"""

#!/bin/bash

# List of different flags for each run
FLAGS_LIST=(
"--d_hid 150 --d_emb 20 --n_layers 4 --dataset icu --device cuda:0 --phi_net masked --act elu --model_id 0 --device cuda:0 --norm None --early_stop 40 --cv_folds 5"
"--d_hid 150 --d_emb 20 --n_layers 4 --dataset fico --device cuda:1 --phi_net masked --act elu --model_id 0 --device cuda:1 --norm None --early_stop 40 --cv_folds 5"
"--d_hid 150 --d_emb 20 --n_layers 4 --dataset spambase --device cuda:2 --phi_net masked --act elu --model_id 0 --device cuda:2 --norm None --early_stop 40 --cv_folds 5"
"--d_hid 150 --d_emb 20 --n_layers 4 --dataset adult --device cuda:3 --phi_net masked --act elu --model_id 0 --device cuda:3 --norm None --early_stop 40 --cv_folds 5"
"--d_hid 150 --d_emb 20 --n_layers 4 --dataset icu --device cuda:0 --phi_net masked --act elu --model_id 0 --device cuda:0 --norm None --early_stop 40 --cv_folds 5 --norm layer"
"--d_hid 150 --d_emb 20 --n_layers 4 --dataset fico --device cuda:1 --phi_net masked --act elu --model_id 0 --device cuda:1 --norm None --early_stop 40 --cv_folds 5 --norm layer"
"--d_hid 150 --d_emb 20 --n_layers 4 --dataset spambase --device cuda:2 --phi_net masked --act elu --model_id 0 --device cuda:2 --norm None --early_stop 40 --cv_folds 5 --norm layer"
"--d_hid 150 --d_emb 20 --n_layers 4 --dataset adult --device cuda:3 --phi_net masked --act elu --model_id 0 --device cuda:3 --norm None --early_stop 40 --cv_folds 5 --norm layer"
"--d_hid 500 --d_emb 30 --n_layers 4 --dataset icu --device cuda:0 --phi_net masked --act elu --model_id 0 --device cuda:0 --norm None --early_stop 40 --cv_folds 5 --cont"
"--d_hid 500 --d_emb 30 --n_layers 4 --dataset fico --device cuda:1 --phi_net masked --act elu --model_id 0 --device cuda:1 --norm None --early_stop 40 --cv_folds 5 --cont"
"--d_hid 500 --d_emb 30 --n_layers 4 --dataset spambase --device cuda:2 --phi_net masked --act elu --model_id 0 --device cuda:2 --norm None --early_stop 40 --cv_folds 5 --cont"
"--d_hid 500 --d_emb 30 --n_layers 4 --dataset adult --device cuda:3 --phi_net masked --act elu --model_id 0 --device cuda:3 --norm None --early_stop 40 --cv_folds 5 --cont"
)

# Name prefix for each screen session
# E as in experiments
SESSION_PREFIX="E" 

# Path to the script
SCRIPT="main.py"

# Loop over the flags and start a new screen for each
for i in "${!FLAGS_LIST[@]}"; do
  SESSION_NAME="${SESSION_PREFIX}$i"
  FLAGS="${FLAGS_LIST[$i]}"
  
  echo "Launching $SCRIPT with flags: $FLAGS in screen: $SESSION_NAME"

  # Start a detached screen session that runs main.py with the given flags
  screen -dmS "$SESSION_NAME" 
  bash -c "conda activate torchrl"
  bash -c "python $SCRIPT $FLAGS; exec bash"
done