#!/bin/bash
device=$1
experiment_name=$2
log_file=log_$experiment_name.txt
CUDA_VISIBLE_DEVICES=$device python3 -u scripts/train.py --config=config/config.py --experiment_name=$experiment_name 2>&1 | tee $log_file