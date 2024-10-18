#!/bin/bash
device=$1
epoch=$2
weather=$3
checkpoint_path=./workdirs/MonoWAD/checkpoint/MonoWAD_$epoch.pth
CUDA_VISIBLE_DEVICES=$device python3 scripts/eval.py --config=config/config.py --experiment_name=$experiment_name --checkpoint_path=$checkpoint_path --weather=$weather 2>&1 | tee test_log.txt