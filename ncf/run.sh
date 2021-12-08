#!/bin/bash

# Set the variables below
export DATA_DIR= # Path to test and training data files
export SHARED_PATH= #a shared path visible to all processes for rendezvous
export RANK= #rank of this process
export WANDB_API_KEY= #wandb api key
export WANDB_ENTITY= #wandb entity

echo $DATA_DIR

########################## Exact ###################################################
reducer='exact'
python ncf.py --mode train --data=${DATA_DIR} --backend=nccl --shared_path=${SHARED_PATH} --reducer=$reducer --seed=1 --rank=$RANK --use_wandb
####################################################################################

########################### Top-k ###################################################
# reducer='topk'
# python ncf.py --mode train --data=${DATA_DIR} --backend=nccl --shared_path=${SHARED_PATH} --reducer=$reducer --seed=1 --comp_ratio=0.095 --rank=$RANK --use_wandb
#######################################################################################

########################### Threshold ################################################
# reducer='thresh'
# python ncf.py --mode train --data=${DATA_DIR} --backend=nccl --shared_path=${SHARED_PATH} --reducer=$reducer --seed=1 --thresh=1.3e-6 --rank=$RANK --use_wandb
########################################################################################

########################### Top-k ################################################
# reducer='topk'
# python ncf.py --mode train --data=${DATA_DIR} --backend=nccl --shared_path=${SHARED_PATH} --reducer=$reducer --seed=1 --comp_ratio=0.095 --rank=$RANK --use_wandb
########################################################################################