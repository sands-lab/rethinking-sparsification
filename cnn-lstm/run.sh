#!/bin/bash

# Set the below environment variables
export RANK= # enter the rank of the current node
export DATA= # Path to test and training data files
export WANDB_ENTITY= # enter your wandb entity
export WANDB_API_KEY= # Your wandb key
export OUTPUT_DIR= # Directoy for output files. 
export DIST_INIT= # Path of torch.distributed distributed init file for rendezvous
export USE_WANDB=0 # Whether to use wandb or not

########################################## Tasks #########################################

##### ResNet18-CIFAR10 #######
# task='resnet18'
##############################

##### ResNet18-CIFAR100#######
# task='resnet18_cifar100'
##############################

##### GoogleNet-CIFAR10 #######
# task='googlenet'
###############################

##### GoogleNet-CIFAR100 #######
# task='googlenet_cifar100'
################################

##### SENet18-CIFAR10 #######
# task='senet'
#############################

##### SENet18-CIFAR100 #######
# task='senet_cifar100'
##############################

###### LSTM-WikiText ###########
# task='wikitext2'
################################


########################################################################################

#################################### Reducers ##########################################
world_size=8
#########################


# Using the formula \lambda = \frac{1}{2\sqrt{k}} for the threshold. Here k is the expected per-iteration element transmission.
# For the ACCORDION experiments, k is derived from the minimum k used in ACCORDION for each model in Table 1 and 7.
# |-------------------------------------------------------------------------------|
# |                           CIFAR 10                                            |
# |-------------------------------------------------------------------------------|
# | Network   | Total paramaters (d) | k_min (Top-%)      |\frac{1}{2\sqrt{k_min}}|
# |-----------|----------------------|--------------------|-----------------------|
# | ResNet18  | 11173962             | 11174   (Top-0.1%) | 4.73 x 10e-3          |
# | SeNet     | 11260354             | 11260   (Top-0.1%) | 4.68 x 10e-3          |
# | GoogLeNet | 6166250              | 6166   (Top-0.1%)  | 6.37 x 10e-3          |
# |-------------------------------------------------------------------------------|

# |-------------------------------------------------------- ---------------------|
# |                             CIFAR 100                                        |
# |-----------|----------------------|------------------- |----------------------|
# | Network   | Total paramaters (d) | k_min (Top-%)      |\frac{2}{\sqrt{k_min}}|
# |-----------------------------------------------------------------------------|
# | ResNet18  | 11220132             | 11220   (Top-0.1%) | 4.72 x 10e-3        |
# | SeNet     | 11436256             | 11436   (Top-0.1%) | 4.68 x 10e-3        |
# | GoogLeNet | 6258500              | 6258    (Top-0.1%) | 6.32 x 10e-3        |
# |-----------------------------------------------------------------------------|

########### Exact ################
reducer='exact'
if ((${RANK} == 0)); then rm -f ${DIST_INIT}; fi
python run.py --world-size=$world_size --task='resnet18' --seed=1 --reducer=$reducer --rank=${RANK}
##############################

########### Threshold ############ 
# reducer='thresh'
# if ((${RANK} == 0)); then rm -f ${DIST_INIT}; fi
# python run.py --world-size=$world_size --task='resnet18' --seed=1 --reducer=$reducer --thresh=0.00473 --rank=${RANK}
##################################


########### Top-k ############
# reducer='topk'
# if ((${RANK} == 0)); then rm -f ${DIST_INIT}; else sleep 15; fi
# python run.py --world-size=$world_size --task='resnet18_cifar100' --seed=1 --reducer=$reducer --comp_ratio=0.001 --rank=${RANK}
###############################

### Entire model Top-k ########
# reducer='gtopk'
# if ((${RANK} == 0)); then rm -f ${DIST_INIT}; else sleep 15; fi
# python run.py --world-size=$world_size --task='resnet18_cifar100' --seed=1 --reducer=$reducer --comp_ratio=0.001 --rank=${RANK}
###############################

###### AccordionTopK #########
# reducer='accordiontopk'
# if ((${RANK} == 0)); then rm -f ${DIST_INIT}; fi
# python run.py --world-size=$world_size --task='resnet18' --seed=1 --reducer=$reducer --k_low=0.001 --k_high=0.01  --rank=${RANK}
#############################################################################################################################################
