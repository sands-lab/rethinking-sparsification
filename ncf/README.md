# Neural Collaborative Filtering (NCF) experiment

## Code organization

The [ncf.py](ncf.py) script contains most of the training and validation logic. Data loading and preprocessing code is located in [dataloading.py](dataloading.py).
The model architecture is defined in [neumf.py](neumf.py). Some initial data preprocessing is located in [convert.py](convert.py).

## Getting the data

The NCF model was trained on the ML-20m dataset.
For each user, the interaction with the latest timestamp was included in the test set and the rest of the examples are used as the training data. 

This repository contains the `./prepare_dataset.sh` script which will automatically download and preprocess the training and validation datasets. 
By default, data will be downloaded to the `/data` directory. The preprocessed data will be placed in `/data/cache`.

## Training process
Use [run.sh](run.sh) that executes [ncf.py](ncf.py) to train on multiple gpus using `torch.distributed`.

The main result of the training are checkpoints stored by default in `/data/checkpoints/`. This location can be controlled
by the `--checkpoint_dir` command-line argument.

The validation metric is Hit Rate at 10 (HR@10) with 100 test negative samples. This means that for each positive sample in 
the test set 100 negatives are sampled. All resulting 101 samples are then scored by the model. If the true positive sample is
among the 10 samples with highest scores we have a "hit" and the metric is equal to 1, otherwise it's equal to 0.
The HR@10 metric is the number of hits in the entire test set divided by the number of samples in the test set. 

## Reproducibility
All the hyperparameters used in the experiments are provided in our paper's appendix. 

## Acknowledgements
Our code is built on top of the `NCF` code in [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF). We are grateful to the authors of the original code.