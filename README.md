# Rethinking Gradient Sparsification as Total error minimization

## Deep Neural Network experiments
Our DNN experiments consist of three tasks: Image Classification using CNNs, Language Modelling using LSTMs, and Recommendation using NCF. Image Classification and Language Modelling experiments are in the `cnn-lstm` directory, and Recommendation experiment is in the `ncf` directory. 

## Logistic Regression experiments
Our logistic regression experiment is implemented in the `logistic-regression` directory.
## Create the Conda environment

To install the necessary dependencies, use the provided `environment.yml` to create a Conda envrironment by running the following command.

```bash
$ conda env create --prefix ./env --file environment.yml
```

Once the new environment has been created you can activate the environment with the following 
command.

```bash
$ conda activate ./env
```
<!--
Note that the `env` directory is *not* under version control as it can always be re-created from 
the `environment.yml` file as necessary.

### Updating the Conda environment

If you add (remove) dependencies to (from) the `environment.yml` file after the environment has 
already been created, then you can update the environment with the following command.

```bash
$ conda env update --prefix ./env --file environment.yml --prune
```

### Listing the full contents of the Conda environment

The list of explicit dependencies for the project are listed in the `environmen.yml` file. Too see the full lost of packages installed into the environment run the following command.

```bash
conda list --prefix ./env
```
-->

# Reference

If you use this code, please cite the following [paper](https://arxiv.org/abs/2108.00951)

    @inproceedings{sda+2021rethinking-sparsification,
      author = {Sahu, Atal Narayan and Dutta, Aritra and Abdelmoniem, Ahmed M. and Banerjee, Trambak and Canini, Marco and Kalnis, Panos},
      title = "{Rethinking gradient sparsification as total error minimization}",
      booktitle = {NeurIPS 2021 - Advances in Neural Information Processing Systems},
      year = 2021,
      url = {https://arxiv.org/abs/2108.00951}
    }