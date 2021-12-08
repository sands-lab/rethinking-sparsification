# Logistic Regression experiments

## Data and Setup
Download the respective [LIBSVM datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) in .txt format in a new `datasets` directory. Also, create `dump` and `plot` directories to store the models and plots respectively.

## Code organization

-   [distrib_algs.py](distrib_algs.py) contains implementation of methods.
-   [utils.py](utils.py) contains useful tools for saving results and making plots.
-   [functions.py](functions.py) contains implementation of oracles and sparsifiers.

Use the jupyter notebooks to reproduce our experiments. 

## Acknowledgements
Our code is built on top of the [ef-sigma-k code](https://github.com/eduardgorbunov/ef_sigma_k). If you use the  `logistic-regression` section of this repository, please cite the paper [Linearly Converging Error Compensated SGD](https://arxiv.org/abs/2010.12292)

    @inproceedings{gorbunov2020linearly,
    title={{Linearly Converging Error Compensated SGD}},
    author={Gorbunov, E. and Kovalev, D. and Makarenko, D. and Richtarik, P.},
    booktitle={NeurIPS 2020 - Advances in Neural Information Processing Systems},
    year={2020},
    url = {https://arxiv.org/abs/2010.12292}
    }

