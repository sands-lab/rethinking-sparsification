# CNN and LSTM experiments

## Code organization
### A few pointers

-   [train.py](train.py) is the entrypoint.
-   [gradient_reducers.py](gradient_reducers.py) implements communication algorithms.
-   [hard-threshold sparsifier](gradient_reducers.py#L179).
-   Optimization problems can be found under [tasks/](tasks/__init__.py).

Use [run.sh](run.sh) that executes [run.py](run.py) to train on multiple GPUs using `torch.distributed`.

### Reproducibility
All the hyperparameters used in the experiments are provided in our paper's appendix. 
## Acknowledgements

Our code is built on top of the [PowerSGD code](https://github.com/epfml/powersgd). If you use the  `cnn-lstm` section of this repository, please cite the [PowerSGD paper](https://arxiv.org/abs/1905.13727)

    @inproceedings{vkj2019powerSGD,
      author = {Vogels, Thijs and Karimireddy, Sai Praneeth and Jaggi, Martin},
      title = "{{PowerSGD}: Practical Low-Rank Gradient Compression for Distributed Optimization}",
      booktitle = {NeurIPS 2019 - Advances in Neural Information Processing Systems},
      year = 2019,
      url = {https://arxiv.org/abs/1905.13727}
    }
