# MNIST_mlhub
Starting point for workshop on how to open source an MNIST model with `mlhub`

`mlhub` packages ML models so they are easily accessed, run, rebuilt and deployed. To make a git project repository `mlhub`-friendly, simply include a `MLHUB.yaml` configuration file. To help others understand your model, ideally also include a script to 1. get data eg. `dataloader.py` and 2. demonstrate what your model does eg. `demo.py`

A user can then install and run the pre-built model. 

`$ pip install mlhub\
 $ ml install MNIST_mlhub\
 $ ml configure MNIST_mlhub\
 $ ml demo MNIST_mlhub`
