# MNIST_mlhub
Starting point for workshop on how to open source an MNIST model with `mlhub`

`mlhub` packages ML models so they are easily accessed, run, rebuilt and deployed. 

First, to train the model, run `python model.py`. The output of this training script is a model saved as `best.pth.tar` and a set of weights `metrics_val_best_weights.json`. 

Given this model checkpoint, can we reload the model into a format an end user can get predictions from? 

If you don't manage to package the model from the checkpoint, `vgg.py` contains an example of how to load a pretrained model stored on aws and use the pretrained model to predict the labels of an image. 


# General mlhub concepts 

To make a git project repository `mlhub`-friendly, simply include a `MLHUB.yaml` configuration file. To help others understand your model, ideally also include a script to 1. get data eg. `dataloader.py` and 2. demonstrate what your model does eg. `demo.py`

A user can then install and run the pre-built model. 

1. `$ pip install mlhub`    
2. `$ ml install MNIST_mlhub`     
3. `$ ml configure MNIST_mlhub`      
4. `$ ml demo MNIST_mlhub`
