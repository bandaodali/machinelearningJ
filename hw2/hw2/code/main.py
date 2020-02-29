""" Main file. This is the starting point for your code execution.
You shouldn't have to make any code changes here.
"""


import os
import json
import pickle
import argparse as ap

import numpy as np

import models
from data import load_data


def get_args():
    p = ap.ArgumentParser()

    # Meta arguments
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                    help="Operating mode: train or test.")
    p.add_argument("--train-data", type=str, help="Training data file")
    p.add_argument("--test-data", type=str, help="Test data file")
    p.add_argument("--model-file", type=str, required=True,
                    help="Where to store and load the model parameters")
    p.add_argument("--predictions-file", type=str,
                    help="Where to dump predictions")
    p.add_argument("--algorithm", type=str,
                   choices=['pegasos', 'kernel-pegasos'],
                    help="The type of model to use.")
    p.add_argument("--pegasos-projection", type=bool, default=False,
                    help="Perform gradient projection on weights update.")

    # Model Hyperparameters
    p.add_argument("--pegasos-lambda", type=float, default=1e-3,
                    help="Model learning rate")
    p.add_argument("--train-epochs", type=int, default=5,
                    help="Number of epochs to train the model")
    p.add_argument("--kernel-degree", type=int, default=1,
                    help="Number of degrees for polynomial kernel function.")
   
    return p.parse_args()


def check_args(args):
    mandatory_args = {'mode', 'model_file', 'test_data', 'train_data',
                      'predictions_file', 'algorithm',
                      'train_epochs'}

    if not mandatory_args.issubset(set(dir(args))):
        raise Exception(("You're missing essential arguments!"
                         "We need these to run your code."))

    if args.model_file is None:
        raise Exception("--model-file should be specified in either mode")

    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified during training")
        if args.model_file is None:
            raise Exception("--model-file should be specified during training")
        if args.train_data is None:
            raise Exception("--train-data should be specified during training")
        elif not os.path.exists(args.train_data):
            raise Exception("data file specified by --train-data does not exist.")
    elif args.mode.lower() == "test":
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified during testing")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")
        if args.test_data is None:
            raise Exception("--test-data should be specified during testing")
        elif not os.path.exists(args.test_data):
            raise Exception("data file specified by --test-data does not exist.")
    else:
        raise Exception("invalid mode")


def test(args):
    """ Make predictions over the input test dataset, and store the predictions.
    """
    # load dataset and model
    X, _ = load_data(args.test_data)
    model = pickle.load(open(args.model_file, 'rb'))

    # predict labels for dataset
    preds = model.predict(X)
    
    # output model predictions
    np.savetxt(args.predictions_file, preds, fmt='%d')


def train(args):
    """ Fit a model's parameters given the parameters specified in args.
    """
    X, y = load_data(args.train_data)

    # build the appropriate model
    if args.algorithm == "pegasos":
        model = models.Pegasos(nfeatures=X.shape[1], lmbda=args.pegasos_lambda)
    elif args.algorithm == "kernel-pegasos":
        model = models.KernelPegasos(nexamples=X.shape[0], nfeatures=X.shape[1], lmbda=args.pegasos_lambda, kernel_degree=args.kernel_degree)
    else:
        raise Exception("Algorithm argument not recognized")

    # Run the training loop
    for epoch in range(args.train_epochs):
        model.fit(X=X, y=y)

    # Save the model
    pickle.dump(model, open(args.model_file, 'wb'))


if __name__ == "__main__":
    args = get_args()
    check_args(args)

    if args.mode.lower() == 'train':
        train(args)
    elif args.mode.lower() == 'test':
        test(args)
    else:
        raise Exception("Mode given by --mode is unrecognized.")
