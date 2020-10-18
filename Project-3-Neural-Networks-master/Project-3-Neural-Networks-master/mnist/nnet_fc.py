#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
print(sys.path)
from utils import get_MNIST_data
from train_utils import batchify_data, run_epoch, train_model

def main():
    # Load the dataset
    # num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Baseline
    test1 = {"name":"Baseline", "activation": nn.ReLU(), "batch_size": 32, "lr": 0.1, "momentum": 0}
    # Batch size 64
    test2 = {"name":"Batch size 64", "activation": nn.ReLU(), "batch_size": 64, "lr": 0.1, "momentum": 0}
    # Learning Rate 0.001
    test3 = {"name":"Learning Rate 0.001",  "activation": nn.ReLU(), "batch_size": 32, "lr": 0.01, "momentum": 0}
    # Momentum 0.9
    test4 = {"name":"Momentum 0.9", "activation": nn.ReLU(), "batch_size": 32, "lr": 0.1, "momentum": 0.9}
    # LeakyReLU activation
    test5 = {"name":"LeakyReLU activation", "activation": nn.LeakyReLU(negative_slope=0.01), "batch_size": 32, "lr": 0.1, "momentum": 0}

    tests = [test1, test2, test3, test4, test5]
    results =[]
    for test in tests:
        print("\nTest {}".format(test["name"]))
        torch.manual_seed(12321)  # for reproducibility
        # model = nn.Sequential(
        #       nn.Linear(784, 10),
        #       test["activation"],
        #       nn.Linear(10, 10),
        #     )
        model = nn.Sequential(
              nn.Linear(784, 128),
              test["activation"],
              nn.Linear(128, 10),
            )
        # Split dataset into batches
        train_batches = batchify_data(X_train, y_train, test["batch_size"])
        dev_batches = batchify_data(X_dev, y_dev, test["batch_size"])
        test_batches = batchify_data(X_test, y_test, test["batch_size"])

        train_model(train_batches, dev_batches, model, lr=test["lr"], momentum=test["momentum"])
        ## Evaluate the model on test data
        loss, accuracy = run_epoch(test_batches, model.eval(), None)
        results.append({"loss":loss, "accuracy": accuracy})
    
    print("")
    for i in range(len(tests)):
        print("Test {}".format(tests[i]["name"]))
        print ("Loss on test set:"  + str(results[i]["loss"]) + " Accuracy on test set: " + str(results[i]["accuracy"]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
