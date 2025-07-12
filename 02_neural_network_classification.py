# Neural Network classification with Pytorch
# Uses a toy dataset which is used for experimentation
# and get a handle on the fundamentals

import torch
from torch import nn ## pytorch neural networks
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from pathlib import Path

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

## Make 1000 samples
n_samples = 1000

# Create Circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(f"First 5 samples of X: {X[:5]}")
print(f"First 5 samples of y: {y[:5]}")

#Make a dataframe of circle data where the first column of values
# in X is X1, the second column of values in X is X2, and
# y will be the label

circles = pd.DataFrame({"X1": X[:,0], #first column
                        "X2": X[:,1], #second column
                        "label": y
                        })

print(circles.head(10))

#Plot the data
plt.scatter(x=X[:,0], # x axis
            y=X[:,1], # y axis
            c = y, # color with the labels
            cmap=plt.cm.RdYlBu)
# plt.show()

print(X.shape) # 1000 samples with 2 features
print(y.shape) # 1000 samples of a scalar

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]

print(f"X Sample: {X_sample} \ny sample: {y_sample}")
print(f"X Sample shape: {X_sample.shape}\n y sample shape: {y_sample.shape}")

print(type(X), X.dtype) #numpy arrays
print(type(y), y.dtype) #numpy arrays

# turn the data into tensors for input and output shapes
X = torch.from_numpy(X).type(torch.float) # make them float32
y = torch.from_numpy(y).type(torch.float) # make them float32

#split the training into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, #20% of data will be test 
                                                    random_state=42)

# Check the size of the train and test data
print(X_test.size(), X_train.size())
print(y_test.size(), y_train.size())