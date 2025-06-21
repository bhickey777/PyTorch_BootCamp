# pytorch workflow

import torch
from torch import nn ## pytorch neural networks
import numpy as np
import matplotlib.pyplot as plt
import time

#Check pytorch version
print(torch.__version__)

# 1. Preparing an loading
# Can be images, videos, audio, podcasts, or texts

# Example 1 - Use a linear regression to make a straight line with known parameters
# Y = a + bX

weight = 0.7 # a is the intercept or the value of y when x is zero
bias = 0.3 # b is the slope of the line

# model will estimate the values of x and y.
start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim = 1)
print(X)

y = weight * X + bias

# View the first 10 sets
print(X[:10].squeeze())
print(y[:10].squeeze())
print(len(X), len(y))

# 2. Split the data into training (60-80% of data) and test sets (10-20%)
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:],y[train_split:]

#Check the lengths of each set 
print(len(X_train), len(y_train), len(X_test), len(y_test))

#make the training and test sets visual
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions = None):
    
# plot training data, test data, and compares predictions
#matplotlib scatter graph
    plt.figure(figsize=(10, 7))

# plot the training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label = "Training Data") # color of blue

#plot the test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test Data")

    if predictions is not None:
        #plot the predications
        plt.scatter(test_data, predictions, c="r", s=4, label="Predications")

    plt.legend(prop={"size": 14})

plot_predictions();
