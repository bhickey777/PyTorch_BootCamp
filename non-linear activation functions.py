import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from helper_function import accuracy_fn
from helper_function import plot_predictions, plot_decision_boundary

A = torch.arange(-10, 10, 1, dtype=torch.float32)

#Visualize the tensor

#plt.plot(A)
# plt.show()

# plt.plot(torch.relu(A))

#Custom relu function
def relu(x)-> torch.Tensor:
    return torch.maximum(torch.tensor(0), x)

# plt.plot(relu(A))
# plt.show()

#Custom Sigmoid function
def sigmoid(x)-> torch.Tensor:
    return  1 / (1 + torch.exp(-x))

plt.plot(sigmoid(A))
plt.show()
