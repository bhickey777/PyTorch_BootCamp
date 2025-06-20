import torch
import time

# Can also import the common abbreviation "nn" for "Neural Networks"
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## splicing or selecting data from tensors (similar to numpy)
x = torch.arange(1, 10).reshape(1, 3, 3)
print(f"my tensor: {x}")

print(x[0])
print(x[0][0])
print(x[0][0])
print(x[0][-1])

print(x[0][0][0])
print(x[0][1][1])
print(x[0][1][2])

#print out 9 that is in the tensor
print(x[0][2][2])

#print out next to last element in last row
print(x[0][2][:-1])

# You can use ":" to select "all of a target dimension"
print(x[:, 0])

#Print all of the values of 0th and 1st dimensions but only index 1 of 2nd dimensions
print(x[: :, 1])

#Get index of 0 of 0th dimension and 1st dimension and all values of 2nd dimension
print(x[0, 0, :])

#PyTorch Tensors and Numpy
# Usually will start with data in numpy from pandas and you need to get it into tensors
# torch.from_numpy(ndarray)
# torch.Tensor.numpy()

#Numpy array to tensor (default dtype for pytorch is float32)
x = np.arange(1., 8.)
y = torch.from_numpy(x).type(torch.float64)

print(x)
print(f"numpy datatype: {x.dtype}")
print(y)
print(f"tensor datatype: {y.dtype}")

# change the value of x doesnt change the tensor
x = x + 1
print(y)

# go from tensor back to numpy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(numpy_tensor.dtype)


