## Introduction to PyTorch Tensors

import torch

# Can also import the common abbreviation "nn" for "Neural Networks"
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# validate that pytorch is installed
x = torch.rand(5, 3)
print(x)

# Almost everything in PyTorch is called a "Module" (you build neural networks by stacking together Modules)
this_is_a_module = nn.Linear(in_features=1,out_features=1)
print(type(this_is_a_module))

#Introduction to tensors
#scalar tensor
scalar_tensor = torch.tensor(7)
print(scalar_tensor)
print(scalar_tensor.ndim)

#get tensor back as a python number
print(scalar_tensor.item())

#get tensor as a vector
vector_tensor = torch.tensor([[7, 8, 9], [5, 6, 7], [8, 10, 11]])
print(vector_tensor)
print(vector_tensor.ndim)
print(vector_tensor.shape)

#TENSOR (float) Note: Use uppercase for tensors that are matrices or higher dimensions
TENSOR = torch.tensor([[[7, 8, 9], [5, 6, 7], [8, 10, 11]]], dtype=torch.float32)
print(TENSOR)
print(TENSOR.dtype)
print(TENSOR.shape)
print(TENSOR.ndim)
print(TENSOR[0])
print(TENSOR[1:1])

#Random tensors - useful for initializing weights in neural networks
#Random numbers are then adjusted to better represent the data
#Create a random tensor of size (3, 4) with values between 0 and
random_tensor = torch.rand(2, 3, 4)
print(random_tensor)
print(random_tensor.ndim)
print(random_tensor.shape)

#create random tensor with similar shape to an image tensor
#Image tensors are usually 3D (height, width, color channels)
random_image_tensor = torch.rand(size=(224, 224, 3))  # height, width, color channels (RGB)
print(random_image_tensor)
print(random_image_tensor.shape)
print(random_image_tensor.ndim)

### Zeros and ones tensors (used for masks or filtering)
#Create a tensor of all zeros
zeros_tensor = torch.zeros(size=(3, 4))
print(zeros_tensor)
print(zeros_tensor.shape)

#say you wanted to ignore the first column of a tensor
#You could create a mask tensor of zeros and ones
mask_tensor = torch.zeros(size=(3, 4))
mask_tensor[1] = 1  # Set the first column to 1
print(mask_tensor)
print(mask_tensor.shape)

#now multiply this mask tensor with the previous random tensor
masked_tensor = random_tensor * mask_tensor
print(masked_tensor)

#create a tensor based on a range of numbers
#torch.arange(start, end, step)
arange_tensor = torch.arange(start=0, end=1000, step=77)
print(arange_tensor)

#creating tensors like numpy arrays
#torch.tensor() can convert numpy arrays to tensors
numpy_array = np.array([1, 2, 3, 4, 5])
tensor_from_numpy = torch.tensor(numpy_array)
print(tensor_from_numpy)

#creating tensors like an existing shape
#torch.empty_like() creates a tensor with the same shape as another tensor
existing_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
empty_like_tensor = torch.empty_like(existing_tensor)
print(empty_like_tensor)

zeros_like = torch.zeros_like(existing_tensor)
print(zeros_like)

torch.ones_like = torch.ones_like(existing_tensor)
print(torch.ones_like)