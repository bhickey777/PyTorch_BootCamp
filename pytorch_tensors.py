## Introduction to PyTorch Tensors

import torch
import time

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

#tensor data-types
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False)
print(float_32_tensor.dtype)

# change data type to float 
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor.dtype)

#Tensor datatypes is one of the 3 big issues with deep learning(level of precision)
# 1. Tensors not right datatype
# 2. Tensors not right shape
# 3. Tensors not on the right device

# Examples of device parameters (in PyTorch):
# "cpu": Specifies the CPU.
# "cuda": Specifies the current GPU.
# "cuda:0": Specifies a specific GPU (e.g., the first one)

# When to Use requires_grad=True: (tracking of gradients)
# Model Parameters: Typically, you set requires_grad=True for the weights and biases of 
# your neural network layers as you want to learn these parameters during training.
# Inputs (Less Common): You would set requires_grad=True for input tensors only if 
# you need to calculate gradients with respect to the input itself, 
# which is less common in standard training scenarios. #

#getting tensor attributes (mixing data types)
int32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(float_32_tensor * int32_tensor)
print(f"Device Type: {float_32_tensor.device}")
print(f"Shape: {float_32_tensor.shape}")
print(f"Grad: {float_32_tensor.requires_grad}")

#manipulating tensors (operations)

### Tensor Operations include:
# 1. Addition
# 2. Subtraction
# 3. Multiplication
# 4. Division
# 5. Matrix Multiplication

tensor = torch.tensor([1, 2, 3])
tensor = tensor + 10
print(tensor)

tensor = tensor * 10
print(tensor)

tensor = tensor / 10
print(tensor)

tensor = tensor - 10
print(tensor)

tensor2 = torch.tensor([5, 4, 3], dtype=torch.float32)
tensor = tensor2 * tensor
print(tensor)

#Matrix Multiplication (can use element wise or dot-product)
# dot product is multiplying matching members and then sum

#matrix multiplication via torch will be faster then python operation

tensor = tensor.matmul(tensor2) # can also use @ sign but matmul is better

print(tensor)
print(tensor.shape)
print(tensor2.shape)
#Main rules for operations on large matrixes
# 1. The **inner dimensions must match (3, 2) @ (3, 2) won't work but (2, 3) @ (3, 2) will
# 2. The resulting matrix has the shape of the outer dimensions
try:
    tensor = tensor.matmul(tensor2)
except:
    print("Error due to mismatch of dimension")

tensor1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(tensor1.shape)
tensor2 = torch.tensor([[5, 6, 7], [8, 9, 10]])
print(tensor2.shape)

#new tensor has shape (3, 3)
print(tensor1.mm(tensor2).shape) #mm is an alias for matmul
           
#Finding the mean, max, min and sum of tensors
tensor1 = torch.arange(0, 100, 10)
print(tensor1.shape)

print(f"Sum: {torch.sum(tensor1)}")
print(f"Minimum: {torch.min(tensor1)}")
print(f"Maximum: {torch.max(tensor1)}")
print(f"Avg: {torch.mean(x.type(torch.float32))}") # for avg need floats or longs

#finding the positional mean and max(splicing)
x = torch.arange(1, 100, 10)

#argmin returns the index position of the minimum value
print(x.argmin(), x[0])
print(x.argmax(), x[9])

x = torch.rand(2, 3, 4)
print(x)
print(f"Min index: {x.argmin()}")
print(f"Max index: {x.argmax()}")

#Reshaping Viewing Sqeezing UnSqueezing and Stacking Tensors
# Reshaping - reshapes the input tensor to a defined shape
# View - return a view of an input tensor of a certain shape but keep same memory
# Stacking - combined multiple tensors on top of each other (vstack) or (hstack)
# Squeeze - removes all 1 dimensions from a tensor
# Unsqueeze - adds a 1 dimension to a tensor
# Permute - return a vie wof the input with dimensions permuted (swapped) in a fashion

x = torch.arange(1. , 12.)
print(x)

# Add an extra dimension
x_reshaped = x.reshape(11, 1)
print(x_reshaped)

x_reshaped = x.reshape(1, 11)
print(x_reshaped)

#Change the view. when you change view you change the original tensor
x_view = x.view(1, 11)
print(x_view)
x_view[:, 0] = 5
print(x_view)
print(x)

#Stack tensors on top of each other
y = torch.arange(1., 12.)
z1 = torch.stack([x, y], dim=1) # takes a list of tensors 
z2 = torch.stack([x, y], dim=0) # takes a list of tensors 
print(z1)
print(z2)

z3 = torch.vstack([x, y]) #same as dim = 0
print(z3)
z4 = torch.hstack([x, y])
print(z4)

#Squeeze and Unsqueeze
z = torch.squeeze(x)
print(z)

z = torch.unsqueeze(x, dim=1)
print(z)

# returns a view of the original tensor often used with images
x = torch.rand(size=(224, 224, 3)) # [height, width, color_channels (rgb)

#switch the color channels
z = x.permute(2, 0, 1)
print(x.shape)
print(z.shape)


