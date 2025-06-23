import torch
from torch import nn ## pytorch neural networks
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import time

from pathlib import Path

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

epoch_count = []
loss_values = []
test_loss_values = []

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

weight = 0.7 # a is the intercept or the value of y when x is zero
bias = 0.3 # b is the slope of the line

# model will estimate the values of x and y.
start = 0
end = 1
step = 0.02

# for complex datasets can use torch.utils.data.Dataset and torch.utils.data.DataLoader
# rather then generating random numbers

X = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * X + bias


# 2. Split the data into training (60-80% of data) and test sets (10-20%)
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:],y[train_split:]

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        #Use nn.Linear() for model parameters
        #One input maps to one output
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
model1 = LinearRegressionModelV2()

#set the model device
model1.to(device)
model1.eval()

with torch.inference_mode():
    y_preds = model1(X_test)
    print(y_preds)

loss_fn = nn.L1Loss()
#Setup an optimizer (ways to adjust the parameters)
optimizer = torch.optim.SGD(
        params = model1.parameters(), 
        lr=0.01, #learning rate is a hyperparameter (higher learning more adjustment)
        momentum=0.9) #

epochs = 100
for epoch in range(epochs):
    model1.train()
    y_preds = model1(X_train)
    loss = loss_fn(y_preds, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Test the results
    model1.eval()

    #Now with the model having been trained lets check with some test data
    with torch.inference_mode(): # turns off gradient tracking 

        # 1. do a forward pass thru the model
        y_preds_new = model1(X_test) 

        #  2. Calculate the loss on the test data 
        test_loss = loss_fn(y_preds_new, y_test)

    if epoch % 10 == 0: #every 10 epochs
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")





