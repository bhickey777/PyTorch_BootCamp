# pytorch workflow (setting up model plotting predictions)

import torch
from torch import nn ## pytorch neural networks
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import time

#Check pytorch version
print(torch.__version__)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

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

# for complex datasets can use torch.utils.data.Dataset and torch.utils.data.DataLoader
# rather then generating random numbers

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

#3. Build the model for predicting the linear regression values
class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # can use random or torch.optim to optimize the data
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    # use the formula to provide the computation of the model
    # ideally model will run random weights and bias and come close to value
    # as prescribed by the linear regression formula
    # forward is invoked by Module for each cell defined by the parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
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
    return plt

def showPlot(file_name, plt):

    plt.savefig(file_name) 
    plt.show()

# plt = plot_predictions();
# showPlot('linear regression.png', plt)

#create a instance of your model
model_0 = LinearRegressionModel()
list_parms = list(model_0.parameters()) #this is the generator
parm_names = model_0.state_dict()

print(list_parms)
print(parm_names)

#based on our formula want the weights and bias to be (0.7, 0.3)

#can make predictions using 'torch.inference_mode() 
#predict 'y_test' based on 'x_test'

with torch.inference_mode(): #inference mode keeps the context of the gradients
    y_preds = model_0(X_test)
    print(y_preds)
    # compare to y_test
    print(y_test)

    plt = plot_predictions(predictions=y_preds);
    showPlot('linear regression grad.png', plt)
   
with torch.no_grad(): #no gradients are used
    y_preds = model_0(X_test)
    print(y_preds)
    # compare to y_test
    print(y_test)

    plt = plot_predictions(predictions=y_preds);
    showPlot('linear regression nograd.png', plt)

#4. How to train the model - move from some unknown to some known parameters
#   You can use a loss function or criterion to measure the effectiveness of a model 

#   Optimizer - takes into account the loss of a model and adjusts the models weights and bias 
#   to improve the loss function. 

#   Need a training and a testing loop in order to achieve this 

    print(model_0.state_dict())

    #Setup a loss function
    loss_fn = nn.L1Loss()
    print(loss_fn)
    #Setup an optimizer
    optimizer = torch.optim.SGD(
        params = model_0.parameters(), 
        lr=0.01, #learning rate is a hyperparameter (higher learning more adjustment)
        momentum=0.9) #
    