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

import requests
from pathlib import Path

#Download helper functions from Learn PyTorch repo if not already downloaded
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")    
    with open("helper_function.py", "wb") as f:
        f.write(request.content)

from helper_function import plot_predictions, plot_decision_boundary

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Can use this when using CUDA
torch.cuda.manual_seed(RANDOM_SEED)

#make the code device agnostic 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

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

#set the training and test data to the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

## 2. Build the associated model
## Will classify between the two types of circles

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        #Use nn.Linear() for model parameters
        #Create 2 nn.linear layers capable of handling the shapes of our data
        #takes in 2 features and upscales to 5
        self.layer_1 = nn.Linear(in_features=2, out_features=5) 
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x-> layer1->layer2->output
        return self.layer_2(self.layer_1(x))

#Calculate accuracy - out of 100 examples what percentage does out model get right
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

#Let's replicate the model using nn.Sequential
#steps thru each layer in a sequential fashion
model0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5, bias=True),
    nn.Linear(in_features=5, out_features=1, bias=True)
).to(device)
    
#Our raw outputs for this model are logits. These need to be 
#converted into prediction probabilities using a sigmoid function.
#this then needs to be converted to 1s or 0x depending on their
#values

# doing a raw sigmoid function activation
model0.eval()
with torch.inference_mode():
    y_logits = model0(X_test.to(device))[:5]
    y_pred_probs = torch.sigmoid(y_logits)
    y_preds = torch.round(y_pred_probs)
    # in full
    y_pred_labels = torch.round(torch.sigmoid(model0(X_test.to(device))[:5]))
    print(y_pred_labels)

## option(2) using a pytorch loss function and optimizer

loss_fn1 = nn.BCEWithLogitsLoss()#uses sigmoid activation function
loss_fn2 = nn.BCELoss()

#Setup an optimizer (ways to adjust the parameters)
optimizer = torch.optim.SGD(
        params = model0.parameters(), 
        lr=0.01, #learning rate is a hyperparameter (higher learning more adjustment)
        momentum=0.9) #

epochs = 1000
for epoch in range(epochs):
    model0.train()
    y_logits = model0(X_train).squeeze()  #1
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss/accuracy

    #loss function in this case expects raw logits
    loss1 = loss_fn1(y_logits, y_train)      #2
    #loss function in this case expects predictions
    #loss2 = loss_fn2(torch.sigmoid(y_logits), y_train)

    acc = accuracy_fn(y_true =y_train,y_pred=y_pred)
    optimizer.zero_grad()                 #3
    loss1.backward()                       #4
    # loss2.backward()
    optimizer.step()                      #5

    model0.eval()
    with torch.inference_mode():
        test_logits = model0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn1(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        # print out results
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss1:.5f} | Acc: {acc:.2f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc: .2f}%")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model0, X_test, y_test)

# plt.show()