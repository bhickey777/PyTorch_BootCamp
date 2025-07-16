import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from helper_function import accuracy_fn
from helper_function import plot_predictions, plot_decision_boundary

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Can use this when using CUDA
torch.cuda.manual_seed(RANDOM_SEED)

#make the code device agnostic 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
      
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
#plt.show()

#Convert data to tensors and then train and test splits
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# 2. Split the data into training (60-80% of data) and test sets (10-20%)
train_split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train[:5])
print(X_train.shape)
print(X_test.shape)

print(y_train[:5])
print(y_train.shape)
print(y_test.shape)

class CircleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CircleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_features=output_size)
        self.relu = nn.ReLU()        


    def forward(self, x):
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        return x

# Example usage (assuming you have 3 input features and want a single output)
input_size = 2  # e.g., x, y, radius
hidden_size = 10
output_size = 1
model0 = CircleModel(input_size, hidden_size, output_size).to(device)

loss_fn = nn.BCEWithLogitsLoss()#uses sigmoid activation function

#set the training and test data to the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

#Setup an optimizer (ways to adjust the parameters)
optimizer = torch.optim.SGD(
        params = model0.parameters(), 
        lr=0.1, #learning rate is a hyperparameter (higher learning more adjustment)
        momentum=0.9) #

epochs = 1000
for epoch in range(epochs):
    model0.train()
    y_logits = model0(X_train).squeeze()  #1
    y_pred = torch.round(torch.sigmoid(y_logits))

    #loss function in this case expects raw logits
    loss = loss_fn(y_logits, y_train)      #2

    acc = accuracy_fn(y_true =y_train,y_pred=y_pred)
    optimizer.zero_grad()                 #3
    loss.backward()                       #4
    optimizer.step()                      #5

    model0.eval()
    with torch.inference_mode():
        test_logits = model0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        # print out results
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc: .2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model0, X_test, y_test)

plt.show()
