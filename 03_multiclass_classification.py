import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from helper_function import accuracy_fn
from helper_function import plot_predictions, plot_decision_boundary


RANDOM_SEED = 42
NUM_CLASSES = 4
NUM_FEATURES = 2
torch.manual_seed(RANDOM_SEED)

# Can use this when using CUDA
torch.cuda.manual_seed(RANDOM_SEED)

#make the code device agnostic 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

# create a toy multi-class dataset

n_samples = 1000
X, y = make_blobs(n_samples,
                n_features=NUM_FEATURES,
                centers=NUM_CLASSES,
                cluster_std=1.5,
                random_state=RANDOM_SEED)

#convert numpy arrays to float
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#set the training and test data to the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print(X_train[:10])
print(y_train[:10])

plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=y,cmap=plt.cm.RdYlBu)
# plt.show()

#Uses activation function torch.softmax
def convertToLabels(logits, dim=1):
    pred_probs = torch.softmax(logits, dim) #prediction probabilities
    preds = torch.argmax(pred_probs, dim) #prediction labels
    return preds

class BlobModel(nn.Module):
    def __init__(self, input_size, output_size=4, hidden_size=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

model0 = BlobModel(input_size=NUM_FEATURES).to(device)

loss_fn = nn.CrossEntropyLoss()

#Setup an optimizer (ways to adjust the parameters)
optimizer = torch.optim.SGD(
        params = model0.parameters(), 
        lr=0.1)

#Get some prediction probabilites and see what the output looks like (logits)
#Need to convert preds/logits to prediction probabilities and then to prediction labels
model0.eval()
with torch.inference_mode():
    y_logits = model0(X_test.to(device))
    #convert to labels ala activation function
    y_preds = convertToLabels(y_logits)
    print(y_preds[:10])

epochs = 1000
for epoch in range(epochs):
    model0.train()
    y_logits = model0(X_train)
    y_pred = convertToLabels(y_logits)
    loss = loss_fn(y_logits, y_train) #pass in logits to get loss not the preds
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimizer.zero_grad()                 
    loss.backward()                      
    optimizer.step()                      

    model0.eval()
    with torch.inference_mode():
        test_logits = model0(X_test)
        test_pred = convertToLabels(test_logits)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        # print out results
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Test loss: {test_loss:.4f} | Test acc: {test_acc: .2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model0, X_test, y_test)

plt.show()