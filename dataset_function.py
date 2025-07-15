import torch

def linearDataset(bias=0.3, weight=0.7, step=0.02):
    start = 0
    end = 1
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias #linear regression formula

    return (X, y)


# data = linearDataset()
# print(data)