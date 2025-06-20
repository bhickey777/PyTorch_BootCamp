import torch


# Can also import the common abbreviation "nn" for "Neural Networks"
from torch import nn
import pandas as pd
import numpy as np

RANDOM_SEED = 64
torch.manual_seed = RANDOM_SEED

a = torch.rand(1, 3, 5)

torch.manual_seed = RANDOM_SEED
b = torch.rand(1, 3, 5)

c = a * b

torch.manual_seed = RANDOM_SEED
d = torch.rand(1, 5, 5)

torch.manual_seed = RANDOM_SEED
e = torch.matmul(a, d)

print(c == e)

print(torch.min(e))
print(torch.max(e))
