import torch
from torch import nn
from torch.nn import L1Loss, Conv2d, MaxPool2d, Flatten, Linear


inputs=torch.tensor([1,2,3], dtype=torch.float32)
targets=torch.tensor([1,2,5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)
print(result)             # tensor(0.6667)

loss2 = L1Loss(reduction='sum')
result2 = loss2(inputs, targets)
print(result2)           # tensor(2.0000)

