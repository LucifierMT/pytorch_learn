import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))
        return x+1

mynn = MyNn()
x = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
output = mynn(x)
print(output)


"""
卷积操作，以二维卷积为例
对应位置相乘相加
"""

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [4, 0, 5, 6, 2],
    [2, 3, 1, 2, 3],
    [5, 2, 1, 3, 1],
    [6, 1, 0, 2, 2]]
)

kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])
print(input.shape)
print(kernel.shape)

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input.shape)
print(kernel.shape)

# 输入必须是四维
output1 = F.conv2d(input, kernel, stride=1)
print(f"output1:{output1}")
print("output1.shape:", output1.shape)

output2 = F.conv2d(input, kernel, stride=2)
print(f"output2:{output2}")
print("output2.shape:", output2.shape)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(f"output3:{output3}")
print("output3.shape:", output3.shape)

output4 = F.conv2d(input, kernel, stride=1, padding=1, dilation=2)
print(f"output4:{output4}")
print("output4.shape:", output4.shape)
