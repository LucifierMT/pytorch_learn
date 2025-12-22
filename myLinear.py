import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyData

root_dir="./data/CIFAR10_images/test"
cat_label_dir="cat"
CatDataset = MyData(root_dir, cat_label_dir)
dataloader= DataLoader(CatDataset, batch_size=64, shuffle=True, num_workers=0)

class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)
    def forward(self, input):
        output = self.linear1(input)
        return output

my_linear = MyLinear()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)            # torch.Size([64, 3, 32, 32])
    output = torch.flatten(imgs)
    print(output.shape)
    output = my_linear(output)
    print(output.shape)