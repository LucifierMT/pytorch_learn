import torch
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from myTensorboard import writer
from my_dataset import MyData


root_dir="./data/CIFAR10_images/test"
cat_label_dir="cat"
CatDataset = MyData(root_dir, cat_label_dir)
dataloader= DataLoader(CatDataset, batch_size=64, shuffle=True, num_workers=0)


class MyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3, stride=1, padding=2)
    def forward(self, x):
        x = self.conv1(x)
        return x


my_conv1 = MyConv()
print(my_conv1)

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    outputs = my_conv1(imgs)
    print(imgs.shape)     # torch.Size([64, 3, 32, 32])
    print(outputs.shape)  # torch.Size([64, 6, 34, 34])
    writer.add_images("input", imgs, step)
    outputs = torch.reshape(outputs, [-1, 3, 34, 34])
    writer.add_images("output", outputs, step)
    step += 1
writer.close()