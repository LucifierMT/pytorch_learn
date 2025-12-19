import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyData

root_dir="./data/CIFAR10_images/test"
cat_label_dir="cat"
CatDataset = MyData(root_dir, cat_label_dir)
dataloader= DataLoader(CatDataset, batch_size=64, shuffle=True, num_workers=0)

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

class MyNonLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()    # 以ReLu为例，默认inplace=False
    def forward(self, input):
        output = self.relu(input)
        return output

my_nonlinear1 = MyNonLinear()
output = my_nonlinear1(input)
print(output)

writer = SummaryWriter("logs-nonlinear")
step = 0

for data in dataloader:
    imgs, targets = data
    outputs = my_nonlinear1(imgs)
    print(imgs.shape)
    print(outputs.shape)
    writer.add_images("input", imgs, step)
    writer.add_images("output", outputs, step)
    step += 1

writer.close()