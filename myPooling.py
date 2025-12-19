import torch
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyData



root_dir="./data/CIFAR10_images/test"
cat_label_dir="cat"
CatDataset = MyData(root_dir, cat_label_dir)
dataloader= DataLoader(CatDataset, batch_size=64, shuffle=True, num_workers=0)

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [4, 0, 5, 6, 2],
    [2, 3, 1, 2, 3],
    [5, 2, 1, 3, 1],
    [6, 1, 0, 2, 2]]
)
input = torch.reshape(input, (-1, 1, 5, 5))  # -1表示自动计算
print(input.shape)  # torch.Size([1, 1, 5, 5])


class MyPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


my_pooling1 = MyPooling()
output = my_pooling1(input)
print(output)
print(output.shape)  # torch.Size([1, 1, 2, 2])

writer = SummaryWriter("logs-maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    outputs = my_pooling1(imgs)
    print(imgs.shape)      # torch.Size([64, 3, 32, 32])
    print(outputs.shape)   # torch.Size([64, 3, 11, 11])
    writer.add_images("input", imgs, step)
    writer.add_images("output", outputs, step)
    step += 1

writer.close()
