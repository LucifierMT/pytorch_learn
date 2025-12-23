import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from my_dataset import MyData

root_dir="./data/CIFAR10_images/test"
cat_label_dir="cat"
CatDataset = MyData(root_dir, cat_label_dir)
dataloader= DataLoader(CatDataset, batch_size=64, shuffle=True, num_workers=0)

class MySeqNn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
my_seq_nn = MySeqNn()
optim = torch.optim.SGD(my_seq_nn.parameters(), lr=0.01)

for data in dataloader:
    imgs, targets = data
    label_to_idx = {"cat": 0, "dog": 1, "bird": 2, "plane": 3, "car": 4,
                    "deer": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
    # 将targets转换为整数索引
    targets_idx = torch.tensor([label_to_idx[label] for label in targets])
    outputs = my_seq_nn(imgs)
    result_loss = loss(outputs, targets_idx)
    optim.zero_grad()
    result_loss.backward()
    optim.step()
    print(result_loss)
