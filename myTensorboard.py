from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "data/CIFAR10_images/train/airplane/29.png"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 1, dataformats="HWC")
# y=x
# 使用  tensorboard --logdir=logs 查看
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()