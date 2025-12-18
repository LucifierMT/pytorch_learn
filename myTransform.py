"""
输入      PIL         Image.open()
输出      tensor      ToTensor()
作用      narrays     cv.imread()
"""


from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


writer = SummaryWriter("logs")
img_path = "data/CIFAR10_images/train/dog/107.png"

# ToTensor
img = Image.open(img_path)
img_tensor = transforms.ToTensor()(img)
writer.add_image("test", img_tensor)

# Normalize
print(img_tensor[0][0][0])
# [0, 1] -> [-1, 1]
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("norm", img_norm)

# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize_tensor = transforms.ToTensor()(img_resize)
writer.add_image("resize", img_resize_tensor)

# Compose
trans_resize_2 = transforms.Resize((128,256))
# 管道：缩放 -> 转为张量 -> 标准化
trans_compose = transforms.Compose(
    [trans_resize_2,
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
img_compose = trans_compose(img)
writer.add_image("compose", img_compose)

# RandomCrop
trans_random_crop = transforms.RandomCrop(12)
trans_compose_2 = transforms.Compose(
    [trans_random_crop,
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
for i in range(10):
    img_compose_2 = trans_compose_2(img)
    writer.add_image("RandomCrop", img_compose_2, i)

writer.close()
