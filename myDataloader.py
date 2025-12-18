import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


# 自定义数据集类
class CIFARTestDatasetShip(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.data_dir = os.path.join(root_dir, label_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_mapping = {"ship": 0}

        if os.path.isdir(self.data_dir):
            for img_name in os.listdir(self.data_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(self.data_dir, img_name))
                    self.labels.append(self.label_mapping[label_dir])

        print(f"找到 {len(self.images)} 张图像在目录: {self.data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建数据集实例
test_dataset = CIFARTestDatasetShip(
    root_dir="data/CIFAR10_images/test",
    label_dir="ship",
    transform=transform
)

# 检查数据集是否为空
if len(test_dataset) > 0:
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    # 访问数据集中的样本
    img, target = test_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Target: {target}")
    print(f"Dataset size: {len(test_dataset)}")

    # 遍历数据加载器示例
    print("遍历数据加载器:")
    for batch_idx, (data, targets) in enumerate(test_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, targets shape {targets.shape}")
        break  # 只显示第一个批次
else:
    print("数据集中没有找到图像文件，请检查以下事项：")
    print("1. 路径是否正确：data/CIFAR10_images/test/ship")
    print("2. 该目录是否存在且包含图像文件")
    print("3. 图像格式是否为 png, jpg, jpeg")
