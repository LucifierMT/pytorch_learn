import os
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ===================== 核心配置 =====================
# 数据集根目录（生成的图片会存在这里，可修改）
DATA_ROOT = "./data"
# 解决国内下载慢：阿里云镜像
os.environ['TORCH_VISION_DATASETS_MIRROR'] = 'https://mirrors.aliyun.com/pytorch-vision-datasets/'
os.makedirs(DATA_ROOT, exist_ok=True)

# ===================== 1. MNIST 转可视化图片（手写数字，单通道） =====================
def convert_mnist_to_images():
    print("===== 下载并转换MNIST为可查看的图片 =====")
    # 加载MNIST原始数据集（不做归一化，方便保存图片）
    mnist_transform = transforms.Compose([transforms.ToTensor()])  # 仅转张量，不归一化
    train_mnist = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=mnist_transform)
    test_mnist = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=mnist_transform)

    # 定义MNIST保存路径（train/test + 数字类别）
    mnist_train_root = os.path.join(DATA_ROOT, "MNIST_images/train")
    mnist_test_root = os.path.join(DATA_ROOT, "MNIST_images/test")

    # 保存训练集图片
    for idx, (img_tensor, label) in enumerate(train_mnist):
        # 张量转PIL图片（单通道：[1,28,28] → [28,28]）
        img_np = img_tensor.squeeze(0).numpy() * 255  # 从[0,1]转回[0,255]
        img_pil = Image.fromarray(img_np.astype(np.uint8), mode='L')  # L=灰度图
        # 创建类别文件夹
        label_dir = os.path.join(mnist_train_root, str(label))
        os.makedirs(label_dir, exist_ok=True)
        # 保存图片（命名：idx.png）
        img_path = os.path.join(label_dir, f"{idx}.png")
        img_pil.save(img_path)
        # 进度提示（每10000张打印一次）
        if idx % 10000 == 0 and idx > 0:
            print(f"MNIST训练集已保存 {idx} 张图片")

    # 保存测试集图片
    for idx, (img_tensor, label) in enumerate(test_mnist):
        img_np = img_tensor.squeeze(0).numpy() * 255
        img_pil = Image.fromarray(img_np.astype(np.uint8), mode='L')
        label_dir = os.path.join(mnist_test_root, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(label_dir, f"{idx}.png")
        img_pil.save(img_path)

    print(f"✅ MNIST图片转换完成！")
    print(f"   训练集路径：{mnist_train_root}")
    print(f"   测试集路径：{mnist_test_root}")

# ===================== 2. CIFAR-10 转可视化图片（彩色，按类别命名） =====================
def convert_cifar10_to_images():
    print("\n===== 下载并转换CIFAR-10为可查看的图片 =====")
    # CIFAR-10类别名称（0-9对应）
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    # 加载CIFAR-10原始数据集
    cifar_transform = transforms.Compose([transforms.ToTensor()])
    train_cifar10 = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=cifar_transform)
    test_cifar10 = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=cifar_transform)

    # 定义CIFAR-10保存路径
    cifar_train_root = os.path.join(DATA_ROOT, "CIFAR10_images/train")
    cifar_test_root = os.path.join(DATA_ROOT, "CIFAR10_images/test")

    # 保存训练集图片
    for idx, (img_tensor, label) in enumerate(train_cifar10):
        # 张量转PIL图片（彩色：[3,32,32] → [32,32,3]）
        img_np = img_tensor.permute(1, 2, 0).numpy() * 255  # 通道从C×H×W→H×W×C
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        # 创建类别文件夹（用中文易理解的名称）
        label_name = cifar10_classes[label]
        label_dir = os.path.join(cifar_train_root, label_name)
        os.makedirs(label_dir, exist_ok=True)
        # 保存图片
        img_path = os.path.join(label_dir, f"{idx}.png")
        img_pil.save(img_path)
        if idx % 10000 == 0 and idx > 0:
            print(f"CIFAR-10训练集已保存 {idx} 张图片")

    # 保存测试集图片
    for idx, (img_tensor, label) in enumerate(test_cifar10):
        img_np = img_tensor.permute(1, 2, 0).numpy() * 255
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        label_name = cifar10_classes[label]
        label_dir = os.path.join(cifar_test_root, label_name)
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(label_dir, f"{idx}.png")
        img_pil.save(img_path)

    print(f"✅ CIFAR-10图片转换完成！")
    print(f"   训练集路径：{cifar_train_root}")
    print(f"   测试集路径：{cifar_test_root}")

# ===================== 3. 验证生成结果 =====================
def verify_visual_datasets():
    print("\n===== 验证生成的图片数据集 =====")
    # 验证MNIST
    mnist_train_0 = os.path.join(DATA_ROOT, "MNIST_images/train/0")
    mnist_test_1 = os.path.join(DATA_ROOT, "MNIST_images/test/1")
    print(f"MNIST训练集0类图片数：{len(os.listdir(mnist_train_0)) if os.path.exists(mnist_train_0) else '不存在'}")
    print(f"MNIST测试集1类图片数：{len(os.listdir(mnist_test_1)) if os.path.exists(mnist_test_1) else '不存在'}")

    # 验证CIFAR-10
    cifar_train_airplane = os.path.join(DATA_ROOT, "CIFAR10_images/train/airplane")
    cifar_test_cat = os.path.join(DATA_ROOT, "CIFAR10_images/test/cat")
    print(f"CIFAR-10训练集飞机图片数：{len(os.listdir(cifar_train_airplane)) if os.path.exists(cifar_train_airplane) else '不存在'}")
    print(f"CIFAR-10测试集猫图片数：{len(os.listdir(cifar_test_cat)) if os.path.exists(cifar_test_cat) else '不存在'}")

    print("\n🎉 所有可查看的图片数据集生成完成！")
    print(f"📂 总目录：{os.path.abspath(DATA_ROOT)}")
    print("🔍 直接打开该目录，可看到MNIST_images/CIFAR10_images文件夹，里面train/test分好类，双击图片即可查看！")

# ===================== 主函数 =====================
if __name__ == "__main__":
    convert_mnist_to_images()
    convert_cifar10_to_images()
    verify_visual_datasets()