import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置项 =====================
# 数据集保存根目录（可修改为自己的路径，如D:/data）
DATA_ROOT = "./data"
# 解决国内下载慢：设置TorchVision镜像（阿里云）
os.environ['TORCH_HOME'] = DATA_ROOT  # 缓存目录
os.environ['TRANSFORMERS_CACHE'] = DATA_ROOT
os.environ['HTTP_PROXY'] = ''  # 如有代理可填，无则留空
os.environ['HTTPS_PROXY'] = ''

# ===================== 1. 下载TorchVision数据集（MNIST/CIFAR-10） =====================
def download_torchvision_datasets():
    print("===== 开始下载MNIST数据集 =====")
    # MNIST预处理
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 下载训练集+测试集
    train_mnist = datasets.MNIST(
        root=DATA_ROOT, train=True, download=True, transform=mnist_transform
    )
    test_mnist = datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=mnist_transform
    )
    print(f"MNIST下载完成 | 训练集：{len(train_mnist)}条 | 测试集：{len(test_mnist)}条")

    print("\n===== 开始下载CIFAR-10数据集 =====")
    # CIFAR-10预处理
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # 下载训练集+测试集
    train_cifar10 = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=cifar_transform
    )
    test_cifar10 = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=cifar_transform
    )
    print(f"CIFAR-10下载完成 | 训练集：{len(train_cifar10)}条 | 测试集：{len(test_cifar10)}条")

# ===================== 2. 生成Iris/Wine数据集（本地拆分train/test） =====================
def generate_sklearn_datasets():
    print("\n===== 生成Iris（鸢尾花）数据集（本地拆分train/test） =====")
    # 加载Iris并拆分8:2
    iris = load_iris()
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    # 保存为npz格式（方便后续加载）
    iris_save_path = os.path.join(DATA_ROOT, "iris")
    os.makedirs(iris_save_path, exist_ok=True)
    torch.save({
        "X_train": X_iris_train, "X_test": X_iris_test,
        "y_train": y_iris_train, "y_test": y_iris_test
    }, os.path.join(iris_save_path, "iris_dataset.pt"))
    print(f"Iris生成完成 | 训练集：{len(X_iris_train)}条 | 测试集：{len(X_iris_test)}条")

    print("\n===== 生成Wine（葡萄酒）数据集（本地拆分train/test） =====")
    # 加载Wine并拆分8:2
    wine = load_wine()
    X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    # 保存为npz格式
    wine_save_path = os.path.join(DATA_ROOT, "wine")
    os.makedirs(wine_save_path, exist_ok=True)
    torch.save({
        "X_train": X_wine_train, "X_test": X_wine_test,
        "y_train": y_wine_train, "y_test": y_wine_test
    }, os.path.join(wine_save_path, "wine_dataset.pt"))
    print(f"Wine生成完成 | 训练集：{len(X_wine_train)}条 | 测试集：{len(X_wine_test)}条")

# ===================== 3. 验证所有数据集 =====================
def verify_datasets():
    print("\n===== 验证数据集完整性 =====")
    # 验证MNIST
    mnist_train_path = os.path.join(DATA_ROOT, "MNIST/raw/train-images-idx3-ubyte")
    mnist_test_path = os.path.join(DATA_ROOT, "MNIST/raw/t10k-images-idx3-ubyte")
    print(f"MNIST训练集文件存在：{os.path.exists(mnist_train_path)}")
    print(f"MNIST测试集文件存在：{os.path.exists(mnist_test_path)}")

    # 验证CIFAR-10
    cifar_path = os.path.join(DATA_ROOT, "cifar-10-batches-py/data_batch_1")
    print(f"CIFAR-10训练集文件存在：{os.path.exists(cifar_path)}")

    # 验证Iris/Wine
    iris_path = os.path.join(DATA_ROOT, "iris/iris_dataset.pt")
    wine_path = os.path.join(DATA_ROOT, "wine/wine_dataset.pt")
    print(f"Iris数据集文件存在：{os.path.exists(iris_path)}")
    print(f"Wine数据集文件存在：{os.path.exists(wine_path)}")

    print("\n===== 所有数据集下载/生成完成！=====")
    print(f"数据集根目录：{os.path.abspath(DATA_ROOT)}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 创建根目录
    os.makedirs(DATA_ROOT, exist_ok=True)
    # 执行下载/生成
    download_torchvision_datasets()
    generate_sklearn_datasets()
    verify_datasets()