import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import os

# --- 确保模型定义与训练时一致 ---
class OptimizedCnnNet(nn.Module):
    """
    一个针对MNIST进行优化的CNN模型。
    """
    def __init__(self, classes=10):
        super(OptimizedCnnNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- 自定义噪声添加模块 ---
class AddGaussianNoise(object):
    """
    向张量图像中添加高斯噪声的自定义transform。
    """
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def train_model():
    """
    完整的训练和验证流程函数，包含高级数据增强。
    """
    # 1. 定义包含高级数据增强的训练数据转换器
    train_transform = transforms.Compose([
        # 随机仿射变换：在-10到10度之间随机旋转，并进行最大10%的随机平移
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        # 转换为张量 (将图像从PIL格式转为Tensor，像素值从0-255归一化到0-1)
        transforms.ToTensor(),
        # 添加高斯噪声
        AddGaussianNoise(mean=0., std=0.1),
        # 标准化：使用MNIST数据集的全局均值和标准差
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 测试/验证数据转换器 (通常不需要数据增强)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. 加载数据集
    data_root = r"D:\code\DL\mnist_Qt\mnist\MNIST"
    try:
        train_dataset = datasets.MNIST(root=data_root, train=True, download=False, transform=train_transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, transform=test_transform)
    except Exception as e:
        print(f"加载或下载MNIST数据集失败: {e}")
        return

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 3. 初始化模型、优化器和学习率调度器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedCnnNet(classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # 训练循环
    num_epochs = 15
    print(f"开始训练 {num_epochs} 个周期，使用设备: {device}...")
    print(f"训练数据增强流程: {train_transform}")

    for epoch in range(1, num_epochs + 1):
        # --- 训练阶段 ---
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'训练周期: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

        # --- 验证阶段 ---
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\n测试集 (周期 {epoch}): 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

        # 4. 更新学习率
        scheduler.step()

    torch.save(model.state_dict(), "mnist_augmented_cnn.pth")
    print("训练完成，模型已保存至 mnist_augmented_cnn.pth")


if __name__ == '__main__':
    train_model()
