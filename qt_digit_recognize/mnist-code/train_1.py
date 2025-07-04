import torch
import torch.optim as optim
import random
import os
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from CNN_1 import OptimizedCnnNet
from mnist_dataset import MnistData

def train_model():
    # 1. 定义包含数据增强的训练数据转换器
    train_transform = transforms.Compose([
        # 随机仿射变换：包括旋转(-10到10度)，平移(最多10%)
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(), # 将图像转为Tensor并归一化到[0, 1]
        transforms.Normalize((0.1307,), (0.3081,)) # 使用MNIST数据集的均值和标准差进行标准化
    ])

    # 测试/验证数据转换器 (通常不需要增强)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. 加载数据集

    train_dataset = datasets.MNIST(r"D:\code\DL-master\DL-master\classification\mnist\MNIST", train=True, download=False, transform=train_transform)
    test_dataset = datasets.MNIST(r"D:\code\DL-master\DL-master\classification\mnist\MNIST", train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 3. 初始化模型、优化器和学习率调度器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = OptimizedCnnNet(classes=10).to(device) 
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001) 
    
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # 训练循环
    num_epochs = 15
    for epoch in range(num_epochs):
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
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

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
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

        # 4. 更新学习率
        scheduler.step()


    torch.save(model.state_dict(), "mnist_optimized_cnn.pth")
    print("Training finished and model saved to mnist_optimized_cnn.pth")


train_model()
