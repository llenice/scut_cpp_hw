import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import os
from PIL import Image

# --- 1. 定制版 ResNet 模型 ---
def create_resnet_for_mnist(arch='resnet18', pretrained=False):
    """
    创建一个适用于 MNIST 数据集的、经过修改的 ResNet 模型。

    :param arch: 'resnet18', 'resnet34', etc.
    :param pretrained: 是否加载在 ImageNet 上预训练的权重。
    :return: 修改后的 ResNet 模型。
    """

    model = getattr(models, arch)(pretrained=pretrained)

    # 1. 修改第一个卷积层以接受单通道 (灰度) 输入
    #    原始 ResNet 的 self.conv1 是 nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #    我们将其替换为一个更适合小尺寸灰度图的卷积层
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. (可选但推荐) 移除初始的最大池化层
    #    原始 ResNet 在 conv1 后有一个 MaxPool2d，对于28x28的图像来说过于激进。
    #    我们用一个恒等层 (Identity) 替换它，相当于直接跳过这一步。
    model.maxpool = nn.Identity()

    # 3. 将最后的全连接层从1000类改为10类（MNIST的0-9数字）
    num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数
    model.fc = nn.Linear(num_ftrs, 10)  # 替换为10个输出（对应数字0-9）
    
    print(f"✅ ResNet模型已修改:")
    print(f"   - 输入通道: 3 → 1 (灰度图)")
    print(f"   - 输出类别: 1000 → 10 (MNIST数字)")
    print(f"   - 全连接层: {num_ftrs} → 10")
    
    return model

# --- 2. 训练流程 ---
def main_train():
    # --- 数据加载与增强 ---
    data_root = r"D:\code\DL\mnist_Qt\mnist\MNIST"
    os.makedirs(data_root, exist_ok=True)
    
    # 为训练集添加数据增强，提升模型鲁棒性
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 测试集通常不需要增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print(f"正在从本地路径 '{data_root}' 加载 MNIST 数据...")
    try:
        train_dataset = dsets.MNIST(root=data_root, train=True, transform=train_transform, download=True)
        test_dataset = dsets.MNIST(root=data_root, train=False, transform=test_transform, download=True)
    except Exception as e:
        print(f"加载或下载 MNIST 数据时出错: {e}")
        return
        
    print("MNIST 数据加载成功。")

    # --- 训练设置 ---
    batch_size = 128 # 对于更深的模型，可以适当调整批大小
    num_epochs = 10
    learning_rate = 0.01

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_resnet_for_mnist().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # 使用学习率调度器，在训练后期降低学习率以精细调整
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"开始使用定制版 ResNet 在设备 '{device}' 上训练 {num_epochs} 个周期...")

    # --- 训练与评估循环 ---
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'周期 [{epoch+1}/{num_epochs}], 步骤 [{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}')
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'--- 周期 {epoch+1} 结束 ---')
        print(f'测试集准确率: {accuracy:.2f} %')
        print('-' * 25)
        
        scheduler.step() # 更新学习率

    # 保存模型
    model_save_path = 'mnist_resnet_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"训练完成，模型已保存至 '{model_save_path}'")

#  推理函数
def main_infer(model_path, image_path):
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 '{model_path}'")
        return
    if not os.path.exists(image_path):
        print(f"错误：找不到图片文件 '{image_path}'")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载与训练时完全相同的定制版 ResNet 架构
    model = create_resnet_for_mnist().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型 '{model_path}' 加载成功，已设置为评估模式。")

    # 推理的预处理流程必须与训练时的测试集/验证集完全一致
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        img = Image.open(image_path).convert('L')
        print(f"✅ 图片加载成功: {image_path}")
        print(f"📏 图片尺寸: {img.size}")
    except Exception as e:
        print(f"打开或处理图片时出错: {e}")
        return

    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"🔧 输入张量形状: {img_tensor.shape}")

    with torch.no_grad():
        outputs = model(img_tensor)
        print(f"🎯 模型输出形状: {outputs.shape}")  # 应该是 [1, 10]
        print(f"🎯 原始输出值: {outputs.squeeze()}")
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_digit = predicted.item()
    confidence_score = confidence.item()

    print("\n--- 推理结果 ---")
    print(f"图片: '{os.path.basename(image_path)}'")
    print(f"模型识别出的数字是: {predicted_digit}")
    print(f"置信度: {confidence_score:.2%}")
    
    print("\n所有类别的概率分布:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  数字 {i}: {prob.item():.2%}")
        
    # 验证输出维度
    if probabilities.shape[1] != 10:
        print(f"⚠️  警告: 模型输出维度不正确! 期望10个类别，实际得到{probabilities.shape[1]}个")
    else:
        print(f"✅ 模型输出维度正确: {probabilities.shape[1]}个类别")

# ONNX转换函数 
def convert_to_onnx(model_path='mnist_resnet_model.pth', onnx_path='mnist_resnet_model.onnx'):
    """
    将训练好的ResNet模型转换为ONNX格式的便捷函数
    """
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 '{model_path}'")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建与训练时相同的模型架构
    model = create_resnet_for_mnist().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    
    try:
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11,
            do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"✅ ONNX模型已保存至: {onnx_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX转换失败: {e}")
        return False


if __name__ == '__main__':

    # main_train() 
    
    #  转换为ONNX 
    convert_to_onnx()
    
    # --- 运行推理 ---
    # MODEL_FILE = 'mnist_resnet_model.pth'
    # IMAGE_FILE_TO_INFER = r"C:\Users\hui\Desktop\9_0.png"

    # if not os.path.exists(MODEL_FILE):
    #      print(f"找不到模型文件 '{MODEL_FILE}'。请先运行 main_train() 来训练并保存模型。")
    # elif not os.path.exists(IMAGE_FILE_TO_INFER):
    #     print(f"找不到用于推理的图片 '{IMAGE_FILE_TO_INFER}'。请修改该变量为你自己的图片路径。")
    # else:
    #     main_infer(model_path=MODEL_FILE, image_path=IMAGE_FILE_TO_INFER)