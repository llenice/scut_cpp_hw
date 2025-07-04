import torch
import torch.nn as nn
import os

# --- 步骤 1: 确保模型定义与训练时完全一致 ---
class OptimizedCnnNet(nn.Module):
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

def convert_model_to_onnx(pytorch_model_path, onnx_export_path):
    """
    加载PyTorch模型权重，并将其转换为ONNX格式。

    :param pytorch_model_path: 已训练好的PyTorch模型权重文件 (.pth) 的路径。
    :param onnx_export_path: 要导出的ONNX模型文件的保存路径。
    """
    # 检查PyTorch模型文件是否存在
    if not os.path.exists(pytorch_model_path):
        print(f"错误: 找不到PyTorch模型文件 '{pytorch_model_path}'")
        return

    # 选择设备 (CPU或GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # --- 步骤 2: 创建模型实例并加载权重 ---
    # 创建模型架构的实例
    model = OptimizedCnnNet(classes=10).to(device)
    # 加载训练好的状态字典 (权重)
    model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    model.eval()
    print("模型加载成功并已设置为评估模式。")

    # --- 步骤 3: 创建一个符合模型输入的虚拟输入 (dummy input) ---
    # 我们的模型期望的输入形状是 [batch_size, channels, height, width]
    # 对于我们的MNIST模型，应该是 [batch_size, 1, 28, 28]
    # 我们设置一个批次大小为1的虚拟输入
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)
    print(f"创建的虚拟输入形状: {dummy_input.shape}")

    # --- 步骤 4: 执行转换 ---
    try:
        print(f"开始将模型转换为ONNX格式，保存到 '{onnx_export_path}'...")
        torch.onnx.export(
            model,                          # 要转换的模型
            dummy_input,                    # 虚拟输入
            onnx_export_path,               # 输出的ONNX文件名
            export_params=True,             
            opset_version=11,              
            do_constant_folding=True,      
            input_names=['input'],          
            output_names=['output'],        
            dynamic_axes={                 
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print("-" * 40)
        print(f"模型成功转换为ONNX格式，并已保存！")
        print(f"文件位置: {os.path.abspath(onnx_export_path)}")
        print("-" * 40)

    except Exception as e:
        print(f"模型转换过程中发生错误: {e}")

if __name__ == '__main__':
    PYTORCH_MODEL_PATH = "mnist_augmented_cnn.pth"

    ONNX_EXPORT_PATH = "mnist_cnn_augmented.onnx" 

    convert_model_to_onnx(
        pytorch_model_path=PYTORCH_MODEL_PATH,
        onnx_export_path=ONNX_EXPORT_PATH
    )
