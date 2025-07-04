from CNN_1 import OptimizedCnnNet
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class DigitPredictor:
    def __init__(self, model_path):
        """
        初始化预测器。
        :param model_path: 已训练好的模型权重文件路径 (.pth)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 创建模型实例，然后加载训练好的权重 (state_dict)
        self.model = OptimizedCnnNet(classes=10).to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 2. 将模型设置为评估模式 (这会关闭Dropout等)
        self.model.eval()
        
        # 3. 定义与训练时匹配的图像预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            # 关键：使用与训练时完全相同的均值和标准差进行归一化
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
         # 4. 定义类别标签
        self.labels = [str(i) for i in range(10)]

        print(f"预测器初始化完成，使用设备: {self.device}")

    def predict(self, image_path):
        """
        对单张图片进行预测。
        :param image_path: 待预测的图片文件路径
        :return: (预测的数字, 置信度) 的元组
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件未找到: {image_path}")
            
        try:
            # 关键：加载图片并转换为单通道灰度图 ('L')
            img = Image.open(image_path).convert('L')
        except Exception as e:
            raise IOError(f"无法打开或处理图片文件: {image_path}. 错误: {e}")

        # 应用预处理
        img_tensor = self.transform(img)
        
        # 增加一个批次维度，形状变为 [1, 1, 28, 28]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 在不计算梯度的上下文中进行预测
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # 应用softmax得到概率
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        prob_numpy = probabilities.cpu().numpy()[0]
        
        # 构建并返回包含所有类别及其概率的字典
        all_probs = {self.labels[i]: prob_numpy[i] for i in range(len(self.labels))}

        # return predicted_digit, confidence_score
        return all_probs
    
if __name__ == '__main__':
    MODEL_PATH = "mnist_augmented_cnn.pth"
    
    IMAGE_TO_PREDICT = r"C:\Users\hui\Desktop\0_1.png" 

    # 输出单个概率
    # try:
    #     # 3. 创建预测器实例
    #     predictor = DigitPredictor(model_path=MODEL_PATH)
        
    #     # 4. 执行预测
    #     predicted_digit, confidence = predictor.predict(image_path=IMAGE_TO_PREDICT)
        
    #     # 5. 打印结果
    #     print(f"\n预测的图片: {IMAGE_TO_PREDICT}")
    #     print(f"模型识别出的数字是: {predicted_digit}")
    #     print(f"置信度: {confidence:.2%}")

    # except (FileNotFoundError, IOError) as e:
    #     print(f"发生错误: {e}")
    #     print("请确保模型文件路径和图片路径都正确无误。")
    # except Exception as e:
    #     print(f"发生了未预料的错误: {e}")

    # 输出全部概率
    try:
        # 3. 创建预测器实例
        predictor = DigitPredictor(model_path=MODEL_PATH)
        
        # 4. 执行预测，获取包含所有概率的字典
        all_probabilities = predictor.predict(image_path=IMAGE_TO_PREDICT)
        
        # 5. 打印结果
        print(f"\n对图片 '{os.path.basename(IMAGE_TO_PREDICT)}' 的预测概率分布:")
        print("-" * 40)
        
        # 寻找概率最高的项，用于最终决策
        highest_prob = 0
        predicted_digit = None
        
        # 格式化并打印每一个类别的概率
        for digit, probability in all_probabilities.items():
            print(f"数字 {digit}: {probability:.4%}") # 以百分比形式打印，保留4位小数
            if probability > highest_prob:
                highest_prob = probability
                predicted_digit = digit
        
        print("-" * 40)
        print(f"最终识别结果: {predicted_digit} (置信度: {highest_prob:.2%})")

    except (FileNotFoundError, IOError) as e:
        print(f"发生错误: {e}")
        print("请确保模型文件路径和图片路径都正确无误。")
    except Exception as e:
        print(f"发生了未预料的错误: {e}")
