import onnxruntime as ort
from torchvision import transforms
from scipy.special import softmax
from PIL import Image
import numpy as np

# 初始化ONNX运行时会话
onnx_path = "mnist_cnn.onnx"
ort_session = ort.InferenceSession(onnx_path)

# 定义预处理转换
transform = transforms.Compose([
    transforms.ToTensor()
])

def predict(img_path):
    # 打开图像并进行预处理
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.view(1, 3, 28, 28).numpy()

    # 定义输入和输出名称
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # 运行推理
    outputs = ort_session.run([output_name], {input_name: img})

    # 处理输出
    output = np.array(outputs[0])
    output = softmax(output, axis=1)
    pred = np.argmax(output, axis=1)[0]

    return pred

img_path = r"D:\code\DL-master\DL-master\classification\mnist\test\6\116.jpg"
result = predict(img_path)
print(f"预测结果: {result}")