scut_cpp_hw
cpp大作业一、二


### **大作业一**

基于 QT 的高精度长整数运算

采用链表存储大数。每个链表节点存储数字的一部分。

通过BigNumber 类封装链表、符号、小数点位置以及错误信息

加法/减法 ：从低位到高位逐个节点相加/相减，并处理进位/借位

乘法:`product[i + j] += digits_a[i] * digits_b[j]`，再处理进位

除法:被除数*e10倍(避免精度消失),再进行长除法(不断相减)

<img src="https://github.com/user-attachments/assets/abca61a3-bcee-4e7c-8fd4-215c2edf36c8"  width="300"  />






### **大作业二**

一个基于深度学习的手写数字识别系统

功能特性

- **实时手写识别**：在画布上书写，应用将通过ONNX Runtime进行推理
- **加载本地图片**：可以选择任意包含数字的图片，应用会将其加载到画布上并进行识别
- **模型动态切换**：内置CNN和的ResNet
- **多语言支持**：界面支持中文和英文两种语言

依赖环境

1. Qt Creator：6.9.1
2. ONNX Runtime：1.22.1


<img width="380" alt="recognitize_pic" src="https://github.com/user-attachments/assets/86ca8574-0088-4b21-97a6-6fba6fa20379" />






