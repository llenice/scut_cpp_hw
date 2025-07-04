scut_cpp_hw
cpp大作业一、二

大作业一
基于 QT 的高精度长整数运算
采用链表存储大数。每个链表节点存储数字的一部分。
通过BigNumber 类封装链表、符号、小数点位置以及错误信息

加法/减法 ：从低位到高位逐个节点相加/相减，并处理进位/借位
乘法:`product[i + j] += digits_a[i] * digits_b[j]`，再处理进位
除法:被除数*e10倍(避免精度消失),再进行长除法(不断相减)

![Screenshot 2025-07-02 234319](https://github.com/user-attachments/assets/abca61a3-bcee-4e7c-8fd4-215c2edf36c8 )







