# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 23:31
# @Author  : Justus
# @FileName: backpropagation2.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import torch

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# 定义初始权重
w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
# 计算梯度
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True
# 定义学习率
lr = 0.01


# 定义模型(y=w1x^2+w2x+b)
def forward(x):
    return x * x * w1 + w2 * x + b


# 计算单个损失
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("Predict (before training):", 4, forward(4).item())
for epoch in range(5000):
    for x, y in zip(x_data, y_data):
        # 前馈计算loss
        l = loss(x, y)
        # 反向传播
        l.backward()
        print("\tgrad:", x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        # 更新权重
        w1.data = w1.data - lr * w1.grad.data
        w2.data = w2.data - lr * w2.grad.data
        b.data = b.data - lr * b.grad.data
        # 梯度数据清零,下轮重新计算
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:", epoch, l.item())
print("Predict (after training):", 4, forward(4).item())
