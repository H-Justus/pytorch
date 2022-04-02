# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 22:07
# @Author  : Justus
# @FileName: backpropagation.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import torch

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# 定义初始权重
w = torch.Tensor([1.0])
# 计算梯度
w.requires_grad = True
# 定义学习率
lr = 0.01


# 定义模型(y=wx)
def forward(x):
    return x * w


# 计算单个损失
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("Predict (before training):", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 前馈计算loss
        l = loss(x, y)
        # 反向传播
        l.backward()
        print("\tgrad:", x, y, w.grad.item())
        # 更新权重
        w.data = w.data - lr * w.grad.data
        # 梯度数据清零,下轮重新计算
        w.grad.data.zero_()
    print("progress:", epoch, l.item())
print("Predict (after training):", 4, forward(4).item())

