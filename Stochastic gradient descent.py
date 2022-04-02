# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 12:04
# @Author  : Justus
# @FileName: Stochastic gradient descent.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# 定义初始权重
w = 1.0
# 定义学习率
lr = 0.01


# 定义模型
def forward(x):
    return x * w


# 计算单个损失
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 计算梯度
def gradient(x, y):
    return 2 * x * (x * w - y)


print("Predict (before training):", 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        # 更新权重
        w -= lr * grad
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
    print("Epoch:", epoch, "w=", w, "loss=", l)
print("Predict (after training):", 4, forward(4))
