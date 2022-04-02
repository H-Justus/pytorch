# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 22:39
# @Author  : Justus
# @FileName: gradient descent.py
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


# 计算平均损失
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# 计算梯度
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print("Predict (before training):", 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    # 更新权重
    w -= lr * grad_val
    print("Epoch:", epoch, "w=", w, "loss=", cost_val)
print("Predict (after training):", 4, forward(4))
