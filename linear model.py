# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 21:08
# @Author  : Justus
# @FileName: linear model.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义模型
def forward(x):
    return x * w


# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2


# 计算并记录权重和损失值
w_list = []
mse_list = []
# w从0.0~4.0遍历
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    # l_sum记录损失值之和
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print("MSE=", l_sum/len(x_data))
    w_list.append(w)
    mse_list.append(l_sum/len(x_data))

# 绘图
plt.plot(w_list, mse_list)
plt.ylabel("Loss")
plt.xlabel('w')
plt.show()
