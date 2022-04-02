# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 18:12
# @Author  : Justus
# @FileName: Logistic regression.py
# @Software: PyCharm
import torch
import numpy as np
import matplotlib.pyplot as plt

# 创建数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# 定义模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 增加非线性变换
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()
# BCE损失
criterion = torch.nn.BCELoss(reduction='sum')
# SGD优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(1000):
    # 预测
    y_pred = model(x_data)
    # 梯度清零
    optimizer.zero_grad()
    # 计算损失
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    # 反向传播
    loss.backward()
    # 更新权重
    optimizer.step()

# 0~10采集200个数据点
x = np.linspace(0, 10, 200)
# reshape为200*1
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
# Tensor转为numpy
y = y_t.data.numpy()
plt.plot(x, y)
plt.xlabel("Hours")
plt.ylabel("Probability of Pass")
plt.grid()
plt.show()
