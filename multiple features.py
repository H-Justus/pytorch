# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 13:36
# @Author  : Justus
# @FileName: multiple features.py
# @Software: PyCharm

import numpy as np
import torch

# 加载数据集， delimiter分隔符， dtype数据类型
xy = np.loadtxt("diabetes.csv.gz", delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


# 定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# BCE损失
criterion = torch.nn.BCELoss(reduction='mean')
# SGD优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练
for epoch in range(10000):
    # 预测
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新权重
    optimizer.step()

# 测试
# 0.43左右
print(model(torch.zeros(1, 8)).item())
# 0.1左右
print(model(torch.ones(1, 8)).item())
