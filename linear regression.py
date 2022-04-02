# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 16:47
# @Author  : Justus
# @FileName: linear regression.py
# @Software: PyCharm

import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 定义模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
# MSE损失，size_average是否求均值，reduce是否要求和降维
criterion = torch.nn.MSELoss(size_average=False)
# 优化器，lr学习率，momentum冲量
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(1300):
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

# 打印权重
print("w=", model.linear.weight.item())
# 打印偏置
print("b=", model.linear.bias.item())

# 测试
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y-pred=", y_test.data)
