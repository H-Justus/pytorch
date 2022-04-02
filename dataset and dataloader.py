# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 10:40
# @Author  : Justus
# @FileName: dataset and dataloader.py
# @Software: PyCharm

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 准备数据集
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # 下标操作
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 数据条数
    def __len__(self):
        return self.len


dataset = DiabetesDataset("diabetes.csv.gz")
# 加载器，dataset传递数据集，batch_size批量容量大小，shuffle是否打乱，num_workers并行进程数
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)


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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    # 训练
    for epoch in range(20000):
        for i, data in enumerate(train_loader, 0):
            # 加载数据
            inputs, labels = data
            # 预测
            y_pred = model(inputs)
            # 计算损失
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
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
