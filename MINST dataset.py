# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 14:01
# @Author  : Justus
# @FileName: MINST dataset.py
# @Software: PyCharm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import numpy as np

# 准备数据集
# 归一化，0.1307和0.3081分别为均值和标准差，output = (input - mean) / std
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, ))
                                ])
train_dataset = datasets.MNIST(root="./mnist", train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root="./mnist",
                              train=False,
                              transform=transform,
                              download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        # 交叉熵损失中包含了softmax激活函数，所以最后一层直接输出
        return self.linear5(x)


model = Net()

# 交叉熵损失已包含
criterion = torch.nn.CrossEntropyLoss()
# SGD优化器, momentum冲量值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 加载数据
        inputs, target = data
        # 预测
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, target)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d, %4d]loss:%.3f" % (epoch+1, batch_idx+1, running_loss/300))


# 测试
def test():
    correct = 0
    total = 0
    # 不计算梯度，强制之后的内容不进行计算图构建
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            # 取每行最大值的下标，dim=1按行
            _, predicted = torch.max(outputs.data, dim=1)
            # 取labels的第0个元素，total最终值为样本总数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set:%d %%" % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
