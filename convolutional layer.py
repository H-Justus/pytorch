# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 13:35
# @Author  : Justus
# @FileName: convolutional layer.py
# @Software: PyCharm

import torch

# 1、卷积
in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = (3, 3)
batch_size = 1

# randn正态分布，rand平均分布，输入维度N，C，H，W
input1 = torch.randn(batch_size,
                     in_channels,
                     height,
                     width)
# 卷积层
conv_layer1 = torch.nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size)

output1 = conv_layer1(input1)

print(input1.shape, "\tinput1.shape")
print(output1.shape, "\toutput1.shape")
print(conv_layer1.weight.shape, "\tconv_layer1.weight.shape")

# 2、带padding的卷积
input2 = [3, 4, 6, 5, 7,
          2, 4, 6, 8, 2,
          1, 6, 7, 8, 4,
          9, 7, 4, 6, 2,
          3, 7, 5, 4, 1]
input2 = torch.Tensor(input2).view(1, 1, 5, 5)
# padding在外层填充0，bias偏置量
cov_layer2 = torch.nn.Conv2d(1,
                             2,
                             kernel_size=kernel_size,
                             stride=(2, 2),
                             padding=1,
                             bias=False)
# 卷积核
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
# 初始化卷积层权重
cov_layer2.weight.data = kernel.data

output2 = cov_layer2(input2)
print("output2:", output2)

# 3、MaxPooling下采样
input3 = [3, 4, 6, 5,
          2, 4, 6, 8,
          1, 6, 7, 8,
          9, 7, 4, 6]
input3 = torch.Tensor(input3).view(1, 1, 4, 4)

# kernel_size时自动将stride也设置为2
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output3 = maxpooling_layer(input3)
print("output3:", output3)
