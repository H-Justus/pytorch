# -*- coding: utf-8 -*-
# @Time    : 2022/7/11 
# @Author  : Justus
# @FileName: 1张量.py
# @Software: PyCharm

import numpy as np
import torch

# 1=====直接创建张量
# arr = np.ones((3, 3))
# print("ndarray的数据类型：", arr.dtype)
# t = torch.tensor(arr, device='cuda')
# print(t)

# 2=====通过torch.from_numpy创建张量，共享内存
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# print("numpy array：", arr)
# t = torch.from_numpy(arr)
# print("tensor：", t)
# arr[0, 0] = 0
# print("修改后numpy array：", arr)
# print("tensor：", t)
# t[0, 0] = -1
# print("修改后tensor：", t)
# print("numpy array：", arr)

# 3=====依据数值创建
# out_t = torch.tensor([1])
# t = torch.zeros((3, 3), out=out_t)
# print(t, '\n', out_t)
# print(id(t), id(out_t), id(t) == id(out_t))

# 4=====通过torch.full创建全10张量
# t = torch.full((3, 3), 10)
# print(t)

# 5=====通过torch.arange创建等差数列张量（左闭右开）
# t = torch.arange(2, 10, 2)
# print(t)

# 6=====通过torch.linspace创建均分数列张量（左闭右闭）
# t = torch.linspace(2, 10, 6)
# print(t)

# 7=====依据概率分布创建张量
# mean：张量 std：张量
# mean = torch.arange(1, 5, dtype=torch.float)
# std = torch.arange(1, 5, dtype=torch.float)
# t_normal = torch.normal(mean, std)
# print("mean:", mean)
# print("std:", std)
# print(t_normal)

# mean：标量 std：标量
# t_normal = torch.normal(0., 1., size=(4,))
# print(t_normal)

# mean：张量 std：标量
mean = torch.arange(1, 5, dtype=torch.float)
std = 1
t_normal = torch.normal(mean, std)
print("mean:", mean)
print("std:", std)
print(t_normal)
