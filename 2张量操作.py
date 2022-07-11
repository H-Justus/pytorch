# -*- coding: utf-8 -*-
# @Time    : 2022/7/11 
# @Author  : Justus
# @FileName: 2张量操作.py
# @Software: PyCharm

import torch
torch.manual_seed(1)

# ======================================= example 1 =======================================
# torch.ones不会拓展维度

# t = torch.ones((2, 3))
# t_0 = torch.cat([t, t], dim=0)
# t_1 = torch.cat([t, t], dim=1)
# print("t_0:{} \nshape:{}\nt_1:{} \nshape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

# ======================================= example 2 =======================================
# torch.stack会拓展维度

# t = torch.ones((2, 3))
# t_stack = torch.stack([t, t, t], dim=0)
# print("\nt_stack:{} shape:{}".format(t_stack, t_stack.shape))


# ======================================= example 3 =======================================
# torch.chunk按维度平均切分：input：要切分的张量 chunks：要切分的份数 dim：要切分的维度

# a = torch.ones((2, 7))  # 7
# list_of_tensors = torch.chunk(a, dim=1, chunks=3)   # 3
#
# for idx, t in enumerate(list_of_tensors):
#     print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))


# ======================================= example 4 =======================================
# torch.split指定切分长度切分 tensor：要切分的张量 split_size_or_sections：为int时表示每份的长度，
# 为list时按元素进行切分 dim：要切分的维度

# t = torch.ones((2, 5))
# print(t)
#
# list_of_tensors = torch.split(t, [2, 1, 2], dim=1)
# for idx, t in enumerate(list_of_tensors):
#     print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))

# ======================================= example 5 =======================================
# torch.index_select 在维度dim上按index索引数据

# t = torch.randint(0, 9, size=(3, 3))
# idx = torch.tensor([0, 2], dtype=torch.long)    # float
# t_select = torch.index_select(t, dim=0, index=idx)
# print("t:\n{}\nt_select:\n{}".format(t, t_select))

# ======================================= example 6 =======================================
# torch.masked_select 按mask为True进行索引

# t = torch.randint(0, 9, size=(3, 3))
# mask = t.le(5)  # ge：>=    gt: >    le：<=    lt：<
# t_select = torch.masked_select(t, mask)
# print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))


# ======================================= example 7 =======================================
# torch.reshape 共享内存

# t = torch.randperm(8)
# t_reshape = torch.reshape(t, (-1, 2, 2))    # -1：不关心的维度，自动计算
# print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
#
# t[0] = 1024
# print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
# print("t.data 内存地址:{}".format(id(t.data)))
# print("t_reshape.data 内存地址:{}".format(id(t_reshape.data)))


# ======================================= example 8 =======================================
# torch.transpose 交换张量的两个维度
# 二维张量转置torch.t()，相当于torch.transpose（input, 0, 1）

# t = torch.rand((2, 3, 4))
# t_transpose = torch.transpose(t, dim0=1, dim1=2)    # c*h*w     h*w*c
# print("t shape:{}\nt_transpose shape: {}".format(t.shape, t_transpose.shape))

# ======================================= example 9 =======================================
# torch.squeeze 压缩长度为1的轴，如果dim为None则移除所有，如果指定dim则只移除指定的

# t = torch.rand((1, 2, 3, 1))
# t_sq = torch.squeeze(t)
# t_0 = torch.squeeze(t, dim=0)
# t_1 = torch.squeeze(t, dim=1)
# print(t.shape)
# print(t_sq.shape)
# print(t_0.shape)
# print(t_1.shape)

# ======================================= example 8 =======================================
# torch.add 逐元素计算input + alpha * other

t_0 = torch.randn((3, 3))
t_1 = torch.ones_like(t_0)
t_add = torch.add(t_0, t_1, alpha=10)

print("t_0:\n{}\nt_1:\n{}\nt_add_10:\n{}".format(t_0, t_1, t_add))
