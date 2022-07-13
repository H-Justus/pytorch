# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 
# @Author  : Justus
# @FileName: 4自动求梯度.py
# @Software: PyCharm

import torch

# =====retain_graph   y=(x+w)*(w+1)

# w = torch.tensor([1.], requires_grad=True)
# x = torch.tensor([2.], requires_grad=True)
#
# a = torch.add(w, x)
# b = torch.add(w, 1)
# y = torch.mul(a, b)
#
# y.backward(retain_graph=True)
# print(w.grad)  # dy/dw=2w+x+1
# print(x.grad)  # dy/dx=w+1

# =====backward(gradient=grad_tensors) 权值

# w = torch.tensor([1.], requires_grad=True)
# x = torch.tensor([2.], requires_grad=True)
#
# a = torch.add(w, x)     # retain_grad()
# b = torch.add(w, 1)
#
# y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)
# y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2
#
# loss = torch.cat([y0, y1], dim=0)       # [y0, y1]
# grad_tensors = torch.tensor([1., 2.])
#
# loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors

# print(w.grad)

# =====torch.autograd.grad 高阶求导

# x = torch.tensor([3.], requires_grad=True)
# y = torch.pow(x, 2)     # 平方 y = x**2
#
# grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
# print(grad_1)
#
# grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
# print(grad_2)

# =====tips: 1 梯度不会自动清零，不加grad.zero_梯度会一直累加

# w = torch.tensor([1.], requires_grad=True)
# x = torch.tensor([2.], requires_grad=True)
#
# for i in range(4):
#     a = torch.add(w, x)
#     b = torch.add(w, 1)
#     y = torch.mul(a, b)
#
#     y.backward()
#     print(w.grad)
#
#     # w.grad.zero_()


# =====tips: 2 依赖于叶子结点的节点，requires_grad为Ture

# w = torch.tensor([1.], requires_grad=True)
# x = torch.tensor([2.], requires_grad=True)
#
# a = torch.add(w, x)
# b = torch.add(w, 1)
# y = torch.mul(a, b)
#
# print(a.requires_grad, b.requires_grad, y.requires_grad)

# =====tips: 3 叶子结点不可进行in-place操作

a = torch.ones((1, ))
print(id(a), a)

a = a + torch.ones((1, ))  # 地址不变的是in-place操作
print(id(a), a)

a += torch.ones((1, ))  # 地址不变的是in-place操作
print(id(a), a)


w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

# w.add_(1)  # 做in-place操作会报错

y.backward()

