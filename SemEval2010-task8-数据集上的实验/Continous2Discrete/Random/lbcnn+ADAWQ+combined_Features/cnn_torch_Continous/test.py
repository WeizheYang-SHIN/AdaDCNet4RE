# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2022/11/5 17:01 
# @Author : wz.yang 
# @File : test.py
# @desc :

import torch
a= torch.Tensor([
    [4,1,2,0,0],
    [2,4,0,0,0],
    [1,1,1,6,5],
    [1,2,2,2,2],
    [3,0,0,0,0],
    [2,2,0,0,0]])
index = torch.LongTensor([[3],[2],[5],[5],[1],[2]])
print(a.size(),index.size())
b = torch.gather(a, 1,index-1)
print(b)