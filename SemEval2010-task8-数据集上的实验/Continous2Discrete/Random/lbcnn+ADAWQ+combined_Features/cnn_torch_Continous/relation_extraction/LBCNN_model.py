# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2022/11/7 16:53 
# @Author : wz.yang 
# @File : LBCNN_model.py
# @desc :
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data_utils import *
import math

here = os.path.dirname(os.path.abspath(__file__))

class ConvLBP(nn.Conv2d): # 定义一个二维卷积类
    def __init__(self, in_channels, out_channels, kernel_size, sparsity):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters()) # 通过next(self.parameters()）获取当前模型的参数
        # 比如有个张量a，那么a.normal_()就表示用标准正态分布填充a，是in_place操作
        # 比如有个张量b，那么b.fill_(0)就表示用0填充b，是in_place操作
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5) # fill_()将tensor中的所有值都填充为指定的value（0.5）
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad_(False)

# class EntityConvLBP(nn.Conv2d): # 定义一个二维卷积类
#     def __init__(self, in_channels, out_channels, kernel_size, sparsity):
#         super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
#         weights = next(self.parameters()) # 通过next(self.parameters()）获取当前模型的参数
#         # 比如有个张量a，那么a.normal_()就表示用标准正态分布填充a，是in_place操作
#         # 比如有个张量b，那么b.fill_(0)就表示用0填充b，是in_place操作
#         matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5) # fill_()将tensor中的所有值都填充为指定的value（0.5）
#         binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
#         mask_inactive = torch.rand(matrix_proba.shape) > sparsity
#         binary_weights.masked_fill_(mask_inactive, 0)
#         weights.data = binary_weights
#         weights.requires_grad_(False)

class BlockLBP(nn.Module):

    def __init__(self, batch_size, out_dim, kernel_size, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(batch_size) # 对一个batch_size的数据归一化处理，所以参数为批次大小
        # self.conv_lbp = ConvLBP(batch_size, out_dim, 3,sparsity=sparsity)  # 自定义二维卷积操作
        # self.conv_1x1 = nn.Conv2d(out_dim, batch_size, 1)
        self.conv_lbp = ConvLBP(batch_size, out_dim, kernel_size=(kernel_size, 2), sparsity=sparsity) # 自定义二维卷积操作
        self.conv_1x1 = nn.Conv2d(out_dim, batch_size, kernel_size=(kernel_size, 1))

    def forward(self, x):
        # residual = x # [32 32 137 1]
        x = self.batch_norm(x) # 输入[ batch_size,batch_size, 卷积核卷积后的大小（max_len - kernel_size + 1） 1] 输出同等size
        x = self.conv_lbp(x) # 输出：[batch_size, 100, 124,1]
        x = F.relu(x)
        x = self.conv_1x1(x) # 32 24 124 1
        # x.add_(residual)
        return x

class Lbcnn(nn.Module): # 使用nn.Module类定义神经网络

    def __init__(self, hparams, vocab):
        super().__init__()

        # self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'  # 预训练模型
        self.num_words = hparams.num_words
        self.max_len = hparams.max_len
        self.out_dim = hparams.out_dim
        self.embedding_dim = hparams.embedding_dim  # embedding维度
        self.stride = 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.tagset_size = hparams.tagset_size
        self.activation = nn.Sigmoid()  # 激活函数，去线性化，解决线性模型的局限性
        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)  # 线性变换

        # Embedding layer definition 词典长度：self.num_words + 1、向量维度 self.embedding_dim
        self.embedding = nn.Embedding(self.num_words, self.embedding_dim, padding_idx=0)
        embeding_vector = load_word2vec(hparams.pretrained_word_vectors, self.embedding_dim , vocab)
        self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  # 表示的是在反向传播的时候, 是否对这些词向量进行求导更新

        self.preprocess_block1 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size1, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        self.preprocess_block2 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size2, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        self.preprocess_block3 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size3, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        self.preprocess_block4 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size4, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        self.preprocess_block5 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size5, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        self.preprocess_block6 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size6, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        self.preprocess_block7 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size7, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]


        self.preprocess_block8 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size8, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        self.preprocess_block9 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size9, hparams.embedding_dim)),
            # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
            nn.BatchNorm2d(hparams.train_batch_size),
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        # self.preprocess_block10 = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
        #     nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size10, hparams.embedding_dim)),
        #     # nn.Conv2d(1, hparams.train_batch_size, hparams.kernel_size, padding=1),
        #     nn.BatchNorm2d(hparams.train_batch_size),
        #     # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
        #     nn.ReLU(inplace=True)
        # )  # 输入x [hparams.train_batch_size,1,128,768] 输出x [hparams.train_batch_size,hparams.train_batch_size,1,1] [batch_size,inChannel,]

        chain1 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size1, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks1 = nn.Sequential(*chain1) # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        chain2 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size2, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks2 = nn.Sequential(*chain2)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        chain3 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size3, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks3 = nn.Sequential(*chain3)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        chain4 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size4, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks4 = nn.Sequential(*chain4)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        chain5 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size5, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks5 = nn.Sequential(*chain5)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        chain6 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size6, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks6 = nn.Sequential(*chain6)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        chain7 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size7, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks7 = nn.Sequential(*chain7)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        chain8 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size8, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks8 = nn.Sequential(*chain8)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        chain9 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size9, hparams.sparsity) for i in
                  range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.chained_blocks9 = nn.Sequential(*chain9)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表
        # chain10 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size10, hparams.sparsity) for i in
        #           range(hparams.depth)]  # 传入通道数、权重、稀疏度
        # self.chained_blocks10 = nn.Sequential(*chain10)  # 不带*的是列表，带*的是元素, nn.Sequential()的参数不能是列表

        self.pool1 = nn.AvgPool2d(kernel_size=(hparams.kernel_size1, 1), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(hparams.kernel_size2, 1), stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=(hparams.kernel_size3, 1), stride=3)
        self.pool4 = nn.MaxPool2d(kernel_size=(hparams.kernel_size4, 1), stride=4)
        self.pool5 = nn.AvgPool2d(kernel_size=(hparams.kernel_size5, 1), stride=5)
        self.pool6 = nn.MaxPool2d(kernel_size=(hparams.kernel_size6, 1), stride=6)
        self.pool7 = nn.AvgPool2d(kernel_size=(hparams.kernel_size7, 1), stride=7)
        self.pool8 = nn.MaxPool2d(kernel_size=(hparams.kernel_size8, 1), stride=8)
        self.pool9 = nn.AvgPool2d(kernel_size=(hparams.kernel_size9, 1), stride=9)
        # self.pool10 = nn.MaxPool2d(kernel_size=(hparams.kernel_size10, 1), stride=10)

        # self.fc1 = nn.Linear(hparams.train_batch_size * (hparams.max_len - hparams.kernel_size + 1), hparams.full )
        # self.fc1 = nn.Linear(9216, hparams.full)
        # self.fc2 = nn.Linear(hparams.full , self.tagset_size)
        self.fc1 = nn.Linear(30272, self.tagset_size)


    def forward(self, x, e1_mask, e2_mask): # 至此依然是正常的，word_embedding也没问题
        x_emb = self.embedding(x) # [batch_size, max_length, embedding_dim]

        # 每个实体的所有token向量的平均值
        # e1_h = self.entity_average(x_emb, e1_mask)
        # e2_h = self.entity_average(x_emb, e2_mask)
        # e1_h = self.activation(self.dense(e1_h))  # [8, 768] 线性变换+非线性变换
        # e2_h = self.activation(self.dense(e2_h)) # [8, 768] 线性变换+非线性变换

        sequence_output = x_emb.unsqueeze(1) # [batch_size,1,max_len,embedding_dim]
        x = self.preprocess_block1(sequence_output) # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks1(x) # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x1 = self.pool1(x) # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x = self.preprocess_block2(sequence_output)  # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks2(x)  # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x2 = self.pool2(x)  # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x = self.preprocess_block3(sequence_output)  # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks3(x)  # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x3 = self.pool3(x)  # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x = self.preprocess_block4(sequence_output)  # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks4(x)  # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x4 = self.pool4(x)  # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x = self.preprocess_block5(sequence_output)  # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks5(x)  # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x5 = self.pool5(x)  # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x = self.preprocess_block6(sequence_output)  # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks6(x)  # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x6 = self.pool6(x)  # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x = self.preprocess_block7(sequence_output)  # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks7(x)  # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x7 = self.pool7(x)  # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x = self.preprocess_block8(sequence_output)  # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks8(x)  # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x8 = self.pool8(x)  # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x = self.preprocess_block9(sequence_output)  # 输入x [8,1,128,768] 执行二维卷积、归一化、激活，输出x  [batch_size,inChannel,]
        x = self.chained_blocks9(x)  # 输入x [8,8,1,1] 输出x [batch_size, 卷积核卷积之后的大小,8,1]
        x9 = self.pool9(x)  # 输入x [8,8,1,1] 输出x [8,8,1,1] [batch_size,batch_size,卷积核卷积之后的大小，1] [3 3 156 1]

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x4 = x4.view(x4.shape[0], -1)
        x5 = x3.view(x5.shape[0], -1)
        x6 = x4.view(x6.shape[0], -1)
        x7 = x7.view(x7.shape[0], -1)
        x8 = x8.view(x8.shape[0], -1)
        x9 = x9.view(x9.shape[0], -1)
        # x10 = x10.view(x10.shape[0], -1)

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9], 1)
        x = self.fc1(self.dropout(x))
        out = self.dropout(x)

        out = out.squeeze()

        return out