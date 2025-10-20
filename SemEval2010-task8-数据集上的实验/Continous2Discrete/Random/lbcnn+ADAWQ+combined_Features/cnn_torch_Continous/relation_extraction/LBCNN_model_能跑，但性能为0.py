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

from transformers import BertTokenizer, BertModel

here = os.path.dirname(os.path.abspath(__file__))

class ConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad_(False)

class BlockLBP(nn.Module):

    def __init__(self, numChannels, numWeights, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(numChannels)
        self.conv_lbp = ConvLBP(numChannels, numWeights, kernel_size=3, sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(numWeights, numChannels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.batch_norm(x)
        x = F.relu(self.conv_lbp(x))
        x = self.conv_1x1(x)
        x.add_(residual)
        return x

class Lbcnn(nn.Module): # 使用nn.Module类定义神经网络

    def __init__(self, hparams):
        super().__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'  # 预训练模型
        self.embedding_dim = hparams.embedding_dim  # embedding维度
        self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size

        self.preprocess_block = nn.Sequential(  # torch.nn.Sequential是一个Sequential容器,快速构建网络
            nn.Conv2d(1, hparams.train_batch_size, (hparams.max_len, hparams.embedding_dim)),
            nn.BatchNorm2d(8),  # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
            nn.ReLU(inplace=True)
        )

        chain = [BlockLBP(hparams.numChannels, hparams.numWeights, hparams.sparsity) for i in range(hparams.depth)]  # 传入通道数、权重、稀疏度
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)  # 加载预训练模型
        self.chained_blocks = nn.Sequential(*chain)
        self.pool = nn.AvgPool2d(kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hparams.numChannels, hparams.full)
        self.fc2 = nn.Linear(hparams.full, 10)

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids,
                                                         attention_mask=attention_mask, return_dict=False)
        # sequence_output [8, 128, 768] [batch_size, max_len, embedding_dim]
        # Given groups=1, weight of size 1 8 128 768, expected input[8, 1, 128, 768] to have 8 channels, but got 1 channels instead
        sequence_output = sequence_output.unsqueeze(1) # [8 1 128 768]
        x = self.preprocess_block(sequence_output) # sequence_output, [8, 128, 768] [batch_size, max_len, embedding_dim]
        x = self.chained_blocks(x) # x [8,8,1,1]
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(self.dropout(x))
        x = F.relu(x)
        x = self.fc2(self.dropout(x))
        return x

    #     self.tagset_size = hparams.tagset_size
    #     self.kernel_list = "128"
    #     self.kernel_sizes = [int(k) for k in self.kernel_list.split(',')]
    #
    #     self.bert_model = BertModel.from_pretrained(self.pretrained_model_path) # 加载预训练模型
    #
    #     self.conv = nn.Conv2d(1, 8, (128, 768))
    #     # self.convs = nn.ModuleList([nn.Conv2d(1, 8, (K, 768)) for K in self.kernel_sizes])
    #     self.fc1 = nn.Linear(len(self.kernel_list) * 768, self.tagset_size)
    #
    #     self.dense = nn.Linear(self.embedding_dim, self.embedding_dim) # 线性变换
    #     self.drop = nn.Dropout(self.dropout) # 丢掉dropout的数据
    #     self.activation = nn.Tanh() # 激活函数，去线性化，解决线性模型的局限性
    #     self.norm = nn.LayerNorm(self.embedding_dim * 2) # 归一化
    #     self.hidden2tag = nn.Linear(self.embedding_dim * 2, self.tagset_size) # 线性变换、分类
    #
    # def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
    #     sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)
    #
    #     sequence_output = sequence_output.unsqueeze(1) # [8,1,128,768]
    #     conv_result = self.conv(sequence_output) # [8, 8, 1, 1]
    #     x = conv_result.squeeze(3) # [8, 8, 1]
    #     x = x.squeeze(2) # [8, 8]
    #     # x = [conv(sequence_output).squeeze(3) for conv in self.convs]
    #     # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    #
    #     # 每个实体的所有token向量的平均值
    #     sequence_output = sequence_output.squeeze(1)
    #     e1_h = self.entity_average(sequence_output, e1_mask)
    #     e2_h = self.entity_average(sequence_output, e2_mask)
    #     e1_h = self.activation(self.dense(e1_h)) # 线性变换+非线性变换
    #     e2_h = self.activation(self.dense(e2_h))
    #
    #     # [cls] + 实体1 + 实体2
    #     # concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
    #     concat_h = torch.cat([e1_h, e2_h], dim=-1)
    #     concat_h = self.norm(concat_h) # 归一化
    #     logits = self.hidden2tag(self.drop(concat_h)) # 分类
    #
    #     return logits
    #
    # @staticmethod
    # def entity_average(hidden_output, e_mask):
    #     """
    #     Average the entity hidden state vectors (H_i ~ H_j) (j-i+1)个tokens的向量表示
    #     :param hidden_output: [batch_size, j-i+1, dim]          [batch_size, max_seq_len, dim]
    #     :param e_mask: [batch_size, max_seq_len]               [batch_size, max_seq_len]
    #             e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
    #     :return: [batch_size, dim]                           [batch_size, dim]
    #     """
    #     e_mask_unsqueeze = e_mask.unsqueeze(1)  # [batch_size, 1, max_seq_len]，unsqueeze()升维、squeeze()降维
    #     length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1] # 求实体长度
    #     # torch.bmm 矩阵乘法：[b,h,w]*[b,w,m]=[b,h,m]
    #     sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
    #     avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
    #     return avg_vector # 实体表示的平均值
