# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2022/11/12 9:39 
# @Author : wz.yang 
# @File : CNN_model.py
# @desc :
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from .data_utils import *

class CNN(nn.Module):
    def __init__(self, hparams, vocab):
        super(CNN, self).__init__()
        # 数据参数定义
        self.max_len = hparams.max_len
        self.num_words = hparams.num_words
        self.embedding_dim = hparams.embedding_dim
        self.dropout = nn.Dropout(0.20)
        self.tagset_size = hparams.tagset_size
        self.activation = nn.Sigmoid()
        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)  # 线性变换

        # CNN参数定义
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # Output size of each convolution
        self.out_dim = hparams.out_dim
        # Number of strides for each convolution
        self.stride = hparams.stride
        # Embedding layer definition 词典长度：self.num_words + 1、向量维度 self.embedding_dim
        self.embedding = nn.Embedding(self.num_words, self.embedding_dim, padding_idx=0)
        embeding_vector = load_word2vec(hparams.pretrained_word_vectors, 100, vocab)
        self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector)) # 表示的是在反向传播的时候, 是否对这些词向量进行求导更新
        # self.embedding.weight.requires_grad = False
        # Convolution layer definition
        # self.max_len = 20
        self.conv_1 = nn.Conv1d(self.max_len, self.out_dim, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.max_len, self.out_dim, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.max_len, self.out_dim, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.max_len, self.out_dim, self.kernel_4, self.stride)
        # Max pooling layer definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)
        # Fully connection layer definition
        # self.fc = nn.Linear(38000, self.tagset_size)
        self.fc = nn.Linear(self.in_feature_fc(), self.tagset_size)

    def in_feature_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling
           Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
           Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
           source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
           '''
        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_dim - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1) # 输入为一个数字，返回值为一个整型数字，表示向下取整
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_dim - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)  # 输入为一个数字，返回值为一个整型数字，表示向下取整
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_dim - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.embedding_dim - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_dim

    def conv_operation(self, x):
        # Convolution layer 1 is applied
        x1 = self.conv_1(x) # 32 50 2
        x1 = torch.relu(x1) # 32 50 2
        x1 = self.pool_1(x1) #

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu(x2)
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        # Convolution layer 4 is applied
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        # The output of each conv is concated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2) # 32 50 8
        union = union.reshape(union.size(0), -1)
        return union # 32 400

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j) (j-i+1)个tokens的向量表示
        :param hidden_output: [batch_size, j-i+1, dim]          [batch_size, max_seq_len, dim]
        :param e_mask: [batch_size, max_seq_len]               [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]                           [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [batch_size, 1, max_seq_len]，unsqueeze()升维、squeeze()降维
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1] # 求实体长度
        # torch.bmm 矩阵乘法：[b,h,w]*[b,w,m]=[b,h,m]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector  # 实体表示的平均值

    # def batch_embedding(self, x_emb, e_mask):
    #     batch_e_emb = []
    #     batch_size = x_emb.size()[0]
    #     for i in range(0, batch_size):
    #         e_emb = []
    #         e_index_start = e_mask[i].index(1)
    #         e_index_end = e_index_start + 10
    #         if e_index_end > len(e_mask[i]):
    #             e_index_end = len(e_mask[i])
    #
    #         # e_index = [j for j, k in enumerate(e_mask[i]) if k == 1]
    #         # if len(e_index) > 10:
    #         #     e_index = e_index[0:10]
    #         # else:
    #         #     e_index.extend([0] * (10 - len(e_index)))
    #         sentence_emb = x_emb[i]
    #         e_emb = sentence_emb[e_index_start:e_index_end]
    #         if len(e_emb) < 10:
    #             e_emb = e_emb.extend(sentence_emb[0] * (10 - len(e_emb)))
    #         # for p in e_index:
    #         #     e_emb.append(sentence_emb[p])
    #         e_emb = torch.stack(e_emb, 0)
    #         # e1_emb = torch.LongTensor(e1_emb)
    #         # e1_emb = torch.LongTensor(e1_emb)
    #         # print(e1_emb)
    #         # print(e1_emb.size())
    #         batch_e_emb.append(e_emb)
    #     batch_e_emb = torch.stack(batch_e_emb, 0)
    #     return batch_e_emb

    def batch_embedding(self, x_emb, e_mask, length):
        batch_e_emb = []
        batch_size = x_emb.size()[0]
        for i in range(0, batch_size):
            e_mask_list = e_mask[i].cpu().numpy().tolist()
            e_index_start = e_mask_list.index(1)
            e_index_end = e_index_start + length
            if e_index_end > len(e_mask[i]):
                e_index_end = len(e_mask[i])
            sentence_emb = x_emb[i]
            e_emb = sentence_emb[e_index_start:e_index_end]
            ex = sentence_emb[0].unsqueeze(0)
            if len(e_emb) < length:
                extend_emb = torch.cat([ex] * (length - len(e_emb)), 0)
                e_emb = torch.cat([extend_emb, e_emb], 0)
            batch_e_emb.append(e_emb)
        batch_e_emb = torch.stack(batch_e_emb, 0)
        return batch_e_emb

    def forward(self, x, e1_mask, e2_mask): # x:[batch_size, max_len] e1_mask:[batch_size, max_len] e2_mask:[batch_size, max_len]
        # Embedding 层映射输入
        x_emb = self.embedding(x) # [batch_size, seq_len, embedding_dim]:[32,158,768]
        # e1_emb = self.batch_embedding(x_emb, e1_mask, 20)
        # e2_emb = self.batch_embedding(x_emb, e2_mask, 20)
        conv_x = self.conv_operation(x_emb)
        # conv_e1 = self.conv_operation(e1_emb) # 32, 19000
        # conv_e2 = self.conv_operation(e2_emb) # 32, 19000
        # union = torch.cat((conv_e1, conv_e2), 1)  # 32 38000
        # fully connection
        out = self.fc(conv_x)
        # Dropout
        out = self.dropout(out)
        # activation function is applied
        # out1 = torch.sigmoid(out)
        # out2 = F.softmax(out)

        out = out.squeeze()

        return out

