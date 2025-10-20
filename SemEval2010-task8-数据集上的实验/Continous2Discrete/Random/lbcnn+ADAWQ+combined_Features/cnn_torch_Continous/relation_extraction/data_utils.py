import re
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from relation_extraction.hparams import hparams

here = os.path.dirname(os.path.abspath(__file__))


def tokenize(text):
    result = []
    for word in text.split():
        result.append(word)
    return result

class MyTokenizer(object):
    def __init__(self, pretrained_model_path=None, mask_entity=False):
        # self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        # self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path) # 分词器，未识别出的会用unk替代
        self.mask_entity = mask_entity

    def mytokenize(self, text, e1, e2, entity_1_type, entity_2_type):
        sentence = text # 拿到句子
        # if "1,200" in sentence:
        #     sentence = sentence.replace("1,200","1200")
        # if "5,000" in sentence:
        #     sentence = sentence.replace("5,000","5000")
        # if "CitigroupInc.,C" in sentence:
        #     sentence = sentence.replace("CitigroupInc.,C", "CitigroupInc.C")
        try:
            start1, end1 = sentence.index(e1), sentence.index(e1) + len(e1)
            start2, end2 = sentence.index(e2), sentence.index(e2) + len(e2)
            # start1, end1 = sentence.index('<e1>'), sentence.index('<e1>') + len(e1 + '</e1>')
            # start2, end2 = sentence.index('<e2>'), sentence.index('<e2>') + len(e2 + '</e2>')
        except:
            print(sentence)
            print(e1)
            print(e2)
        pos_head = [start1, end1] # 头实体的位置，[start，end]
        pos_tail = [start2, end2] # 尾实体的位置，[start，end]
        if pos_head[0] > pos_tail[0]: # 如果头实体在后边
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else: # 如果头实体在前边
            pos_min = pos_head
            pos_max = pos_tail
            rev = False
        # pos_min 保存前边实体的位置，pos_max保存后边实体的位置

        sent0 = tokenize(sentence[:pos_min[0]])
        ent0 = tokenize(sentence[pos_min[0]:pos_min[1]]) # 对头实体分词
        sent1 = tokenize(sentence[pos_min[1]:pos_max[0]]) # 对中间部分分词
        ent1 = tokenize(sentence[pos_max[0]:pos_max[1]]) # 对尾实体分词
        sent2 = tokenize(sentence[pos_max[1]:]) # 对右边部分分词

        # asent0 = self.bert_tokenizer.tokenize(sentence[:pos_min[0]]) # 对左边部分分词
        # aent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]]) # 对头实体分词
        # asent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]]) # 对中间部分分词
        # aent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]]) # 对尾实体分词
        # asent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:]) # 对右边部分分词

        if rev: # 如果头实体和尾实体位置反转
            if self.mask_entity: # 如果实体掩盖
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)] # 尾实体的位置
            pos_head = [ # 头实体的位置
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)] # 头实体的位置
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2 # 把实体对在文本中的位置调整对

        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = [0, 0]
        pos2 = [0, 0]
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1[0] = len(re_tokens)
                # re_tokens.append('[unused1]')
                re_tokens.append('<e1>')
            if cur_pos == pos_tail[0]:
                pos2[0] = len(re_tokens)
                #re_tokens.append('[unused2]')
                re_tokens.append('<e2>')
            re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                #re_tokens.append('[unused3]')
                re_tokens.append('</e1>')
                pos1[1] = len(re_tokens)
            if cur_pos == pos_tail[1] - 1:
                # re_tokens.append('[unused4]')
                re_tokens.append('</e2>')
                pos2[1] = len(re_tokens)
            cur_pos += 1
        re_tokens.append('[SEP]')
        return re_tokens[1:-1], pos1, pos2 # 在实体对两边加入标签token，并算入实体的开始结束位置，返回的tokens序列不包含CLS、SEP字符，但是实体位置算入该占位符

def get_relation_list():
    f = open(hparams.tagset_file, 'r', encoding='utf-8')
    lines = f.readlines()
    label = []
    for line in lines:
        label.append(line.strip())
    f.close()
    return label

def combine(f1 ,f2):
    return str(f1) + '_' + str(f2)

class Preprocessing:

    def __init__(self, hparams):
        # self.data = './data/tweets.csv'
        self.train_data = hparams.train_file
        self.val_data = hparams.validation_file
        self.test_data = hparams.test_file
        self.data = self.train_data
        self.num_words = hparams.num_words
        self.seq_len = hparams.max_len
        self.vocabulary = None
        self.x_tokenized = None
        self.x_padded = None
        self.x_raw = None
        self.y = None

        self.x_train = None
        # self.x_test = None
        self.y_train = None
        # self.y_test = None
        self.train_size = None
        self.val_size = None


    def load_data(self):
        # Reads the raw csv file and split into
        # sentences (x) and target (y)
        relation_list = get_relation_list()
        with open(self.train_data, 'r', encoding='utf-8') as tr, open(self.test_data, 'r', encoding='utf-8') as te:
            all_lines = json.load(tr)
            train_lines = all_lines[0:int(len(all_lines) * 0.8)]
            val_lines = all_lines[int(len(all_lines) * 0.8):]
            test_lines = json.load(te)
            self.train_size = len(train_lines)
            self.val_size = len(val_lines)
            self.test_size = len(test_lines)
            lines = train_lines + val_lines + test_lines
        num_instance = 0
        tokenizer = None
        tokens_list = []  # 该列表的元素是每句话的tokens组成的列表，也就是一句话由berttokenizer分词后的列表，包含实体标签unused1、unused2、unused3、unused4，unk
        e1_mask_list = []  # 该列表的元素是每句话的e1位置为1，其余位置为0组成的列表
        e2_mask_list = []  # 该列表的元素是每句话的e2位置为1，其余位置为0组成的列表
        max_len = self.seq_len
        tags = []
        # relation_set, entity_types = get_relation_entity_types(self.data)
        for line in tqdm(lines):
            entity_position = line['entity_position'].strip()  # 实体相对位置
            left = ''.join(line['left'].strip())  # 左边部分
            entity_1 = ''.join(line['e1'].strip())  # 实体1
            middle = ''.join(line['middle'].strip())
            entity_2 = ''.join(line['e2'].strip())
            right = ''.join(line['right'].strip())
            entity_1_type = ''.join(line['e1_type'].strip())
            entity_2_type = ''.join(line['e2_type'].strip())

            LeftPos_Entity1 = ''.join(line['LeftPOSofE1'].strip())
            RightPos_Entity1 = ''.join(line['RightPOSofE1'].strip())
            LeftPos_Entity2 = ''.join(line['LeftPOSofE2'].strip())
            RightPos_Entity2 = ''.join(line['RightPOSofE2'].strip())

            Feature0 = combine(entity_1, entity_2)
            Feature1 = combine(RightPos_Entity1, entity_1_type)
            Feature2 = combine(LeftPos_Entity1, entity_1_type)
            Feature3 = combine(RightPos_Entity2, entity_2_type)
            Feature4 = combine(LeftPos_Entity2, entity_2_type)
            Feature5 = combine(entity_1_type, entity_2_type)

            sentence = ''.join(line['sentence'].strip())
            relation = line['relation'].strip()
            if tokenizer is None:
                tokenizer = MyTokenizer()

            sentence = sentence.replace('<e1>','').replace('</e1>','').replace('<e2>','').replace('</e2>','')
            entity1_instead = ' <' + entity_1_type + '_1> ' + entity_1 + ' </' + entity_1_type + '_1> '
            entity2_instead = ' <' + entity_2_type + '_2> ' + entity_2 + ' </' + entity_2_type + '_2> '
            atomic_features = entity_position + ' ' + entity1_instead + ' ' + entity_1_type + ' ' + LeftPos_Entity1 + ' ' + RightPos_Entity1 + ' ' + entity2_instead + ' ' + entity_2_type + ' ' + LeftPos_Entity2 + ' ' + RightPos_Entity2
            combined_features = entity_position + ' ' + Feature0 + ' ' + Feature1 + ' ' + Feature2 + ' ' + Feature3 + ' ' + Feature4 + ' ' + Feature5

            all_features = combined_features + ' ' + sentence
            tokens, pos_e1, pos_e2 = tokenizer.mytokenize(all_features, entity_1, entity_2, entity_1_type, entity_2_type)  # 拿到包含实体标签的tokens，以及实体开始结束位置
            if num_instance < 5:
                print('text：', all_features)
                print('processed_text:', tokens)
                print('e1:', entity_1)
                print('e2:', entity_2)
                num_instance += 1
            # if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
            #         pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:  # 把实体位置在max_len之外的句子抛弃
            tokens_list.append(tokens)
            e1_mask = convert_pos_to_mask(pos_e1, max_len)
            e2_mask = convert_pos_to_mask(pos_e2, max_len)
            e1_mask_list.append(e1_mask)
            e2_mask_list.append(e2_mask)
            tag = relation
            tags.append(tag)
        self.x_raw = tokens_list # 文本列表
        self.e1_mask = e1_mask_list
        self.e2_mask = e2_mask_list
        self.y = tags


    def clean_text(self):
        # Removes special symbols and just keep
        # words in lower or upper form

        self.x_raw = [x.lower() for x in self.x_raw] # 英文单词大写转小写
        self.x_raw = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_raw]

    def text_tokenization(self):
        # Tokenizes each sentence by implementing the nltk tool
        self.x_raw = [word_tokenize(x) for x in self.x_raw]

    def build_vocabulary(self):
        # Builds the vocabulary and keeps the "x" most frequent words
        self.vocabulary = dict()
        fdist = nltk.FreqDist()

        for sentence in self.x_raw:
            for word in sentence:
                fdist[word] += 1

        common_words = fdist.most_common(self.num_words)

        for idx, word in enumerate(common_words[1:]):
            self.vocabulary[word[0]] = (idx + 1)

    def word_to_idx(self):
        # By using the dictionary (vocabulary), it is transformed
        # each token into its index based representation
        self.x_tokenized = list()

        for sentence in self.x_raw:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
            self.x_tokenized.append(temp_sentence) # 把句子中无效字（不在字典中的）去除

    def padding_sentences(self): # 长的剪裁、短的补0
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0

        pad_idx = 0
        self.x_padded = list() # 列表里边套列表

        for sentence in self.x_tokenized:
            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            # self.x_padded.append(sentence)
            # if len(sentence) != self.seq_len:
            #     pass
            if len(sentence) > self.seq_len:
                self.x_padded.append(sentence[0:self.seq_len])
            else:
                self.x_padded.append(sentence)

        self.x_padded = np.array(self.x_padded) # array类型的数据，里边存放的是列表
        # print("我就想看一眼数据类型")
        self.x_train, self.y_train = self.x_padded[0:self.train_size], self.y[0:self.train_size]
        self.x_val, self.y_val = self.x_padded[self.train_size:self.train_size + self.val_size], self.y[self.train_size:self.train_size + self.val_size]
        self.x_test, self.y_test = self.x_padded[self.train_size + self.val_size:], self.y[self.train_size + self.val_size:]

        self.e1_mask_train,self.e2_mask_train = self.e1_mask[0:self.train_size], self.e2_mask[0:self.train_size]
        self.e1_mask_val,self.e2_mask_val = self.e1_mask[self.train_size:self.train_size + self.val_size], self.e2_mask[self.train_size:self.train_size + self.val_size]
        self.e1_mask_test, self.e2_mask_test = self.e1_mask[self.train_size + self.val_size:], self.e2_mask[self.train_size + self.val_size:]

def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        try:
            e_pos_mask[i] = 1
        except:
            e_pos_mask[max_len - 1] = 1
    return e_pos_mask # 返回长度为128，实体位置为1，其余位置为0的列表


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip()) # 将字符串按照空格分割成一个列表
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict

def prepare_data(hparams):
    # Preprocessing pipeline
    pr = Preprocessing(hparams) # 初始化类变量，包括词典、分词器、句长、文本、标签
    pr.load_data() # 加载文本、标签
    # pr.clean_text()
    # pr.text_tokenization()
    pr.build_vocabulary() # 生成字典存入vocabulary（dict类型变量），元素是    字：id
    pr.word_to_idx() # 把每一句话中的字转换成id，即把句子转换成id列表，去掉字典中没有的字
    pr.padding_sentences() # 把存放字id的列表扩充0到长度为max_len，将每一个列表放入ndarray类型的变量中
    # pr.split_data() # 把数据集分为train和test
    train_dataset = {'x': pr.x_train, 'y': pr.y_train, 'e1_mask': pr.e1_mask_train, 'e2_mask': pr.e2_mask_train}
    val_dataset = {'x': pr.x_val, 'y': pr.y_val, 'e1_mask': pr.e1_mask_val, 'e2_mask': pr.e2_mask_val}
    test_dataset = {'x': pr.x_test, 'y': pr.y_test, 'e1_mask': pr.e1_mask_test, 'e2_mask': pr.e2_mask_test}

    return train_dataset, val_dataset, test_dataset, pr.vocabulary

class CNN_MapDataTorch(Dataset):
    def __init__(self, data, tagset_path):
        self.x = data['x']
        self.y = data['y']
        self.e1_mask = data['e1_mask']
        self.e2_mask = data['e2_mask']
        self.tag2idx = get_tag2idx(tagset_path)  # 一个关系类型对应数字的字典

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.x[idx]
        sample_e1_mask = self.e1_mask[idx]
        sample_e2_mask = self.e2_mask[idx]
        sample_tag = self.y[idx]

        sample_tag_id = self.tag2idx[sample_tag]

        sample = {
            # 'token_ids': torch.tensor(sample_tokens),
            # 'e1_mask': torch.tensor(sample_e1_mask),
            # 'e2_mask': torch.tensor(sample_e2_mask),
            # 'tag_id': torch.tensor(sample_tag_id)

            'token_ids': torch.LongTensor(sample_tokens),
            'e1_mask': torch.LongTensor(sample_e1_mask),
            'e2_mask': torch.LongTensor(sample_e2_mask),
            'tag_id': torch.tensor(sample_tag_id)
        }
        return sample

# 加载中文字向量wiki_100.txt
# def load_word2vec(embedding_path, embedding_dim, vocab):
#
#     # initial matrix with random uniform
#     # initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
#     initW = np.random.randn(len(vocab)+1, embedding_dim).astype(np.float32) / np.sqrt(len(vocab))
#     # load any vectors from the word2vec
#     print("Load word2vec file {0}".format(embedding_path))
#     with open(embedding_path, "r",encoding='utf-8' ) as f:
#         lines = f.readlines()
#         # vocab_size = len(lines)
#         for line in lines:
#             word = line.split(' ')[0]
#             embedding = ' '.join(line.split(' ')[1:])
#             idx = vocab.get(word)
#             if idx != 0 and idx is not None:
#                 word_embedding = np.fromstring(embedding, dtype='float32',sep = ' ')
#                 initW[idx] = word_embedding
#             else:
#                 continue
#     return initW

def load_word2vec(embedding_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab)+1, embedding_dim).astype(np.float32) / np.sqrt(len(vocab))
    # load any vectors from the word2vec
    # print("Load word2vec file {0}".format(embedding_path))
    # with open(embedding_path, "rb" ) as f:
    #     header = f.readline()
    #     vocab_size, layer_size = map(int, header.split())
    #     binary_len = np.dtype('float32').itemsize * layer_size
    #     for line in range(vocab_size):
    #         word = []
    #         while True:
    #             ch = f.read(1).decode('latin-1')
    #             if ch == ' ':
    #                 word = ''.join(word)
    #                 break
    #             if ch != '\n':
    #                 word.append(ch)
    #         if word in vocab:
    #             idx = vocab[word]
    #         else:
    #             idx = 0
    #         if idx != 0:
    #             initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
    #         else:
    #             f.read(binary_len)
    return initW


# class SentenceREDataset(Dataset): # Dataloader的处理逻辑是先通过Dataset类里面的 __getitem__ 函数获取单个的数据，然后组合成batch
#     def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128):
#         self.data_file_path = data_file_path
#         self.tagset_path = tagset_path
#         self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
#         self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)
#         self.max_len = max_len
#         self.tokens_list, self.e1_mask_list, self.e2_mask_list, self.tags = read_data(data_file_path, tokenizer=self.tokenizer, max_len=self.max_len)
#         self.tag2idx = get_tag2idx(self.tagset_path) # 一个关系类型对应数字的字典
#
#     def __len__(self):
#         return len(self.tags)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         sample_tokens = self.tokens_list[idx]
#         sample_e1_mask = self.e1_mask_list[idx]
#         sample_e2_mask = self.e2_mask_list[idx]
#         sample_tag = self.tags[idx]
#         encoded = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
#         sample_token_ids = encoded['input_ids']
#         sample_token_type_ids = encoded['token_type_ids']
#         sample_attention_mask = encoded['attention_mask']
#         sample_tag_id = self.tag2idx[sample_tag]
#
#         sample = {
#             'token_ids': torch.tensor(sample_token_ids),
#             'token_type_ids': torch.tensor(sample_token_type_ids),
#             'attention_mask': torch.tensor(sample_attention_mask),
#             'e1_mask': torch.tensor(sample_e1_mask),
#             'e2_mask': torch.tensor(sample_e2_mask),
#             'tag_id': torch.tensor(sample_tag_id)
#         }
#         return sample
