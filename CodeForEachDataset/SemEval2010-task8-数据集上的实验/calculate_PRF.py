# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2024/3/22 9:15 
# @Author : wz.yang 
# @File : calculate_PRF.py
# @desc :

from sklearn import metrics
import re
tagset_file = './tags.txt'
true_file = './true.txt'
chatgpt_file = './chatgpt.txt'
chatgpt4_file = './chatgpt4.txt'
sgan_file = './DisBiCoN.txt'

def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))

def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))

idx2tag = get_idx2tag(tagset_file)
tag2idx = get_tag2idx(tagset_file)
truths = []
preds = []

true_labels = open(true_file,'r',encoding='utf-8').readlines()
pred_labels = open(sgan_file,'r',encoding='utf-8').readlines()
for id, true_label in enumerate(true_labels):
    test_label = true_label.strip()
    truths.append(tag2idx[test_label])
    pred_label = pred_labels[id].strip()
    preds.append(tag2idx[pred_label])

# for labels in pred_labels:
#     if len(labels) != 0:
#         preds.append(labels[0])
#     else:
#         preds.append(0)

cls_results = metrics.classification_report(truths, preds,
                                            labels=list(idx2tag.keys())[1:len(idx2tag)],
                                            target_names=list(idx2tag.values())[1: len(idx2tag)],
                                            digits=4)
print(cls_results)
cls_results_list = cls_results.split('\n')
f1 = 0
for line in cls_results_list:
    if 'macro avg' in line:
        line = line.strip()
        f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(
            line.split('\t')[0].split('    ')[2].strip()) / (
                         float(line.split('\t')[0].split('    ')[1].strip()) + float(
                     line.split('\t')[0].split('    ')[2].strip()))
print("f1:", f1)
# wr_line = open('./output/PRF.txt', 'a+', encoding='utf-8')
# check_p_name = open('./output/models/checkpoint', 'r', encoding='utf-8').readline()
# wr_line.write(check_p_name + '\n')
# wr_line.write(cls_results)
# wr_line.close()
# wr_line = open('./output/this_PRF.txt', 'w', encoding='utf-8')
# wr_line.write(cls_results)
# wr_line.close()
# lines = open('./output/this_PRF.txt', 'r', encoding='utf-8').readlines()
# f1 = 0
# for line in cls_results:
#     print(line)
#     if 'macro avg' in line:
#         f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(
#             line.split('\t')[0].split('    ')[2].strip()) / (
#                      float(line.split('\t')[0].split('    ')[1].strip()) + float(
#                  line.split('\t')[0].split('    ')[2].strip()))
# print('The True F1 is ', f1)