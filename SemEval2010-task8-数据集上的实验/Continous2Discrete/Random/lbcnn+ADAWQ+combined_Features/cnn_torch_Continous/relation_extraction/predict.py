import os
import re
import torch

from .data_utils import MyTokenizer, get_idx2tag, convert_pos_to_mask, SentenceREDataset, CNN_MapDataTorch
from torch.utils.data import DataLoader
from .Bert_model import SentenceRE
from .LBCNN_model import Lbcnn
from tqdm import tqdm
from sklearn import metrics
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def predict(hparams, data, vocab):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    test_file = hparams.test_file

    idx2tag = get_idx2tag(tagset_file)
    max_len = hparams.max_len
    validation_batch_size = hparams.test_batch_size
    test_batch_size = hparams.test_batch_size
    hparams.tagset_size = len(idx2tag)
    model = Lbcnn(hparams,vocab).to(device)
    # model = SentenceRE(hparams).to(device)
    model.load_state_dict(torch.load(model_file)) # '../saved_models/model.bin' 模型保存路径
    model.eval()
    tokenizer = MyTokenizer(pretrained_model_path)

    # test_dataset = SentenceREDataset(test_file, tagset_path=tagset_file,
    #                                        pretrained_model_path=pretrained_model_path,
    #                                        max_len=max_len)
    # test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    test_data = data
    test_dataset = CNN_MapDataTorch(test_data, tagset_path=tagset_file)  # 转为torch.Dataset类
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=True, shuffle=False)

    with torch.no_grad():
        tags_true = []  # 真实值
        tags_pred = []  # 预测值
        for test_i_batch, test_sample_batched in enumerate(tqdm(test_loader,desc='testing')):
            token_ids = test_sample_batched['token_ids'].to(device)
            # token_type_ids = test_sample_batched['token_type_ids'].to(device)
            # attention_mask = test_sample_batched['attention_mask'].to(device)
            e1_mask = test_sample_batched['e1_mask'].to(device)
            e2_mask = test_sample_batched['e2_mask'].to(device)
            tag_ids = test_sample_batched['tag_id']
            logits = model(token_ids, e1_mask, e2_mask)  # 验证集上的预测结果
            logits = logits.to(torch.device('cuda'))
            pred_tag_ids = logits.argmax(1)
            tags_true.extend(tag_ids.tolist())
            tags_pred.extend(pred_tag_ids.tolist())
        cls_results = metrics.classification_report(tags_true, tags_pred, labels=list(idx2tag.keys())[1:hparams.tagset_size],
                                                    target_names=list(idx2tag.values())[1: hparams.tagset_size], digits=4)
        print(cls_results)
        f1 = metrics.f1_score(tags_true, tags_pred, average='macro')
        precision = metrics.precision_score(tags_true, tags_pred, average='macro')
        recall = metrics.recall_score(tags_true, tags_pred, average='macro')

        wr_line = open('./output/PRF.txt', 'a+', encoding='utf-8')
        wr_line.write(cls_results)
        wr_line.close()
        print("predicting F1 score on test_data:***:P:", precision,"--R:", recall, "--F:", f1)
        count_max_f('./output/PRF.txt')

def count_max_f(file):
    lines = open(file,'r',encoding='utf-8').readlines()
    max_f = 0
    line_count = 0
    max_line_count = 0
    k = -10
    for line in lines:
        line_count += 1
        if 'macro avg' in line:
            element_list = line.split('    ')
            p = float(element_list[1])
            r = float(element_list[2])
            if (p + r) == 0:
                continue
            f1 = (2 * p * r) / (p + r)
            if(f1 > max_f):
                max_f = f1
                max_line_count = line_count
    for i in range(11):
        print(lines[max_line_count + k].strip())
        k += 1
    print('The true max f1:',max_f)
    print('最大值所在行占总行数的：', max_line_count / len(lines))

