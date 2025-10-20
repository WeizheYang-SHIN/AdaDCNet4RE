import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from sklearn import metrics

from .data_utils import CNN_MapDataTorch, get_idx2tag, load_checkpoint, save_checkpoint, prepare_data
from .binirilize_lbcnn_model import Lbcnn, save_convlbp_weights_to_txt, ConvLBP#, quantize_weights_hook
# from .LBCNN_adjust_model import Lbcnn
# from .Bert_model import SentenceRE
# from .CNN_model import CNN
# from .predict_CNN import predict
from .predict_LBCNN import predict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
here = os.path.dirname(os.path.abspath(__file__))

# def quantize_weights_hook(module, grad_input, grad_output):
#     if isinstance(module, ConvLBP):
#         module.weight.data = module.quantize_weights(module.weight.data)


def train(hparams):
    with open('get_convlbp_weights.txt','w',encoding='utf-8') as w:
        pass
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    # pretrained_model_path = hparams.pretrained_model_path
    # train_file = hparams.train_file
    validation_file = hparams.validation_file
    log_dir = hparams.log_dir
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    checkpoint_file = hparams.checkpoint_file

    train_batch_size = hparams.train_batch_size
    validation_batch_size = hparams.validation_batch_size
    epochs = hparams.epochs

    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay

    # Preprocessing CNN
    train_data, val_data, test_data, vocab = prepare_data(hparams)
    train_dataset = CNN_MapDataTorch(train_data, tagset_path=tagset_file)# 转为torch.Dataset类
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=False, shuffle=True)

    # train_dataset
    # train_dataset = SentenceREDataset(train_file, tagset_path=tagset_file,
    #                                   pretrained_model_path=pretrained_model_path,
    #                                   max_len=max_len) # 处理训练集数据，将tokens、tags、e_mask等信息取出
    # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True ,shuffle=True) # 把数据分批次

    # model
    idx2tag = get_idx2tag(tagset_file) # 得到[数字：关系类型]的字典数据
    hparams.tagset_size = len(idx2tag) # 关系类型数量
    model = Lbcnn(hparams, vocab).to(device) # 初始化LBCNN模型
    # model = Lbcnn(hparams).to(device) # 初始化LBCNN模型
    # model = SentenceRE(hparams).to(device) # 构造Bert神经网络，将数据传到device上运算， tensor和numpy都是矩阵，前者能在GPU上运行，后者只能在CPU运行，所以要注意数据类型的转换。

    # load checkpoint if one exists
    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_file)) # 加载保存的模型
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0

    # optimizer 构建优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 注册钩子，在每次反向传播后量化权重
    # for name, module in model.named_modules():
    #     if isinstance(module, ConvLBP):
    #         module.register_backward_hook(quantize_weights_hook)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    running_loss = 0.0
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        model.train() # model.train()，作用是 启用 batch normalization 和 dropout
        for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
            token_ids = sample_batched['token_ids'].to(device)
            # token_type_ids = sample_batched['token_type_ids'].to(device)
            # attention_mask = sample_batched['attention_mask'].to(device)
            e1_mask = sample_batched['e1_mask'].to(device)
            e2_mask = sample_batched['e2_mask'].to(device)
            tag_ids = sample_batched['tag_id'].to(device)
            # tag_ids = sample_batched['tag_id'].type(torch.FloatTensor).to(device)
            model.zero_grad() # model.zero_grad()的作用是将所有模型参数的梯度置为0
            logits = model(token_ids, e1_mask, e2_mask) # [batch_size, tagset_size] [3,7]
            # logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            loss = criterion(logits, tag_ids)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()  # 清零梯度，为下一轮迭代做准备
        
            running_loss += loss.item()
            if i_batch % 10 == 9:
                writer.add_scalar('Training/training loss', running_loss / 10, epoch * len(train_loader) + i_batch)
                running_loss = 0.0
        # 每个epoch结束时，更新ConvLBP中的权重
        # if epoch % 1 == 0:
        #     for block in model.chained_blocks1:
        #         block.conv_lbp.update_weight()

        if validation_file:
            # validation_data = prepare_data(hparams, mode=hparams.validation_file)
            validation_dataset = CNN_MapDataTorch(val_data, tagset_path=tagset_file)  # 转为torch.Dataset类
            val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, drop_last=False, shuffle=False)

            # validation_dataset = SentenceREDataset(validation_file, tagset_path=tagset_file,
            #                                        pretrained_model_path=pretrained_model_path,
            #                                        max_len=max_len)
            # val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
            model.eval()
            with torch.no_grad(): # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
                tags_true = [] # 真实值
                tags_pred = [] # 预测值
                for val_i_batch, val_sample_batched in enumerate(tqdm(val_loader, desc='Validation')):
                    token_ids = val_sample_batched['token_ids'].to(device)
                    # token_type_ids = val_sample_batched['token_type_ids'].to(device)
                    # attention_mask = val_sample_batched['attention_mask'].to(device)
                    e1_mask = val_sample_batched['e1_mask'].to(device)
                    e2_mask = val_sample_batched['e2_mask'].to(device)
                    tag_ids = val_sample_batched['tag_id']
                    logits = model(token_ids, e1_mask, e2_mask)  # 验证集上的预测结果
                    # logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask) # 验证集上的预测结果
                    pred_tag_ids = logits.argmax(1)
                    tags_true.extend(tag_ids.tolist())
                    tags_pred.extend(pred_tag_ids.tolist())

                # print(metrics.classification_report(tags_true, tags_pred, labels=list(idx2tag.keys())[1:],target_names=list(idx2tag.values())[1:], digits=4))
                # print(metrics.classification_report(tags_true, tags_pred, labels=list(idx2tag.keys()), target_names=list(idx2tag.values()),digits=4))
                # f1 = metrics.f1_score(tags_true, tags_pred, average='weight')
                f1 = metrics.f1_score(tags_true, tags_pred, average='macro')
                precision = metrics.precision_score(tags_true, tags_pred, average='macro')
                recall = metrics.recall_score(tags_true, tags_pred, average='macro')
                accuracy = metrics.accuracy_score(tags_true, tags_pred)
                writer.add_scalar('Validation/f1', f1, epoch)
                writer.add_scalar('Validation/precision', precision, epoch)
                writer.add_scalar('Validation/recall', recall, epoch)
                writer.add_scalar('Validation/accuracy', accuracy, epoch)

                if checkpoint_dict.get('epoch_f1'):
                    checkpoint_dict['epoch_f1'][epoch] = f1
                else:
                    checkpoint_dict['epoch_f1'] = {epoch: f1}
                if f1 > best_f1:
                    best_f1 = f1
                    checkpoint_dict['best_f1'] = best_f1
                    checkpoint_dict['best_epoch'] = epoch
                    torch.save(model.state_dict(), model_file) # model.state_dict() 能够获取 模型中的所有参数，包括可学习参数和不可学习参数，其返回值是一个有序字典 OrderedDict。
                print("Model is training epoch:", epoch, ", validition score on val_data-----------------------P:",precision, "--R:", recall, "--F:", f1,'--best f1:',best_f1 )
                save_checkpoint(checkpoint_dict, checkpoint_file)

                # 在每个 epoch 结束时保存权重
                save_convlbp_weights_to_txt(epoch, model, f'get_convlbp_weights.txt')
                predict(hparams, test_data, vocab)

    writer.close()
