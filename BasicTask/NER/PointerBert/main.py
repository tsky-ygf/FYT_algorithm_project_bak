#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/08 15:22
# @Author  : Czq
# @File    : run_qa.py
# @Software: PyCharm
import os
# from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from BasicTask.NER.BertNer.metrics import SpanEntityScore

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
from pprint import pprint
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from BasicTask.NER.PointerBert.utils import load_data, set_seed, ReaderDataset, batchify, read_config_to_label, \
    evaluate_index, batchify_cluener, evaluate_entity_wo_category, bert_extract_item, read_config_to_label_aux
from BasicTask.NER.PointerBert.model_NER import PointerNERBERT,  PointerNERBERT2


def compute_kl_loss(self, p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.concat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.concat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss



def train(args, train_loader, model, optimizer):
    print('-' * 50 + 'training' + '-' * 50)
    total_loss = 0
    num_samples = 0
    entities = []
    for i, samples in enumerate(train_loader):
        optimizer.zero_grad()
        encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
        sentences = samples[4]
        starts_aux, ends_aux = samples[5]
        start_prob, end_prob, start_prob2, end_prob2 = model(encoded_dicts)
        # start_loss = torch.nn.functional.binary_cross_entropy(input=start_prob, target=starts, reduction="sum")
        # end_loss = torch.nn.functional.binary_cross_entropy(input=end_prob, target=ends, reduction="sum")
        start_loss = torch.nn.functional.binary_cross_entropy(input=start_prob, target=starts)
        end_loss = torch.nn.functional.binary_cross_entropy(input=end_prob, target=ends)
        start_loss_aux = torch.nn.functional.binary_cross_entropy(input=start_prob2, target=starts_aux)
        end_loss_aux = torch.nn.functional.binary_cross_entropy(input=end_prob2, target=ends_aux)

        # start_prob = start_prob.contiguous().view(-1, start_prob.shape[-1])
        # end_prob = end_prob.contiguous().view(-1,end_prob.shape[-1])
        # starts = starts.contiguous().view(-1, starts.shape[-1])
        # ends = ends.contiguous().view(-1,ends.shape[-1])
        # start_loss = multilabel_categorical_crossentropy(y_true=starts, y_pred=start_prob)
        # end_loss = multilabel_categorical_crossentropy(y_true=ends, y_pred=end_prob)

        # PFLoss = Poly1FocalLoss(num_classes=len(args.labels),
        #                       reduction='sum',
        #                       label_is_onehot=True,
        #                       pos_weight=None)
        # start_loss = PFLoss(logits=start_prob, labels=starts)
        # end_loss = PFLoss(logits=end_prob, labels=ends)

        # start_prob = start_prob.contiguous().view(-1,len(args.labels))
        # end_prob = end_prob.contiguous().view(-1,len(args.labels))
        # starts = starts.view(-1)
        # ends = ends.view(-1)
        # start_loss = F.cross_entropy(input=start_prob, target=starts)
        # end_loss = F.cross_entropy(input=end_prob, target=ends)
        loss = torch.sum(start_loss) + torch.sum(end_loss) + 0.8 * (torch.sum(start_loss_aux) + torch.sum(end_loss_aux))
        loss.backward()

        # total_loss += loss.item()
        # num_samples += len(samples)

        optimizer.step()
        if i % args.logging_steps == 0:
            print("loss: ", loss.item())
    # print("loss: ", total_loss / num_samples)


def main(args):

    train_data = load_data(args.train_path)
    dev_data = load_data(args.dev_path)

    set_seed(args.seed)

    # model = PointerNERBERT(args).to(args.device)
    model = PointerNERBERT2(args).to(args.device)
    # model = BertSpanForNer(config).to(args.device)
    # ===============================================================================
    # state = torch.load("DocumentReview/PointerBert/model_src/pBert0921_cluener.pt")
    # torch.load(map_location="cpu")
    # model.load_state_dict(state['model_state'])
    # ===============================================================================

    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.learning_rate)
    print("numbers train", len(train_data))
    train_dataset = ReaderDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batchify)

    dev_dataset = ReaderDataset(dev_data)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=batchify)
    # model.zero_grad()
    best_f1 = 0
    for e in range(args.num_epochs):
        # train
        model.train()
        train(args, train_loader, model, optimizer)
        # evaluate
        model.eval()
        print('-' * 50 + 'evaluating' + '-' * 50)
        y_true = torch.FloatTensor([]).to(args.device)
        y_pred = torch.FloatTensor([]).to(args.device)
        entities = []
        true_entities = []

        for i, samples in enumerate(dev_loader):

            encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
            sentences = samples[4]

            start_prob, end_prob, _, _ = model(encoded_dicts)
            # bs, seq_len
            '''
            start_pred = torch.argmax(start_prob, dim=-1, keepdim=False)
            end_pred = torch.argmax(end_prob, dim=-1, keepdim=False)
            for bi in range(len(start_pred)):
                sentence = sentences[bi]
                for label_index in range(len(args.labels)):
                    if label_index == 0:
                        continue
                    start_index = []
                    end_index = []
                    for ti in range(len((start_pred[bi]))):
                        if start_pred[bi][ti] == label_index:
                            start_index.append(ti)
                        if end_pred[bi][ti] == label_index:
                            end_index.append(ti)
                    if len(start_index) == len(end_index):
                        for s_index, e_index in zip(start_index, end_index):
                            entities.append([args.label(label_index), sentence[s_index:e_index]])
                    else:
                        minnum = min(len(start_index), len(end_index))
                        for k in range(minnum):
                            entities.append([args.labels[label_index], sentence[start_index[k]:end_index[k]]])
            '''
            thred = torch.FloatTensor([0.5]).to(args.device)
            start_pred = start_prob > thred
            end_pred = end_prob > thred
            start_pred = start_pred.transpose(2, 1)
            end_pred = end_pred.transpose(2, 1)
            for bi in range(len(start_pred)):
                sentence = sentences[bi]
                for li in range(len(start_pred[bi])):
                    start_seq = start_pred[bi][li]
                    end_seq = end_pred[bi][li]
                    start_index = []
                    end_index = []
                    for start_ind in range(len(start_seq)):
                        if start_seq[start_ind]:
                            start_index.append(start_ind)
                    for end_ind in range(len(end_seq)):
                        if end_seq[end_ind]:
                            end_index.append(end_ind)
                    if len(start_index) == len(end_index):
                        for start_ind, end_ind in zip(start_index, end_index):
                            entities.append([args.labels[li], sentence[start_ind:end_ind]])
                    else:
                        min_len = min(len(start_index), len(end_index))
                        for mi in range(min_len):
                            entities.append([args.labels[li], sentence[start_index[mi]:end_index[mi]]])

            true_entities.extend(labels)

        print('pred entities: ', len(entities))
        if len(entities)>0:
            print(entities[0])
        print('true_entities: ', len(true_entities))
        print(true_entities[0])
        # precision, recall, f1 = evaluate_entity_wo_category(true_entities, entities)
        cir = SpanEntityScore()
        cir.update(true_entities, entities)
        print('cal...')
        score, class_info = cir.result()
        f1 = score['f1']
        print("epoch:", e, "  p: {0}, r: {1}, f1: {2}".format(score['acc'], score['recall'], score['f1']))
        pprint(class_info)

        if f1 > best_f1:
            print("f1 score increased  {0}==>{1}".format(best_f1, f1))
            best_f1 = f1
            PATH = args.model_save_path
            state = {'model_state': model.state_dict(), 'e': e, 'optimizer': optimizer.state_dict()}
            # model cpu?
            torch.save(state, PATH)
            # 加载
            # PATH = './model.pth'  # 定义模型保存路径
            # state = torch.load(PATH, map_location="cpu")
            # model.load_state_dict(state['model_state'])
            # optimizer.load_state_dict(state['optimizer'])
        '''
        if args.is_inference:
            entities = []
            true_entities = []
            for i, samples in enumerate(dev_loader):
                encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
                sentences = samples[4]
                start_prob, end_prob = model(encoded_dicts)
                print(start_prob.shape)
                thred = torch.FloatTensor([0.5]).to(args.device)
                start_pred = start_prob > thred
                end_pred = end_prob > thred
                # batch_size, number_of_label, sentence_length
                start_pred = start_pred.transpose(2, 1)
                end_pred = end_pred.transpose(2, 1)
                if True in start_pred:
                    print("true in start_pred")
                if True in end_pred:
                    print("true in end_pred")
                # 0-1 seq to entity
                for bi in range(len(start_pred)):
                    index_bias = 0
                    sentence = sentences[bi]
                    for li in range(len(start_pred[bi])):
                        start_seq = start_pred[bi][li]
                        end_seq = end_pred[bi][li]
                        start_index = []
                        end_index = []
                        if True in start_seq:
                            for start_ind in range(len(start_seq)):
                                if start_seq[start_ind]:
                                    start_index.append(start_ind)
                                    print("label:", args.labels[li], "start:", start_ind + index_bias)
                        if True in end_seq:
                            for end_ind in range(len(end_seq)):
                                if end_seq[end_ind]:
                                    end_index.append(end_ind)
                                    print("label:", args.labels[li], "end:", end_ind + index_bias)
                        if len(start_index) == len(end_index):
                            for start_ind, end_ind in zip(start_index, end_index):
                                entities.append({'start': start_ind + index_bias, 'end': end_ind + index_bias,
                                                 'entity': sentence[start_ind:end_ind]})
                        else:
                            min_len = min(len(start_index), len(end_index))
                            for mi in range(min_len):
                                entities.append(
                                    {'start': start_index[mi] + index_bias, 'end': end_index[mi] + index_bias,
                                     'entity': sentence[start_index[mi]:end_index[mi]]})
            print("entities", entities)
        '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--is_inference", default=False, type=bool)
    parser.add_argument("--model_save_path", default='model/PointerBert/PBert1010_common_all_20sche_aux.pt')
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length. "
                                                                     "Sequences longer than this will be split automatically.")
    parser.add_argument("--bert_emb_size", default=768, type=int, help="The embedding size of pretrained model")
    parser.add_argument("--hidden_size", default=200, type=int, help="The hidden size of model")
    parser.add_argument("--num_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=200, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=100, type=int,
                        help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="cuda",
                        help="Select which device to train model, defaults to gpu.")
    # parser.add_argument("--model", choices=["uie-base", "uie-tiny", "uie-medium", "uie-mini", "uie-micro", "uie-nano"],
    #                     default="uie-base", type=str, help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--model", default="uie-base", type=str,
                        help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--init_from_ckpt", default=None, type=str,
                        help="The path of model parameters for initialization.")


    args = parser.parse_args()
    args.train_path = 'data/data_src/common_all/train.json'
    args.dev_path = 'data/data_src/common_all/dev.json'
    # args.train_path = 'data/cluener/train.json'
    # args.dev_path = 'data/cluener/dev.json'
    args.model = 'model/language_model/chinese-roberta-wwm-ext'
    labels, alias2label = read_config_to_label(args)
    labels_aux, label2new_label = read_config_to_label_aux()
    args.labels = labels
    args.labels_aux = labels_aux
    pprint(args)

    main(args)
    # export PYTHONPATH=$(pwd):$PYTHONPATH
    # nohup python -u BasicTask/NER/PointerBert/main.py > log/PointerBert/pBert_1010_common_all_20sche_aux.log 2>&1 &