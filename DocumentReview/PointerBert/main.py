#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/08 15:22
# @Author  : Czq
# @File    : main.py
# @Software: PyCharm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
from pprint import pprint
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from DocumentReview.PointerBert.utils import load_data, set_seed, ReaderDataset, batchify, read_config_to_label
from BasicTask.NER.BertNer.ModelStructure.bert_ner_model import PointerNERBERT


def train(args, train_loader, model, optimizer):
    print('-'*50+'training'+'-'*50)
    total_loss = 0
    num_samples = 0
    for i, samples in enumerate(train_loader):
        optimizer.zero_grad()
        encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
        start_prob, end_prob = model(encoded_dicts)
        start_loss = torch.nn.functional.binary_cross_entropy(input=start_prob, target=starts, reduction="sum")
        end_loss = torch.nn.functional.binary_cross_entropy(input=end_prob, target=ends, reduction="sum")
        loss = start_loss + end_loss

        total_loss += loss.item()
        num_samples += len(samples)

        loss.backward()
        optimizer.step()
        if i % args.logging_steps == 0:
            print("loss: ", total_loss / num_samples)
    print("loss: ", total_loss / num_samples)


def main(args):
    labels = read_config_to_label(args)
    args.labels = labels
    train_data = load_data(args.train_path)
    dev_data = load_data(args.dev_path)

    set_seed(args.seed)

    model = PointerNERBERT(args).to(args.device)
    # ======================
    # state = torch.load("ContractNER/model_src/PointerBert/pBert0920_bs1.pt")
    # model.load_state_dict(state['model_state'])
    # ======================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset = ReaderDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batchify)

    dev_dataset = ReaderDataset(dev_data)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=batchify)

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
        for i, samples in enumerate(dev_loader):
            encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
            start_prob, end_prob = model(encoded_dicts)
            y_true = torch.cat([y_true,starts])
            y_true = torch.cat([y_true,ends])
            thred = torch.FloatTensor([0.5]).to(args.device)
            start_pred = start_prob>thred
            end_pred = end_prob>thred
            y_pred = torch.cat([y_pred,start_pred])
            y_pred = torch.cat([y_pred,end_pred])


        # calculate p r f1
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        print("numbers of Correct prediction ", np.sum(y_pred))
        # print("numbers of Correct prediction ", np.sum(y_true))   # 418
        TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
        FP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
        FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
        precision = TP/(TP+FP) if (TP+FP)!=0 else 0
        recall = TP/(TP+FN) if (TP+FN)!=0 else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall !=0 else 0
        print("epoch:", e, "  p: {0}, r: {1}, f1: {2}".format(precision,recall,f1))
        if f1>best_f1:
            print("f1 score increased  {0}==>{1}".format(best_f1,f1))
            best_f1 = f1
            PATH = args.model_save_path
            state = {}
            # TODO model cpu？
            state['model_state'] = model.state_dict()
            state['e'] = e
            state['optimizer'] = optimizer.state_dict(),
            torch.save(state, PATH)
            # 加载
            # PATH = './model.pth'  # 定义模型保存路径
            # state = torch.load(PATH)
            # model.load_state_dict(state['model_state'])
            # optimizer.load_state_dict(state['optimizer'])

        if args.is_inference:
            entities = []
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
                                entities.append({'start': start_index[mi] + index_bias, 'end': end_index[mi] + index_bias,
                                                 'entity': sentence[start_index[mi]:end_index[mi]]})
            print("entities",entities)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--is_inference", default=True, type=bool)
    parser.add_argument("--model_save_path", default='DocumentReview/PointerBert/model_src/pBert0920.pt')
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
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
    args.train_path = 'data/data_src/new/train.txt'
    args.dev_path = 'data/data_src/new/dev.txt'
    args.model = 'model/language_model/chinese-roberta-wwm-ext'
    pprint(args)

    main(args)
