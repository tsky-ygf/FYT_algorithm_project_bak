#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/08 15:22
# @Author  : Czq
# @File    : run_qa.py
# @Software: PyCharm
import os
# from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from BasicTask.NER.BertNer.metrics import SpanEntityScore

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import argparse
from pprint import pprint
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from DocumentReview.PointerBert.utils import load_data, set_seed, ReaderDataset, batchify, read_config_to_label, \
    evaluate_index, batchify_cluener, evaluate_entity_wo_category, bert_extract_item
from DocumentReview.PointerBert.model_NER import PointerNERBERT, BertSpanForNer


def train(args, train_loader, model, optimizer):
    print('-' * 50 + 'training' + '-' * 50)
    total_loss = 0
    num_samples = 0
    entities = []
    for i, samples in enumerate(train_loader):
        optimizer.zero_grad()
        encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
        sentences = samples[4]
        start_prob, end_prob = model(encoded_dicts)
        # start_loss = torch.nn.functional.binary_cross_entropy(input=start_prob, target=starts, reduction="sum")
        # end_loss = torch.nn.functional.binary_cross_entropy(input=end_prob, target=ends, reduction="sum")
        start_loss = torch.nn.functional.binary_cross_entropy(input=start_prob, target=starts)
        end_loss = torch.nn.functional.binary_cross_entropy(input=end_prob, target=ends)
        loss = start_loss + end_loss
        loss.backward()

        total_loss += loss.item()
        num_samples += len(samples)

        optimizer.step()
        if i % args.logging_steps == 0:
            print("loss: ", loss.item())
    # print("loss: ", total_loss / num_samples)


def main(args):
    labels = read_config_to_label(args)
    # labels2id = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position',
    #              'scene']
    args.labels = labels

    # config_class, model_class, tokenizer_class = BertConfig, BertSpanForNer, BertTokenizer
    # config = config_class.from_pretrained(args.model, num_labels=len(labels2id))
    # config.loss_type = 'lsr'
    # config.soft_label = True

    train_data = load_data(args.train_path)
    dev_data = load_data(args.dev_path)

    set_seed(args.seed)

    model = PointerNERBERT(args).to(args.device)
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

            start_prob, end_prob = model(encoded_dicts)
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
                    # if True in start_seq:
                    for start_ind in range(len(start_seq)):
                        if start_seq[start_ind]:
                            start_index.append(start_ind)
                            # print("label:", args.labels[li], "start:", start_ind)
                    # if True in end_seq:
                    for end_ind in range(len(end_seq)):
                        if end_seq[end_ind]:
                            end_index.append(end_ind)
                            # print("label:", args.labels[li], "end:", end_ind)
                    if len(start_index) == len(end_index):
                        for start_ind, end_ind in zip(start_index, end_index):
                            entities.append([args.labels[li], sentence[start_ind:end_ind]])
                            # true_entities.append([labels[bi][0], sentence[labels[bi][1]:labels[bi][2]+1]])
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
        print(class_info)
        # ev = evaluate_index(y_pred, y_true)
        # print("epoch:", e, "  p: {0}, r: {1}, f1: {2}".format(ev[0], ev[1], ev[2]))
        # score, class_info = metric.result()
        # f1 = score['f1']
        # print("epoch:", e, "  p: {0}, r: {1}, f1: {2}".format(score['acc'], score['recall'], score['f1']))


        if f1 > best_f1:
            print("f1 score increased  {0}==>{1}".format(best_f1, f1))
            best_f1 = f1
            PATH = args.model_save_path
            state = {'model_state': model.state_dict(), 'e': e, 'optimizer': optimizer.state_dict()}
            # TODO model cpu？
            torch.save(state, PATH)
            # 加载
            # PATH = './model.pth'  # 定义模型保存路径
            # state = torch.load(PATH, map_location="cpu")
            # model.load_state_dict(state['model_state'])
            # optimizer.load_state_dict(state['optimizer'])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--is_inference", default=False, type=bool)
    parser.add_argument("--model_save_path", default='DocumentReview/PointerBert/model_src/PBert0923_old.pt')
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
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
    args.train_path = 'data/data_src/old/train_split.json'
    args.dev_path = 'data/data_src/old/dev_split.json'
    # args.train_path = 'data/cluener/train.json'
    # args.dev_path = 'data/cluener/dev.json'
    args.model = 'model/language_model/chinese-roberta-wwm-ext'
    pprint(args)

    main(args)

    """nohup python -u DocumentReview/PointerBert/main.py > log/PointerBert/pBert_old_0923.log 2>&1 &"""