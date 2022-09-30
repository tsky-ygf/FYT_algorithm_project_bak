#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/08 15:22
# @Author  : Czq
# @File    : run_qa.py
# @Software: PyCharm
import os
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from BasicTask.NER.BertNer.metrics import SpanEntityScore

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import argparse
from pprint import pprint
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from BasicTask.NER.PointerBert.utils import load_data, set_seed, ReaderDataset, batchify, read_config_to_label, \
    evaluate_index, batchify_cluener, evaluate_entity_wo_category, bert_extract_item
from BasicTask.NER.PointerBert.model_NER import PointerNERBERT, BertSpanForNer


def train(args, train_loader, model, optimizer):
    print('-' * 50 + 'training' + '-' * 50)
    total_loss = 0
    num_samples = 0
    for i, samples in enumerate(train_loader):
        optimizer.zero_grad()
        encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
        # outputs = model(encoded_dicts['input_ids'], token_type_ids=encoded_dicts['token_type_ids'],
        #                 attention_mask=encoded_dicts['attention_mask'],
        #                 start_positions=starts, end_positions=ends)
        # loss = outputs[0]
        # loss.backward()
        start_prob, end_prob = model(encoded_dicts)
        # start_loss = torch.nn.functional.binary_cross_entropy(input=start_prob, target=starts, reduction="sum")
        # end_loss = torch.nn.functional.binary_cross_entropy(input=end_prob, target=ends, reduction="sum")
        # start_loss = torch.nn.functional.binary_cross_entropy(input=start_prob, target=starts)
        # end_loss = torch.nn.functional.binary_cross_entropy(input=end_prob, target=ends)
        start_prob = start_prob.contiguous().view(-1, 11)
        end_prob = end_prob.contiguous().view(-1, 11)
        starts = starts.view(-1)
        ends = ends.view(-1)
        start_loss = torch.nn.functional.cross_entropy(start_prob, starts)
        end_loss = torch.nn.functional.cross_entropy(end_prob, ends)
        loss = start_loss + end_loss
        loss.backward()
        total_loss += loss.item()
        num_samples += len(samples)

        optimizer.step()
        if i % args.logging_steps == 0:
            print("loss: ", loss.item())
    # print("loss: ", total_loss / num_samples)


def main(args):
    # labels = read_config_to_label(args)
    labels2id = ['O','address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position',
                 'scene']
    args.labels = labels2id

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
    # model.load_state_dict(state['model_state'])
    # ===============================================================================

    # no_decay = ["bias", "LayerNorm.weight"]
    # bert_parameters = model.bert.named_parameters()
    # start_parameters = model.start_fc.named_parameters()
    # end_parameters = model.end_fc.named_parameters()
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
    #      "weight_decay": args.weight_decay, 'lr': args.learning_rate},
    #     {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
    #         , 'lr': args.learning_rate},
    #
    #     {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
    #      "weight_decay": args.weight_decay, 'lr': 0.001},
    #     {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
    #         , 'lr': 0.001},
    #
    #     {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
    #      "weight_decay": args.weight_decay, 'lr': 0.001},
    #     {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
    #         , 'lr': 0.001},
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.learning_rate)
    print("numbers train", len(train_data))
    train_dataset = ReaderDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batchify_cluener)

    dev_dataset = ReaderDataset(dev_data)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=batchify_cluener)
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
        # metric = SpanEntityScore()
        for i, samples in enumerate(tqdm(dev_loader)):

            encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
            sentences = samples[4]
            start_prob, end_prob = model(encoded_dicts)
            # y_true = torch.cat([y_true, starts])
            # y_true = torch.cat([y_true, ends])
            start_pred = torch.argmax(start_prob, dim=-1)
            end_pred = torch.argmax(end_prob, dim=-1)
            # thred = torch.FloatTensor([0.5]).to(args.device)
            # start_pred = start_prob > thred
            # end_pred = end_prob > thred
            # y_pred = torch.cat([y_pred, start_pred])
            # y_pred = torch.cat([y_pred, end_pred])
            # start_pred = start_pred.transpose(2, 1)
            # end_pred = end_pred.transpose(2, 1)
            for bi in range(len(start_pred)):
                sentence = sentences[bi]
                for li, l in enumerate(args.labels):
                    if li == 0:
                        continue
                    start_index = []
                    end_index = []
                    for ti in range(len(start_pred[bi])):
                        if start_pred[bi][ti] == li:
                            start_index.append(ti)
                        if end_pred[bi][ti] == li:
                            end_index.append(ti)
                    min_l = min(len(start_index), len(end_index))
                    for mi in range(min_l):
                        entities.append([l, sentence[start_index[mi]:end_index[mi]]])

            '''
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
                            entities.append([labels2id[li], sentence[start_ind:end_ind+1]])
                            # true_entities.append([labels[bi][0], sentence[labels[bi][1]:labels[bi][2]+1]])
                    else:
                        min_len = min(len(start_index), len(end_index))
                        for mi in range(min_len):
                            entities.append([labels2id[li], sentence[start_index[mi]:end_index[mi]+1]])
            '''
            true_entities.extend(labels)

        print('entities: ', len(entities))
        # print(entities[0])
        print('true_entities: ', len(true_entities))
        # print(true_entities[0])
        # precision, recall, f1 = evaluate_entity_wo_category(true_entities, entities)
        cir = SpanEntityScore()
        cir.update(true_entities, entities)
        print('cal...')
        score, class_info = cir.result()
        f1 = score['f1']
        print("epoch:", e, "  p: {0}, r: {1}, f1: {2}".format(score['acc'], score['recall'], score['f1']))

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
            # torch.save(state, PATH)
            # 加载
            # PATH = './model.pth'  # 定义模型保存路径
            # state = torch.load(PATH)
            # model.load_state_dict(state['model_state'])
            # optimizer.load_state_dict(state['optimizer'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--is_inference", default=False, type=bool)
    parser.add_argument("--model_save_path", default='model/PointerBert/PBert0922_cluener.pt')
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
    # args.train_path = 'data/data_src/new/train.txt'
    # args.dev_path = 'data/data_src/new/dev.txt'
    args.train_path = 'data/cluener/train.json'
    args.dev_path = 'data/cluener/dev.json'
    args.model = 'model/language_model/bert-base-chinese'
    pprint(args)

    main(args)

    """nohup python -u DocumentReview/PointerBert/main.py --model "model/language_model/chinese-roberta-wwm-ext" \
> log/PointerBert/cluener_0921.log 2>&1 &"""