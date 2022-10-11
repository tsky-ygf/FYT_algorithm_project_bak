#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/08 14:07
# @Author  : Czq
# @File    : main.py
# @Software: PyCharm
import argparse
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from BasicTask.NER.BertNer.metrics import SpanEntityScore
from BasicTask.NER.PointerBert.model_NER import PointerNERBERT
from BasicTask.NER.PointerBertMRC.model_MRC import PointerMRCBERT
from BasicTask.NER.PointerBertMRC.utils import read_config_to_label, load_data, set_seed, ReaderDataset


def batchify_mrc(batch):
    sentences = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    start_seqs = []
    end_seqs = []
    labels = []
    window_length = 510  # add 101 102 to 512
    window_length = 510 - 15  # add sep and question
    batch_size = len(batch)

    for b in batch:
        text = b['text']

        res_list = b['entities']
        start_seq = [[0] * 510 for _ in range(len(args.labels))]
        end_seq = [[0] * 510 for _ in range(len(args.labels))]
        if not res_list:
            start_seqs.append(start_seq)
            end_seqs.append(end_seq)
        else:
            for res in res_list:
                label = res['label']
                if label not in args.labels:
                    continue
                start = res['start_offset']
                end = res['end_offset']
                if start is not None and end is not None:
                    entity_text = text[start:end]
                    labels.append([label, entity_text])
                label_id = args.labels.index(label)
                if start is not None:
                    start_seq[label_id][start] = 1
                if end is not None:
                    end_seq[label_id][end] = 1
            start_seqs.append(start_seq)
            end_seqs.append(end_seq)
        # 循环拼接question
        for label in args.labels:
            ques = label + "是什么？"
            sentences.append(text + '[SEP]' + ques)
            assert len(text) + len(ques) + 3 <= 512, len(text)
            input_i = [101] + args.tokenizer.convert_tokens_to_ids(list(text)) + [102] + \
                      args.tokenizer.convert_tokens_to_ids(list(ques)) + [102]
            input_id = input_i.copy() + [0] * (512 - len(input_i))
            atten_mask = [1] * len(input_i) + [0] * (512 - len(input_i))
            token_type_id = [0] * (len(text) + 2) + [1] * (len(ques) + 1) + [0] * (512 - len(input_i))
            assert len(input_id) == 512 == len(atten_mask) == len(token_type_id), [len(input_id), len(token_type_id)]

            input_ids.append(input_id)
            attention_mask.append(atten_mask)
            token_type_ids.append(token_type_id)

    encoded_dict = {
        'input_ids': torch.LongTensor(input_ids).to('cuda'),
        'attention_mask': torch.LongTensor(attention_mask).to('cuda'),
        'token_type_ids': torch.LongTensor(token_type_ids).to('cuda')
    }

    # 要把num_labels 放到bath_size上  batch_size*num_labels, seq_len, 1(pad)
    start_seqs = torch.FloatTensor(start_seqs).to('cuda')
    end_seqs = torch.FloatTensor(end_seqs).to('cuda')
    start_seqs = start_seqs.reshape(-1, start_seqs.shape[2]).unsqueeze(-1)
    end_seqs = end_seqs.reshape(-1, end_seqs.shape[2]).unsqueeze(-1)

    return encoded_dict, start_seqs, end_seqs, labels, sentences


def train(args, train_loader, model, optimizer):
    print('-' * 50 + 'training' + '-' * 50)
    for i, samples in enumerate(train_loader):
        optimizer.zero_grad()
        encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
        sentences = samples[4]
        start_prob, end_prob = model(encoded_dicts)
        start_loss = torch.nn.functional.binary_cross_entropy(input=start_prob, target=starts)
        end_loss = torch.nn.functional.binary_cross_entropy(input=end_prob, target=ends)

        loss = torch.sum(start_loss) + torch.sum(end_loss)
        loss.backward()
        optimizer.step()
        if i % args.logging_steps == 0:
            print("loss: ", loss.item())


def main(args):
    train_data = load_data(args.train_path)
    dev_data = load_data(args.dev_path)
    print("numbers train", len(train_data))

    set_seed(args.seed)

    model = PointerMRCBERT(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset = ReaderDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batchify_mrc)

    dev_dataset = ReaderDataset(dev_data)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=batchify_mrc)

    best_f1 = 0
    for e in range(args.num_epochs):
        # train
        model.train()
        train(args, train_loader, model, optimizer)
        # evaluate
        model.eval()
        print('-' * 50 + 'evaluating' + '-' * 50)

        entities = []
        true_entities = []
        num_label = len(args.labels)
        for i, samples in enumerate(dev_loader):
            encoded_dicts, starts, ends, labels = samples[0], samples[1], samples[2], samples[3]
            sentences = samples[4]

            start_prob, end_prob = model(encoded_dicts)
            start_prob = start_prob.squeeze()
            start_prob = start_prob.reshape(-1, num_label, start_prob.shape[1])
            end_prob = end_prob.squeeze()
            end_prob = end_prob.reshape(-1, num_label, end_prob.shape[1])

            thred = torch.FloatTensor([0.5]).to(args.device)
            start_pred = start_prob > thred
            end_pred = end_prob > thred
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
        if len(entities) > 0:
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
            torch.save(state, PATH)
            # 加载
            # PATH = './model.pth'  # 定义模型保存路径
            # state = torch.load(PATH, map_location="cpu")
            # model.load_state_dict(state['model_state'])
            # optimizer.load_state_dict(state['optimizer'])


def ttt():
    text = '今天阴天'
    inputs = args.tokenizer.encode_plus(text=text, text_pair="明天晴天")
    print(inputs)


# using MRC method to long schema
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", default='model/PointerBert/PMRCBert1010_common_long.pt')
    parser.add_argument("--train_path", default='data/data_src/common_mrc/train.json', type=str,
                        help="The path of train set.")
    parser.add_argument("--dev_path", default='data/data_src/common_mrc/dev.json', type=str,
                        help="The path of dev set.")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--bert_emb_size", default=768, type=int, help="The embedding size of pretrained model")
    parser.add_argument("--hidden_size", default=200, type=int, help="The hidden size of model")
    parser.add_argument("--num_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=200, type=int, help="The interval steps to logging.")
    parser.add_argument("--model", default='model/language_model/chinese-roberta-wwm-ext', type=str,
                        help="The path of model parameters for initialization.")
    parser.add_argument('--device', default="cuda",
                        help="Select which device to train model, defaults to gpu.")
    args = parser.parse_args()
    torch.cuda.set_device(1)
    _labels, alias2label = read_config_to_label(args)
    args.labels = _labels
    args.alias2label = alias2label
    args.tokenizer = BertTokenizer.from_pretrained(args.model)
    pprint(args)
    main(args)
    # nohup python -u BasicTask/NER/PointerBertMRC/main.py > log/PointerBert/PMRCBert_1010_common_long.log 2>&1 &
