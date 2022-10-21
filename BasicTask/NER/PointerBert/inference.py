#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/21 11:16
# @Author  : Czq
# @File    : inference.py
# @Software: PyCharm
import argparse
import os
from collections import OrderedDict, defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from BasicTask.NER.PointerBert.model_NER import PointerNERBERT
from DocumentReview.src.ParseFile import read_docx_file

tokenizer = BertTokenizer.from_pretrained('model/language_model/chinese-roberta-wwm-ext')


class TestDataset(Dataset):
    def __init__(self, data):
        self.test_data = data

    def __getitem__(self, item):
        return self.test_data[item]

    def __len__(self):
        return len(self.test_data)


def batchify_test(batch):
    sentences = []
    index_biass = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    is_head_tail_list = []
    for b in batch:
        text = b['text']
        index_bias = b['index_bias']
        sentences.append(text)
        index_biass.append(index_bias)
        is_head_tail_list.append(b['is_head_tail'])

        input_i = [101] + tokenizer.convert_tokens_to_ids(list(text)) + [102]
        input_id = input_i.copy() + [0] * (512 - len(input_i))
        atten_mask = [1] * len(input_i) + [0] * (512 - len(input_i))
        token_type_id = [0] * 512

        assert len(input_id) == 512, len(input_id)
        input_ids.append(input_id)
        attention_mask.append(atten_mask)
        token_type_ids.append(token_type_id)

    encoded_dict = {
        'input_ids': torch.LongTensor(input_ids).to('cpu'),
        'attention_mask': torch.LongTensor(attention_mask).to('cpu'),
        'token_type_ids': torch.LongTensor(token_type_ids).to('cpu')
    }
    return encoded_dict, index_biass, sentences, is_head_tail_list


class CommonPBAcknowledgement:
    def __init__(self, common_model_args, contract_type_list, config_path_format):
        self.contract_type_list = contract_type_list
        self.common2alias = dict()
        _labels, _common2alias_dict = self._read_common_schema(common_model_args.common_schema_path)
        self.common_labels = _labels
        self.common2alias_dict = _common2alias_dict
        common_model_args.labels = _labels

        self.common_model = PointerNERBERT(common_model_args).to('cpu')
        state = torch.load(common_model_args.model_load_path, map_location="cpu")
        self.common_model.load_state_dict(state['model_state'])
        self.common_model.eval()

        self.contract_type = ""
        self.config = dict()
        self.review_result = OrderedDict()
        self.data = ""
        self.usr = ""

        config_path = config_path_format.format(contract_type_list[0])
        config = pd.read_csv(config_path).fillna("")
        self.config = config

        self.wind = common_model_args.wind
        self.step = common_model_args.step

    def review_main(self, content, mode, contract_type, usr):
        self.contract_type = contract_type
        self.data = self.read_origin_content(content, mode)
        extraction_res = self.check_data_func()
        self.usr = usr
        return extraction_res

    def check_data_func(self):
        self.common2alias = self.common2alias_dict[self.contract_type]
        res_common = self._get_common_result()
        print('PB use time', time.time() - localtime)
        print("common predict result", len(res_common))
        print(res_common)
        return res_common

    def _split_text(self, text):
        text_list = []
        wind = self.wind
        step = self.step

        if len(text) <= wind:
            return [{'index_bias': 0, 'text': text, 'is_head_tail': False}]

        text_head_tail = text[:wind // 2] + "\n" + text[-wind // 2 + 1:]
        assert len(text_head_tail) == wind
        # index bias 是tail的开始坐标
        text_list.append({'index_bias': len(text) - wind // 2 + 1, 'text': text_head_tail, 'is_head_tail': True})

        text = text[wind // 2:-wind // 2 + 1]
        for i in range(0, len(text), step):
            text_list.append({'index_bias': i + wind // 2,
                              'text': text[i + wind // 2:i + wind // 2 + wind],
                              'is_head_tail': False})
        return text_list

    def post_process(self, start_prob, end_prob, samples):
        _, index_biass, sentences, is_head_tail_list = samples
        thred = torch.FloatTensor([0.5]).to('cpu')
        start_pred = start_prob > thred
        end_pred = end_prob > thred
        # batch_size, number_of_label, sentence_length
        start_pred = start_pred.transpose(2, 1)
        end_pred = end_pred.transpose(2, 1)
        for bi in range(len(start_pred)):
            index_bias = index_biass[bi]
            is_head_tail = is_head_tail_list[bi]
            sentence = sentences[bi]
            for li in range(len(start_pred[bi])):
                start_seq = start_pred[bi][li]
                end_seq = end_pred[bi][li]
                # ================================================================
                if is_head_tail:
                    start_index = []
                    end_index = []
                    for start_ind in range(len(start_seq) // 2):
                        if start_seq[start_ind]:
                            start_index.append(start_ind)
                    for end_ind in range(len(end_seq) // 2):
                        if end_seq[end_ind]:
                            end_index.append(end_ind)
                    self.index2entity(start_index, end_index, bi, li, 0, sentence, start_prob, end_prob)
                    # .................
                    start_index = []
                    end_index = []
                    for start_ind in range(len(start_seq) // 2, len(start_seq)):
                        if start_seq[start_ind]:
                            start_index.append(start_ind)
                    for end_ind in range(len(end_seq) // 2, len(end_seq)):
                        if end_seq[end_ind]:
                            end_index.append(end_ind)
                    self.index2entity(start_index, end_index, bi, li, index_bias, sentence, start_prob, end_prob)
                # ================================================================

                else:
                    start_index = []
                    end_index = []
                    for start_ind in range(len(start_seq)):
                        if start_seq[start_ind]:
                            start_index.append(start_ind)
                    for end_ind in range(len(end_seq)):
                        if end_seq[end_ind]:
                            end_index.append(end_ind)
                    # start_index, end_index, li, index_bias, sentence
                    self.index2entity(start_index, end_index, bi, li, index_bias, sentence, start_prob, end_prob)

    def index2entity(self, start_index, end_index, bi, li, index_bias, sentence, start_prob, end_prob):
        if len(start_index) == len(end_index):
            for _start, _end in zip(start_index, end_index):
                new_label = self.common2alias[self.common_labels[li]]
                if new_label == "无":
                    continue
                tmp_entity = {'text': sentence[_start:_end],
                              'start': _start + index_bias,
                              'end': _end + index_bias}
                if tmp_entity not in self.entities[new_label]:
                    self.entities[new_label].append(tmp_entity)
        elif not start_index:
            return
        elif not end_index:
            return
        elif start_index[0] > end_index[-1]:
            return
        else:
            while start_index and end_index and start_index[0] > end_index[0]:
                end_index = end_index[1:]
            while start_index and end_index and start_index[-1] > end_index[-1]:
                start_index = start_index[:-1]
            # 1. 数量相等
            if len(start_index) == len(end_index):
                pass
            # 2. 数量不等, 删去置信度最低的
            else:
                diff = abs(len(start_index) - len(end_index))
                if len(start_index) > len(end_index):
                    start_index_wt_prob = [[_start, start_prob[bi][_start][li].item()] for _start in
                                           start_index]
                    start_index_wt_prob.sort(key=lambda x: x[1])
                    start_index_wt_prob = start_index_wt_prob[diff:]
                    start_index = [_[0] for _ in start_index_wt_prob]
                    start_index.sort()
                elif len(start_index) < len(end_index):
                    end_index_wt_prob = [[_end, end_prob[bi][_end][li].item()] for _end in end_index]
                    end_index_wt_prob.sort(key=lambda x: x[1])
                    end_index_wt_prob = end_index_wt_prob[diff:]
                    end_index = [_[0] for _ in end_index_wt_prob]
                    end_index.sort()
            for _start, _end in zip(start_index, end_index):
                new_label = self.common2alias[self.common_labels[li]]
                if new_label == "无":
                    continue
                tmp_entity = {'text': sentence[_start:_end],
                              'start': _start + index_bias,
                              'end': _end + index_bias}
                if tmp_entity not in self.entities[new_label]:
                    self.entities[new_label].append(tmp_entity)

    def _get_common_result(self):
        # 在通用条款识别前， 需要进行文本分割等操作
        text_list = self._split_text(self.data)
        dataset = TestDataset(text_list)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=batchify_test)
        self.entities = defaultdict(list)
        for samples in dataloader:
            inputs = samples[0]
            # batch_size, sentence_length, number_of_label
            start_prob, end_prob = self.common_model(inputs)

            self.post_process(start_prob, end_prob, samples)

        return self.entities

    def _read_common_schema(self, path):
        schema_df = pd.read_csv(path)
        schemas = schema_df['schema'].values
        common2alias_dict = dict()
        for cont_type in self.contract_type_list:
            columns = schema_df[cont_type].values
            common2alias = dict()
            for sche, alias in zip(schemas, columns):
                if sche in ['争议解决', '通知与送达', '甲方解除合同', '乙方解除合同', '未尽事宜', '金额']:
                    continue
                sche = sche.strip()
                alias = alias.strip()
                common2alias[sche] = alias
            common2alias_dict[cont_type] = common2alias
        schemas = schemas.tolist()
        schemas.remove('争议解决')
        schemas.remove('通知与送达')
        schemas.remove('甲方解除合同')
        schemas.remove('乙方解除合同')
        schemas.remove('未尽事宜')
        schemas.remove('金额')

        return schemas, common2alias_dict

    def read_origin_content(self, content="", mode="text"):
        if mode == "text":
            text_list = content  # 数据在通过接口进入时就会清洗整理好， 只使用text模式； 本地使用，只使用docx格式
        elif mode == "docx":
            text_list = read_docx_file(docx_path=content)
        elif mode == "txt":
            with open(content, encoding='utf-8', mode='r') as f:
                text_list = f.readlines()
                text_list = [line.strip() for line in text_list]
        else:
            raise Exception("mode error")

        return text_list


if __name__ == "__main__":
    import time

    contract_type = "maimai"

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_load_path", default='model/PointerBert/PBert1021_common_all_20sche_cat.pt', type=str)
    parser.add_argument("--model", default='model/language_model/chinese-roberta-wwm-ext', type=str)
    parser.add_argument("--common_schema_path", default='DocumentReview/Config/config_common.csv', type=str,
                        help="The hidden size of model")
    parser.add_argument("--bert_emb_size", default=768, type=int, help="The embedding size of pretrained model")
    parser.add_argument("--hidden_size", default=200, type=int, help="The hidden size of model")
    parser.add_argument("--wind", default=510, type=int)
    parser.add_argument("--step", default=400, type=int)
    common_model_args = parser.parse_args()

    print('=' * 50, '模型初始化', '=' * 50)
    acknowledgement = CommonPBAcknowledgement(contract_type_list=[contract_type],
                                              config_path_format="DocumentReview/Config/schema/{}.csv",
                                              common_model_args=common_model_args)
    print('=' * 50, '开始预测', '=' * 50)
    localtime = time.time()
    acknowledgement.review_main(content="data/DocData/{}/maimai1.docx".format(contract_type), mode="docx",
                                contract_type=contract_type, usr="Part B")
    print('=' * 50, '结束', '=' * 50)
    print('use time: {}'.format(time.time() - localtime))
    pass
