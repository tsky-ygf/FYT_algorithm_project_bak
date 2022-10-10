#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/09 10:01
# @Author  : Czq
# @File    : common_contract.py
# @Software: PyCharm
import argparse
import os
from collections import defaultdict

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from BasicTask.NER.PointerBert.model_NER import PointerNERBERT
from DocumentReview.src.basic_contract import BasicUIEAcknowledgement
from pprint import pformat, pprint

tokenizer = BertTokenizer.from_pretrained('model/language_model/chinese-roberta-wwm-ext')


def _split_text(text):
    text_list = []
    wind = 500
    step = 400
    for i in range(0, len(text), step):
        text_list.append({'index_bias': i, 'text': text[i:i + wind]})
    return text_list


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
    for b in batch:
        text = b['text']
        index_bias = b['index_bias']
        sentences.append(text)
        index_biass.append(index_bias)

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
    return encoded_dict, index_biass, sentences


class BasicPBAcknowledgement(BasicUIEAcknowledgement):

    def __init__(self, common_model_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common2alias = dict()
        _labels, _common2alias_dict = self._read_common_schema(common_model_args.common_schema_path)
        self.common_labels = _labels
        self.common2alias_dict = _common2alias_dict
        common_model_args.labels = _labels

        self.common_model = PointerNERBERT(common_model_args).to('cpu')
        state = torch.load(common_model_args.model_load_path, map_location="cpu")
        self.common_model.load_state_dict(state['model_state'])
        self.common_model.eval()

    def check_data_func(self):
        if self.device == "cpu":
            self.logger.debug(self.data)
            self.common2alias = self.common2alias_dict[self.contract_type]
            res = self.predictor_dict[self.contract_type].predict([self.data])
            res_common = self._get_common_result()
            print("common predict result", len(res_common))
            print(res_common)
            # note: res 外多套了层list
            res_final = self._merge_result(res[0], res_common)

        else:
            assert False, "dont use gpu"
            res = self.ie(self.data)
        self.logger.debug(pformat(res_final))

        return res_final

    def _merge_result(self, uies, commons):
        for common in commons:
            uies[common] = commons[common]
        return uies

    def _get_common_result(self):
        # 在通用条款识别前， 需要进行文本分割等操作
        text_list = _split_text(self.data)
        dataset = TestDataset(text_list)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=batchify_test)
        entities = defaultdict(list)
        for inputs, index_biass, sentences in dataloader:
            # batch_size, sentence_length, number_of_label
            start_prob, end_prob = self.common_model(inputs)
            thred = torch.FloatTensor([0.5]).to('cpu')
            start_pred = start_prob > thred
            end_pred = end_prob > thred
            # batch_size, number_of_label, sentence_length
            start_pred = start_pred.transpose(2, 1)
            end_pred = end_pred.transpose(2, 1)
            for bi in range(len(start_pred)):
                index_bias = index_biass[bi]
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
                    min_len = min(len(start_index), len(end_index))
                    for mi in range(min_len):
                        # 针对具体合同类型， 转换schema名称。
                        new_label = self.common2alias[self.common_labels[li]]
                        tmp_entity = {'text': sentence[start_index[mi]:end_index[mi]],
                         'start': start_index[mi] + index_bias,
                         'end': end_index[mi] + index_bias}
                        if tmp_entity not in entities[new_label]:
                            entities[new_label].append(tmp_entity)

        return entities

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


if __name__ == "__main__":
    import time

    contract_type = "fangwuzulin"

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_load_path", default='model/PointerBert/PBert1009_common_all_20sche_tr.pt', type=str)
    parser.add_argument("--model", default='model/language_model/chinese-roberta-wwm-ext', type=str)
    parser.add_argument("--common_schema_path", default='DocumentReview/Config/config_common.csv', type=str,
                        help="The hidden size of model")
    # parser.add_argument("--contract_type", default=contract_type)
    parser.add_argument("--bert_emb_size", default=768, type=int, help="The embedding size of pretrained model")
    parser.add_argument("--hidden_size", default=100, type=int, help="The hidden size of model")
    common_model_args = parser.parse_args()

    print('=' * 50, '模型初始化', '=' * 50)
    acknowledgement = BasicPBAcknowledgement(contract_type_list=[contract_type],
                                             config_path_format="DocumentReview/Config/schema/{}.csv",
                                             model_path_format="model/uie_model/export_cpu/{}/inference",
                                             common_model_args=common_model_args,
                                             log_level="INFO",
                                             device="cpu")
    print('=' * 50, '开始预测', '=' * 50)
    localtime = time.time()
    acknowledgement.review_main(content="data/DocData/{}/fwzl1.docx".format(contract_type), mode="docx",
                                contract_type=contract_type, usr="Part B")
    print('=' * 50, '结束', '=' * 50)
    print('use time: {}'.format(time.time() - localtime))
    pass
