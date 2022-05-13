#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 11:21
# @Author  : Adolf
# @Site    : 
# @File    : infer_ner.py
# @Software: PyCharm
import os
import torch
from Tools.infer_tool import BaseInferTool
from transformers import BertTokenizer, BertConfig
from LawEntityExtraction.BertNer.ModelStructure.bert_ner_model import BertSpanForNer, BertCrfForNer
from LawEntityExtraction.BertNer.model_ner_dataset import ClueNerSpanDataset, ClueNerCRFDataset
from LawEntityExtraction.BertNer.metrics import SpanEntityScore

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    "bert_crf": (BertConfig, BertCrfForNer, BertTokenizer, ClueNerCRFDataset),
    "bert_span": (BertConfig, BertSpanForNer, BertTokenizer, ClueNerSpanDataset),
}

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

class NerInferTool(BaseInferTool):
    def __init__(self, config_path):
        model_name = "bert_crf"
        self.bert_config, self.bert_model, self.bert_tokenizer, self.bert_dataset = MODEL_CLASSES[model_name]
        self.num_labels = len(self.bert_dataset.label_list)

        super(NerInferTool, self).__init__(config_path)
        # self.label_list = self.test_dataset.label_list
        self.id2label = self.test_dataset.id2label
        self.label2id = self.test_dataset.label2id

        self.metric = SpanEntityScore(self.id2label)

    def init_model(self):
        tokenizer = self.bert_tokenizer.from_pretrained(self.config["pre_train_tokenizer"], do_lower_case=True)
        bert_model_config = self.bert_config.from_pretrained(self.config["pre_train_model"], num_labels=self.num_labels)
        bert_model_config.soft_label = self.config["soft_label"]
        bert_model_config.loss_type = self.config["loss_type"]
        model = self.bert_model.from_pretrained(self.config["pre_train_model"], config=bert_model_config)
        model.load_state_dict(torch.load(self.config['model_path']))
        return tokenizer, model

    def init_dataset(self, *args, **kwargs):
        test_dataset = self.bert_dataset(data_dir="data/cluener/train.json",
                                          tokenizer=self.tokenizer,
                                          mode="test",
                                          max_length=self.config["max_length"])

        return test_dataset

    def data_collator(self, batch):
        return self.bert_dataset.data_collator_test(batch)

    def pred_data_process(self, text, start_logits, end_logits):
        R = self.bert_extract_item(start_logits, end_logits)
        if R:
            label_entities = [[self.id2label[x[0]], x[1], x[2]] for x in R]
        else:
            label_entities = []

        words = list(text)
        json_d = {'label': {}}
        if len(label_entities) != 0:
            for subject in label_entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]

        return json_d

    def infer(self, text):
        inputs = self.tokenizer(text, add_special_tokens=True,
                                max_length=self.config["max_length"],
                                truncation=True,
                                return_offsets_mapping=False,
                                return_tensors="pt")

        outputs = self.model(input_ids=inputs['input_ids'],
                             token_type_ids=inputs['token_type_ids'],
                             attention_mask=inputs['attention_mask'])

        start_logits, end_logits = outputs[:2]
        return self.pred_data_process(text, start_logits, end_logits)

    def test_data(self):
        # pbar = ProgressBar(n_total=len(self.test_dataloader), desc="Testing")
        for step, batch in enumerate(self.test_dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2]}
            # start_positions = batch[3]
            # end_positions = batch[4]

            outputs = self.model(**inputs)
            # print(outputs)
            # start_logits, end_logits = outputs[:2]

            # text = "".join(self.tokenizer.convert_ids_to_tokens(list(batch[0]), skip_special_tokens=True))
            # print(text)
            for one_ids in range(batch[0].shape[0]):
                # print(one_text)
                # one_text = "".join(self.tokenizer.convert_ids_to_tokens(batch[0][one_ids], skip_special_tokens=True))
                one_start_logits = outputs[0][one_ids].unsqueeze(0)
                one_end_logits = outputs[1][one_ids].unsqueeze(0)
                # res_json = self.pred_data_process(one_text, one_start_logits, one_end_logits)
                R = self.bert_extract_item(one_start_logits, one_end_logits)
                T = [(self.label2id[one[0]], one[1], one[2]) for one in batch[3][one_ids]]
                # T = batch[3][one_ids]
                self.metric.update(true_subject=T, pred_subject=R)

            eval_info, entity_info = self.metric.result()
            self.logger.info(eval_info)
            self.logger.info(entity_info)

            return eval_info, entity_info

    def test_crf_data(self):
        for step, batch in enumerate(self.test_dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2]}

            outputs = self.model(**inputs)
            # logits = outputs[:2]
            tags = self.model.crf.decode(outputs[0], inputs['attention_mask'])
            # out_label_ids = inputs['labels'].cpu().numpy().tolist()
            tags = tags.squeeze(0).cpu().numpy().tolist()
            for one_ids in range(batch[0].shape[0]):
                # print(one_ids)
                # R = self.bert_extract_item(one_start_logits, one_end_logits)
                R = get_entity_bio(tags[one_ids],self.id2label)
                T = get_entity_bio(batch[3][one_ids],self.id2label)
                # T = batch[3][one_ids]
                self.metric.update(true_subject=T, pred_subject=R)

            eval_info, entity_info = self.metric.result()
            self.logger.info(eval_info)
            self.logger.info(entity_info)
            return eval_info, entity_info
    @staticmethod
    def bert_extract_item(_start_logits, _end_logits):
        S = []
        start_pred = torch.argmax(_start_logits, -1).cpu().numpy()[0][1:-1]
        end_pred = torch.argmax(_end_logits, -1).cpu().numpy()[0][1:-1]
        for i, s_l in enumerate(start_pred):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end_pred[i:]):
                if s_l == e_l:
                    S.append((s_l, i, i + j))
                    break
        return S


if __name__ == '__main__':
    ner_tool = NerInferTool(config_path="LawEntityExtraction/BertNer/Config/base_ner_infer.yaml")
    # print(ner_tool.infer(text="突袭黑暗雅典娜》中Riddick发现之前抓住他的赏金猎人Johns，"))
    # ner_tool.test_data()
    ner_tool.test_crf_data()