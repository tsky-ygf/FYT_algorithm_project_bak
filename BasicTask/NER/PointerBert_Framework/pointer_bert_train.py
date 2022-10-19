#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 15:15
# @Author  : Czq
# @File    : pointer_bert_train.py
# @Software: PyCharm
import json
import os
from pprint import pprint, pformat

import torch
from transformers import BertTokenizer

from BasicTask.NER.BertNer.metrics import SpanEntityScore
from BasicTask.NER.PointerBert_Framework.model_NER import PointerNERBERTInFramework
from Tools.data_pipeline import InputExample
from Tools.train_tool import BaseTrainTool
from BasicTask.NER.PointerBert_Framework.utils import read_config_to_label

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class TrainPointerBert(BaseTrainTool):
    def __init__(self, config_schema, is_long, *args, **kwargs):
        self.schema2id, self.id2schema, self.num_labels = read_config_to_label(config_schema, is_long=is_long)
        super().__init__(*args, **kwargs)

    def init_model(self, *args, **kwargs):
        tokenizer = BertTokenizer.from_pretrained(self.model_args.tokenizer_name)

        self.model_args.mode = self.model_args.model_name_or_path
        self.model_args.num_labels = self.num_labels
        model = PointerNERBERTInFramework(self.model_args)
        # model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    def create_examples(self, data_path, mode="train"):
        self.logger.info("Creating {} examples".format(mode))
        self.logger.info("Creating examples from {} ".format(data_path))

        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                # -2 for cls and sep
                start_seq = [[0] * (self.data_train_args.max_length-2) for _ in range(self.num_labels)]
                end_seq = [[0] * (self.data_train_args.max_length-2) for _ in range(self.num_labels)]
                entities = []
                text = line['text']
                for entity in line['entities']:
                    label_id = self.schema2id[entity['label']]
                    start = entity['start_offset']
                    end = entity['end_offset']
                    if start is not None:
                        start_seq[label_id][start] = 1
                    if end is not None:
                        end_seq[label_id][end] = 1
                    if start is not None and end is not None:
                        entities.append([entity['label'], text[start:end]])
                examples.append(InputExample(guid=line["id"], texts=[text],
                                             label={'entity': entities,
                                                    'start_seq': start_seq,
                                                    'end_seq': end_seq,
                                                    'sentence': text
                                                    }))
        return examples

    def prepare_input(self, example, mode="train"):
        text = example.texts[0]
        label = example.label

        # inputs = self.tokenizer(text,
        #                         add_special_tokens=True,
        #                         max_length=self.data_train_args.max_length,
        #                         padding="max_length",
        #                         truncation=True,
        #                         return_offsets_mapping=False,
        #                         return_tensors="pt")
        token_ids = self.tokenizer.convert_tokens_to_ids(list(text))
        input_ids = [101]+token_ids+[102]+[0]*(512-len(token_ids)-2)
        attention_mask = [1]*(len(token_ids)+2)+[0]*(512-len(token_ids)-2)
        token_type_ids = [0] * 512
        inputs = {'input_ids': torch.LongTensor([input_ids]),
                  'attention_mask': torch.LongTensor([attention_mask]),
                  'token_type_ids': torch.LongTensor([token_type_ids]),
                  'label': label}

        assert sum(inputs['attention_mask'][0]) == len(text)+2, text
        assert len(inputs['input_ids'][0]) == len(inputs['attention_mask'][0]) == 512
        return inputs

    def data_collator(self, batch):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        start_seqs = []
        end_seqs = []
        labels = []
        sentences = []
        for b in batch:
            input_ids.append(b['input_ids'])
            token_type_ids.append(b['token_type_ids'])
            attention_mask.append(b['attention_mask'])
            start_seqs.append(b['label']['start_seq'])
            end_seqs.append(b['label']['end_seq'])
            labels.extend(b['label']['entity'])
            sentences.append(b['label']['sentence'])
        encoded_dict = {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attention_mask),
            'token_type_ids': torch.cat(token_type_ids)
        }

        # no sense for softmax method
        start_seqs = torch.FloatTensor(start_seqs).transpose(1, 2)
        end_seqs = torch.FloatTensor(end_seqs).transpose(1, 2)

        return encoded_dict, [start_seqs, end_seqs, labels, sentences]

    def cal_output_loss(self, batch, **kwargs):
        outputs, loss = self.model(batch)
        return outputs, loss

    def eval_epoch(self):
        eval_loss_res = 0
        cir = SpanEntityScore()
        for step, batch in enumerate(self.eval_dataloader):
            self.model.eval()
            with torch.no_grad():
                output, eval_loss = self.cal_output_loss(batch)
                eval_loss_res += eval_loss.item()

                true_entities, pred_entities = self.post_process_function(batch, output)
                cir.update(true_entities, pred_entities)
        self.logger.info("number of true_entities " + str(len(cir.origins)))
        if len(cir.origins)>0:
            self.logger.info(cir.origins[0])
        self.logger.info("number of pred_entities" + str(len(cir.founds)))
        if len(cir.founds)>0:
            self.logger.info(cir.founds[0])
        score, class_info = cir.result()
        f1 = score['f1']
        self.logger.info("p: {0}, r: {1}, f1: {2}".format(score['acc'], score['recall'], score['f1']))
        self.logger.info(pformat(class_info))


        eval_loss_res /= len(self.eval_dataloader)
        return eval_loss_res

    def post_process_function(self, batch, output):
        _, [starts, ends, labels, sentences] = batch
        start_prob, end_prob = output

        pred_entities = []
        true_entities = []

        thred = torch.FloatTensor([0.5]).to('cuda')
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
                min_len = min(len(start_index), len(end_index))
                for mi in range(min_len):
                    pred_entities.append([self.id2schema[li], sentence[start_index[mi]:end_index[mi]]])
        true_entities.extend(labels)

        return true_entities, pred_entities


    def save_model(self, model_path):
        self.accelerator.wait_for_everyone()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        state = {'model_state': self.model.state_dict()}
        torch.save(state, model_path+"/pytorch_model.bin")

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    t = TrainPointerBert(config_path="BasicTask/NER/PointerBert_Framework/base_p.yaml",
                         config_schema='data/data_src/config.csv', is_long=False)
    t.run()
    pass
    # export PYTHONPATH=$(pwd):$PYTHONPATH
    # nohup python -u BasicTask/NER/PointerBert_Framework/pointer_bert_train.py > log/PointerBert/framework/pb_1017_nohup.log 2>&1 &

