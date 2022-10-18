#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/21 14:07
# @Author  : Adolf
# @Site    : 
# @File    : extractive_qa_train.py
# @Software: PyCharm
# import torch
import json
import torch
from Tools.train_tool import BaseTrainTool
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering

from Tools.data_pipeline import InputExample

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TrainExtractQA(BaseTrainTool):
    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name)
        # model = MultiLabelClsModel(self.config)
        model_config = AutoConfig.from_pretrained(self.model_args.config_name)
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            config=model_config,
        )
        return tokenizer, model

    def create_examples(self, data_path, mode="train"):
        self.logger.info("Creating {} examples".format(mode))
        self.logger.info("Creating examples from {} ".format(data_path))

        with open(data_path, 'r') as f:
            content = json.load(f)
        content = content["data"]
        examples = []
        for i, line in enumerate(content):
            for one_paragraph in line["paragraphs"]:
                for qas in one_paragraph["qas"]:
                    text = [ans['text'] for ans in qas['answers']]
                    answer_start = [ans['answer_start'] for ans in qas['answers']]
                    label = {'text': text, 'answer_start': answer_start}
                    examples.append(
                        InputExample(guid=qas["id"], texts=[qas["question"], one_paragraph["context"]], label=label))
        return examples

    def prepare_input(self, example, mode="train"):
        doc_stride = 32
        pad_on_right = self.tokenizer.padding_side == "right"
        # 既要对examples进行truncation（截断）和padding（补全）还要还要保留所有信息，所以要用的切片的方法。
        # 每一个一个超长文本example会被切片成多个输入，相邻两个输入之间会有交集。
        tokenized_examples = self.tokenizer(
            example.texts[0 if pad_on_right else 1],
            example.texts[1 if pad_on_right else 0],
            truncation="only_second" if pad_on_right else "only_first",
            # max_length=self.data_train_args.max_length,
            max_length=128,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        # 我们使用overflow_to_sample_mapping参数来映射切片片ID到原始ID。
        # 比如有2个example被切成4片，那么对应是[0, 0, 1, 1]，前两片对应原来的第一个example。
        # sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples.pop("overflow_to_sample_mapping")
        # offset_mapping也对应4片
        # offset_mapping参数帮助我们映射到原始输入，由于答案标注在原始输入上，所以有助于我们找到答案的起始和结束位置。
        # if mode == "train":
        offset_mapping = tokenized_examples.pop("offset_mapping")
        # else:
        #     offset_mapping = tokenized_examples["offset_mapping"]

        # 重新标注数据
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offset in enumerate(offset_mapping):
            # sample_idx = sample_mapping[i]
            answer = example.label
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                tokenized_examples["start_positions"].append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                tokenized_examples["end_positions"].append(idx + 1)

        tokenized_examples["start_positions"] = torch.tensor(tokenized_examples["start_positions"], dtype=torch.long)
        tokenized_examples["end_positions"] = torch.tensor(tokenized_examples["end_positions"], dtype=torch.long)
        return tokenized_examples

    def post_process_function(self, batch, output):
        n_best_size = 20
        max_answer_length = 30

        start_logits = output.start_logits[0].cpu().numpy()
        end_logits = output.end_logits[0].cpu().numpy()
        # 收集最佳的start和end logits的位置:
        start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
        end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index <= end_index:  # 如果start小雨end，那么合理的
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "pred_s_d": [start_index, end_index]  # 后续需要根据token的下标将答案找出来
                        }
                    )

        true_start_logits = batch["start_positions"][0].cpu().numpy()
        true_end_logits = batch["end_positions"][0].cpu().numpy()

        # # context = datasets["validation"][0]["context"]
        #

        exit()


if __name__ == '__main__':
    TrainExtractQA(config_path="BasicTask/QA/Extractive-QA/base_qa.yaml").run()
