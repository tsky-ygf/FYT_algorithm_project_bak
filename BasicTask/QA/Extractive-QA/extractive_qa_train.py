#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/21 14:07
# @Author  : Adolf
# @Site    : 
# @File    : extractive_qa_train.py
# @Software: PyCharm
# import torch
import json
from Tools.train_tool import BaseTrainTool
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering

from Tools.data_pipeline import InputExample


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def create_squad_examples(data_path, *args, **kwargs):
    with open(data_path, 'r') as f:
        content = json.load(f)
    content = content["data"]
    examples = []
    for i, line in enumerate(content):
        for one_paragraph in line["paragraphs"]:
            for qas in one_paragraph["qas"]:
                examples.append(InputExample(guid=qas["id"], texts=[qas["question"], one_paragraph["context"]],
                                             label=qas["answers"]))
    return examples


def prepare_input_for_squad(example, tokenizer, max_length, *args, **kwargs):
    doc_stride = 128
    pad_on_right = tokenizer.padding_side == "right"
    # 既要对examples进行truncation（截断）和padding（补全）还要还要保留所有信息，所以要用的切片的方法。
    # 每一个一个超长文本example会被切片成多个输入，相邻两个输入之间会有交集。
    tokenized_examples = tokenizer(
        example.texts[0 if pad_on_right else 1],
        example.texts[1 if pad_on_right else 0],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # print(example)
    # 我们使用overflow_to_sample_mapping参数来映射切片片ID到原始ID。
    # 比如有2个example被切成4片，那么对应是[0, 0, 1, 1]，前两片对应原来的第一个example。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset_mapping也对应4片
    # offset_mapping参数帮助我们映射到原始输入，由于答案标注在原始输入上，所以有助于我们找到答案的起始和结束位置。
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 重新标注数据
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 对每一片进行处理
        # 将无答案的样本标注到CLS上
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 区分question和context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 拿到原始的example 下标.
        sample_index = sample_mapping[i]
        answers = example.label[sample_index]
        # 如果没有答案，则使用CLS所在的位置为答案.
        # if len(answers["answer_start"]) == 0:
        if len(answers) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 答案的character级别Start/end位置.
            start_char = answers["answer_start"]
            end_char = start_char + len(answers["text"])

            # 找到token级别的index start.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # 找到token级别的index end.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 检测答案是否超出文本长度，超出的话也适用CLS index作为标注.
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 如果不超出则找到答案token的start和end位置。.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    # print(tokenized_examples)
    # exit()
    return tokenized_examples


class TrainExtractQA(BaseTrainTool):
    def __init__(self, config_path):
        super(TrainExtractQA, self).__init__(config_path=config_path,
                                             data_func=create_squad_examples,
                                             prepare_input=prepare_input_for_squad)

    # self.create_examples = self

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name)
        # model = MultiLabelClsModel(self.config)
        model_config = AutoConfig.from_pretrained(self.model_args.config_name)
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            config=model_config,
        )
        # self.logger.debug(model)

        # for param in model.longformer.parameters():
        #     param.requires_grad = False

        return tokenizer, model

    # def cal_loss(self, batch):
    #     self.logger.debug(batch)
    # self.logger.debug(batch)
    # labels = batch['labels']
    # input_data = {'input_ids': batch['input_ids'],
    #               'atention_mask': batch['attention_mask'],
    #               'token_type_ids': batch['token_type_ids']}
    # pred = self.model(batch['input_ids'].squeeze())
    # loss = self.criterion(torch.sigmoid(pred.logits), labels.squeeze())
    # self.logger.debug(loss)
    # return loss


if __name__ == '__main__':
    TrainExtractQA(config_path="BasicTask/QA/Extractive-QA/base_qa.yaml").run()
