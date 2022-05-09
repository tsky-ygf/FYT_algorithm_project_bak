#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:04
# @Author  : Adolf
# @Site    : 
# @File    : model_ner_dataset.py
# @Software: PyCharm
import os
import json
import torch
from Tools.data_pipeline import DataProcessor
from torch.utils.data import Dataset


class ClueNerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(self.read_json(data_dir), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(self.read_json(data_dir), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(self.read_json(data_dir), "test")

    @staticmethod
    def get_labels():
        """See base class."""
        return ["O", "address", "book", "company", 'game', 'government', 'movie', 'name', 'organization', 'position',
                'scene']

    @classmethod
    def read_json(cls, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                # words = list(text)
                label_entities = line.get('label', {})
                labels = []
                if len(label_entities) > 0:
                    for key, value in label_entities.items():
                        for pos in [*value.values()]:
                            labels.append([key, pos[0][0], pos[0][1]])
                # print(label_entities)
                # print(labels)
                lines.append({"text": text, "labels": labels})
                # print(lines)
        # print(lines)
        return lines


class ClueNerDataset(Dataset):
    def __init__(self, data_dir, tokenizer, mode, max_length=128):
        self.processor = ClueNerProcessor()
        self.mode = mode
        if mode == "train":
            self.examples = self.processor.get_train_examples(data_dir)
        elif mode == "dev":
            self.examples = self.processor.get_dev_examples(data_dir)
        elif mode == "test":
            self.examples = self.processor.get_test_examples(data_dir)
        else:
            raise ValueError("Invalid mode: %s" % mode)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label_list = self.processor.get_labels()
        self.id2label = {index: label for index, label in enumerate(self.label_list)}
        self.label2id = {label: index for index, label in enumerate(self.label_list)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        text = example.text_a
        subjects = example.subject
        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                max_length=self.max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt",
                                is_split_into_words=True)

        start_ids = [0] * self.max_length
        end_ids = [0] * self.max_length
        # print(subjects)
        for subject in subjects:
            label = subject[0]
            start = subject[1] + 1
            end = subject[2] + 1
            start_ids[start] = self.label2id[label]
            end_ids[end] = self.label2id[label]
            # subjects_id.append((self.label2id[label], start, end))
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        # print("input_ids: %s", " ".join([str(x) for x in input_ids]))
        # print("start_ids: %s" % " ".join([str(x) for x in start_ids]))
        # print("end_ids: %s" % " ".join([str(x) for x in end_ids]))
        # labels = {"start_ids": torch.tensor(start_ids),
        #           "end_ids": torch.tensor(end_ids)}
        if self.mode == "test":
            return input_ids, attention_mask, token_type_ids, subjects
        else:
            return input_ids, attention_mask, token_type_ids, torch.tensor(start_ids), torch.tensor(end_ids)

    @staticmethod
    def data_collator(batch):
        """
        batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        input_ids, attention_mask, token_type_ids, start_ids, end_ids = map(torch.squeeze,
                                                                            map(torch.stack, zip(*batch)))
        return input_ids, attention_mask, token_type_ids, start_ids, end_ids

    @staticmethod
    def data_collator_test(batch):
        in_data = [one[:3] for one in batch]
        subjects = [one[3] for one in batch]
        input_ids, attention_mask, token_type_ids = map(torch.squeeze, map(torch.stack, zip(*in_data)))
        return input_ids, attention_mask, token_type_ids, subjects


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer_ = BertTokenizer.from_pretrained('model/language_model/bert-base-chinese')
    clue_ner_dataset = ClueNerDataset(data_dir="data/cluener", tokenizer=tokenizer_, mode="dev")
    for i in range(5):
        print(clue_ner_dataset[i])
