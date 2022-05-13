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
from Tools.data_pipeline import DataProcessor, InputExample
from torch.utils.data import Dataset


def get_span_labels():
    """See base class."""
    return ["O", "address", "book", "company", 'game', 'government', 'movie', 'name', 'organization', 'position',
            'scene']


def get_crf_labels():
    """See base class."""
    return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
            'B-organization', 'B-position', 'B-scene', "I-address",
            "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
            'I-organization', 'I-position', 'I-scene',
            "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
            'S-name', 'S-organization', 'S-position',
            'S-scene', 'O']


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

    # def create_examples(self, lines, set_type):
    #     """Creates examples for the training and dev sets."""
    #     examples = []
    #     for (i, line) in enumerate(lines):
    #         if i == 0:
    #             continue
    #         guid = "%s-%s" % (set_type, i)
    #         text_a= line['words']
    #         # BIOS
    #         labels = []
    #         for x in line['labels']:
    #             if 'M-' in x:
    #                 labels.append(x.replace('M-','I-'))
    #             elif 'E-' in x:
    #                 labels.append(x.replace('E-', 'I-'))
    #             else:
    #                 labels.append(x)
    #         examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
    #     return examples

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
    def __init__(self, data_dir, tokenizer, mode, max_length=128, label_list=None):
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

        self.label_list = label_list
        self.id2label = {index: label for index, label in enumerate(self.label_list)}
        self.label2id = {label: index for index, label in enumerate(self.label_list)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def data_collator_test(batch):
        in_data = [one[:3] for one in batch]
        subjects = [one[3] for one in batch]
        input_ids, attention_mask, token_type_ids = map(torch.stack, zip(*in_data))
        return input_ids, attention_mask, token_type_ids, subjects


class ClueNerSpanDataset(ClueNerDataset):
    label_list = get_span_labels()
    def __init__(self, data_dir, tokenizer, mode, max_length=128):
        # self.label_list = get_span_labels()
        super().__init__(data_dir, tokenizer, mode, max_length, self.label_list)

    def __getitem__(self, item):
        example = self.examples[item]
        text = example.text_a
        subjects = example.subject
        # print(text)
        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                max_length=self.max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt",
                                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
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
        input_ids = inputs["input_ids"].long().squeeze(0)
        attention_mask = inputs["attention_mask"].long().squeeze(0)
        token_type_ids = inputs["token_type_ids"].long().squeeze(0)
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
        input_ids, attention_mask, token_type_ids, start_ids, end_ids = map(torch.stack, zip(*batch))
        return input_ids, attention_mask, token_type_ids, start_ids, end_ids


class ClueNerCRFDataset(ClueNerDataset):
    label_list = get_crf_labels()
    def __init__(self, data_dir, tokenizer, mode, max_length=128):
        # self.label_list = get_crf_labels()
        super().__init__(data_dir, tokenizer, mode, max_length, self.label_list)

    def __getitem__(self, item):
        example = self.examples[item]
        text = example.text_a
        subjects = example.subject
        # print(text)
        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                max_length=self.max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt",
                                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                                is_split_into_words=True)

        words = list(text)
        labels = ['O'] * len(words)
        if subjects is not None:
            for sub_name, start_index, end_index in subjects:
                if start_index == end_index:
                    labels[start_index] = 'S-' + sub_name
                else:
                    labels[start_index] = 'B-' + sub_name
                    for index in range(start_index + 1, end_index + 1):
                        labels[index] = 'I-' + sub_name
        # print(labels)
        label_ids = [self.label2id[x] for x in labels]
        # label_ids = label_ids[: (self.max_length - 2)]
        # label_ids += [self.label2id['O']]
        label_ids = label_ids[: (self.max_length - 2)]
        label_ids = [self.label2id['O']] + label_ids + [self.label2id['O']]
        label_ids += [0] * (self.max_length - len(label_ids))
        # print(subjects)
        input_ids = inputs["input_ids"].long().squeeze(0)
        attention_mask = inputs["attention_mask"].long().squeeze(0)
        token_type_ids = inputs["token_type_ids"].long().squeeze(0)

        if self.mode == "test":
            return input_ids, attention_mask, token_type_ids, label_ids
        else:
            return input_ids, attention_mask, token_type_ids, torch.tensor(label_ids)

    @staticmethod
    def data_collator(batch):
        input_ids, attention_mask, token_type_ids, label_ids = map(torch.stack, zip(*batch))
        return input_ids, attention_mask, token_type_ids, label_ids


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer_ = BertTokenizer.from_pretrained('model/language_model/bert-base-chinese')
    clue_ner_dataset = ClueNerCRFDataset(data_dir="data/cluener/train.json", tokenizer=tokenizer_, mode="train")
    # for i in range(5):
    #     print(clue_ner_dataset[i])
    print(clue_ner_dataset[0])
