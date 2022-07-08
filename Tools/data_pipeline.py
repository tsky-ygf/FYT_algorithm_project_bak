#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:10
# @Author  : Adolf
# @Site    : 
# @File    : data_pipeline.py
# @Software: PyCharm
import json
import copy

import torch.utils.data as data


class InputExample:
    """A single training/test example for token classification."""

    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, create_examples=None):
        if create_examples is not None:
            self.create_examples = create_examples
        else:
            self.create_examples = self.read_examples

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self.create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self.create_examples(data_dir, "test")

    @staticmethod
    def read_examples(data_path, set_type):
        """Creates examples for the training and dev sets."""
        with open(data_path, 'rb') as f:
            lines = json.load(f)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line['text']
            label = line['label']
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    # @staticmethod
    # def read_data(data_dir):
    #     raise NotImplementedError()


class BaseDataset(data.Dataset):
    def __init__(self, data_dir_dict, tokenizer, mode, max_length=128, create_examples=None, is_debug=False):
        self.processor = DataProcessor(create_examples=create_examples)
        self.mode = mode

        if mode == "train":
            self.examples = self.processor.get_train_examples(data_dir_dict['train'])
        elif mode == "dev":
            self.examples = self.processor.get_dev_examples(data_dir_dict['dev'])
        elif mode == "test":
            self.examples = self.processor.get_test_examples(data_dir_dict['test'])
        else:
            raise ValueError("Invalid mode: %s" % mode)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if is_debug:
            self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        inputs = self.prepare_input(self.examples[item], self.tokenizer, max_len=self.max_length)
        return inputs

    @staticmethod
    def prepare_input(example, tokenizer, max_len=512):
        text = example.text
        label = example.label

        inputs = tokenizer(text,
                           add_special_tokens=True,
                           max_length=max_len,
                           padding="max_length",
                           truncation=True,
                           return_offsets_mapping=False,
                           return_tensors="pt")
        inputs['label'] = label

        return inputs
