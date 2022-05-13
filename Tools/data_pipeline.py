#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:10
# @Author  : Adolf
# @Site    : 
# @File    : data_pipeline.py
# @Software: PyCharm
import json
import copy

class InputExample:
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, subject):
        self.guid = guid
        self.text_a = text_a
        self.subject = subject

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @staticmethod
    def create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['text']
            subject = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples
