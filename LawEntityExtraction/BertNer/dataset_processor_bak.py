#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 15:11
# @Author  : Adolf
# @Site    : 
# @File    : dataset_processor_bak.py
# @Software: PyCharm
import os
import torch
import copy
import json
from torch.utils.data import TensorDataset



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


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, start_ids, end_ids, subjects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.input_len = input_len
        self.end_ids = end_ids
        self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class CluenerProcessor:
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    @staticmethod
    def get_labels():
        """See base class."""
        return ["O", "address", "book", "company", 'game', 'government', 'movie', 'name', 'organization', 'position',
                'scene']

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = line['labels']
            subject = get_entities(labels, id2label=None, markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                lines.append({"words": words, "labels": labels})
        return lines


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
        :param seq:
        :param id2label:
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
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


def get_entity_bio(seq, id2label):
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
        :param seq:
        :param id2label:
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


def get_entities(seq, id2label, markup='bios'):
    """
    :param seq:
    :param id2label:
    :param markup:
    :return:
    """
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def load_and_cache_examples(data_dir, data_type, tokenizer, max_seq_length=128):
    processor = CluenerProcessor()

    cached_features_file = os.path.join(data_dir, 'cached_span-{}_{}'.format('bert', 'cluener'))
    print(cached_features_file)

    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(data_dir)
        else:
            examples = processor.get_test_examples(data_dir)

        label2id = {label: i for i, label in enumerate(label_list)}
        features = []

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                print("Writing example %d of %d", ex_index, len(examples))
            textlist = example.text_a
            subjects = example.subject
            if isinstance(textlist, list):
                textlist = " ".join(textlist)
            tokens = tokenizer.tokenize(textlist)
            start_ids = [0] * len(tokens)
            end_ids = [0] * len(tokens)
            subjects_id = []
            for subject in subjects:
                label = subject[0]
                start = subject[1]
                end = subject[2]
                start_ids[start] = label2id[label]
                end_ids[end] = label2id[label]
                subjects_id.append((label2id[label], start, end))
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                start_ids = start_ids[: (max_seq_length - special_tokens_count)]
                end_ids = end_ids[: (max_seq_length - special_tokens_count)]
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SESEPP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            pad_token_segment_id = 0
            cls_token_segment_id = 0
            
            cls_token = tokenizer.cls_token
            sep_token = tokenizer.sep_token
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            
            tokens += [sep_token]
            start_ids += [0]
            end_ids += [0]
            sequence_a_segment_id = 0
            
            segment_ids = [sequence_a_segment_id] * len(tokens)
            cls_token_at_end = False
            pad_on_left = False
            mask_padding_with_zero = True
            
            if cls_token_at_end:
                tokens += [cls_token]
                start_ids += [0]
                end_ids += [0]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                start_ids = [0] + start_ids
                end_ids = [0] + end_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                start_ids = ([0] * padding_length) + start_ids
                end_ids = ([0] * padding_length) + end_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                start_ids += ([0] * padding_length)
                end_ids += ([0] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(start_ids) == max_seq_length
            assert len(end_ids) == max_seq_length

            if ex_index < 5:
                print("*** Example ***")
                print("guid: %s", example.guid)
                print("tokens: %s", " ".join([str(x) for x in tokens]))
                print("input_ids: %s", " ".join([str(x) for x in input_ids]))
                print("input_mask: %s", " ".join([str(x) for x in input_mask]))
                print("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                print("start_ids: %s" % " ".join([str(x) for x in start_ids]))
                print("end_ids: %s" % " ".join([str(x) for x in end_ids]))

            features.append(InputFeature(input_ids=input_ids,
                                         input_mask=input_mask,
                                         segment_ids=segment_ids,
                                         start_ids=start_ids,
                                         end_ids=end_ids,
                                         subjects=subjects_id,
                                         input_len=input_len))

        # print(features)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids,
                            all_input_lens)
    return dataset


if __name__ == '__main__':
    from transformers import BertTokenizer

    tokenizer_ = BertTokenizer.from_pretrained('model/language_model/bert-base-chinese')
    load_and_cache_examples("data/cluener_public", "dev", tokenizer_, 128)
