# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from LawsuitPrejudgment.src.civil.modules import tokenization, bert_modeling
import tensorflow as tf
import os
from LawsuitPrejudgment.src.civil.common.config_loader import model_path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# Required parameters
bert_config_file = model_path + 'bert/bert_config.json'
vocab_file = model_path + 'bert/vocab.txt'
checkpoint = model_path + 'bert/law'
do_lower_case = True
max_seq_length = 64


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_single_example(example, max_seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, tokens


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, input_ids, input_mask, segment_ids):
    model = bert_modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False,
        scope='bert')
    output_layer = model.get_pooled_output()
    return output_layer


bert_config = bert_modeling.BertConfig.from_json_file(bert_config_file)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
input_ids_placeholder = tf.placeholder(tf.int32, shape=(None, max_seq_length), name='input_ids')
input_mask_placeholder = tf.placeholder(tf.int32, shape=(None, max_seq_length), name='input_mask')
segment_ids_placeholder = tf.placeholder(tf.int32, shape=(None, max_seq_length), name='segment_ids')
output_layer = create_model(
    bert_config, input_ids=input_ids_placeholder, input_mask=input_mask_placeholder, segment_ids=segment_ids_placeholder)

init_checkpoint = tf.train.latest_checkpoint(checkpoint)
tvars = tf.trainable_variables()
(assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
saver = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
saver.restore(sess, init_checkpoint)

tvars = tf.global_variables()
not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
print('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))
for var in not_initialized_vars:
    print('not initialized: %s' % (var.name))


def vectorization(sentence, sentence_b=None):
    input_ids, input_mask, segment_ids, tokens = convert_single_example(InputExample(0, sentence, sentence_b), max_seq_length, tokenizer)
    vector = sess.run(output_layer, feed_dict={input_ids_placeholder: [input_ids],
                                               input_mask_placeholder: [input_mask],
                                               segment_ids_placeholder: [segment_ids]})
    return vector, tokens


if __name__=='__main__':
    tokens, vec = vectorization('我不想离婚')
    print(tokens)
    print(vec)