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

import os
import sys
import traceback
import configparser

import tensorflow as tf
from LawsuitPrejudgment.common import prob_ps_desc
from LawsuitPrejudgment.modules import bert_modeling, tokenization

config = configparser.ConfigParser()
config.read('LawsuitPrejudgment/main/config.ini')
max_seq_length = 512
print(config.sections())
label_list = [0, 1]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def create_example(problem, suqiu, inputs):
    text_a = problem + '。' + suqiu + '。' + prob_ps_desc[problem+'_'+suqiu]
    text_b = tokenization.convert_to_unicode(inputs)
    return InputExample(guid=0, text_a=text_a, text_b=text_b)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
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

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def create_model(bert_config, input_ids, input_mask, segment_ids, num_labels):
    """Creates a classification model."""
    model = bert_modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    with tf.variable_scope('output'):

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.sigmoid(logits)

    return probabilities

def init_sess(section):
    bert_config = bert_modeling.BertConfig.from_json_file(config.get(section, 'bert_config_file'))
    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tokenizer = tokenization.FullTokenizer(vocab_file=config.get(section, 'vocab_file'), do_lower_case=True)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    input_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_mask")
    segment_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="segment_ids")

    probabilities = create_model(bert_config, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=segment_ids_p,
                                 num_labels=len(label_list))

    tvars = tf.trainable_variables()
    init_checkpoint = tf.train.latest_checkpoint(config.get(section, 'model_path'))
    (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    init_vars = [v for v in tf.global_variables() if v.name in initialized_variable_names or 'moving' in v.name]
    init_vars_name = [v.name for v in init_vars]
    saver = tf.train.Saver(init_vars, max_to_keep=1)
    saver.restore(sess, init_checkpoint)

    tvars = tf.global_variables()
    not_initialized_vars = [v for v in tvars if v.name not in init_vars_name]
    tf.logging.info('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
    for v in not_initialized_vars:
        tf.logging.info('not initialized: %s' % (v.name))
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
    return sess, probabilities, input_ids_p, input_mask_p, segment_ids_p, tokenizer


sess, probabilities, input_ids_p, input_mask_p, segment_ids_p, tokenizer = None, None, None, None, None, None
g1 = tf.Graph()
with g1.as_default():
    sess, probabilities, input_ids_p, input_mask_p, segment_ids_p, tokenizer = init_sess('modelPath')

sess2, probabilities2, input_ids_p2, input_mask_p2, segment_ids_p2, tokenizer2 = None, None, None, None, None, None
g2 = tf.Graph()
with g2.as_default():
    sess2, probabilities2, input_ids_p2, input_mask_p2, segment_ids_p2, tokenizer2 = init_sess('newModelPath')


def predict(problem, suqiu, inputs):
    try:
        example = create_example(problem, suqiu, inputs)
        if problem in ['租赁合同']:
            with g2.as_default():
                input_ids, input_mask, segment_ids = convert_single_example(example, max_seq_length, tokenizer2)
                result = sess2.run(probabilities2, feed_dict={input_ids_p2: [input_ids],
                                                              input_mask_p2: [input_mask],
                                                              segment_ids_p2: [segment_ids]})
        else:
            with g1.as_default():
                input_ids, input_mask, segment_ids = convert_single_example(example, max_seq_length, tokenizer)
                result = sess.run(probabilities, feed_dict={input_ids_p: [input_ids],
                                                            input_mask_p: [input_mask],
                                                            segment_ids_p: [segment_ids]})
        return result[0][1]
    except:
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print(predict('婚姻继承', '离婚', '分居两年办，想离婚'))
