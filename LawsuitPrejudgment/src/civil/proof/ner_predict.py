# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
from LawsuitPrejudgment.src.civil.modules import tokenization, bert_modeling

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

flags = tf.flags

FLAGS = flags.FLAGS

# parameters
flags.DEFINE_string(
    "data_dir", "../data/proof",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", '../model/bert/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", '../model/bert/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", '../model/bert/proof',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
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
        self.label = label


def convert_single_example(example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    tokens_a = tokenizer.tokenize(example.text_a)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for i, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

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
    output_layer = model.get_sequence_output()
    seq_length = output_layer.shape[1].value
    hidden_size = output_layer.shape[2].value
    output_layer = tf.reshape(output_layer, shape=(-1, hidden_size))

    output_weights = tf.get_variable(
        "output_weights", [hidden_size, num_labels],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        logits = tf.matmul(output_layer, output_weights)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, shape=(-1, seq_length, num_labels))

    return logits


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)

input_ids_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_ids")
input_mask_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_mask")
segment_ids_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="segment_ids")

label_list = ['B', 'M', 'E', 'S', 'O']
bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
logits = create_model(bert_config, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=segment_ids_p, num_labels=len(label_list))

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(FLAGS.init_checkpoint))


def get_label_sentence(sentence, label):
    result = []
    temp = ''
    add = False
    for i in range(min(len(sentence), len(label))):
        if label[i]=='O':
            temp = ''
            add = False
            continue
        elif label[i]=='S':
            result.append(sentence[i])
            temp = ''
            add = False
        elif label[i]=='B':
            temp += sentence[i]
            add = True
        elif add:
            temp += sentence[i]
            if label[i]=='E':
                result.append(temp)
                temp = ''
                add = False
    return result


def predict(sentences, batch_size=32):
    input_ids = []
    input_mask = []
    segment_ids = []
    for i, sentence in enumerate(sentences):
        ids, mask, seg = convert_single_example(InputExample(guid=i, text_a=sentence), FLAGS.max_seq_length, tokenizer)
        input_ids.append(ids)
        input_mask.append(mask)
        segment_ids.append(seg)

    proba_result = []
    for epoch in range(len(sentences)//batch_size+1):
        probas = sess.run(logits, feed_dict={input_ids_p: input_ids[batch_size*epoch: batch_size*(epoch+1)],
                                             input_mask_p: input_mask[batch_size*epoch: batch_size*(epoch+1)],
                                             segment_ids_p: segment_ids[batch_size*epoch: batch_size*(epoch+1)]})
        proba_result += probas.tolist()

    result = []
    for sentence, proba in zip(sentences, proba_result):
        labels = ''.join([label_list[np.argmax(p)] for p in proba][1:-1])
        labels = get_label_sentence(sentence, labels)
        if len(labels)>0:
            result.append('、'.join(labels))
    return result
