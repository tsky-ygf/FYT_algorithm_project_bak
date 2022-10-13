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

import collections
import os
import sys
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
from LawsuitPrejudgment.src.civil.common import prob_ps, prob_ps_desc, chaming_fact_extract
from LawsuitPrejudgment.src.civil.modules import optimization, tokenization, bert_modeling
import configparser

config = configparser.ConfigParser()
config.read('../main/bert_train_config.ini')
section = sys.argv[1]
if not config.has_section(section):
    sys.exit(-1)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get(section, 'cuda_visible_devices'))

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", config.get(section, 'data_dir'),
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", '../model/bert/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", '../model/bert/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", config.get(section, 'output_dir'),
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", '../model/bert/chinese/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_valid", True, "Whether to run training.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("valid_batch_size", 16, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 3e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 10000,
                     "How often to save the model checkpoint.")


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


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_valid_examples(self, problem):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        data = pd.read_csv(input_file)
        data = data.sample(frac=1)
        tf.logging.info('file %s; size %s' % (input_file, len(data)))
        return data


# self inference_with_reason_Processor
class ClassificationProcessor(DataProcessor):
    """Processor for the self data set."""

    def get_train_examples(self):
        """See base class."""
        examples = self._create_examples(self._read_tsv(config.get(section, 'train_data_path')))
        permutation = np.random.permutation(np.arange(len(examples)))
        examples = np.array(examples)[permutation].tolist()
        return examples

    def get_valid_examples(self, problem):
        """See base class."""
        data = self._read_tsv(config.get(section, 'valid_data_path'))
        data = data[data['problem']==problem]
        examples = self._create_examples(data)
        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, data):
        """Creates examples for the training and dev sets."""
        examples = []
        data = data[data['label'] != -1].copy()
        data = pd.concat([data[data['biancheng'].isna()], data[~data['biancheng'].isna()][:300000]], sort=False)
        data = data.sample(frac=1)
        tf.logging.info('bian cheng filter data size %s' % (len(data)))
        data['chaming_fact'] = data['chaming'].apply(chaming_fact_extract)
        for index, row in data.iterrows():
            guid = "%s" % (index)
            label = row['label']
            problem = row['problem']
            suqiu = row['suqiu']
            text_a = problem + '。' + suqiu + '。' + prob_ps_desc[problem+'_'+suqiu]
            text_b = tokenization.convert_to_unicode(row['chaming_fact'])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


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


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
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

    label_ids = label_map[example.label]

    if ex_index < 2:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_ids))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_ids])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    """Creates a classification model."""
    model = bert_modeling.BertModel(
        config=bert_config,
        is_training=is_training,
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

    with tf.variable_scope("output"):

        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.sigmoid(logits)

        loss = 0
        if is_training:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            # log_probs = tf.nn.log_softmax(logits, axis=-1)
            # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            # label_weights = tf.cast(label_weights, tf.float32)
            # loss = tf.reduce_mean(label_weights * per_example_loss)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
            loss = tf.reduce_mean(loss)

        return loss, logits, probabilities


def train(config, batch_size, seq_length, init_checkpoint, train_file, model_path, num_labels, init_learning_rate,
          num_train_steps, num_warmup_steps, save_checkpoint_steps, num_epochs):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example, name_to_features):
        parsed_features = tf.parse_single_example(example, name_to_features)
        for name in list(parsed_features.keys()):
            t = parsed_features[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            parsed_features[name] = t
        return (parsed_features["input_ids"], parsed_features["input_mask"], parsed_features["segment_ids"],
                parsed_features["label_ids"])

    with tf.Graph().as_default():

        dataset = tf.data.TFRecordDataset(train_file)
        dataset = dataset.map(lambda x: _parse_function(x, name_to_features))
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True).repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        one_batch = iterator.get_next()

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        with tf.Session(config=gpu_config) as sess:

            input_ids_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_ids")
            input_mask_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_mask")
            segment_ids_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="segment_ids")
            label_id_p = tf.placeholder(tf.int32, [None, ], name="label_id")

            (loss, logits, probabilities) = create_model(
                config, is_training=True, input_ids=input_ids_p, input_mask=input_mask_p,
                segment_ids=segment_ids_p, labels=label_id_p, num_labels=num_labels)

            train_op, learning_rate, global_step = optimization.create_optimizer(
                loss, init_learning_rate, num_train_steps, num_warmup_steps, False)

            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            saver = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
            saver.restore(sess, init_checkpoint)

            tvars = tf.global_variables()
            not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
            tf.logging.info('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            for v in not_initialized_vars:
                tf.logging.info('not initialized: %s' % (v.name))

            saver = tf.train.Saver([v for v in tvars if 'moving' in v.name or ('adam_v' not in v.name and 'adam_m' not in v.name)], max_to_keep=3)
            losses = []
            step = 0
            lr = init_learning_rate
            ls = 0
            try:
                while step < num_train_steps:
                    input_ids, input_mask, segment_ids, label_id = sess.run(one_batch)
                    ls, lr, step, _ = sess.run([loss, learning_rate, global_step, train_op], feed_dict={input_ids_p: input_ids,
                                                                                                        input_mask_p: input_mask,
                                                                                                        segment_ids_p: segment_ids,
                                                                                                        label_id_p: label_id})
                    losses.append(ls)
                    if step % 100 == 0:
                        tf.logging.info('step: %s; learning_rate: %s; loss: %s' % (step, lr, ls))
                    if step > 0 and step % save_checkpoint_steps == 0:
                        saver.save(sess, model_path, global_step=step)
                        np.save('../model/bert/losses.npy', losses)
            except tf.errors.OutOfRangeError:
                pass
            tf.logging.info('step: %s; learning_rate: %s; loss: %s' % (step, lr, ls))
            saver.save(sess, model_path, global_step=step)
            np.save('../model/bert/losses.npy', losses)


def predict(config, batch_size, seq_length, init_checkpoint, predict_file, num_labels):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example, name_to_features):
        parsed_features = tf.parse_single_example(example, name_to_features)
        for name in list(parsed_features.keys()):
            t = parsed_features[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            parsed_features[name] = t
        return (parsed_features["input_ids"], parsed_features["input_mask"], parsed_features["segment_ids"],
                parsed_features["label_ids"])

    with tf.Graph().as_default():
        dataset = tf.data.TFRecordDataset(predict_file)
        dataset = dataset.map(lambda x: _parse_function(x, name_to_features))
        dataset = dataset.batch(batch_size, drop_remainder=False)
        iterator = dataset.make_one_shot_iterator()
        one_batch = iterator.get_next()

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        with tf.Session(config=gpu_config) as sess:

            input_ids_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_ids")
            input_mask_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_mask")
            segment_ids_p = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="segment_ids")
            label_ids_p = tf.placeholder(tf.int32, [None], name="label_ids")

            (loss, logits, probabilities) = create_model(
                config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=segment_ids_p,
                labels=label_ids_p, num_labels=num_labels)

            tvars = tf.trainable_variables()
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

            proba_result = []
            label_result = []
            try:
                while True:
                    input_ids, input_mask, segment_ids, label_ids = sess.run(one_batch)
                    probas = sess.run(probabilities, feed_dict={input_ids_p: input_ids,
                                                            input_mask_p: input_mask,
                                                            segment_ids_p: segment_ids,
                                                            label_ids_p: label_ids})
                    proba_result += probas.tolist()
                    label_result += label_ids.tolist()
            except tf.errors.OutOfRangeError:
                pass

            true_label = []
            prediction = []
            for i, label in enumerate(label_result):
                true_label.append(label)
                prediction.append(np.argmax(proba_result[i]))
            return prediction, true_label


def main(_):
    if not FLAGS.do_train and not FLAGS.do_valid:
        raise ValueError(
            "At least one of `do_train` or `do_valid` must be True.")

    bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_file = os.path.join(FLAGS.data_dir, "train.tf_record")
    valid_file = os.path.join(FLAGS.data_dir, "valid.tf_record")

    processor = ClassificationProcessor()
    label_list = processor.get_labels()

    if FLAGS.do_train:
        train_examples = processor.get_train_examples()
        file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # num_train_steps = int(512404 / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        tf.logging.info("***** Running training *****")
        # tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train(config=bert_config, batch_size=FLAGS.train_batch_size, seq_length=FLAGS.max_seq_length,
            init_checkpoint=FLAGS.init_checkpoint, train_file=train_file, model_path=os.path.join(FLAGS.output_dir, 'model.ckpt'),
            num_labels=len(label_list), init_learning_rate=FLAGS.learning_rate, num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps, save_checkpoint_steps=FLAGS.save_checkpoints_steps,
            num_epochs=FLAGS.num_train_epochs)

    if FLAGS.do_valid:
        predictions = []
        true_labels = []
        for problem, suqius in prob_ps.items():
            valid_examples = processor.get_valid_examples(problem)
            file_based_convert_examples_to_features(valid_examples, label_list, FLAGS.max_seq_length, tokenizer, valid_file)
            tf.logging.info("***** Running validing*****")
            tf.logging.info('  problem: %s' % (problem))
            tf.logging.info("  Num examples = %d", len(valid_examples))
            tf.logging.info("  Batch size = %d", FLAGS.valid_batch_size)
            if len(valid_examples)==0:
                continue
            prediction, true_label = predict(config=bert_config, batch_size=FLAGS.valid_batch_size,
                                             seq_length=FLAGS.max_seq_length, init_checkpoint=tf.train.latest_checkpoint(FLAGS.output_dir),
                                             predict_file=valid_file, num_labels=len(label_list))
            predictions += prediction
            true_labels += true_label
            prediction = np.array(prediction)
            true_label = np.array(true_label)
            tf.logging.info('accuracy: %s' % (accuracy_score(true_label, prediction)))
            tf.logging.info('f1_score: %s' % (f1_score(true_label, prediction, average='macro')))
            tf.logging.info(classification_report(true_label, prediction))
            tf.logging.info(confusion_matrix(true_label, prediction))

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        tf.logging.info('accuracy: %s' % (accuracy_score(true_labels, predictions)))
        tf.logging.info('f1_score: %s' % (f1_score(true_labels, predictions, average='macro')))
        tf.logging.info(classification_report(true_labels, predictions))
        tf.logging.info(confusion_matrix(true_labels, predictions))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('*'*100)
    tf.logging.info('bert performance')
    tf.logging.info('*'*100)
    tf.app.run()
