# coding=utf-8
import random
import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../modules'))
from modules import bert_modeling, optimization, tfm_modeling, tokenization

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def convert_tokens_to_ids(data_file, target_file, max_seq_length, tokenizer):
    source_file = open(data_file, 'r')
    target_file = open(target_file, 'w')
    for line in source_file.readlines():
        tokens = tokenizer.tokenize(line.strip())
        if len(tokens) > max_seq_length:
            tokens = tokens[0:max_seq_length]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
        target_file.write(' '.join(str(ids) for ids in input_ids) + '\n')
    source_file.close()
    target_file.close()


def convert_ids_to_tokens(data_file, target_file, tokenizer):
    source_file = open(data_file, 'r')
    target_file = open(target_file, 'w')
    for line in source_file.readlines():
        ids = [int(i) for i in line.strip().split()]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        target_file.write(''.join(tokens) + '\n')
    source_file.close()
    target_file.close()



#######################################################################################################################################
#       DATALOADER
#######################################################################################################################################

class GenPretrainDataLoader(object):
    def __init__(self, input_ids, start_token_index):
        self.input_ids = []
        self.decode_ids = []
        self.target_ids = []
        for input_id in input_ids:
            for i in range(len(input_id)-1):
                if input_id[i+1] == 0:
                    break
            self.input_ids.append(input_id)
            self.decode_ids.append([start_token_index] + input_id[:i] + [0] * (len(input_id) - i -1))
            self.target_ids.append(input_id[:i+1] + [0] * (len(input_id) - i -1))
        self.input_ids = np.array(self.input_ids)
        self.decode_ids = np.array(self.decode_ids)
        self.target_ids = np.array(self.target_ids)
        self.pointer = 0
        tf.logging.info('input_ids %s' % (self.input_ids[0]))
        tf.logging.info('decode_ids %s' % (self.decode_ids[0]))
        tf.logging.info('target_ids %s' % (self.target_ids[0]))

    def next_batch(self, batch_size):
        self.num_batch = len(self.input_ids) // batch_size
        if self.pointer % self.num_batch == 0:
            shuffle_indices = np.random.permutation(np.arange(len(self.input_ids)))
            self.input_ids = self.input_ids[shuffle_indices]
            self.decode_ids = self.decode_ids[shuffle_indices]
            self.target_ids = self.target_ids[shuffle_indices]

            self.input_ids_batch = np.split(self.input_ids[:self.num_batch * batch_size], self.num_batch, 0)
            self.decode_ids_batch = np.split(self.decode_ids[:self.num_batch * batch_size], self.num_batch, 0)
            self.target_ids_batch = np.split(self.target_ids[:self.num_batch * batch_size], self.num_batch, 0)
            self.pointer = 0

        batch_data = self.input_ids_batch[self.pointer], self.decode_ids_batch[self.pointer], self.target_ids_batch[self.pointer]
        self.pointer = self.pointer + 1
        return batch_data

    def reset_pointer(self):
        self.pointer = 0


class GenTrainDataLoader(object):
    def __init__(self, input_ids, start_token_index):
        self.input_ids = input_ids
        self.decode_ids = []
        for input_id in self.input_ids:
            self.decode_ids.append([start_token_index] + [0] * (len(input_id) - 1))
        self.input_ids = np.array(self.input_ids)
        self.decode_ids = np.array(self.decode_ids)
        self.pointer = 0
        tf.logging.info('input_ids %s' % (self.input_ids[0]))
        tf.logging.info('decode_ids %s' % (self.decode_ids[0]))

    def next_batch(self, batch_size):
        self.num_batch = len(self.input_ids) // batch_size
        if self.pointer % self.num_batch == 0:
            shuffle_indices = np.random.permutation(np.arange(len(self.input_ids)))
            self.input_ids = self.input_ids[shuffle_indices]
            self.decode_ids = self.decode_ids[shuffle_indices]

            self.input_ids_batch = np.split(self.input_ids[:self.num_batch * batch_size], self.num_batch, 0)
            self.decode_ids_batch = np.split(self.decode_ids[:self.num_batch * batch_size], self.num_batch, 0)
            self.pointer = 0

        batch_data = self.input_ids_batch[self.pointer], self.decode_ids_batch[self.pointer]
        self.pointer = self.pointer + 1
        return batch_data

    def reset_pointer(self):
        self.pointer = 0


class DisDataLoader(object):
    def __init__(self, positive_ids, negative_ids):
        self.input_ids = positive_ids + negative_ids
        self.input_mask = []
        self.segment_ids = []
        for input_id in self.input_ids:
            self.input_mask.append([1 if ii!=0 else 0 for ii in input_id])
            self.segment_ids.append([0 for ii in input_id])
        self.labels = [1 for _ in positive_ids] + [0 for _ in negative_ids]
        self.input_ids = np.array(self.input_ids)
        self.input_mask = np.array(self.input_mask)
        self.segment_ids = np.array(self.segment_ids)
        self.labels = np.array(self.labels)
        self.pointer = 0
        tf.logging.info('input_ids %s' % (self.input_ids[-1]))
        tf.logging.info('labels %s' % (self.labels[-1]))
        with open('../data/generator/positive_ids.txt', 'w', encoding='utf-8') as f:
            for ids in positive_ids:
                f.write(','.join(str(i) for i in ids) + '\n')
        with open('../data/generator/negative_ids.txt', 'w', encoding='utf-8') as f:
            for ids in negative_ids:
                f.write(','.join(str(i) for i in ids) + '\n')

    def next_batch(self, batch_size):
        self.num_batch = len(self.input_ids) // batch_size
        if self.pointer % self.num_batch == 0:
            shuffle_indices = np.random.permutation(np.arange(len(self.input_ids)))
            self.input_ids = self.input_ids[shuffle_indices]
            self.input_mask = self.input_mask[shuffle_indices]
            self.segment_ids = self.segment_ids[shuffle_indices]
            self.labels = self.labels[shuffle_indices]

            self.input_ids_batch = np.split(self.input_ids[:self.num_batch * batch_size], self.num_batch, 0)
            self.input_mask_batch = np.split(self.input_mask[:self.num_batch * batch_size], self.num_batch, 0)
            self.segment_ids_batch = np.split(self.segment_ids[:self.num_batch * batch_size], self.num_batch, 0)
            self.labels_batch = np.split(self.labels[:self.num_batch * batch_size], self.num_batch, 0)
            self.pointer = 0
        batch_data = self.input_ids_batch[self.pointer], self.input_mask_batch[self.pointer], \
                     self.segment_ids_batch[self.pointer], self.labels_batch[self.pointer]
        self.pointer = self.pointer + 1
        return batch_data

    def reset_pointer(self):
        self.pointer = 0



#######################################################################################################################################
#       DISCRIMINATOR
#######################################################################################################################################

# BERT
# class Discriminator(object):
#     def __init__(self, name, max_seq_length, label_list, bert_config_file):
#         self.name = name
#         self.max_seq_length = max_seq_length
#         self.label_list = label_list
#         self.bert_config = bert_modeling.BertConfig.from_json_file(bert_config_file)
#
#         self.input_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_ids")
#         self.input_mask_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_mask")
#         self.segment_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="segment_ids")
#         self.labels_p = tf.placeholder(tf.int32, [None, ], name="labels")
#
#     def inference(self, input_ids, input_mask, segment_ids, training=True):
#         model = bert_modeling.BertModel(
#             config=self.bert_config,
#             is_training=training,
#             input_ids=input_ids,
#             input_mask=input_mask,
#             token_type_ids=segment_ids,
#             use_one_hot_embeddings=False,
#             scope='bert')
#         output_layer = model.get_pooled_output()
#         hidden_size = output_layer.shape[-1].value
#
#         with tf.variable_scope(self.name):
#             if training:
#                 output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
#
#             output_weights = tf.get_variable(
#                 "output_weights", [len(self.label_list), hidden_size],
#                 initializer=tf.truncated_normal_initializer(stddev=0.02))
#             output_bias = tf.get_variable(
#                 "output_bias", [len(self.label_list)], initializer=tf.zeros_initializer())
#
#             logits = tf.matmul(output_layer, output_weights, transpose_b=True)
#             logits = tf.nn.bias_add(logits, output_bias)
#
#         return logits
#
#     def train(self, learning_rate, total_steps, warmup_steps):
#         logits = self.inference(self.input_ids_p, self.input_mask_p, self.segment_ids_p, True)
#         log_probs = tf.nn.log_softmax(logits, axis=-1)
#         one_hot_labels = tf.one_hot(self.labels_p, depth=len(self.label_list), dtype=tf.float32)
#
#         per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
#         loss = tf.reduce_mean(per_example_loss)
#
#         train_op, learning_rate, global_step = optimization.create_optimizer(
#             loss, learning_rate, total_steps, warmup_steps, False)
#
#         return loss, train_op, learning_rate, global_step
#
#     def predict(self):
#         logits = self.inference(self.input_ids_p, self.input_mask_p, self.segment_ids_p, False)
#         probabilities = tf.nn.softmax(logits)
#         return probabilities
#
#     def init_model(self, sess, init_checkpoint):
#         tvars = tf.trainable_variables()
#         (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
#         saver = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
#         saver.restore(sess, init_checkpoint)
#
#         tvars = tf.global_variables()
#         not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
#         tf.logging.info('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
#         if len(not_initialized_vars):
#             sess.run(tf.variables_initializer(not_initialized_vars))
#         for v in not_initialized_vars:
#             tf.logging.info('not initialized: %s' % (v.name))
#
#     def load_model(self, sess, model_path):
#         tvars = tf.trainable_variables()
#         init_checkpoint = tf.train.latest_checkpoint(model_path)
#         (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
#         saver = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
#         saver.restore(sess, init_checkpoint)
#
#         tvars = tf.global_variables()
#         not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
#         tf.logging.info('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
#         if len(not_initialized_vars):
#             sess.run(tf.variables_initializer(not_initialized_vars))
#
#     def save_model(self, sess, model_path, step):
#         saver = tf.train.Saver(max_to_keep=1)
#         saver.save(sess, model_path, global_step=step)


class Discriminator(object):
    def __init__(self, name, config_file, label_list, max_seq_length, dropout=0.9):
        self.name = name
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        self.bert_config = bert_modeling.BertConfig.from_json_file(config_file)

        self.input_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_ids")
        self.input_mask_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_mask")
        self.segment_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="segment_ids")
        self.labels_p = tf.placeholder(tf.int32, [None, ], name="labels")

    def inference(self, input_ids, input_mask, segment_ids, is_training=True):
        '''
        inputs:
        memory: encoder outputs. (batch_size, max_seq_length, hidden_size)
        '''
        model = bert_modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            output_layer = model.get_pooled_output()
            hidden_size = output_layer.shape[-1].value

            output_weights_deep = tf.get_variable(
                "output_weights", [len(self.label_list), hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [len(self.label_list)], initializer=tf.zeros_initializer())

            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=self.dropout)

            logits = tf.matmul(output_layer, output_weights_deep, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            return logits

    def train(self, learning_rate, total_steps, warmup_steps):
        logits = self.inference(self.input_ids_p, self.input_mask_p, self.segment_ids_p, True)

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(self.labels_p, depth=len(self.label_list), dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        train_op, learning_rate, global_step = optimization.create_optimizer(
            loss, learning_rate, total_steps, warmup_steps, False)

        return loss, train_op, learning_rate, global_step

    def predict(self):
        logits = self.inference(self.input_ids_p, self.input_mask_p, self.segment_ids_p, False)
        probabilities = tf.nn.softmax(logits)
        return probabilities



#######################################################################################################################################
#       GENERATOR
#######################################################################################################################################

class Generator(object):
    def __init__(self, name, vocab_size, num_blocks, num_heads, max_seq_length, hidden_size=512, ff_size=2048, dropout=0.1):
        self.name = name
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.dropout = dropout

        self.input_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_ids")
        self.decode_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="decode_ids")
        self.target_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="target_ids")
        self.rewards_p = tf.placeholder(tf.float32, [None, max_seq_length], name="rewards")
        self.noise_p = tf.placeholder(tf.float32, [None, max_seq_length, hidden_size], name="noise")

    def encode(self, input_ids, training=True):
        '''
        inputs:
        memory: encoder outputs. (batch_size, max_seq_length, hidden_size)
        '''
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.embeddings = tfm_modeling.get_token_embeddings(self.vocab_size, self.hidden_size, zero_pad=True)

            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                # embedding
                encoding = tf.nn.embedding_lookup(self.embeddings, input_ids)  # (batch_size, max_seq_length, hidden_size)
                encoding *= self.hidden_size**0.5  # scale

                encoding += tfm_modeling.positional_encoding(encoding, self.max_seq_length)
                encoding = tf.layers.dropout(encoding, self.dropout, training=training)

                # Blocks
                for i in range(self.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # self-attention
                        encoding = tfm_modeling.multihead_attention(queries=encoding,
                                                                    keys=encoding,
                                                                    values=encoding,
                                                                    num_heads=self.num_heads,
                                                                    dropout_rate=self.dropout,
                                                                    training=training,
                                                                    causality=False)
                        # feed forward
                        encoding = tfm_modeling.ff(encoding, num_units=[self.ff_size, self.hidden_size])
        return encoding

    def decode(self, decode_ids, memory, training=True):
        '''
        decodes: (batch_size, length)
        memory: encoder outputs. (batch_size, max_seq_length, hidden_size)
        '''
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                # embedding
                decoding = tf.nn.embedding_lookup(self.embeddings, decode_ids)  # (batch_size, length, hidden_size)
                decoding *= self.hidden_size ** 0.5  # scale

                decoding += tfm_modeling.positional_encoding(decoding, decoding.shape[1].value)
                decoding = tf.layers.dropout(decoding, self.dropout, training=training)

                # Blocks
                for i in range(self.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # Masked self-attention (Note that causality is True at this time)
                        decoding = tfm_modeling.multihead_attention(queries=decoding,
                                                                    keys=decoding,
                                                                    values=decoding,
                                                                    num_heads=self.num_heads,
                                                                    dropout_rate=self.dropout,
                                                                    training=training,
                                                                    causality=True,
                                                                    scope="self_attention")

                        # Vanilla attention
                        decoding = tfm_modeling.multihead_attention(queries=decoding,
                                                                    keys=memory,
                                                                    values=memory,
                                                                    num_heads=self.num_heads,
                                                                    dropout_rate=self.dropout,
                                                                    training=training,
                                                                    causality=False,
                                                                    scope="vanilla_attention")
                        # Feed Forward
                        decoding = tfm_modeling.ff(decoding, num_units=[self.ff_size, self.hidden_size])

        return decoding

    def generate_sentence(self, input_ids, decode_ids, given_length, training, add_noise):
        """
        循环生成句子
        :param given_length:
        :param training:
        :return:
        """
        memory = self.encode(input_ids, training)
        if add_noise:
            memory = memory + self.noise_p
        probabilities = None
        predictions = None
        for i in range(given_length, self.max_seq_length):
            decoding = self.decode(decode_ids, memory, training)
            weights = tf.transpose(self.embeddings)  # (hidden_size, vocab_size)
            logits = tf.einsum('ntd,dk->ntk', decoding, weights)  # (batch_size, seq_length, vocab_size)
            probabilities = tf.nn.softmax(logits, axis=-1)  # (batch_size, seq_length)
            predictions = tf.to_int32(tf.argmax(logits, axis=-1))  # (batch_size, seq_length)

            pred_y = tf.squeeze(predictions[:, i:i + 1])
            if tf.reduce_sum(pred_y) == 0:
                break
            decode_ids = tf.concat([decode_ids[:, 0:i+1], predictions[:, i:i+1], decode_ids[:, i+2:]], axis=-1)
        return probabilities, predictions

    def inference(self, training):
        # forward
        memory = self.encode(self.input_ids_p, training)
        decoding = self.decode(self.decode_ids_p, memory, training)

        weights = tf.transpose(self.embeddings)  # (hidden_size, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', decoding, weights)  # (batch_size, seq_length, vocab_size)
        predictions = tf.to_int32(tf.argmax(logits, axis=-1))  # (batch_size, seq_length)

        return logits, predictions

    def pretrain(self, learning_rate, total_steps, warmup_steps):
        logits, prediction = self.inference(True)

        labels = tfm_modeling.label_smoothing(tf.one_hot(self.target_ids_p, depth=self.vocab_size))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

        nonpadding = tf.to_float(tf.not_equal(self.target_ids_p, 0))
        loss = tf.reduce_sum(loss * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-8)

        train_op, learning_rate, global_step = optimization.create_optimizer(
            loss, learning_rate, total_steps, warmup_steps, False)

        return loss, train_op, learning_rate, global_step

    def train(self, learning_rate, total_steps, warmup_steps):
        """

        :param given_length:
        :param rewards:
        :param learning_rate:
        :param total_steps:
        :param warmup_steps:
        :return:
        """
        probabilities, _ = self.generate_sentence(self.input_ids_p, self.decode_ids_p, 0, True, False)

        predictions = tf.log(tf.clip_by_value(tf.reshape(probabilities, [-1, self.vocab_size]), 1e-20, 1.0))
        targets = tf.one_hot(tf.to_int32(tf.reshape(self.target_ids_p, [-1])), self.vocab_size, 1.0, 0.0)

        loss = -tf.reduce_sum(tf.reduce_sum(targets * predictions, 1) * tf.reshape(self.rewards_p, [-1]))
        train_op, learning_rate, global_step = optimization.create_optimizer(
            loss, learning_rate, total_steps, warmup_steps, False)

        return loss, train_op, learning_rate, global_step

    def predict(self, given_length, add_noise):
        _, predictions = self.generate_sentence(self.input_ids_p, self.decode_ids_p, given_length, False, add_noise)
        return predictions



#######################################################################################################################################
#       ROLLOUT
#######################################################################################################################################

class Rollout(object):
    def __init__(self, generator):
        self.vocab_size = generator.vocab_size
        self.num_blocks = generator.num_blocks
        self.num_heads = generator.num_heads
        self.max_seq_length = generator.max_seq_length
        self.hidden_size = generator.hidden_size
        self.ff_size = generator.ff_size
        self.dropout = generator.dropout
        self.generator = Generator('G',
                                   self.vocab_size,
                                   self.num_blocks,
                                   self.num_heads,
                                   self.max_seq_length,
                                   self.hidden_size,
                                   self.ff_size,
                                   self.dropout)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.predictions = []
        for given_length in range(1, self.max_seq_length):
            self.predictions.append(self.generator.predict(given_length, False))

    def reset_parameter(self, model_path):
        tf.logging.info('Loading Model')
        tvars = tf.trainable_variables()
        init_checkpoint = tf.train.latest_checkpoint(model_path)
        (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        saver = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
        saver.restore(self.sess, init_checkpoint)

        tvars = tf.global_variables()
        not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
        tf.logging.info('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def get_reward_predictions(self, input_ids, rollout_num, start_token_index):
        predict_ids = []
        for i in range(rollout_num):
            pred_ids = []
            # given_num between 1 to sequence_length - 1 for a part completed sentence
            for given_length in range(1, self.max_seq_length):
                decode_ids = []
                for input_id in input_ids:
                    decode_ids.append([start_token_index] + input_id[:given_length].tolist() + [0] * (self.max_seq_length - given_length -1))
                preds = self.sess.run(self.predictions[given_length-1],
                                      feed_dict={self.generator.input_ids_p: input_ids,
                                                 self.generator.decode_ids_p: decode_ids})
                pred_ids.append(preds)
            predict_ids.append(pred_ids)
        return predict_ids



#######################################################################################################################################
#       SEQUENCE GAN
#######################################################################################################################################

class SeqGAN(object):
    def __init__(self, generator, discriminator, rollout, rollout_num, vocab_file, source_file, premodel_path, model_path):
        self.generator = generator
        self.discriminator = discriminator
        self.rollout = rollout

        self.rollout_num = rollout_num
        self.max_seq_length = generator.max_seq_length
        self.hidden_size = generator.hidden_size
        self.premodel_path = premodel_path
        self.model_path = model_path

        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.start_token_index = self.tokenizer.convert_tokens_to_ids(['<S>'])[0]
        self.end_token_index = self.tokenizer.convert_tokens_to_ids(['<T>'])[0]
        tf.logging.info('start token index: %s' % (self.start_token_index))
        tf.logging.info('end token index: %s' % (self.end_token_index))
        self.input_ids = []
        with open(source_file, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line.strip()))
                token_ids = token_ids[:self.max_seq_length-1]
                token_ids.append(self.end_token_index)
                while len(token_ids) < self.max_seq_length:
                    token_ids.append(0)
                self.input_ids.append(token_ids)
        self.data_size = len(self.input_ids)

    def model_pretrain(self, sess, batch_size, learning_rate, num_epoch, save_checkpoint_steps):
        total_steps = num_epoch * self.data_size // batch_size

        # create model
        predictions = self.generator.predict(0, True)
        dis_loss, dis_train_op, dis_learning_rate, dis_global_step = self.discriminator.train(learning_rate, total_steps, None)
        gen_loss, gen_train_op, gen_learning_rate, gen_global_step = self.generator.pretrain(learning_rate, total_steps, None)

        # model load
        tvars = tf.trainable_variables()
        init_checkpoint = '../model/bert/chinese/bert_model.ckpt'
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
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver([v for v in tvars if 'adam_v' not in v.name and 'adam_m' not in v.name], max_to_keep=1)

        # generator pretrain
        tf.logging.info('Generator Pretrain. steps: %s' % (total_steps))
        gen_data = GenPretrainDataLoader(self.input_ids, self.start_token_index)
        for step in range(total_steps):
            input_ids, decode_ids, target_ids = gen_data.next_batch(batch_size)
            ls, lr, _ = sess.run([gen_loss, gen_learning_rate, gen_train_op],
                                 feed_dict={self.generator.input_ids_p: input_ids,
                                            self.generator.decode_ids_p: decode_ids,
                                            self.generator.target_ids_p: target_ids})
            if step % 1000 == 0:
                tf.logging.info('step: %s; learning_rate: %s; loss: %s' % (step, lr, ls))
            if step > 0 and step % save_checkpoint_steps == 0:
                saver.save(sess, self.premodel_path + 'model.ckpt', global_step=step)
        saver.save(sess, self.premodel_path + 'model.ckpt', global_step=step)

        # generate sentences
        tf.logging.info('Generate Sentence. steps: %s' % (self.data_size // batch_size // 10))
        gen_data = GenTrainDataLoader(self.input_ids, self.start_token_index)
        positive_ids = []
        negative_ids = []
        for i in range(self.data_size // batch_size // 10):
            input_ids, decode_ids = gen_data.next_batch(batch_size)
            noise = 2*np.random.randn(batch_size, self.max_seq_length, self.hidden_size) + 0.1
            preds = sess.run(predictions, feed_dict={self.generator.input_ids_p: input_ids,
                                                     self.generator.decode_ids_p: decode_ids,
                                                     self.generator.noise_p: noise})
            negative_ids += preds.tolist()
            positive_ids += input_ids.tolist()
            if i % 1000 == 0:
                tf.logging.info('step %s, positive data size %s, negative data size %s' % (i, len(positive_ids), len(negative_ids)))

        # discriminator pretrain
        tf.logging.info('Discriminator Pretrain. steps: %s' % (total_steps))
        dis_data = DisDataLoader(positive_ids, negative_ids)
        for step in range(total_steps):
            input_ids, input_mask, segment_ids, labels = dis_data.next_batch(batch_size)
            ls, lr, _ = sess.run([dis_loss, dis_learning_rate, dis_train_op],
                                 feed_dict={self.discriminator.input_ids_p: input_ids,
                                            self.discriminator.input_mask_p: input_mask,
                                            self.discriminator.segment_ids_p: segment_ids,
                                            self.discriminator.labels_p: labels})
            if step % 1000 == 0:
                tf.logging.info('step: %s; learning_rate: %s; loss: %s' % (step, lr, ls))
            if step > 0 and step % save_checkpoint_steps == 0:
                saver.save(sess, self.premodel_path + 'model.ckpt', global_step=step)
        saver.save(sess, self.premodel_path + 'model.ckpt', global_step=step)

    def model_train(self, sess, batch_size, learning_rate, num_epoch):
        gen_total_steps = self.data_size // batch_size // 10
        dis_total_steps = 4 * self.data_size // batch_size
        tf.logging.info('Generator total steps: %s' % (gen_total_steps))
        tf.logging.info('Discriminator total steps: %s' % (dis_total_steps))

        # create model
        predictions = self.generator.predict(0, True)
        probabilities = self.discriminator.predict()
        gen_loss, gen_train_op, gen_learning_rate, gen_global_step = self.generator.train(learning_rate, gen_total_steps, None)
        dis_loss, dis_train_op, dis_learning_rate, dis_global_step = self.discriminator.train(learning_rate, dis_total_steps, None)

        # load model
        self.load_model(sess, self.premodel_path)
        saver = tf.train.Saver([v for v in tf.global_variables() if 'adam_v' not in v.name and 'adam_m' not in v.name], max_to_keep=1)

        # train
        for epoch in range(num_epoch):
            # rollout reset parameter
            self.rollout.reset_parameter(self.model_path)

            # generator train
            gen_data = GenTrainDataLoader(self.input_ids, self.start_token_index)

            ls = 0
            tf.logging.info('Generator total steps: %s' % (gen_total_steps))
            for step in range(gen_total_steps):
                input_ids, decode_ids = gen_data.next_batch(batch_size)
                noise = 2*np.random.randn(batch_size, self.max_seq_length, self.hidden_size) + 0.1
                preds = sess.run(predictions, feed_dict={self.generator.input_ids_p: input_ids,
                                                         self.generator.decode_ids_p: decode_ids,
                                                         self.generator.noise_p: noise})

                predict_ids = self.rollout.get_reward_predictions(preds, self.rollout_num, self.start_token_index)
                rewards = []
                for i in range(self.rollout_num):
                    # given_num between 1 to sequence_length - 1 for a part completed sentence
                    for given_length in range(1, self.max_seq_length):
                        pre_ids = predict_ids[i][given_length-1]
                        pre_mask = [[1 if ii!=0 else 0 for ii in inp] for inp in pre_ids]
                        seg_ids = [[0 for _ in inp] for inp in pre_ids]
                        probs = sess.run(probabilities, feed_dict={self.discriminator.input_ids_p: pre_ids,
                                                                   self.discriminator.input_mask_p: pre_mask,
                                                                   self.discriminator.segment_ids_p: seg_ids})
                        ypred = np.array([item[1] for item in probs])
                        if i == 0:
                            rewards.append(ypred)
                        else:
                            rewards[given_length - 1] += ypred

                    # the last token reward
                    pred_mask = [[1 if ii!=0 else 0 for ii in inp] for inp in preds]
                    segm_ids = [[0 for _ in inp] for inp in preds]
                    probs = sess.run(probabilities, feed_dict={self.discriminator.input_ids_p: preds,
                                                               self.discriminator.input_mask_p: pred_mask,
                                                               self.discriminator.segment_ids_p: segm_ids})
                    ypred = np.array([item[1] for item in probs])
                    if i == 0:
                        rewards.append(ypred)
                    else:
                        rewards[self.max_seq_length - 1] += ypred

                rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)  # batch_size x seq_length

                ls, lr, _ = sess.run([gen_loss, gen_learning_rate, gen_train_op],
                                     feed_dict={self.generator.input_ids_p: input_ids,
                                                self.generator.decode_ids_p: decode_ids,
                                                self.generator.target_ids_p: input_ids,
                                                self.generator.rewards_p: rewards})
                if step%20==0:
                    tf.logging.info('step: %s; learning_rate: %s; loss: %s' % (step, lr, ls))
            tf.logging.info('epoch: %s; generator loss: %s' % (epoch, ls))
            saver.save(sess, self.model_path + 'model.ckpt', global_step=epoch)

            # generate sentences
            tf.logging.info('Generate Sentence. steps: %s' % (self.data_size // batch_size // 10))
            positive_ids = []
            negative_ids = []
            for _ in range(self.data_size // batch_size // 10):
                input_ids, decode_ids = gen_data.next_batch(batch_size)
                noise = 2*np.random.randn(batch_size, self.max_seq_length, self.hidden_size) + 0.1
                preds = sess.run(predictions, feed_dict={self.generator.input_ids_p: input_ids,
                                                         self.generator.decode_ids_p: decode_ids,
                                                         self.generator.noise_p: noise})
                negative_ids += preds.tolist()
                positive_ids += input_ids.tolist()

            # discriminator train
            tf.logging.info('Discriminator total steps: %s' % (dis_total_steps))
            dis_data = DisDataLoader(positive_ids, negative_ids)
            for step in range(dis_total_steps):
                input_ids, input_mask, segment_ids, labels = dis_data.next_batch(batch_size)
                ls, lr, _ = sess.run([dis_loss, dis_learning_rate, dis_train_op],
                                     feed_dict={self.discriminator.input_ids_p: input_ids,
                                                self.discriminator.input_mask_p: input_mask,
                                                self.discriminator.segment_ids_p: segment_ids,
                                                self.discriminator.labels_p: labels})
                if step%1000==0:
                    tf.logging.info('step: %s; learning_rate: %s; loss: %s' % (step, lr, ls))
            tf.logging.info('epoch: %s; discriminator loss: %s' % (epoch, ls))
            saver.save(sess, self.model_path + 'model.ckpt', global_step=epoch)

            # 生成语句
            input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('男方提出离婚'))
            input_ids = input_ids[:self.max_seq_length - 1] + [self.end_token_index]
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
            decode_ids = [self.start_token_index] + [0] * (self.max_seq_length - 1)

            for _ in range(10):
                noise = 2 * np.random.randn(self.max_seq_length, self.hidden_size) + 0.1
                pred_ids = sess.run(predictions, feed_dict={self.generator.input_ids_p: [input_ids],
                                                            self.generator.decode_ids_p: [decode_ids],
                                                            self.generator.noise_p: [noise]})
                sentence = self.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())
                tf.logging.info('generate sentence: %s' % (''.join(sentence)))

    def generate_sentence(self, sess, sentence, model_path):
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
        input_ids = input_ids[:self.max_seq_length-1] + [self.end_token_index]
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
        decode_ids = [self.start_token_index] + [0] * (self.max_seq_length - 1)

        tf.logging.info('orignial sentence: %s' % (sentence))
        predictions = self.generator.predict(0, True)
        self.load_model(sess, model_path)
        for _ in range(100):
            noise = np.random.randn(self.max_seq_length, self.hidden_size) + 0.1
            pred_ids = sess.run(predictions, feed_dict={self.generator.input_ids_p: [input_ids],
                                                        self.generator.decode_ids_p: [decode_ids],
                                                        self.generator.noise_p: [noise]})
            sentence = self.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())
            # if '<T>' in sentence:
            #     sentence = sentence[:sentence.index('<T>')]
            tf.logging.info('generate sentence: %s' % (''.join(sentence)))

    def load_model(self, sess, model_path):
        tf.logging.info('Loading Model')
        tvars = tf.trainable_variables()
        init_checkpoint = tf.train.latest_checkpoint(model_path)
        (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        saver = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
        saver.restore(sess, init_checkpoint)

        tvars = tf.global_variables()
        not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
        tf.logging.info('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))


def main():
    random.seed(2019)
    np.random.seed(2019)

    generator = Generator(name='G',
                          vocab_size=21128,
                          num_blocks=3,
                          num_heads=8,
                          max_seq_length=32,
                          hidden_size=512,
                          ff_size=1024,
                          dropout=0.2)
    discriminator = Discriminator(name='D',
                                  config_file='../model/bert/bert_config.json',
                                  label_list=[0, 1],
                                  max_seq_length=32,
                                  dropout=0.2)
    rollout = Rollout(generator=generator)
    model = SeqGAN(generator=generator,
                   discriminator=discriminator,
                   rollout=rollout,
                   rollout_num=8,
                   vocab_file='../model/bert/vocab.txt',
                   source_file='../data/generator/raw_data.txt',
                   premodel_path = '../model/bert/generator/pre_checkpoint/',
                   model_path='../model/bert/generator/checkpoint/')

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        # model.model_pretrain(sess=sess,
        #                      batch_size=64,
        #                      learning_rate=0.0001,
        #                      num_epoch=2,
        #                      save_checkpoint_steps=5000)
        # model.generate_sentence(sess, '男方提出离婚', model.premodel_path)
        # model.model_train(sess=sess,
        #                   batch_size=64,
        #                   learning_rate=0.0001,
        #                   num_epoch=6)
        model.generate_sentence(sess, '男方提出离婚', model.model_path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
