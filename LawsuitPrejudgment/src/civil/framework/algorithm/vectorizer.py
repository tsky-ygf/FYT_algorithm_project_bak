# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from LawsuitPrejudgment.src.civil.modules import tokenization, bert_modeling
import tensorflow as tf
import os
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


########################################################################################################################
#
# 词袋向量化
#
########################################################################################################################


class WordBagVectorizer(object):

    def __init__(self, text_file, n_features):
        with open(text_file, 'r', encoding='utf-8') as file:
            sentences = [' '.join(jieba.lcut(line.strip())) for line in file.readlines()]

        # 词袋模型构建
        self.vectorizer = CountVectorizer().fit(sentences)
        vector = self.vectorizer.transform(sentences).toarray()
        vector[vector > 0] = 1
        print('word bag size: %s' % (len(self.vectorizer.vocabulary_)))

        self.pca = PCA(n_components=n_features).fit(vector)
        print('sigular values: %s' % (self.pca.singular_values_))

    def vectorize(self, sentence):
        vector = self.vectorizer.transform([sentence]).toarray()
        vector = self.pca.transform(vector)
        return vector[0]


########################################################################################################################
#
# Tf-Idf向量化
#
########################################################################################################################


class TFIDFVectorizer(object):

    def __init__(self, text_file, n_features):
        with open(text_file, 'r', encoding='utf-8') as file:
            sentences = [' '.join(jieba.lcut(line.strip())) for line in file.readlines()]

        # 词袋模型构建
        self.vectorizer = TfidfVectorizer().fit(sentences)
        vector = self.vectorizer.transform(sentences).toarray()
        vector[vector > 0] = 1
        print('word bag size: %s' % (len(self.vectorizer.vocabulary_)))

        self.pca = PCA(n_components=n_features).fit(vector)
        print('sigular values: %s' % (self.pca.singular_values_))

    def vectorize(self, sentence):
        vector = self.vectorizer.transform([sentence]).toarray()
        vector = self.pca.transform(vector)
        return vector[0]


########################################################################################################################
#
# Word2Vec向量化
#
########################################################################################################################


class Word2VecVectorizer(object):

    def __init__(self, model_path='../model/word2vec/law.bin'):
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print('word size %s' % (len(self.model.wv.vocab)))

    def vectorize(self, sentence):
        vector = None
        for word in jieba.lcut_for_search(sentence):
            if word in self.model.wv.vocab:
                vector = self.model.wv[word] if vector is None else vector + self.model.wv[word]
        return vector


########################################################################################################################
#
# BERT向量化
#
########################################################################################################################

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


class BertVectorizer(object):

    def __init__(self,
                 max_seq_length=64,
                 checkpoint='../model/bert/law',
                 bert_config_file='../model/bert/bert_config.json',
                 vocab_file='../model/bert/vocab.txt'):

        self.max_seq_length = max_seq_length
        self.checkpoint = checkpoint
        self.bert_config = bert_modeling.BertConfig.from_json_file(bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)

        self.input_ids_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_seq_length), name='input_ids')
        self.input_mask_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_seq_length), name='input_mask')
        self.segment_ids_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_seq_length), name='segment_ids')

        self.output_layer = self.create_model()
        self.load_model()

    def create_model(self):
        model = bert_modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids_placeholder,
            input_mask=self.input_mask_placeholder,
            token_type_ids=self.segment_ids_placeholder,
            use_one_hot_embeddings=False,
            scope='bert')
        output_layer = model.get_pooled_output()
        return output_layer

    def load_model(self):
        init_checkpoint = tf.train.latest_checkpoint(self.checkpoint)
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        saver = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
        saver.restore(self.sess, init_checkpoint)

        tvars = tf.global_variables()
        not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
        print('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))
        for var in not_initialized_vars:
            print('not initialized: %s' % (var.name))

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_single_example(self, example, max_seq_length, tokenizer):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
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

    def vectorize(self, sentence, sentence_b=None):
        input_ids, input_mask, segment_ids, tokens = self.convert_single_example(
            InputExample(0, sentence, sentence_b), self.max_seq_length, self.tokenizer)
        vector = self.sess.run(self.output_layer, feed_dict={self.input_ids_placeholder: [input_ids],
                                                             self.input_mask_placeholder: [input_mask],
                                                             self.segment_ids_placeholder: [segment_ids]})
        return vector[0]


if __name__ == '__main__':
    vectorizer = WordBagVectorizer('../data/道路交通安全违法行为记分分值.txt', 20)
    print(vectorizer.vectorize('驾驶客车违章行驶'))
    vectorizer = TFIDFVectorizer('../data/道路交通安全违法行为记分分值.txt', 20)
    print(vectorizer.vectorize('驾驶客车违章行驶'))
    vectorizer = Word2VecVectorizer()
    print(vectorizer.vectorize('驾驶客车违章行驶'))
