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
"""BERT finetuning runner of classification for online prediction. input is a list. output is a label."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import bert.modeling as modeling
import bert.tokenization as tokenization
import tensorflow as tf
import numpy as np
import logging
import random
import copy

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
BERT_BASE_DIR="../model_files/question_answering/checkpoint_bert_qa_rank_q2a_200w_0525/" # "../model_files/inference_with_reason/checkpoint_bert/"
BERT_BASE_DIR="../model_files/question_answering/checkpoint_bert_qa_rank_q2a_200w_0525/" # "../model_files/inference_with_reason/checkpoint_bert/"

flags.DEFINE_string("bert_config_file", BERT_BASE_DIR+"bert_config_small.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "sentence_pair", "The name of the task to train.")

flags.DEFINE_string("vocab_file", BERT_BASE_DIR+"vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("init_checkpoint", BERT_BASE_DIR, # model.ckpt-66870--> /model.ckpt-66870
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer("max_seq_length", 105,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

# flags.DEFINE_string("c","gunicorn.conf","gunicorn.conf") # data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5--->data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin--->sgns.merge.char

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

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class SentencePairClassificationProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"] # ["-1","0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      label = tokenization.convert_to_unicode(line[0])
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,tokenizer):
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

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature

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

def create_int_feature(values):
  f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return f

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  print("create_model.is_training:", is_training)
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities,model)


tf.logging.set_verbosity(tf.logging.ERROR) # INFO
processors = {
  "sentence_pair":SentencePairClassificationProcessor,
}
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
task_name = FLAGS.task_name.lower()
print("task_name:",task_name)
processor = processors[task_name]()
label_list = processor.get_labels()
#lines_dev=processor.get_dev_examples("./TEXT_DIR")
index2label={i:label_list[i] for i in range(len(label_list))}
print("###index2label:",index2label)
tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


def main(_):
    pass

# init mode and session
# move something codes outside of function, so that this code will run only once during online prediction when predict_online is invoked.
is_training=False
use_one_hot_embeddings=False
batch_size=1
num_labels=len(label_list)
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
#sess=tf.Session(config=gpu_config)
model=None
global graph
input_ids_p,input_mask_p,label_ids_p,segment_ids_p=None,None,None,None
if not os.path.exists(FLAGS.init_checkpoint + "checkpoint"):
    raise Exception("failed to get checkpoint. going to return. init_checkpoint:",FLAGS.init_checkpoint)

graph = tf.Graph()#get_default_graph()
sess=tf.Session(config=gpu_config,graph=graph)
with graph.as_default():
    print("BERT.going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_mask")
    label_ids_p = tf.placeholder(tf.int32, [batch_size], name="label_ids")
    segment_ids_p = tf.placeholder(tf.int32, [FLAGS.max_seq_length], name="segment_ids")
    total_loss, per_example_loss, logits, probabilities,model = create_model(bert_config, is_training, input_ids_p, input_mask_p, segment_ids_p,label_ids_p, num_labels, use_one_hot_embeddings)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.init_checkpoint))

def predict_online_batch(question,answer_list):
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    logging.info("bert.predict_online_batch.question_answering.question:"+question+";answer_list:"+answer_list)
    ################
    label = '1' #tokenization.convert_to_unicode(line[0]) # this should compatible with format you defined in processor.
    text_a = tokenization.convert_to_unicode(question)
    text_b = tokenization.convert_to_unicode(answer)
    example= InputExample(guid=0, text_a=text_a, text_b=text_b, label=label)
    feature = convert_single_example(0, example, label_list,FLAGS.max_seq_length, tokenizer)
    #################
    input_ids = np.reshape([feature.input_ids],(1,FLAGS.max_seq_length))
    input_mask = np.reshape([feature.input_mask],(1,FLAGS.max_seq_length))
    segment_ids =  np.reshape([feature.segment_ids],(FLAGS.max_seq_length))
    label_ids =[feature.label_id]

    global graph
    with graph.as_default():
        feed_dict = {input_ids_p: input_ids, input_mask_p: input_mask,segment_ids_p:segment_ids,label_ids_p:label_ids}

        possibility = sess.run([probabilities], feed_dict)
        possibility=possibility[0][0] # get first label
        possibility_original=copy.deepcopy(possibility)
        label_index=np.argmax(possibility)
        label_predict=index2label[label_index]
        label_predict_oringal=copy.deepcopy(label_predict)

    #print("###bert.predict_online.label:",str(label_predict))
    #print("label_predict:",label_predict)
    #print("possibility:",possibility)
    #logging.info("bert.question_answering.predict_online.label:"+str(label_predict)+";possibility:"+ str(possibility))

    return label_predict,possibility ,possibility_original,label_predict_oringal

def predict_online(question,answer):
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    logging.info("bert.predict_online.question_answering.question:"+question+";answer:"+answer)
    label = '1' #tokenization.convert_to_unicode(line[0]) # this should compatible with format you defined in processor.
    text_a = tokenization.convert_to_unicode(question)
    text_b = tokenization.convert_to_unicode(answer)
    example= InputExample(guid=0, text_a=text_a, text_b=text_b, label=label)
    feature = convert_single_example(0, example, label_list,FLAGS.max_seq_length, tokenizer)
    input_ids = np.reshape([feature.input_ids],(1,FLAGS.max_seq_length))
    input_mask = np.reshape([feature.input_mask],(1,FLAGS.max_seq_length))
    segment_ids =  np.reshape([feature.segment_ids],(FLAGS.max_seq_length))
    label_ids =[feature.label_id]

    global graph
    with graph.as_default():
        feed_dict = {input_ids_p: input_ids, input_mask_p: input_mask,segment_ids_p:segment_ids,label_ids_p:label_ids}
        possibility = sess.run([probabilities], feed_dict)
        possibility=possibility[0][0] # get first label
        print("###predict_online.possibility:",possibility)
        possibility_original=copy.deepcopy(possibility)
        label_index=np.argmax(possibility)
        label_predict=index2label[label_index]
        label_predict_oringal=copy.deepcopy(label_predict)

        #label_predict = '0' if (str(label_predict) == '0' or str(label_predict) == '-1') else '1' # TODO ADD
        #possibility=[possibility[0]+possibility[1],possibility[2]]
    print("###bert.predict_online.label:",str(label_predict),";possibility:",possibility)
    print("label_predict:",label_predict)
    print("possibility:",possibility)
    logging.info("bert.question_answering.predict_online.label:"+str(label_predict)+";possibility:"+ str(possibility))

    return label_predict,possibility ,possibility_original,label_predict_oringal

def evaluate(dev_file):
    print("evaluate.started...")
    acc=0.0
    # 1. read source file
    dev_object=open(dev_file,'r')
    dev_lines=dev_object.readlines()
    random.shuffle(dev_lines)
    dev_lines=dev_lines[0:1000]
    # 2. predict for each sample
    count_right=0
    dict_acc_labels={-1:[0,0],0:[0,0],1:[0,0]} # [right,total]
    print("dict_acc_labels0:",dict_acc_labels)
    num_example=len(dev_lines)
    for i, line in enumerate(dev_lines):
        label,content,type_information=line.strip().split("\t")
        label_predict, possibility,possibility_original,label_predict_original=predict_online(content, type_information) # label_predict:1; possibility:[0.3,0.2,0.5]
        print(i,"label_predict_original:",label_predict_original,";possibility_original:",possibility_original)
        label_int=int(label)
        count_list = dict_acc_labels[label_int]
        if int(label_predict_original)==label_int:
            dict_acc_labels[label_int]=[count_list[0]+1,count_list[1]+1]
        else:
            dict_acc_labels[label_int]=[count_list[0]+0,count_list[1]+1]

        print(i,"label:",label,";type_information:",type_information,";content:",content)
        print(i,"label_predict:",label_predict,";possibility:",possibility)

        label= '0' if (str(label)=='0' or str(label)=='-1') else '1'
        label_predict = '0' if (str(label_predict) == '0' or str(label_predict) == '-1') else '1'
        if label==label_predict:
            count_right+=1

    print("dict_acc_labels1:",dict_acc_labels)
    dict_acc_labels_acc={}
    for l, count_list in dict_acc_labels.items():
        dict_acc_labels_acc[l]=float(count_list[0])/float(count_list[1])
    print("dict_acc_labels_acc2:",dict_acc_labels_acc)

    acc=float(count_right)/float(num_example)
    print("evaluate.ended...")

    return acc

def generate_report_badcase(dev_file):
    # 1. read source file
    dev_object = open(dev_file, 'r')
    dev_lines = dev_object.readlines()
    random.shuffle(dev_lines)
    dev_lines = dev_lines[0:1000]



#if __name__ == "__main__":
if __name__ == "__main__":
    #tf.app.run()
    #content='原告蔡成兵诉称：2015年8月26日起，原告开始在被告的东山领地项目部工作，工作岗位为砖工。直至2015年9月26日，被告并未依法与原告签订劳动合同，2015年10月3日下午5:00，原告在东山领地项目部8号楼工作时摔伤。现原告因工伤认定一事，需确认双方存在劳动关系。为维护原告自身合法权益，特诉请法院判令：1.确认原、被告之间存在事实劳动关系；2.本案诉讼费由被告承担。原告蔡成兵又称：原告系包工头刘强叫来上班，由其安排工作、发放工资。工地现场由刘强的管理人员进行管理。原告的工牌由刘强发放。据原告了解，刘强系从某劳务公司处承包工程，而被告系案涉工程项目的总承包人。'
    #type_information='劳动纠纷。 存在劳动关系。请求确认与单位存在劳动关系'
    question='男女双方自愿登记结果，婚后育有一女，现十岁，因性格不合离婚。婚姻期间，有房屋一套，为婚后购买，房产证只登记男方名字，由男方按揭贷款所购，按揭未还完，请问离婚此房屋产权如何归属，子女抚养权如何归属？'
    answer='你好，具体要看具体的事实和证据情况，如果协商不成，可以到法院提起离婚诉讼。离婚案件，涉及离与不离问题、子女抚养问题、财产分割问题、过错赔偿问题等。关于子女抚养问题，总的原则是有利于子女身心健康成长，保障子女的合法权益。需要结合父母双方的抚养能力和抚养条件等'
    result1=predict_online(question,answer)
    print("######result1:",result1)

    question='男女双方自愿登记结果，婚后育有一女，现十岁，因性格不合离婚。婚姻期间，有房屋一套，为婚后购买，房产证只登记男方名字，由男方按揭贷款所购，按揭未还完，请问离婚此房屋产权如何归属，子女抚养权如何归属？'
    answer='离婚房屋产权归双方共同所有，子女抚养权，是有利于子女身心健康成长，保障子女的合法权益的原则。'
    result2=predict_online(question,answer)
    print("######result2:",result2)
    #pass

    #dev_file='/home/shizai/xul/bert/TEXT_DIR/suqiu_2qi_small/dev.tsv' #'/Users/xuliang/PycharmProjects/ai-bxh-api/inference_with_reason/dev.tsv'
    #eval_acc=evaluate(dev_file)
    #print("eval_acc:",eval_acc)

    #dev_file = '/Users/xuliang/PycharmProjects/ai-bxh-api/inference_with_reason/dev.tsv'
    #eval_acc = evaluate(dev_file)
    #print("eval_acc:", eval_acc)



