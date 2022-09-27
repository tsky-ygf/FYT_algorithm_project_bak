import json, re, time, os
import configparser
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
# from qa_util import load_dict, sequence_padding
import keras.backend.tensorflow_backend as KTF

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3   #进行配置，使用30%的GPU
sess = tf.Session(config=config)
KTF.set_session(sess)

model_file_path = "../model_files/question_answering/text_cnn_model_files/"


def convs_block(data, convs=[3, 4, 5], f=256, name="conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(
            BatchNormalization()(Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)


# 简单的 text_cnn
def get_textcnn(config):
    content = Input(shape=(config["max_seq_len"],), dtype="int32")
    embedding = Embedding(
        name="embedding",
        input_dim=config["dict_len"],
        output_dim=config["embedding_size"])
    trans_content = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(config["hidden_size"]))(embedding(content)))))
    feat = convs_block(trans_content)
    dropfeat = Dropout(config["dropout"])(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(config["hidden_size"])(dropfeat)))
    output = Dense(config["label_size"], activation="softmax")(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # model.summary()
    return model


def load_dict(char2id_path=model_file_path + "char2id_filter_15.json", label2id_path=model_file_path + "label2id.json"):
    """
        加载char、label字典
    :return:
    """
    # Print args
    print("-" * 50 + "\t准备数据\t" + "-" * 50)
    char2id = json.loads(open(char2id_path).read())
    # 新增 <unk>
    char_value_dict_len = len(char2id) + 1

    label2id = json.loads(open(label2id_path).read())

    print("字集合大小:%d", char_value_dict_len)
    print("标签个数:%d", len(label2id))
    return char2id, label2id


def sequence_padding(chars, padding="right", max_len=512):
    """
        对句子进行padding
    :return:
    """
    # list的extend方法没有返回值，是none，结果在原列表中
    l = len(chars)
    if padding == "left":
        if l <= max_len:
            _chars = [0] * (max_len - l) + chars
            # _labels = [0] * (max_len - l) + labels
            # _masks = [0] * (max_len - l) + [1] * l
        else:
            _chars = chars[l - max_len:]
            # _labels = labels[l - max_len:]
            # _masks = [1] * max_len
    elif padding == "right":
        if l <= max_len:
            _chars = chars + [0] * (max_len - l)
            # _labels = labels + [0] * (max_len - l)
            # _masks = [1] * l + [0] * (max_len - l)
        else:
            _chars = chars[:max_len]
            # _labels = labels[:max_len]
            # _masks = [1] * max_len
    else:
        raise Exception
    return _chars  # , _labels  # , _masks


class ModelQuestionLabel:
    """
        # model.get_layer('Dense_1').output
        包含自定义层时，使用 load_weights , load_model会报layer找不到的异常
    """

    def __init__(self, path=model_file_path + "model.conf"):
        conf = configparser.ConfigParser()
        conf.read(path)
        model_use = conf.get("model", "use")
        config = conf[model_use]
        self.char2id, self.label2id = load_dict()
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.char_len = len(self.char2id)
        self.config = {}
        for k, v in config.items():
            try:
                if float(v) < 1:
                    self.config[k] = float(v)
                else:
                    self.config[k] = int(v)
            except:
                self.config[k] = v
        # print(self.config)
        if model_use == "text_cnn":
            self.model = get_textcnn(self.config)
            self.new_model = Model(inputs=self.model.input, outputs=self.model.get_layer('conv_feat').output)
        elif model_use == "lstm_attention":
            self.model = get_char_lstm(self.config)
            self.new_model = Model(inputs=self.model.input, outputs=self.model.get_layer('attention_1').output)

    def text_process(self, text):
        """
            生成 x
        :param text:
        :return:
        """
        x = []
        for c in text.encode("utf-8").decode("utf-8-sig").strip():
            if self.char2id.get(c):
                x.append(self.char2id[c])
            else:
                x.append(self.char_len)
        return sequence_padding(x, max_len=self.config["max_seq_len"])

    def train(self, data_path="./train_data/XY_train_filter_15.npz"):
        """
            训练模型
        :param data_path:
        :param model_name:
        :return:
        """
        XY = np.load(data_path)
        X = XY["X"]
        Y = XY["Y"]
        if not os.path.exists("./bin/"):
            os.mkdir("./bin/")
        # 加载以前的模型
        if os.path.exists('./bin/' + self.config["model_name"]):
            # 该模型是不包含标点符号的模型，需提前分割好[:1000]
            print('加载以前训练的模型')
            self.model.load_weights('./bin/' + self.config["model_name"])
            # self.model = load_model('./bin/' + model_name)
        checkpointer = ModelCheckpoint(filepath='./bin/' + self.config["model_name"], monitor="val_loss", verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False)

        self.model.fit(X, Y, batch_size=256, epochs=3, verbose=1, validation_split=0.1, shuffle=True,
                       callbacks=[checkpointer])

    def evaluate(self, data_path="./train_data/XY_dev_filter_15.npz"):
        """
            评估模型
        :param data_path:
        :return:
        """
        XY = np.load(data_path)
        X = XY["X"]
        Y = XY["Y"]
        result = self.model.evaluate(X, Y, batch_size=128)
        print(result)
        return result

    def load_train_model(self):
        """
            加载模型
        :param model_name:
        :return:
        """
        self.model.load_weights(model_file_path + self.config["model_name"])

    def get_feature(self, text):
        """
            获取模型 中间层 特征
        :param text:
        :return:
        """
        feature = self.new_model.predict(np.array([self.text_process(text)]))[0]
        return feature

    def get_label(self, text):
        """
            获取 纠纷类型 标签
        :param text:
        :return:
        """
        feature = self.model.predict(np.array([self.text_process(text)]))[0]
        index = np.argmax(feature)
        # print(feature[index])
        return self.id2label[index], feature[index]


def get_label(text):
    """
    :param text:
    :return:
    """
    return MQL.get_label(text)


def get_feature(text):
    """
    :param text:
    :return:
    """
    return MQL.get_feature(text)


MQL = ModelQuestionLabel()
MQL.load_train_model()

# 同来识别 纠纷类型 、 提取特征
print(get_label("我想离婚怎么办呢"))
get_feature("我想离婚怎么办呢")


class MT:
    def __init__(self, model_path="./model_files/"):
        self.char2id, label2id = load_dict(char2id_path=model_path + "char2id.json",
                                           label2id_path=model_path + "label2id.json")
        self.id2label = {value: key for key, value in label2id.items()}

        with tf.Graph().as_default() as graph:
            with tf.gfile.FastGFile(model_path + 'text_cnn.pb', "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name="")
        sess = tf.Session(graph=graph)

        self.sess = sess
        self.x = self.sess.graph.get_tensor_by_name("placeholder/x:0")
        self.keep_prob = sess.graph.get_tensor_by_name("placeholder/keep_prob:0")  # is_training
        self.feature = self.sess.graph.get_tensor_by_name("fc/dense/Relu:0")  # fc/dense/Relu  cnn_block/Reshape
        self.p = self.sess.graph.get_tensor_by_name("output/logits:0")

    def predict(self, text):
        """
        预测 主题
        :param text:
        :return:
        """
        x = sequence_padding([self.char2id.get(c, 1) for c in text], max_len=128)
        feed_dict = {
            self.x: [x],
            self.keep_prob: 0.5,
        }
        predicts = self.sess.run([self.p], feed_dict)[0][0]
        max_index = np.argmax(predicts)
        return predicts, predicts[max_index], self.id2label[max_index]

    def get_feature(self, text):
        """
        提取 特征
        :param text:
        :return:
        """
        x = sequence_padding([self.char2id.get(c, 1) for c in text], max_len=128)
        feed_dict = {
            self.x: [x],
            self.keep_prob: 1.0,
        }
        feature = self.sess.run([self.feature], feed_dict)[0][0]
        return feature


mt = MT(model_path="../model_files/question_answering/model_ask_type/")
text = "工资，不发，老板会坐牢吗"


def get_label_new(text):
    """
    :param text:
    :return:
    """
    return mt.predict(text)


def get_feature_new(text):
    """
    :param text:
    :return:
    """
    return mt.get_feature(text)


get_label_new(text)
get_feature_new(text)