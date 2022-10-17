# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from ProfessionalSearch.src.similar_case_retrival.similar_case.rank_util import (
    cosine_similiarity,
)
from ProfessionalSearch.src.similar_case_retrival.similar_case.util import (
    get_index_list,
)
from Tools.parse_argument import parse_config_file
from Utils import print_run_time


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = parse_config_file(config)
        data_training_arguments = self.config["DataTrainingArguments"]
        dataset = data_training_arguments["dataset"]
        embedding_path = data_training_arguments["embedding_path"]
        self.model_name = "TextCNN"
        self.train_path = dataset + "/train_dev/train/train.txt"  # 训练集
        self.dev_path = dataset + "/train_dev/dev/dev.txt"  # 验证集
        self.vocab_path = dataset + "/train_dev/vocab.pkl"  # 词表
        self.save_path = dataset + "/saved_dict/" + self.model_name + ".ckpt"  # 模型训练结果
        self.log_path = dataset + "/log/" + self.model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = 300
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256
        self.wiki_vec = torch.load(embedding_path)
        self.embedding = nn.Embedding.from_pretrained(self.wiki_vec, freeze=False)
        self.num_class = 53  # 类别数
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes]
        )
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_class)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(0)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        self.middle_feature = out.detach().cpu().numpy()
        print("self.middle_feature:", self.middle_feature, self.middle_feature.shape)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    @print_run_time
    def get_mid_feature(self, x):
        text_index = get_index_list(x)
        x = torch.tensor(text_index)
        out = self.embedding(x)
        out = out.unsqueeze(0)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        self.middle_feature = out.detach().cpu().numpy()
        return self.middle_feature


if __name__ == "__main__":
    # TrainClassification(
    #     config="ProfessionalSearch/config/similar_case_retrival/text_cnn_cls.yaml"
    # ).run()
    index_test = "你好啊，朋友们"
    index_test_sim = "你好啊，朋友们"
    index_test_not = "这里的景色不错"
    textCNN = Model(
        config="ProfessionalSearch/config/similar_case_retrival/text_cnn_cls.yaml"
    )
    index_test_mid = textCNN.get_mid_feature(index_test)
    # print("index_test_mid", index_test_mid, index_test_mid.shape)
    index_sim_mid = textCNN.get_mid_feature(index_test_sim)
    # print("index_test_mid", index_test_mid, index_test_mid.shape)
    index_not_mid = textCNN.get_mid_feature(index_test_not)
    # print("index_test_mid", index_test_mid, index_test_mid.shape)

    cos_i_sim = cosine_similiarity(index_test_mid[0], index_sim_mid[0])
    cos_i_not = cosine_similiarity(index_test_mid[0], index_not_mid[0])
    print("sim", cos_i_sim)
    print("not", cos_i_not)
