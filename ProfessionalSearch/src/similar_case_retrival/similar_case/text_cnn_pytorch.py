# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Tools.parse_argument import parse_config_file
from Tools.train_tool import BaseTrainTool


class TrainClassification(BaseTrainTool):
    def init_model(self):
        model = Model(self.config)
        self.logger.debug(model)
        self.criterion = torch.nn.CrossEntropyLoss
        return model

    def create_examples(self):
        pass

    def cal_loss(self, batch):
        self.logger.debug(batch)
        labels = batch["labels"]
        # input_data = {'input_ids': batch['input_ids'],
        #               'atention_mask': batch['attention_mask'],
        #               'token_type_ids': batch['token_type_ids']}
        pred = self.model(batch["input_ids"].squeeze())
        loss = self.criterion(torch.sigmoid(pred.logits), labels.squeeze())
        self.logger.debug(loss)
        return loss


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = parse_config_file(config)
        data_training_arguments = config["DataTrainingArguments"]
        dataset = data_training_arguments.dataset
        embedding_path = data_training_arguments.embedding_path
        self.model_name = "TextCNN"
        self.train_path = dataset + "/train_dev/train/train.txt"  # 训练集
        self.dev_path = dataset + "/train_dev/dev/dev.txt"  # 验证集
        self.class_list = [
            x.strip()
            for x in open(dataset + "/train_dev/class.txt", encoding="utf-8").readlines()
        ]  # 类别名单
        self.vocab_path = dataset + "/train_dev/vocab.pkl"  # 词表
        self.save_path = dataset + "/saved_dict/" + self.model_name + ".ckpt"  # 模型训练结果
        self.log_path = dataset + "/log/" + self.model_name
        self.embedding_pretrained = (
            torch.tensor(
                np.load(embedding_path)["embeddings"].astype("float32")
            )
            if embedding_path != "random"
            else None
        )  # 预训练词向量
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = (
            self.embedding_pretrained.size(1)
            if self.embedding_pretrained is not None
            else 300
        )  # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                data_training_arguments.embedding_path, freeze=False
            )
        else:
            self.embedding = nn.Embedding(
                config.n_vocab, config.embed, padding_idx=config.n_vocab - 1
            )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, config.num_filters, (k, config.embed))
                for k in config.filter_sizes
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(
            config.num_filters * len(config.filter_sizes), config.num_classes
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        self.middle_feature = out.detach().cpu().numpy()
        out = self.fc(out)
        return out


if __name__ == "__main__":
    TrainClassification(
        config="ProfessionalSearch/config/similar_case_retrival/text_cnn_cls.yaml"
    ).run()
