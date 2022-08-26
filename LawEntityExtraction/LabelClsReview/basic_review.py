import re
import uuid

import numpy as np
import torch
import torch.nn.functional as F

import pandas as pd
from collections import OrderedDict
from Utils.parse_file import parse_config_file
from Utils import Logger
from DocumentReview.ParseFile.parse_word import read_docx_file

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BasicSituation:
    def __init__(self, config_path):
        self.config = parse_config_file(config_path)
        self.logger = Logger(name="Situation_{}".format(uuid.uuid1()), level=self.config["log_level"]).logger
        self.logger.info(self.logger.name)
        self.logger.info("log level:{}".format(self.config["log_level"]))

        self.data_list = []

        self.review_result = OrderedDict()

    def get_data_res(self, *args, **kwargs):
        raise NotImplementedError

    def review_main(self, content, mode):
        ress = []

        # self.data_list = self.read_origin_content(content, mode)
        # TODO 根据正则表达式判断是否为诉求内容
        # rqst_patn1 = re.compile(
        #     '(原告|被告)')
        # # rqst_patn2 = re.compile(
        # #     '提出(撤销|判令|责令|确认|执行|责成|判决|撤回).{,20}(请求|要求|申请|起诉)')
        # for content_item in self.data_list:
        #     if not len(content_item):
        #         continue
        #     m = rqst_patn1.search(content_item)
        #     # if not m:
        #     #     m = rqst_patn2.search(content_item)
        #     if m:
        #         st, _ = m.span()
        #         content_item = [content_item.char_span(st, len(content_item) - 1, '诉求')]
        #         ress.append(self.get_data_res(content_item))

        # TODO 取句子中，投票最高的类别 or  不分割为句子，直接预测文本
        res_pro = self.get_data_res(content)
        return res_pro

    @staticmethod
    def read_origin_content(content="", mode="text"):
        if mode == "text":
            content = content.replace(" ", "").replace("\u3000", "")
            text_list = content.split("。")
        elif mode == "docx":
            text_list = read_docx_file(docx_path=content)
        else:
            raise Exception("mode error")
        return text_list


class BasicBertSituation(BasicSituation):
    def __init__(self, device_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer"])
        self.param = torch.load(self.config["task_model"], map_location=lambda storage, loc: storage.cuda(0))
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config["model"], num_labels=self.config["num_labels"])
        self.model.to(device_id)
        self.model.load_state_dict(self.param)
        self.threshold = self.config["threshold"]
        self.logger.info(self.config["model"])

    def get_data_res(self, text):
        map_df = pd.read_csv(self.config["label_mapping_path"])
        pred_map = dict(zip(map_df['index'], map_df['label']))
        res_pro = {}
        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                max_length=self.config['max_len'],
                                padding="max_length",
                                truncation=True,
                                return_offsets_mapping=False,
                                return_tensors="pt")
        inputs = inputs.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits) > 0.5
        predictions = predictions.detach().cpu().numpy().astype(int)

        probability = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy()
        # predictions = np.argmax(outputs.logits.detach().cpu().numpy(), axis=-1)
        # predictions = predictions.astype(int)
        for index, res in enumerate(predictions[0]):
            if res == 1:
                res_pro[pred_map[index]] = float(probability[index])
        self.logger.info(res_pro)
        return res_pro

    def get_key(self, pred_map, value):
        return [k for k, v in pred_map.items() if v == value]
