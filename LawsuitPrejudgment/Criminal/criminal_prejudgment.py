#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 15:01
# @Author  : Adolf
# @Site    : 
# @File    : criminal_prejudgment.py
# @Software: PyCharm
from pprint import pprint, pformat
# from extraction.feature_extraction import init_extract
import requests
from LawsuitPrejudgment.Criminal.basic_prejudgment import PrejudgmentPipeline

from autogluon.text import TextPredictor
import os
from xmindparser import xmind_to_dict
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pd.set_option('display.max_columns', None)


class CriminalPrejudgment(PrejudgmentPipeline):
    def __init__(self, *args, **kwargs):
        super(CriminalPrejudgment, self).__init__(*args, **kwargs)

        # 加载案由预测模型
        self.predictor = TextPredictor.load(self.config.anyou_identify_model_path)
        # self.ie = init_extract(criminal_type=criminal_type)
        if self.config.situation_identify_model_path[:4] == "http":
            self.ie_url = self.config.situation_identify_model_path

        self.logger.info("加载预测模型成功")

    def anyou_identify(self):
        if "anyou" in self.content:
            pass
        sentence = self.content["fact"]
        predictions = self.predictor.predict({"fact": [sentence]})
        self.content["anyou"] = predictions.iloc[0]
        # self.logger.debug(self.content)

    def suqiu_identify(self):
        if "suqiu" in self.content:
            pass
        if self.config.prejudgment_type == "criminal":
            self.content["suqiu"] = "量刑推荐"

    def situation_identify(self):
        if "event" in self.content:
            pass
        if self.content['anyou'] == "盗窃":
            criminal_type = 'theft'
        else:
            raise Exception("暂时不支持该数据格式")

        r = requests.post(self.ie_url, json={"criminal_type": criminal_type, "fact": self.content["fact"]})
        extract_result = r.json()['result'][0]
        self.content["event"] = dict()
        for key, values in extract_result.items():
            for value in values:
                self.content["event"]["事件"] = value["text"]
                relations = value["relations"]
                # self.logger.debug(relations)
                self.content["event"]["物品"] = relations.get("物品", None)
                self.content["event"]["地点"] = relations.get("地点", None)
                self.content["event"]["地点"] = relations.get("地点", None)
                self.content["event"]["人物"] = relations.get("人物", None)
                self.content["event"]["总金额"] = relations.get("总金额", None)
                self.content["event"]["行为"] = relations.get("行为", None)
        # self.logger.debug(self.content)

    def parse_config_file(self):

    def get_question(self):
        self.logger.debug(self.content["question_answers"])
        question_config_df = pd.read_csv(
            "LawsuitPrejudgment/Criminal/question_config/{}罪特征表.csv".format(self.content["anyou"]))

        circumstances_list = self.content['circumstances_graph'].keys()
        if len(self.content["question_answers"]) == 0:
            for circumstance in circumstances_list:
                if self.content['circumstances_graph'][circumstance]['type'] == "大前提":
                    self.logger.debug(self.content['circumstances_graph'][circumstance]['特征'])
        self.logger.debug(question_config_df)

    def generate_report(self):
        evaluation_report = dict()
        evaluation_report["案件事实"] = "根据您的描述，【时间】，在【地点】，【人物】存在【行为】等行为，窃得【物品】、【物品】等财物，盗窃总金额为【总金额】。"
        evaluation_report["评估理由"] = "【盗窃罪XMind评估理由】【量刑情节XMind评估理由】"
        evaluation_report["法律建议"] = "【盗窃罪XMind法律建议】"
        evaluation_report["法律依据"] = "【盗窃罪XMind法律依据】【量刑情节XMind法律依据】"


if __name__ == '__main__':
    criminal_config = {"log_level": "debug",
                       "prejudgment_type": "criminal",
                       "xmind_path": "LawsuitPrejudgment/Criminal/base_config/theft.xmind",
                       "anyou_identify_model_path": "model/gluon_model/accusation",
                       "situation_identify_model_path": "http://172.19.82.199:7777/information_result",
                       }
    criminal_pre_judgment = CriminalPrejudgment(**criminal_config)

    text = "浙江省诸暨市人民检察院指控，2019年7月22日10时30分许，被告人唐志强窜至诸暨市妇幼保健医院，在3楼21号病床床头柜内窃得被害人俞" \
           "某的皮包一只，内有现金￥1500元和银行卡、身份证等财物。"

    question_answers = {}

    # predict
    criminal_pre_judgment(fact=text, question_answers=question_answers)
