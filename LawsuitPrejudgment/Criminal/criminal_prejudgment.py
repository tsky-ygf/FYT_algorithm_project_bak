#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 15:01
# @Author  : Adolf
# @Site    : 
# @File    : criminal_prejudgment.py
# @Software: PyCharm
# from pprint import pprint, pformat
# from extraction.feature_extraction import init_extract
from LawsuitPrejudgment.Criminal.basic_prejudgment import PrejudgmentPipeline, PrejudgmentConfig

from autogluon.text import TextPredictor
import os

# import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class CriminalPrejudgment(PrejudgmentPipeline):
    def __init__(self, *args, **kwargs):
        super(CriminalPrejudgment, self).__init__(*args, **kwargs)

        # 加载案由预测模型
        self.predictor = TextPredictor.load(self.config.anyou_identify_model_path)
        # self.ie = init_extract(criminal_type=criminal_type)

    def anyou_identify(self):
        sentence = self.content["fact"]
        predictions = self.predictor.predict({"fact": [sentence]})
        self.content["anyou"] = predictions.iloc[0]
        self.logger.debug(self.content)

    def suqiu_identify(self):
        if self.config.prejudgment_type == "criminal":
            self.content["suqiu"] = "量刑推荐"

    def situation_identify(self):
        raise NotImplemented

    def parse_xmind(self):
        raise NotImplemented

    def generate_report(self):
        raise NotImplemented

    def get_base_information(self):
        # res = self.ie(content)
        # return res[0]
        pass
    # def handle_information(self, extract_result):
    #     for key, values in extract_result.items():
    #         logger.info(key)
    #         logger.info(values)
    #         for value in values:
    #             trigger = value["text"]
    #             relations = value["relations"]
    #             logger.info(relations)
    #             _thing = relations["物品"]
    #             _place = relations["地点"]
    #             _time = relations["时间"]
    #             _figure = relations["人物"]
    #             _total = relations["总金额"]


if __name__ == '__main__':
    criminal_config = {"log_level": "debug",
                       "prejudgment_type": "criminal",
                       "anyou_identify_model_path": "model/gluon_model/accusation"}
    criminal_pre_judgment = CriminalPrejudgment(**criminal_config)

    text = "浙江省诸暨市人民检察院指控，2019年7月22日10时30分许，被告人唐志强窜至诸暨市妇幼保健医院，在3楼21号病床床头柜内窃得被害人俞" \
           "某的皮包一只，内有现金￥1500元和银行卡、身份证等财物。"

    criminal_pre_judgment(text=text)
