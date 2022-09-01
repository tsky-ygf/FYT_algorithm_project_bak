#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 13:40
# @Author  : Adolf
# @Site    : 
# @File    : feature_extraction.py
# @Software: PyCharm
from pprint import pprint
from DocumentReview.UIETool.deploy.uie_predictor import UIEPredictor

schema_config = {"theft": {'盗窃触发词': ['总金额', '物品', '地点', '时间', '人物', '行为']},
                 "provide_drug": {
                     "容留他人吸毒触发词": ["毒品名称", "容留次数", "毒品种类", "被容留人", "时间", "地点", "行为","人物"]},
                 }


class InferArgs:
    model_path_prefix = ""
    position_prob = 0.5
    max_seq_len = 512
    batch_size = 1
    device = "cpu"
    schema = []


def init_extract(criminal_type=""):
    args = InferArgs()
    args.model_path_prefix = "model/uie_model/export_cpu/{}/inference".format(criminal_type)
    args.schema = schema_config[criminal_type]
    predictor = UIEPredictor(args)
    return predictor


def post_multi_thing(post_result, name):
    _thing = post_result[name]
    _thing = [one_thing["text"] for one_thing in _thing]
    _thing_str = "、".join(_thing)
    post_result[name] = _thing_str


# Post-processing for UIE results
def post_process_uie_results(predictor, criminal_type, fact):
    extract_result = predictor.predict([fact])[0]
    post_result = dict()
    if criminal_type == "theft":
        for key, values in extract_result.items():
            for value in values:
                post_result["事件"] = value["text"]
                relations = value["relations"]
                # self.logger.debug(relations)
                post_result["物品"] = relations.get("物品", [{"text": "一些"}])

                post_multi_thing(post_result, "物品")
                # _thing = post_result["物品"]
                # _thing = [one_thing["text"] for one_thing in _thing]
                # _thing_str = "、".join(_thing)
                # post_result["物品"] = _thing_str

                post_result["时间"] = relations.get("时间", [{"text": ""}])[0]["text"]
                post_result["地点"] = relations.get("地点", [{"text": ""}])[0]["text"]
                post_result["人物"] = relations.get("人物", [{"text": "嫌疑人"}])[0]["text"]
                post_result["总金额"] = relations.get("总金额", [{"text": ""}])[0]["text"]
                post_result["行为"] = relations.get("行为", [{"text": "盗窃"}])[0]["text"]
    elif criminal_type == "provide_drug":
        for key, values in extract_result.items():
            for value in values:
                post_result["事件"] = value["text"]
                relations = value["relations"]
                # self.logger.debug(relations)
                post_result["毒品名称"] = relations.get("毒品名称", [{"text": "毒品"}])
                post_result["毒品种类"] = relations.get("毒品种类", [{"text": "毒品"}])
                post_result["被容留人"] = relations.get("被容留人", [{"text": "其他人"}])

                post_multi_thing(post_result, "毒品名称")
                post_multi_thing(post_result, "毒品种类")
                post_multi_thing(post_result, "被容留人")

                post_result["时间"] = relations.get("时间", [{"text": ""}])[0]["text"]
                post_result["地点"] = relations.get("地点", [{"text": ""}])[0]["text"]
                post_result["行为"] = relations.get("行为", [{"text": "提供毒品"}])[0]["text"]
                post_result["人物"] = relations.get("人物", [{"text": "嫌疑人"}])[0]["text"]
                post_result["容留次数"] = relations.get("容留次数", [{"text": ""}])[0]["text"]

    return post_result


# text = "浙江省诸暨市人民检察院指控，2019年7月22日10时30分许，被告人唐志强窜至诸暨市妇幼保健医院，在3楼21号病床床头柜内窃得被害人俞" \
#        "某的皮包一只，内有现金￥1500元和银行卡、身份证等财物。"

# text = "湖南省涟源市人民检察院指控，2014年8月至2015年1月，被告人刘某甲先后多次容留刘2某、刘某乙、刘1某、刘某丙、袁某等人在其位于本市" \
#        "安平镇田心村二组的家中吸食甲基苯丙胺（冰毒）和甲基苯丙胺片剂（麻古）。具体事实如下：1、2014年8月份的一天，被告人" \
#        "刘某甲容留刘某丙、刘1某等人在其家中卧室吸食甲基苯丙胺和甲基苯丙胺片剂。"
#
# predictor_ = init_extract(criminal_type="provide_drug")
# pprint(post_process_uie_results(predictor=predictor_, criminal_type="provide_drug", fact=text))
