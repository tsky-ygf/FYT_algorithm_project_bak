#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 13:40
# @Author  : Adolf
# @Site    : 
# @File    : feature_extraction.py
# @Software: PyCharm
from DocumentReview.UIETool.deploy.uie_predictor import UIEPredictor

schema_config = {"theft": {'盗窃触发词': ['总金额', '物品', '地点', '时间', '人物', '行为']}}


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


init_extract(criminal_type="theft")
