#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 15:01
# @Author  : Adolf
# @Site    : 
# @File    : criminal_prejudgment.py
# @Software: PyCharm
from loguru import logger
from pprint import pprint, pformat
from extraction.feature_extraction import init_extract


class CriminalPrejudgment:
    def __init__(self, criminal_type=""):
        self.ie = init_extract(criminal_type=criminal_type)

    def get_base_information(self, content):
        res = self.ie(content)
        return res[0]

    def __call__(self, content=""):
        logger.info("starting")
        res = self.get_base_information(content)
        logger.debug(pformat(res))
        for key, value in res.items():
            logger.info(key)
            logger.info(value)
        pass


if __name__ == '__main__':
    criminal_pre_judgment = CriminalPrejudgment(criminal_type="theft")

    text = "浙江省诸暨市人民检察院指控，2019年7月22日10时30分许，被告人唐志强窜至诸暨市妇幼保健医院，在3楼21号病床床头柜内窃得被害人俞" \
           "某的皮包一只，内有现金￥1500元和银行卡、身份证等财物。"

    criminal_pre_judgment(content=text)
