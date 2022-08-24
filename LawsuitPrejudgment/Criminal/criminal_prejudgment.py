#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 15:01
# @Author  : Adolf
# @Site    : 
# @File    : criminal_prejudgment.py
# @Software: PyCharm
from loguru import logger


class CriminalPrejudgment:
    def __init__(self, criminal_type=""):
        pass

    def __call__(self, *args, **kwargs):
        logger.info("starting")
        pass


if __name__ == '__main__':
    criminal_pre_judgment = CriminalPrejudgment()
    criminal_pre_judgment()
