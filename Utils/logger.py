#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:19
# @Author  : Adolf
# @Site    : 
# @File    : logger.py
# @Software: PyCharm
import logging
import time


# 打印日志
def get_module_logger(module_name, level="INFO"):
    module_name = "tst.{}".format(module_name)
    module_logger = logging.getLogger(module_name)

    if level.upper() == "INFO":
        module_logger.setLevel(logging.INFO)
    elif level.upper() == "DEBUG":
        module_logger.setLevel(logging.DEBUG)
    elif level.upper() == "WARNING":
        module_logger.setLevel(logging.WARNING)
    elif level.upper() == "ERROR":
        module_logger.setLevel(logging.ERROR)

    module_logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s %(lineno)s :\n %(message)s \n"
        "----------------------------------------------------------------------"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    module_logger.handlers.append(console_handler)

    return module_logger


# 计算函数执行时间
def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print('current Function [%s] run time is %.2f' % (func.__name__, time.time() - local_time))

    return wrapper
