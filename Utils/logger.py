#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:19
# @Author  : Adolf
# @Site    : 
# @File    : logger.py
# @Software: PyCharm
import logging
import time
import colorlog

log_colors_config = {
    'DEBUG': 'white',  # cyan white
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}


# 打印日志
def get_module_logger(module_name, level="INFO", console=True, logger_file=None):
    # global console_handler, file_handler
    logger = logging.getLogger(module_name)
    console_handler = None
    file_handler = None
    # 输出到控制台
    if console:
        console_handler = logging.StreamHandler()
    # 输出到文件
    if logger_file is not None:
        file_handler = logging.FileHandler(filename=logger_file, mode='a', encoding='utf8')

    if level.upper() == "INFO":
        logger.setLevel(logging.INFO)
        if console:
            console_handler.setLevel(logging.INFO)
        if logger_file is not None:
            file_handler.setLevel(logging.INFO)
    elif level.upper() == "DEBUG":
        logger.setLevel(logging.DEBUG)
        if console:
            console_handler.setLevel(logging.DEBUG)
        if logger_file is not None:
            file_handler.setLevel(logging.DEBUG)
    elif level.upper() == "WARNING":
        logger.setLevel(logging.WARNING)
        if console:
            console_handler.setLevel(logging.WARNING)
        if logger_file is not None:
            file_handler.setLevel(logging.WARNING)
    elif level.upper() == "ERROR":
        logger.setLevel(logging.ERROR)
        if console:
            console_handler.setLevel(logging.ERROR)
        if logger_file is not None:
            file_handler.setLevel(logging.ERROR)

    # 日志输出格式
    file_formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S'
    )
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config
    )
    if console:
        console_handler.setFormatter(console_formatter)
    if logger_file is not None:
        file_handler.setFormatter(file_formatter)

    if not logger.handlers:
        if console:
            logger.addHandler(console_handler)
        if logger_file is not None:
            logger.addHandler(file_handler)

    if console:
        console_handler.close()
    if logger_file is not None:
        file_handler.close()

    return logger


# 计算函数执行时间
def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print('current Function [%s] run time is %.2f s' % (func.__name__, time.time() - local_time))

    return wrapper
