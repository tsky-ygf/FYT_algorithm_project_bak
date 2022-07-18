#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:19
# @Author  : Adolf
# @Site    : 
# @File    : logger.py
# @Software: PyCharm
import contextlib
import functools
import logging
import time
import threading
# from typing import List

import uuid
import colorlog

# from colorama import Fore

# loggers = {}

log_config = {
    'DEBUG': {
        'level': 10,
        'color': 'purple'
    },
    'INFO': {
        'level': 20,
        'color': 'green'
    },
    'TRAIN': {
        'level': 21,
        'color': 'cyan'
    },
    'EVAL': {
        'level': 22,
        'color': 'blue'
    },
    'WARNING': {
        'level': 30,
        'color': 'yellow'
    },
    'ERROR': {
        'level': 40,
        'color': 'red'
    },
    'CRITICAL': {
        'level': 50,
        'color': 'bold_red'
    }
}


class Logger(object):
    """
    Default logger in FYT_Project
    Args:
        name(str) : Logger name, default is 'FYTProject'
    """

    def __init__(self, name: str = None, level: str = 'INFO'):
        name = 'FYTProject-{}'.format(uuid.uuid1()) if not name else name
        self.logger = logging.getLogger(name)
        # self.logger.propagate = False

        # logging.Logger.manager.loggerDict.pop(__name__)
        # 将当前文件的handlers 清空
        # self.logger.handlers = []
        # 然后再次移除当前文件logging配置
        # self.logger.removeHandler(self.logger.handlers)
        # self.logger.handlers.clear()

        for key, conf in log_config.items():
            logging.addLevelName(conf['level'], key)
            self.__dict__[key] = functools.partial(self.__call__, conf['level'])
            self.__dict__[key.lower()] = functools.partial(self.__call__,
                                                           conf['level'])

        self.format = colorlog.ColoredFormatter(
            '\n%(log_color)s[%(asctime)-12s] [%(levelname)4s][%(filename)s -> %(funcName)s line:%(lineno)d]%(reset)s \n'
            '%(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S',
            log_colors={
                key: conf['color']
                for key, conf in log_config.items()
            })

        if not self.logger.handlers:
            self.handler = logging.StreamHandler()
            self.handler.setFormatter(self.format)
            self.logger.addHandler(self.handler)

        self.logLevel = level
        if level.upper() == "INFO":
            self.logger.setLevel(logging.INFO)
        elif level.upper() == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif level.upper() == "WARNING":
            self.logger.setLevel(logging.WARNING)
        elif level.upper() == "ERROR":
            self.logger.setLevel(logging.ERROR)

        self.logger.propagate = False
        self._is_enable = True

    def disable(self):
        self._is_enable = False

    def enable(self):
        self._is_enable = True

    @property
    def is_enable(self) -> bool:
        return self._is_enable

    def __call__(self, log_level: str, msg: str):
        if not self.is_enable:
            return

        self.logger.log(log_level, msg)

    @contextlib.contextmanager
    def use_terminator(self, terminator: str):
        old_terminator = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old_terminator

    @contextlib.contextmanager
    def processing(self, msg: str, interval: float = 0.1):
        """
        Continuously print a progress bar with rotating special effects.
        Args:
            msg(str): Message to be printed.
            interval(float): Rotation interval. Default to 0.1.
        """
        end = False

        def _printer():
            index = 0
            flags = ['\\', '|', '/', '-']
            while not end:
                flag = flags[index % len(flags)]
                with self.use_terminator('\r'):
                    self.info('{}: {}'.format(msg, flag))
                time.sleep(interval)
                index += 1

        t = threading.Thread(target=_printer)
        t.start()
        yield
        end = True


# logger = Logger()

log_colors_config = {
    'DEBUG': 'white',  # cyan white
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}


# 打印日志
def get_module_logger(module_name, level="INFO", console=True, logger_file=None):
    """
    :param module_name: module name
    :param level: logger level
    :param console: use console or not
    :param logger_file: write log to file or not
    :return:
    """
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
