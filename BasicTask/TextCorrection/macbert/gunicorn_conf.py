#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 10:15
# @Author  : Adolf
# @Site    : 
# @File    : gunicorn.conf.py
# @Software: PyCharm
workers = 4
bind = '0.0.0.0:6598'
# 设置守护进程,将进程交给supervisor管理
daemon = False
# 设置超时时间120s，默认为30s。按自己的需求进行设置
timeout = 30
# 设置进程文件目录
reload = True
# 设置访问日志和错误信息日志路径
accesslog = "log/model_log/access.log"
errorlog = "log/model_log/debug.log"
loglevel = "info"
