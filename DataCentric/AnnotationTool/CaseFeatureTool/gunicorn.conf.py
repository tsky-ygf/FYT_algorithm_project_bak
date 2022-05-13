#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 10:15
# @Author  : Adolf
# @Site    : 
# @File    : gunicorn.conf.py
# @Software: PyCharm
# 并行工作进程数
workers = 4

# 指定每个工作者的线程数
threads = 2

# 端口 5000
bind = '0.0.0.0:6021'

# 设置守护进程,将进程交给supervisor管理
daemon = 'false'

# 工作模式协程
worker_class = 'gevent'

# 设置最大并发量
worker_connections = 2000

# 设置进程文件目录
pidfile = 'log/run/gunicorn.pid'

# 设置访问日志和错误信息日志路径
accesslog = "log/annotation_tool/access.log"
errorlog = "log/annotation_tool/debug.log"
loglevel = "debug"

# 设置日志记录水平
loglevel = 'warning'