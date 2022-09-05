import os
import gevent.monkey
gevent.monkey.patch_all()
import multiprocessing

# worker的个数
workers = multiprocessing.cpu_count() * 2 +1
worker_class = 'gevent'
threads = 20
preload_app = True

# 监听内网端口
bind = '0.0.0.0:8120'
# 设置守护进程【关闭连接时，程序仍在运行】
daemon = True
# 设置超时时间120s，默认为30s。按自己的需求进行设置
timeout = 120
#Restart workers when code changes. False:will not restart
reload = False
# 设置访问日志和错误信息日志路径
accesslog = 'log/legal_knowledge/gunicorn_legal_knowledge_api_access.log'
errorlog = 'log/legal_knowledge/gunicorn_legal_knowledge_api_error.log'
loglevel = 'info'
