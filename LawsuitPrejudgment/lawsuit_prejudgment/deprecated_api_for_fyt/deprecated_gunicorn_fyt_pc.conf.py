# worker的个数
workers = 2
# 监听内网端口
bind = '0.0.0.0:5080'
# 设置守护进程【关闭连接时，程序仍在运行】
daemon = True
# 设置超时时间120s，默认为30s。按自己的需求进行设置
timeout = 30
#Restart workers when code changes. False:will not restart
reload = False
# 设置访问日志和错误信息日志路径
accesslog = 'log/lawsuit_prejudgment/deprecated_gunicorn_fyt_pc_access.log'
errorlog = 'log/lawsuit_prejudgment/deprecated_gunicorn_fyt_pc_error.log'
loglevel = 'info'
