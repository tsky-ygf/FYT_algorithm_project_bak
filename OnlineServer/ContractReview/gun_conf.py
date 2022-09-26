# 并行工作进程数(multiprocessing.cpu_count()线程数,官方说可以有：核心数*2+1个)
workers = 2
# 指定每个工作者的线程数
threads = 2
# 监听内网端口5000
bind = '0.0.0.0:8112'
# 设置守护进程,推荐将进程交给supervisor管理(以守护进程形式来运行Gunicorn进程，true其实就是将这个服务放到后台去运行,故此处设置false，交给supervisor开守护进程，因为supervisor不支持后台进程)
daemon = True
# 工作模式协程
worker_class = 'gevent'
# 设置最大并发量
worker_connections = 2000
# 设置进程文件目录
# pidfile = '/var/run/gunicorn.pid'
# 设置访问日志和错误信息日志路径
accesslog = 'log/contract_review/gunicorn_access.log'
errorlog = 'log/contract_review/gunicorn_error.log'
# 日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
loglevel = 'INFO'
