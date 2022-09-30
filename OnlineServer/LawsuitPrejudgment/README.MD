# 诉讼预判相关服务接口

## 1. 服务启动
进入到运行环境: `conda activate lawsuit_prejudgment`

添加项目跟目录: `export PYTHONPATH=$(pwd):$PYTHONPATH`

启动命令：
`nohup python -m OnlineServer.LawsuitPrejudgment.server > log/lawsuit_prejudgment/server.log 2>&1 &`

## 2. 接口在线文档
内网地址: "http://172.19.82.199:8100/docs"

外网地址: "http://101.69.229.138:8100/docs"

## 3. 接口测试
使用`pytest`进行接口测试，测试文件为`OnlineServer/LawsuitPrejudgment/app_test.py`

测试命令:
`pytest OnlineServer/LawsuitPrejudgment/app_test.py`

## 4. 结束服务
kill进程: `ps -ef | grep LawsuitPrejudgment.server | awk {'print $2'} |xargs kill -9`