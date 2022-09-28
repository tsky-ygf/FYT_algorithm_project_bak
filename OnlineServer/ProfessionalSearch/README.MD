# 检索相关服务接口

## 1. 服务启动
进入到运行环境: `conda activate pp-use`

添加项目跟目录: `export PYTHONPATH=$(pwd):$PYTHONPATH`

启动命令：
`nohup python -m OnlineServer.ProfessionalSearch.server >log/ProfessionalSearch/server.log 2>&1 &`

## 2. 接口在线文档
内网地址: "http://172.19.82.199:8135/docs"
外网地址: "http://101.69.229.138:8135/docs"

## 3. 接口测试
使用`pytest`进行接口测试，测试文件为`OnlineServer/ProfessionalSearch/app_test.py`

## 4. 结束服务
kill进程: `ps -ef | grep ProfessionalSearch.server | awk {'print $2'} |xargs kill -9`