# 纠错服务的相关服务接口

## 1. 服务启动
进入到运行环境: `conda activate corrector`

添加项目跟目录: `export PYTHONPATH=$(pwd):$PYTHONPATH`

启动命令：
`nohup python -m OnlineServer.ContractCorrector.server > log/contract_corrector/server.log 2>&1 &`

## 2. 接口在线文档
内网地址: "http://172.19.82.199:6598/docs"
外网地址: "http://101.69.229.138:6598/docs"

## 3. 接口测试
使用`pytest`进行接口测试，测试文件为`OnlineServer/ContractCorrector/app_test.py`

## 4. 结束服务
kill进程: `ps -ef | grep ContractCorrector.server | awk {'print $2'} |xargs kill -9`

## 5. 服务部署
pass

## 6. 测试页面
http://172.19.82.199:6598/