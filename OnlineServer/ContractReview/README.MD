# 合同智审相关服务接口

## 1. 服务启动
进入到运行环境: `conda activate pp-use`

添加项目跟目录: `export PYTHONPATH=$(pwd):$PYTHONPATH`

启动命令：
`nohup python -m OnlineServer.ContractReview.server >log/contract_review/server.log 2>&1 &`

## 2. 接口在线文档
内网地址: "http://172.19.82.199:8112/docs"
外网地址: "http://101.69.229.138:8112/docs"

## 3. 接口测试
使用`pytest`进行接口测试，测试文件为`OnlineServer/ContractReview/app_test.py`

## 4. 结束服务
kill进程: `ps -ef | grep ContractReview.server | awk {'print $2'} |xargs kill -9`

## 5. 服务部署
pass

## 6. 测试页面