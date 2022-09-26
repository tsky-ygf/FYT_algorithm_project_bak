# 合同智审相关服务接口

## 1. 合同智审服务
添加项目跟目录: `export PYTHONPATH=$(pwd):$PYTHONPATH`

启动命令：

`gunicorn -c OnlineServer/ContractReview/gun_conf.py OnlineServer.ContractReview.server:app`