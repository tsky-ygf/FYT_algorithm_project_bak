# 智能咨询算法模块

# 1. 模块介绍
主要用于智能咨询的算法模块，包括了智能咨询的基本算法，以及智能咨询的基本模型。

# 2. 模块结构
```
```

# 3. 部署

### 3.1 部署环境
3.1.1 安装python环境```conda create -n consult python=3.9```

3.1.2 安装相关依赖包
- ```pip install -r common_requirements.txt```
- ```pip install -r IntelligentConsultation/deploy_requirements.txt```

3.1.3 配置相关需求文件

- 需要插入到es的数据文件 ```data/fyt_train_use_data/QA/pro_qa.csv```
- 使用到的模型文件 ```model/similarity_model/simcse-model-topic-qa```

3.1.4 将数据导入到es库中
- export PYTHONPATH=$(pwd):$PYTHONPATH
- python IntelligentConsultation/service_use/insert_data_to_es.py

3.1.5 启动服务
``` nohup python -m OnlineServer.IntelligentConsultation.server >log/intelligent_consultation/server.log 2>&1 & ```