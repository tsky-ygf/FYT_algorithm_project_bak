# 合同智审算法模块

# 1. 模块介绍
主要用于合同智审的算法模块，包括了合同智审的基本算法，以及合同智审的基本模型。

# 2. 模块结构
```
```

# 3. 部署

### 3.1 部署环境
3.1.1 安装python环境```conda create -n review python=3.9.12```

3.1.2 安装相关依赖包
- ```pip install -r common_requirements.txt```
- ```pip install -r DocumentReview/deploy_requirements.txt```


3.1.3 配置相关需求文件

- 需要插入到es的数据文件 ```无```
- 使用到的模型文件 
```
model/PointerBert/PBert1014_common_all_20sche_auged.pt
model/uie_model/export_cpu/caigou/
model/uie_model/export_cpu/fangwuzulin/
model/uie_model/export_cpu/jiekuan/
model/uie_model/export_cpu/jietiao/
model/uie_model/export_cpu/laodong/
model/uie_model/export_cpu/laodongbaomi/
model/uie_model/export_cpu/laowu/
model/uie_model/export_cpu/maimai/
model/uie_model/export_cpu/shangyebaomi/
model/uie_model/export_cpu/yibanzulin/
```
