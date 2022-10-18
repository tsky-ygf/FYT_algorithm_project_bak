# 检索部分相关说明

## 模块
### 1. cofig
relevant_laws: `相关法条配置文件`

similar_case_retrival: `相似类案配置文件`

### 2. src
* relevant_laws: `相关法条核心函数`
  * bert_model: `相关法条语义相似`
  * data_process: `相关法条数据预处理`
  * law_items_graph: `基于图算法的相关法条`
  * process_by_es: `基于es的法条检索、入库`
  * similar_case_retrival: `相似类案核心函数`
    * bert: `用于相似类案`
    * cause_division: `相似语义`
    * process_by_es: `基于es的案例检索、入库`
    * similar_case: `相似类案核心函数`

### 3. service_use
relevant_laws: `相关法条接口`

similar_case_retrival: `相似类案接口`

### 4. test_result
  relevant_laws: `相关法条测试`

  similar_case_retrival: `相似类案测试`


## 部署
### 3.1 部署环境
3.1.1 安装python环境```conda create -n search python=3.9.13```

3.1.2 安装相关依赖包
- ```pip install -r common_requirements.txt```
- ```pip install -r IntelligentConsultation/deploy_requirements.txt```

3.1.3 配置相关需求文件

- 需要插入到es的数据文件
  - ```data/search/elastic_index/case_index_minshi_v2.json```
  - ```data/search/elastic_index/case_index_v2.json```
  - ```data/search/elastic_index/flfg.json```
- 使用到的模型配置文件 
  - ```data/bxh_search_data/textCNN/vector/pad_word_index.json```
  - ```data/bxh_search_data/textCNN/vector/wiki_vec_pad.pt```

3.1.4 将数据导入到es库中
- elasticdump --input data/search/elastic_index/flfg.json --output http://127.0.0.1:9200/ --type=data
- elasticdump --input data/search/elastic_index/case_index_minshi_v2.json --output http://127.0.0.1:9200/ --type=data
- elasticdump --input data/search/elastic_index/case_index_v2.json --output http://127.0.0.1:9200/ --type=data

3.1.5 启动服务
``` nohup python -m OnlineServer.ProfessionalSearch.server >log/ProfessionalSearch/server.log 2>&1 & ```

## TODOLIST
- [ ] 相似度计算方法替换，使用bert,或者其他模型计算相似度；
     * 原方法: `textcnn中间层emb + cosine`
     * 替换方法1: `simbert`
     * 替换方法2: `simcse`
     * 替换方法3: `ernie, pair-wise方式`
- [ ] 统一索引库（narrative_similarity_v2 和 case_index_v2）;
- [ ] tag关键词的计算
- [ ] index（or database）的数据清洗（例如，文书标题含有李某、李某1，过滤掉1）
- [ ] 索引中加入标题

