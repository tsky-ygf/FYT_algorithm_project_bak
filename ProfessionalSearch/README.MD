# 检索部分相关说明

## 1. cofig
relevant_laws: `相关法条配置文件`

similar_case_retrival: `相似类案配置文件`

## 2. src
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

## 3. service_use
relevant_laws: `相关法条接口`

similar_case_retrival: `相似类案接口`

## 4. test_result
relevant_laws: `相关法条测试`

similar_case_retrival: `相似类案测试`


## TODOLIST
- [ ] 相似度计算方法替换，使用bert,或者其他模型计算相似度；
- [ ] 统一索引库（narrative_similarity_v2 和 case_index_v2）;
- [ ] tag关键词的计算

