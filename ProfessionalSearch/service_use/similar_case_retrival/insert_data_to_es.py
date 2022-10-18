from ProfessionalSearch.src.similar_case_retrival.process_by_es.insert_case_to_es import (
    insert_case_to_es,
)

# 使用elasticdump插入索引
# 索引文件位置: data/search/elastic_index/case_index_minshi_v2.json  # 民事类案数据
#             data/search/elastic_index/case_index_v2.json  # 案例检索数据
# 执行命令: elasticdump --input case_index_minshi_v2.json --output http://127.0.0.1:9200/ --type=data

# 从数据库新建索引 ， 暂时写方法入口，方法入参待优化
insert_case_to_es()
