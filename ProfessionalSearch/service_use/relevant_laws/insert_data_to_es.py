from ProfessionalSearch.src.relevant_laws.process_by_es.laws_items_es import (
    insert_law_to_es,
)

# 使用elasticdump
# 索引文件位置: data/search/elastic_index/flfg.json
# 执行命令:  elasticdump --input flfg.json --output http://127.0.0.1:9200/ --type=data

# 从数据库新建索引， 暂时写方法入口，方法入参待优化
insert_law_to_es()
