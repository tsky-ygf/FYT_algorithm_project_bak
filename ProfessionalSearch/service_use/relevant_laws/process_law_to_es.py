import argparse

from ProfessionalSearch.src.relevant_laws.process_by_es.laws_items_es import (
    process_law_to_es,
)

# 使用elasticdump
# 索引文件位置: data/search/elastic_index/flfg.json
# 执行命令:  elasticdump --input flfg.json --output http://127.0.0.1:9200/ --type=data

# 从数据库新建索引， 暂时写方法入口，方法入参待优化
parser = argparse.ArgumentParser()

parser.add_argument("--way", default='update', type=str,
                    help="操作方法，insert or update")
argus = parser.parse_args()
print(argus.way)
# if argus.way == 'insert':
#     process_law_to_es(argus.way)
# elif argus.way == 'update':
#     process_law_to_es(argus.way)
# elif argus.way == 'create':
#     process_law_to_es(argus.way)
# elif argus.way == 'do_nothing':
#     pass
# else:
#     pass
process_law_to_es('update')
