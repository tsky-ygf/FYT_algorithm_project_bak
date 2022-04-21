
import re
import pymysql
import random
import pandas as pd
import os
import sys

import configparser

config = configparser.ConfigParser()
config.read('../main/db_config.ini')
big_data_host = config.get('bigData', 'host')
big_data_port = config.get('bigData', 'port')
big_data_user = config.get('bigData', 'user')
big_data_password = config.get('bigData', 'password')
big_data_db = config.get('bigData', 'db')

connect_big_data = pymysql.connect(host=big_data_host, port=int(big_data_port), user=big_data_user, password=big_data_password, db=big_data_db)

sql1 = '''
    select f1 as title, f2 as doc_id, f3 as province, f5 as serial, f12 as anyou, f14 as date, f41 as court, f7 as raw_content
    from justice_big_data.%s where f12 = '%s' limit %s, %s
''' % ('case_list_original_hetong', '租赁合同纠纷', 0, 1000)
sql2 = 'select * from justice_big_data.case_list_original_hetong limit 0, 10'
cs = connect_big_data.cursor()
cs.execute(sql2)
res = cs.fetchone()
print(res)
temp = pd.read_sql(sql1, con=connect_big_data)
print(len(temp))
