# -*- coding: utf-8 -*-
import pypinyin
from pypinyin import pinyin, lazy_pinyin
import os
import pandas as pd
def add_data_to_graph_fn(config_files_path,target_path):
    """
    将数据批量添加到图数据库
    :return:
    """
    # 1.遍历文件夹下的文件
    for root, dirs, files in os.walk(config_files_path):

        for index,file in enumerate(files):
            if 'structure.csv' not in file and '.csv' in file:
            #if '纠纷类型_诉求_诉求.csv'  in file:
                single_file_process(file, root,target_path,index)
    pass

def single_file_process(file,root,target_path,index):
    """
    处理单个文件
    :param file:
    :param root:
    :return:
    """
    print("file:", file)  # file='纠纷类型_诉求_诉求.csv'
    file_path = os.path.join(root, file)
    print("file_path:", file_path)
    data_temp = pd.read_csv(file_path)
    # 0)获得列名
    column_list = data_temp.columns
    print("column_list:", column_list)
    entity_name_1 = column_list[0]
    print("entity_name_1:", entity_name_1)
    entity_name_2 = column_list[1]
    print("entity_name_2:", entity_name_2)
    relationship = column_list[2]
    # 1) 获得所有实体1
    entity_1_list = list(set(data_temp[entity_name_1]))
    print("entity_1_list:", entity_1_list)
    entity_1_dict = {x: get_id_by_name(x) for i, x in enumerate(entity_1_list)}  # "_".join(lazy_pinyin(u'云浮新兴双线-01'))
    print("entity_1_dict:", entity_1_dict)
    # 2) 获得所有实体2
    entity_2_list = list(set(data_temp[entity_name_2]))
    print("entity_2_list:", entity_2_list)
    entity_2_dict = {x: get_id_by_name(x) for i, x in enumerate(entity_2_list)}  # "_".join(lazy_pinyin(u'云浮新兴双线-01'))
    print("entity_2_dict:", entity_2_dict)
    relationship_list = list(data_temp[column_list[2]])
    print("relationship_list:", relationship_list)
    # 3）构建实体的csv
    list_entity_1 = [[entity_1_dict[entity1], entity1, entity_name_1] for entity1 in entity_1_list]
    print("###list_entity_1:", list_entity_1)

    list_entity_2 = [[entity_2_dict[entity2], entity2, entity_name_2] for entity2 in entity_2_list]
    print("###list_entity_2:", list_entity_2)
    list_entity_1.extend(list_entity_2)
    pd_entity = pd.DataFrame(list_entity_1,columns=['elementId'+':ID', 'name', ':LABEL'])  # elementId:ID	name	:LABEL

    entity_file_name = file.replace(".csv", "")
    # 4）写成一个实体的文件
    pd_entity.to_csv(target_path + entity_file_name + "_entity" + ".csv", index=False)

    # 5）构建关系文件的数据
    list_entity_relationsihp = [] #[[entity_1_dict[entity1], entity1, entity_name_1] for entity1 in entity_1_list]
    for index, row in data_temp.iterrows():
        e1=row[entity_name_1]
        e2=row[entity_name_2]
        r=row[relationship]
        list_entity_relationsihp.append((get_id_by_name(e1),get_id_by_name(e2),r))
    pd_relationship = pd.DataFrame(list_entity_relationsihp, columns=[':START_ID', ':END_ID', ':TYPE'])  # :START_ID	:END_ID	:TYPE
    # 6）写成关系
    pd_relationship.to_csv(target_path + entity_file_name + "_relation" + ".csv", index=False)

def get_id_by_name(name):
    idd="_".join(lazy_pinyin(name)) + '_'+str(abs(hash(name)))
    return idd

config_files_path='./config_data'
target_path='./config_graph_data/'
add_data_to_graph_fn(config_files_path,target_path)

def xxx(config_data_files_path):
    for root, dirs, files in os.walk(config_data_files_path):

        file_list=[ "data_hunying/"+x for x in files if 'relat' in x]
        print("file_list:",file_list)

config_data_files_path='./config_graph_data'
#xxx(config_data_files_path)