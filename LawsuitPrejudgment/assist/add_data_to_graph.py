# -*- coding: utf-8 -*-
from pypinyin import lazy_pinyin
import os
import pandas as pd


def add_data_to_graph_fn(config_files_path):
    """
    将数据批量添加到图数据库
    :return:
    """
    # 1.遍历文件夹下的文件
    list_entity, list_entity_relationsihp = [], []
    unique_id_dict = {}
    for root, dirs, files in os.walk(config_files_path):

        for index, file in enumerate(files):
            if 'structure.csv' not in file and '.csv' in file:  # and '描述_特征_类别.csv' in file:
                print(index, file)
            # if '纠纷类型_诉求_诉求.csv'  in file:
                single_file_process(file, root, list_entity, list_entity_relationsihp, unique_id_dict, index)
    # 4）写成一个实体的文件
    list_entity_new = remove_duplicate_element(list_entity)
    pd_entity = pd.DataFrame(list_entity_new, columns=['elementId' + ':ID', 'name', ':LABEL'])
    pd_entity.to_csv(target_path + "entity.csv", index=False)
    # 5) 写成一个关系的文件
    list_entity_relationsihp_new = remove_duplicate_element(list_entity_relationsihp)
    pd_relation = pd.DataFrame(list_entity_relationsihp_new, columns=[':START_ID', ':END_ID', ':TYPE'])
    pd_relation.to_csv(target_path + "relation.csv", index=False)


def remove_duplicate_element(list_entity):
    list_entity_new = []
    element_id_dict = {}
    for e in list_entity:
        k = e[0] + "_" + e[1]  # +"_"+e[2] # todo 一个名称只使用一个节点
        value = element_id_dict.get(k, None)
        if value is None:
            list_entity_new.append(e)
            element_id_dict[k] = e
    return list_entity_new


def single_file_process(file, root, list_entity, list_entity_relationsihp, unique_id_dict, index):
    """
    处理单个文件
    :param file:
    :param root:
    :return:
    """
    file_path = os.path.join(root, file)
    data_temp = pd.read_csv(file_path)
    # 0)获得列名
    column_list = data_temp.columns
    entity_name_1 = column_list[0]
    entity_name_2 = column_list[1]
    relationship = column_list[2]
    # 1) 获得所有实体1
    entity_1_list = list(set(data_temp[entity_name_1]))
    entity_1_dict = {x: get_id_by_name(x, unique_id_dict, index) for i, x in enumerate(entity_1_list)}  # "_".join(lazy_pinyin(u'云浮新兴双线-01'))
    # 2) 获得所有实体2
    entity_2_list = list(set(data_temp[entity_name_2]))
    entity_2_dict = {x: get_id_by_name(x, unique_id_dict, index) for i, x in enumerate(entity_2_list)}  # "_".join(lazy_pinyin(u'云浮新兴双线-01'))
    relationship_list = list(data_temp[column_list[2]])
    # 3）构建实体的csv
    list_entity_1 = [[entity_1_dict[entity1], entity1, entity_name_1] for entity1 in entity_1_list]
    list_entity_2 = [[entity_2_dict[entity2], entity2, entity_name_2] for entity2 in entity_2_list]
    list_entity_1.extend(list_entity_2)
    list_entity.extend(list_entity_1)

    for ii, row in data_temp.iterrows():
        e1 = row[entity_name_1]
        e2 = row[entity_name_2]
        r = row[relationship]
        list_entity_relationsihp.append((get_id_by_name(e1, unique_id_dict, index), get_id_by_name(e2, unique_id_dict, index), r))


def get_id_by_name(name, unique_id_dict, index):
    idd = "_".join(lazy_pinyin(name)) + '_' + str(abs(hash(name)))
    # if unique_id_dict.get(idd,None) is not None:
    #    idd=idd+"_duplicate_"+str(index)
    # unique_id_dict[idd]=idd
    return idd


config_files_path = './config_data'
target_path = './config_graph_data/'
add_data_to_graph_fn(config_files_path)


def xxx(config_data_files_path):
    for root, dirs, files in os.walk(config_data_files_path):
        file_list = ["data_hunying/" + x for x in files if 'relat' in x]
        print("file_list:", file_list)


config_data_files_path = './config_graph_data'
# xxx(config_data_files_path)
