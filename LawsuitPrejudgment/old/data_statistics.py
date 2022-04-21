# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../common'))
from common import multi_processing_data, config_path, problem_suqius, suqiu_bkw_dict, suqiu_correct


'''
统计因子出现频率，用于候选问题的排序
'''

def feature_stats(source_file, target_file, problem, problem_list, process_num=30):
    """
    统计特征出现频率
    :param source_file: 裁判文书数据路径
    :param target_file: 特征因子权重路径
    :param problem: 纠纷类型
    :param process_num: 进程数
    """
    suqiu_factor_dict = {}
    factor_others_dict = {}
    data = pd.read_csv(source_file, encoding='utf-8')
    data = data[data['problem'].isin(problem_list)]
    data['new_problem'] = data.apply(lambda row: suqiu_correct(row['suqing'], row['problem'], row['suqiu'])[0], axis=1)
    data['new_suqiu'] = data.apply(lambda row: suqiu_correct(row['suqing'], row['problem'], row['suqiu'])[1], axis=1)
    print("length of data:", len(data))

    for suqiu in problem_suqius[problem]:
        data_temp = data[(data['new_problem'] == problem) & (data['new_suqiu'] == suqiu) & (data['label'] == 1)].copy()
        print(suqiu, len(data_temp))
        if len(data_temp)==0:
            continue

        data_x_kw = multi_processing_data(data_temp['chaming'].values, min(process_num, len(data_temp)), problem, suqiu)
        print(data_x_kw.shape)
        for i, f in enumerate(suqiu_bkw_dict[problem+'_'+suqiu].index):
            data_temp[f] = data_x_kw[:, i]

        for index, row in data_temp.iterrows():
            sentence_factor = {f: 1 for f in data_temp.columns[row == 1] if f in suqiu_bkw_dict[problem+'_'+suqiu]}
            factor_co_occur_count(suqiu, factor_others_dict, sentence_factor)
            # print("sentence_factor:",sentence_factor) # sentence_factor: {'家暴遗弃虐待': ['因原告于2011年外出打工', 1], '恶习': ['被告李某甲因吸毒在201
            for factor, v in sentence_factor.items():
                # k='家暴遗弃虐待'; v=['因原告于2011年外出打工', 1]
                key = suqiu + "_" + factor
                suqiu_factor_dict[key] = suqiu_factor_dict.get(key, 0) + 1

    print("suqiu_factor_dict:", suqiu_factor_dict)
    final_list = sorted(suqiu_factor_dict.items(), key=lambda item: item[1], reverse=True)

    list_new = []
    for e in final_list:  # e="离婚_结婚登记:3116"
        suqiu_, factor_ = e[0].split("_")
        if suqiu_ + "_" + factor_ in factor_others_dict:
            sub_dict = factor_others_dict[suqiu_ + "_" + factor_]  # sub_dict:  {'家暴遗弃虐待': 3, '恶习': 2, '结婚登记': 1}
            sub_list = sorted(sub_dict.items(), key=lambda item: item[1], reverse=True)
        else:
            sub_list = []
        count_ = e[1]
        list_new.append((suqiu_, factor_, int(count_), sub_list[:10]))

    pd_weight = pd.DataFrame(list_new, columns=['诉求', '要素', '统计', '相关要素'])
    pd_weight = pd_weight.sort_values(['诉求', '统计'], ascending=False)
    pd_weight.to_csv(target_file, index=False)


def factor_co_occur_count(suqiu_type, factor_others_dict, sentence_factor_dict):
    """
    统计两个要素的共现关系
    :param factor_others_dict: 全局共现的dict
    :param sentence_factor_dict: 单次匹配到的结果. e.g.  {'家暴遗弃虐待': ['因原告于2011年外出打工', 1], '恶习': ['被告李某甲因吸毒在201]}
    :return:
    """
    factor_sub_list = [k for k, v in sentence_factor_dict.items()]  # factor_sub_list=['家暴遗弃虐待','恶习','家暴']
    length = len(factor_sub_list)
    if length <= 1:
        return
    # print("factor_sub_list:",factor_sub_list)
    for i, f1 in enumerate(factor_sub_list):
        for j, f2 in enumerate(factor_sub_list):
            if j != i:
                k = suqiu_type + '_' + f1  # +"_"+f2
                # print("k:",k)
                sub_dict = factor_others_dict.get(k, None)
                if sub_dict is None:
                    sub_dict = {}
                sub_dict[f2] = sub_dict.get(f2, 0) + 1
                factor_others_dict[k] = sub_dict
    # print("factor_others_dict:",factor_others_dict)


# 弃用
# def feature_description(source_file, target_file, problem, suqiu_list):
#     suqiu_factor_dict = {}
#     data = pd.read_csv(source_file)
#     print("length of data:", len(data))
#     for suqiu_type in suqiu_list:
#         data_temp = data[(data['problem'] == problem) & (data['suqiu'] == suqiu_type) & (data['label'] == 1)]
#         data_temp = data_temp.sample(frac=0.01)
#         print(suqiu_type, len(data_temp))
#         for index, row in data_temp.iterrows():
#             content = row['chaming']
#             # print("index:",index,";row:",row)
#             sentence_factor = single_case_match(content, problem, suqiu_type)
#             # print("sentence_factor:",sentence_factor) # sentence_factor: {'家暴遗弃虐待': ['因原告于2011年外出打工', 1], '恶习': ['被告李某甲因吸毒在201
#             for factor, v in sentence_factor.items():
#                 # k='家暴遗弃虐待'; v=['因原告于2011年外出打工', 1]
#                 key = suqiu_type + "_" + factor
#                 if key in suqiu_factor_dict and v[1] in suqiu_factor_dict[key]:
#                     continue
#                 if key in suqiu_factor_dict:
#                     suqiu_factor_dict[key][v[1]] = v[0]
#                 else:
#                     suqiu_factor_dict[key] = {v[1]: v[0]}
#
#     print("suqiu_factor_dict:", suqiu_factor_dict)
#     list_new = []
#     for k, v in suqiu_factor_dict.items():
#         suqiu_, factor_ = k.split("_")
#         list_new.append((suqiu_, factor_, v[1] if 1 in v else None, v[-1] if -1 in v else None))
#     pd_weight = pd.DataFrame(list_new, columns=['suqiu', 'factor', 'positive_description', 'negative_description'])
#     pd_weight.to_csv(target_file, index=False)


if __name__ == '__main__':
    source_file = '/datadisk2/tyin/new/result_train.csv'
    # source_file = '../data/result_train.csv'
    # problem = '劳动社保'
    # problem_list = ['劳动纠纷', '社保纠纷', '工伤赔偿', '提供劳务者受害责任纠纷', '提供劳务者致害责任纠纷']
    problem = '婚姻家庭'
    problem_list = ['婚姻家庭', '继承问题']

    target_file = config_path + problem + '/' + problem + '因子权重.csv'
    feature_stats(source_file, target_file, problem, problem_list)
