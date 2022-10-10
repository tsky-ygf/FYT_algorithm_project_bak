import re
import os
import pandas as pd

def read_pattern(pattern_df):
    general_features = {}
    law_features = {}
    general_features_groups = pattern_df.groupby('一般特征')
    law_features_groups = pattern_df.groupby('法条特征')

    for general_feature,small_df in general_features_groups:
        general_features[general_feature] = list(set(list(small_df['关键词'])))

    for law_feature,small_df in law_features_groups:
        law_features[law_feature] = list(set(list(small_df['一般特征'])))

    features = sorted(list(set(list(pattern_df['一般特征']))))
    law = sorted(list(set(list(pattern_df['法条特征']))))
    key_words = sorted(list(set(list(pattern_df['关键词']))))
    feature2law_adjacency_matrix = [[0 for _ in  range(len(features))] for _ in range(len(law))]
    law2suqiu_adjacency_matrix = [[ 0 for _ in range(len(law))]for _ in range(4*2)]
    general_features = sorted(general_features.items(), key=lambda d: d[0], reverse=False)
    law_features = sorted(law_features.items(),key=lambda d:d[0],reverse=False)
    return features,law,key_words,general_features,law_features,feature2law_adjacency_matrix,law2suqiu_adjacency_matrix


def read_pattern_multi_appeal(pattern_path, suqiu_list, current_suqiu):
    suqiu_list = sorted(suqiu_list)
    pattern_df = pd.read_excel(pattern_path, names=['问题类型', '诉求', '法条特征', '一般特征', '关键词'])
    # pattern_df['诉求'] = pattern_df['诉求'].map(strip_speace)
    general_features = {}
    law_features = {}
    general_features_groups = pattern_df.groupby('一般特征')
    law_features_groups = pattern_df.groupby('法条特征')

    for general_feature, small_df in general_features_groups:
        general_features[general_feature] = list(set(list(small_df['关键词'])))

    for law_feature, small_df in law_features_groups:
        law_features[law_feature] = list(set(list(small_df['一般特征'])))

    features = sorted(list(set(list(pattern_df['一般特征']))))
    law = sorted(list(set(list(pattern_df['法条特征']))))
    key_words = sorted(list(set(list(pattern_df['关键词']))))
    general_features = sorted(general_features.items(), key=lambda d: d[0], reverse=False)
    law_features = sorted(law_features.items(), key=lambda d: d[0], reverse=False)

    # 构建多诉求邻接矩阵
    if os.path.exists('keywords2feature_multi.csv') and os.path.exists('feature2law_multi.csv') and os.path.exists(
            'law2suqiu_multi.csv'):
        kw2fea_adjacency_matrix = pd.read_csv('keywords2feature_multi.csv', index_col=0)
        fea2law_adjacency_matrix = pd.read_csv('feature2law_multi.csv', index_col=0)
        suqiu2law_adjacency = pd.read_csv('law2suqiu_multi.csv', index_col=0)
    else:
        # 构建邻接矩阵
        kw2fea_adjacency_matrix = pd.DataFrame(index=features, columns=key_words)
        kw2fea_adjacency_matrix.fillna(0, inplace=True)
        kw2fea_adjacency_matrix.to_csv('keywords2feature_multi')
        fea2law_adjacency_matrix = pd.DataFrame(index=law, columns=features)
        fea2law_adjacency_matrix.fillna(0, inplace=True)
        fea2law_adjacency_matrix.to_csv('feature2law_multi.csv')
        suqiu2law_adjacency = pd.DataFrame(columns=law, index=pd.MultiIndex.from_product([suqiu_list, ['1', '0']]))
        suqiu2law_adjacency.fillna(0, inplace=True)
        suqiu2law_adjacency.to_csv('law2suqiu_multi.csv')
    ###################################################
    # 在统计权值的时候分诉求统计
    ###################################################

    for suqiu, data in pattern_df.groupby('诉求'):
        if suqiu.strip() == current_suqiu.strip():
            general_features = {}
            law_features = {}
            general_features_groups = data.groupby('一般特征')
            law_features_groups = data.groupby('法条特征')

            for general_feature, small_df in general_features_groups:
                general_features[general_feature] = list(set(list(small_df['关键词'])))

            for law_feature, small_df in law_features_groups:
                law_features[law_feature] = list(set(list(small_df['一般特征'])))

            features = sorted(list(set(list(data['一般特征']))))
            law = sorted(list(set(list(data['法条特征']))))
            key_words = sorted(list(set(list(data['关键词']))))
            general_features = sorted(general_features.items(), key=lambda d: d[0], reverse=False)
            law_features = sorted(law_features.items(), key=lambda d: d[0], reverse=False)
            return features, law, key_words, general_features, law_features, kw2fea_adjacency_matrix, fea2law_adjacency_matrix, suqiu2law_adjacency
        else:
            continue
