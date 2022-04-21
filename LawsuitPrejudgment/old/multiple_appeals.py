import re
from old.utils import read_pattern_multi_appeal
from old.util2 import match_sentence_by_keyword
import pandas as pd

data_path = '/datadisk2/law_data/pinggu/result_train.csv'
pattern_path = '/datadisk4/wukong/推理图谱/data/法律推理图谱0423.xls'

suqiu_list = ['离婚', '返还彩礼']
judge_df = pd.read_csv(data_path)

for suqiu in suqiu_list:
    data_df = judge_df[judge_df['suqiu'] == suqiu]
    features, law, key_words, general_features, law_features, kw2fea_adjacency_matrix, fea2law_adjacency_matrix, suqiu2law_matrix = read_pattern_multi_appeal(
        pattern_path, suqiu_list, suqiu)
    text_texts = list(data_df['renwei'])
    labels = list(data_df['label'])
    for text, label in zip(text_texts, labels):
        print('_______________________________________________________________')
        print(text)
        print("++++++++++++++++")
        cuted_texts = re.split('[，。；：、]', text)
        for item in general_features:
            text_path = []
            neg_match = False
            for kw in item[1]:
                for samll_text in cuted_texts:
                    flag = match_sentence_by_keyword(samll_text, item[0], kw)
                    if flag == 1:
                        if label == 0 and (item[0] == '感情破裂' or item[0] == '感情不和'):
                            break
                        neg_match = False
                        text_path.append(samll_text)
                        text_path.append(kw)
                        # text_path.append(item[0])
                        kw2fea_adjacency_matrix.loc[item[0], kw] = kw2fea_adjacency_matrix.loc[item[0], kw] + 1
                        text_path.append(item[0] + '_' + str(kw2fea_adjacency_matrix.loc[item[0], kw]))
                        break
                    else:
                        neg_match = True
                        continue
                else:
                    continue
                break

            if text_path == [] or neg_match:
                continue
            for law_item in law_features:
                if item[0] in law_item[1]:
                    fea2law_adjacency_matrix.loc[law_item[0], item[0]] = fea2law_adjacency_matrix.loc[
                                                                             law_item[0], item[0]] + 1
                    text_path.append(law_item[0] + '_' + str(fea2law_adjacency_matrix.loc[law_item[0], item[0]]))
                    if label == 1:
                        suqiu2law_matrix.loc[suqiu].loc['1', law_item[0]] = suqiu2law_matrix.loc[suqiu].loc[
                                                                                '1', law_item[0]] + 1
                        text_path.append('1' + '_' + str(suqiu2law_matrix.loc[suqiu].loc['1', law_item[0]]))
                    if label == 0:
                        suqiu2law_matrix.loc[suqiu].loc['0', law_item[0]] = suqiu2law_matrix.loc[suqiu].loc[
                                                                                '0', law_item[0]] + 1
                        text_path.append('0' + '_' + str(suqiu2law_matrix.loc[suqiu].loc['0', law_item[0]]))
            if text_path != []:
                print('-->'.join(text_path))
    kw2fea_adjacency_matrix.to_csv('keywords2feature_multi.csv', index=True)
    fea2law_adjacency_matrix.to_csv('feature2law_multi.csv', index=True)
    suqiu2law_matrix.to_csv('law2suqiu_multi.csv', index=True)
