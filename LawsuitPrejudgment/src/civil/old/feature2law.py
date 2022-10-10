import pandas as pd
from old.utils import read_pattern
import re
from old.util2 import match_sentence_by_keyword

data_path = '/datadisk2/law_data/pinggu/result_train.csv'
pattern_path = '/datadisk4/wukong/推理图谱/data/0425离婚法律推理图谱.xlsx'
pattern_df = pd.read_excel(pattern_path,names=['问题类型','诉求','法条特征','一般特征','关键词'])
features, law, key_words,general_features, law_features, feature2law_adjacency_matrix, law2suqiu_adjacency_matrix = read_pattern(pattern_df)

judge_data = pd.read_csv(data_path)
judge_data = judge_data[judge_data['suqiu'] == '离婚']
judge_data = judge_data.iloc[0:2000]

#构建邻接矩阵
kw2fea_adjacency_matrix = pd.DataFrame(index=features,columns=key_words)
kw2fea_adjacency_matrix.fillna(0,inplace=True)
fea2law_adjacency_matrix = pd.DataFrame(index=law,columns=features)
fea2law_adjacency_matrix.fillna(0,inplace=True)
suqiu = pd.DataFrame(index=['1','0'],columns=law)
suqiu.fillna(0,inplace=True)


test_texts = list(judge_data['renwei'])
labels = list(judge_data['label'])

#test_texts = ['老公殴打我，我们感情已经彻底破裂','老公殴打']
#labels = [1,1]

for text, label in zip(test_texts, labels):
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
                    suqiu.loc['1', law_item[0]] = suqiu.loc['1', law_item[0]] + 1
                    text_path.append('1' + '_' + str(suqiu.loc['1', law_item[0]]))
                if label == 0:
                    suqiu.loc['0', law_item[0]] = suqiu.loc['0', law_item[0]] + 1
                    text_path.append('0' + '_' + str(suqiu.loc['0', law_item[0]]))
        if text_path != []:
            print('-->'.join(text_path))
kw2fea_adjacency_matrix.to_csv('keywords2feature.csv',index=True)
fea2law_adjacency_matrix.to_csv('feature2law.csv',index = True)
suqiu.to_csv('law2suqiu.csv',index = True)