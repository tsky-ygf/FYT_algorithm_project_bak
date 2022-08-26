

import json
import pandas as pd
import  matplotlib.pyplot as plt

# content 长度
# count    48188.000000
# mean       387.631630
# std        255.280813
# min          7.000000
# 25%        226.000000
# 50%        326.000000
# 75%        477.000000
# max       4305.000000
# 最大长度有12149 大于4305的长度还有353条记录   SELECT  LENGTH(content)  situation, factor FROM `labels_law_entity_feature` ORDER BY LENGTH(content) DESC
# Name: text_len, dtype: float64
def content_len(input_file):
    data_loan = pd.read_json(input_file, lines=True)
    data_loan['text_len'] = data_loan['words'].str.len()
    print(data_loan['text_len'].describe())
    _ = plt.hist(data_loan['text_len'], bins=200)
    plt.xlabel('len')
    plt.title('his')
    plt.show()
    return  None

# label 标签分布 situation + factor    标签类别不平衡
def label_distri(input_file, top_n = 50):
    data_loan = pd.read_json(input_file, lines=True)
    data_loan['label_id'].value_counts().plot.bar()
    count = data_loan['label_id'].value_counts()
    for i in range(top_n):
        print(count.index[i])
        if count.index[i] in data_loan['label_id']:
            for item in data_loan.iterrows():
                if str(count.index[i]) == str(item[1][2]):
                    print(count.index[i])
                    print(item[1])
                    print('---')
                    break
    plt.title('label count')
    plt.xlabel('category')
    plt.xticks(rotation=90, fontsize=8)
    plt.xlim(0,top_n)
    plt.show()
    return None

if __name__ == '__main__':
    # data_loan_path = 'data/loan/labels_law_entity_feature_with_situation_and_factor.json'
    data_loan_path = 'data/loan/labels_law_entity_feature_with_situation_factor.json'
    content_len(data_loan_path)
    label_distri(data_loan_path, top_n=50)