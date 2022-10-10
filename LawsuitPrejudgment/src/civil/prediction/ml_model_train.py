# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import jieba
import os
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix

'''
基于关键词特征以及卡方值选择的特征词进行模型训练，诉求作为特征一起训练
'''

data_path = '../data/prediction/'
model_ml_path = '../model/ml/prediction/'


def feature_x_calculate(data, is_train, problem, suqiu, min_df=20, n_features=300):
    data['words'] = data['chaming_fact'].fillna('').apply(lambda x: ' '.join(jieba.lcut(x)))
    if is_train:
        # 特征向量
        # data['no_proof_fact'] = data['no_proof_fact'].apply(lambda x: ' '.join(jieba.lcut(x)))
        # data['no_proof_fact'] = data.apply(lambda row: row['no_proof_fact'] if row['label']==0 else '', axis=1)
        # data['words'] = data['words'] + ' ' + data['no_proof_fact']
        vectorizer = CountVectorizer(min_df=min_df, token_pattern='[\u4E00-\u9FA5#]{2,}').fit(data['words'].values)
        feature_x_wb = vectorizer.transform(data['words'].values).toarray()
        feature_x_wb[feature_x_wb > 0] = 1
        print('word bag size: %s' % (len(vectorizer.vocabulary_)))
        # 特征词选择
        selection = SelectKBest(chi2, n_features).fit(feature_x_wb, data['label'].values)
        feature_x_wb = selection.transform(feature_x_wb)
        print(feature_x_wb.shape)
        joblib.dump(vectorizer, model_ml_path + problem + '_' + suqiu + 'count.pkl')
        joblib.dump(selection, model_ml_path + problem + '_' + suqiu + 'selection.pkl')
    else:
        vectorizer = joblib.load(model_ml_path + problem + '_' + suqiu + 'count.pkl')
        selection = joblib.load(model_ml_path + problem + '_' + suqiu + 'selection.pkl')
        feature_x_wb = vectorizer.transform(data['words'].values).toarray()
        feature_x_wb[feature_x_wb > 0] = 1
        feature_x_wb = selection.transform(feature_x_wb)
        print(feature_x_wb.shape)

    words = vectorizer.get_feature_names()
    support = selection.get_support()
    selected_words = [w for i, w in enumerate(words) if support[i]]

    return selected_words, feature_x_wb


def train_model(data, problem, suqiu):
    """
    每个问题类型训练一个传统机器学习模型
    :param target_path: 目标路径
    """
    # 1.加载训练数据
    selected_words, feature_x_wb_train = feature_x_calculate(data, True, problem, suqiu, len(data)//5000+1)

    # 2.构造相关模型特征向量
    train_x = feature_x_wb_train
    train_y = data['label'].values
    print("train_x:", train_x.shape, ";train_y:", train_y.shape)

    # 3.训练相关模型
    model = RandomForestClassifier(n_estimators=200, max_depth=32, n_jobs=1, random_state=20181024, class_weight='balanced')
    model.fit(train_x, train_y)
    accuracy_train = model.score(train_x, train_y)
    print("accuracy_train:", accuracy_train)
    joblib.dump(model, model_ml_path + problem + '_' + suqiu + 'RF.pkl')


##########################################################################################################

def predict_model(data, problem, suqiu):
    """
    基于训练的模型进行测试数据的预测
    :param target_path: 目标路径
    :param date_string: 日期
    """
    feature_x_wb_test = feature_x_calculate(data, False, problem, suqiu)[1]

    # 载入模型
    print('load model')
    model = joblib.load(model_ml_path + problem + '_' + suqiu + 'RF.pkl')
    test_x = feature_x_wb_test

    # 模型预测并统计平均表现
    predict_label = np.array([1 if p[1] > 0.5 else 0 for p in model.predict_proba(test_x)])
    data['predict_label'] = predict_label
    y_true = data['label'].values
    y_pred = data['predict_label'].values
    print('average performance', problem)
    print('accuracy: %s' % (accuracy_score(y_true, y_pred)))
    print('f1_score: %s' % (f1_score(y_true, y_pred, average='macro')))
    print('f1_score of 2 labels: %s' % (f1_score(y_true[y_true>-1], y_pred[y_true>-1]>0, average='macro')))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    proba = np.array([p[1] for p in model.predict_proba(test_x)])
    proba0 = proba[y_true == 0]
    proba1 = proba[y_true == 1]
    return data, proba0, proba1


def evaluate(problem_list):
    y_true = []
    y_pred = []
    proba0 = []
    proba1 = []
    for problem, suqius in problem_list.items():
        if os.path.exists(data_path + problem + '_valid.csv'):
            data_valid = pd.read_csv(data_path + problem + '_valid.csv')
            for suqiu in suqius:
                data_temp = data_valid[data_valid['new_suqiu']==suqiu].copy()
                print(problem, suqiu, len(data_temp))
                if len(data_temp)==0:
                    continue
                data_valid, p0, p1 = predict_model(data_temp, problem, suqiu)
                y_true += data_valid['label'].values.tolist()
                y_pred += data_valid['predict_label'].values.tolist()
                proba0 += p0.tolist()
                proba1 += p1.tolist()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print('average performance')
    print('accuracy: %s' % (accuracy_score(y_true, y_pred)))
    print('f1_score: %s' % (f1_score(y_true, y_pred, average='macro')))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


##########################################################################################################

if __name__ == '__main__':
    import time
    st = time.time()
    problem_list = {
        '婚姻家庭': [
            '离婚', '返还彩礼', '房产分割', '确认抚养权', '行使探望权', '支付抚养费', '增加抚养费', '减少抚养费',
            '支付赡养费', '确认婚姻无效', '财产分割', '夫妻共同债务', '确认遗嘱有效', '遗产继承'
        ]
    }

    # 训练模型
    for problem, suqius in problem_list.items():
        data_train = pd.read_csv(data_path + problem + '_train.csv')
        for suqiu in suqius:
            data_temp = data_train[data_train['new_suqiu'] == suqiu].copy()
            print(problem, suqiu, len(data_temp))
            train_model(data_temp, problem, suqiu)

    # 验证模型
    print('evaluate model with valid case')
    evaluate(problem_list)
    print('time used', time.time() - st)
