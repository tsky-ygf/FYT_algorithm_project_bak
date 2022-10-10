# coding=utf-8
import pymysql
import pandas as pd
import numpy as np
import re
import random
from sklearn.cluster import SpectralClustering
from textrank4zh import TextRank4Sentence
from LawsuitPrejudgment.src.civil.common.vectorization import vectorization
from LawsuitPrejudgment.src.civil.common import sucheng_extract, cosine_matrix, date_filter, num_filter


def get_data_from_database(anyous, num=100):
    data = []
    connect_big_data = pymysql.connect(host='rm-bp100iyd6uq3s5mtkbo.mysql.rds.aliyuncs.com',
                                       user='justice_user_03', password='justice_user_03_pd_!@#$',
                                       db='justice_big_data')
    for database, anyou_list in anyous.items():
        for anyou in anyou_list:
            sql1 = '''
                select f5 as serial, f12 as anyou, f7 as raw_content
                from justice_big_data.%s where f12 = '%s' limit %s
            ''' % (database, anyou, num)
            temp = pd.read_sql(sql1, con=connect_big_data)
            print("案由：", anyou, "从数据库的表取出了:", len(temp))
            data.append(temp)
    connect_big_data.close()

    data = pd.concat(data, sort=False)
    data = data.reset_index().drop('index', axis=1)
    data['sucheng'] = data['raw_content'].apply(sucheng_extract)
    data = data[data['sucheng'].apply(lambda x: False if x is None or x[0] is None or x[1] is None else True)]
    data['sucheng'] = data['sucheng'].apply(lambda x: x[0].replace(x[1], '').replace('事实和理由：', ''))
    data['sucheng'] = data['sucheng'].apply(lambda x: date_filter(x, '#'))
    data['sucheng'] = data['sucheng'].apply(lambda x: num_filter(x, '#'))

    def _repeat_clear(x):
        while '##' in x:
            x = x.replace('##', '#')
        return x

    data['sucheng'] = data['sucheng'].apply(_repeat_clear)
    data = data[data['sucheng'].str.len()>0]
    return data


def reason_extract(data, num):
    def _extract(text):
        delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n', ':', '：', ',', '，']
        tr4s = TextRank4Sentence(delimiters=delimiters)
        tr4s.analyze(text=text, lower=True, source='all_filters')
        items = sorted(tr4s.get_key_sentences(num=num), key=lambda x: x.index)
        if len(items) == 0:
            return None
        result = items[0].sentence
        for i, item in enumerate(items[1:]):
            if items[i].index == item.index - 1:
                result += '，' + item.sentence
            else:
                result += '。' + item.sentence
        result += '。'
        return result

    data['reason'] = data['sucheng'].apply(_extract)
    data = data.dropna()
    return data['reason'].drop_duplicates().values


def negative_reason_extract(data, num):
    def _extract(text):
        delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n', ':', '：', ',', '，']
        tr4s = TextRank4Sentence(delimiters=delimiters)
        tr4s.analyze(text=text, lower=True, source='all_filters')
        items = tr4s.get_key_sentences(num=num)
        if len(items) == 0:
            return None
        if len(items) == 1:
            return items[0].sentence + '。'
        for item in items[1:]:
            if len(re.findall('[不未没无非]', item.sentence))>0:
                if items[0].index<item.index:
                    return items[0].sentence + '，' + item.sentence + '。'
                else:
                    return item.sentence + '，' + items[0].sentence + '。'
        if items[0].index < items[1].index:
            return items[0].sentence + '，' + items[1].sentence + '。'
        else:
            return items[1].sentence + '，' + items[0].sentence + '。'

    data['reason'] = data['sucheng'].apply(_extract)
    data = data.dropna()
    return data


def get_sim_matrix(reasons):
    reason_vec = []
    for reason in reasons:
        reason_vec.append(vectorization(reason)[0][0])
    sim = cosine_matrix(np.mat(reason_vec), np.mat(reason_vec))
    sim[sim<0.95] = 0
    points = np.arange(len(reasons))[np.sum(np.array(sim), axis=1) > 1.5]
    single_points = np.arange(len(reasons))[np.sum(np.array(sim), axis=1) <= 1.5]
    sim = sim[:, points][points, :]
    single_reasons = reasons[single_points]
    reasons = reasons[points]
    return sim, reasons, single_reasons


def get_spectral_cluster(sim, n_clusters=30):
    cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    cluster.fit(sim)
    result = [[] for i in range(n_clusters)]
    for i, l in enumerate(cluster.labels_):
        result[l].append(i)
    return result


def main():
    anyous = {
        # 'case_list_original_hetong': ['网络服务合同纠纷', '娱乐服务合同纠纷', '凭样品买卖合同纠纷', '教育培训合同纠纷', '房屋买卖合同纠纷', '网络购物合同纠纷', '买卖合同纠纷',
        #                               '服务合同纠纷', '网络购物合同纠纷', '旅游合同纠纷', '凭样品买卖合同纠纷'],
        # 'case_list_original_qinquan': ['产品销售者责任纠纷', '产品生产者责任纠纷'],
        # 'case_list_original_labor_2': ['工伤保险待遇纠纷', '社会保险纠纷', '养老保险待遇纠纷', '失业保险待遇纠纷', '医疗保险待遇纠纷', '生育保险待遇纠纷'],
        # 'case_list_original_labor': ['确认劳动关系纠纷','劳务派遣合同纠纷','经济补偿金纠纷','追索劳动报酬纠纷','人事争议','劳动合同纠纷'],
        'case_list_original_hetong': ['房屋买卖合同纠纷','商品房销售合同纠纷','商品房预售合同纠纷','商品房委托代理销售合同纠纷','商品房预约合同纠纷'],
        # 'case_list_original_hetong': ['租赁合同纠纷','土地租赁合同纠纷','房屋租赁合同纠纷','车辆租赁合同纠纷','建筑设备租赁合同纠纷'],
    }
    data = get_data_from_database(anyous, 3000)
    print('get_data_from_database:', len(data))
    data = negative_reason_extract(data, num=10)
    reasons = data['reason'].drop_duplicates().values
    print('reason_extract:', len(reasons))
    sim, reasons, single_reasons = get_sim_matrix(reasons)
    print('get_sim_matrix:', len(single_reasons))
    cluster = get_spectral_cluster(sim, n_clusters=50)
    print('get_spectral_cluster:', len(cluster))
    result = []
    for c in cluster:
        for i in random.sample(c, min(40, len(c))):
            temp = data[data['reason']==reasons[i]]
            result.append([reasons[i], temp['sucheng'].values[0], temp['serial'].values[0]])
        result.append(['', '', ''])
    for reason in single_reasons:
        if random.randint(0,40)==0:
            temp = data[data['reason']==reason]
            result.append([reason, temp['sucheng'].values[0], temp['serial'].values[0]])
            result.append(['', '', ''])
    pd.DataFrame(result, columns=['情形', '诉称', '案号']).to_csv('../data/reason/租赁纠纷情形.csv', index=False)


if __name__=='__main__':
    main()

    # def _extract(text):
    #     delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n', ':', '：', ',', '，']
    #     tr4s = TextRank4Sentence(delimiters=delimiters)
    #     tr4s.analyze(text=text, lower=True, source='all_filters')
    #     items = sorted(tr4s.get_key_sentences(num=10), key=lambda x: x.index)
    #     if len(items) == 0:
    #         return None
    #     result = items[0].sentence
    #     for i, item in enumerate(items[1:]):
    #         if items[i].index == item.index - 1:
    #             result += '，' + item.sentence
    #         else:
    #             result += '。' + item.sentence
    #     result += '。'
    #     return result
    #
    # print(_extract('原告与李昌甲系兄弟关系。李昌甲与被告张湘雅系夫妻关系。被告张湘雅与被告李曲敏、李曲雯、张幽系母女关系。四被告是李昌甲的法定继承人。#，原告与李昌甲立下祖屋转让契约一份，李昌甲将坐落在永康市古山镇前黄村父母遗留给他的楼屋一间（地号：#-#B-#）折价人民币#元转让给原告，原告即时付款给李昌甲，李昌甲将该屋及相关证、契交付给原告。此后，该房屋一直由原告居住。#，原告请求四被告协助办理该房屋的过户登记手续，但被告张湘雅、李曲敏同意协助，被告李曲雯、张幽不同意协助，致使原告无法办理过户手续。审理过程中，原告申请撤回第二项诉讼请求。'))
