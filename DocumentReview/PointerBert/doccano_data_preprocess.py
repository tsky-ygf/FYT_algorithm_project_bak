#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/22 13:29
# @Author  : Czq
# @File    : doccano_data_preprocess.py
# @Software: PyCharm
import json
import os


def split_text(file, to_file):
    w = open(to_file, 'w', encoding='utf-8')

    window = 510
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            text = line['text'].replace('\xa0',' ')
            used = []
            entities = line['entities']
            for i in range(0, len(text), 400):
                entities_new = []
                bias = i
                text_split = text[i:i+window]
                for entity in entities:
                    if i<= entity['start_offset']<entity['end_offset']<i+window:
                        entities_new.append({'label':entity['label'],
                                             'start_offset':entity['start_offset']-bias, 'end_offset':entity['end_offset']-bias})
                        used.append(entity)
                w.write(json.dumps({'id':line['id'],
                                    'text': text_split,
                                    'entities': entities_new
                                    },ensure_ascii=False)+'\n')

            if len(entities) > len(used):
                print('true', len(entities))
                print('used', len(used))
                print(line)
    w.close()


def divided_train_dev(file, to_path):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    l = len(data)
    p = int(l*0.81)
    train_data = data[:p]
    dev_data = data[p:]

    with open(os.path.join(to_path,'train_split.json'), 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line)
    with open(os.path.join(to_path, 'dev_split.json'),'w', encoding='utf-8') as f:
        for line in dev_data:
            f.write(line)

# 转换为cluener的数据格式
def convert_format(in_file, out_file):

    w = open(out_file, 'w', encoding='utf-8')

    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            text = line['text']
            entities = line['entities']
            labels = dict()
            for entity in entities:
                entity_text = text[entity['start_offset']:entity['end_offset']]
                labels[entity['label']] = {entity_text: [[entity['start_offset'], entity['end_offset']-1]]}

            res = {'text':text, 'label':labels}
            w.write(json.dumps(res, ensure_ascii=False)+"\n")

    w.close()

def t():
    j = {'id': 216464, 'text': '借条借款人：潘红梅身份证号：320405197610268726手机（可接收短信）：15338615075微信号：15338615075联系地址：新疆维吾尔自治区哈尔滨市永川张街B座246807号保证人：郑楠身份证号：620500198503103968手机（可接收短信）：13269912911微信号：13269912911联系地址：西藏自治区成都县徐汇台北路X座980618号\xa0借款人向黄杰（身份证号：141034197310206846，即出借人）借到现金人民币（大写）拾万圆整（￥\xa0100000\xa0\xa0）下称“借款金额”，并承诺如下事项：一、借款用途：\xa0家庭生活支出\xa0。二、利率为：月利率为\xa02\xa0\xa0%；利息支付时间：每年\xa02\xa0月之前支付上年度利息。三、还款期限：\xa0\xa02022\xa0年3月1日前还款。超期还款的，利息按约定利率的\xa030\xa0%计算。四、出借人放款方式按下列第\xa0\xa01\xa0种方式处理：（1）借款人确认：签署本借条时，借款人已经收到出借人以现金支付的全部借款金额；（2）出借人可通过下列账户向借款人支付借款金额：户名：黄杰开户行：中国招商银行商务支行账号：38870768069751五、其它：1.如借款人未按时还本付息发生争议的，由出借人所在地人民法院管辖。2.因催收或诉讼产生的律师费、调查费、公证费、诉讼费、保全费、公告费等费用均由借款人承担。3.如有保证人，则保证期间为自还款期限届满之日起三年。4.前述借款人、保证人地址和联系方式同时作为有效司法送达方式。\xa01签署时间：\xa02022\xa0年\xa02月1日\xa0借款人签字：\xa0保证人签字：\xa0', 'entities': [{'id': 35208, 'label': '标题', 'start_offset': 0, 'end_offset': 2}, {'id': 35217, 'label': '乙方', 'start_offset': 6, 'end_offset': 9}, {'id': 35219, 'label': '乙方身份证号/统一社会信用代码', 'start_offset': 14, 'end_offset': 32}, {'id': 35220, 'label': '乙方联系方式', 'start_offset': 42, 'end_offset': 53}, {'id': 35221, 'label': '乙方联系方式', 'start_offset': 73, 'end_offset': 98}, {'id': 35222, 'label': '甲方', 'start_offset': 195, 'end_offset': 197}, {'id': 35223, 'label': '甲方身份证号/统一社会信用代码', 'start_offset': 203, 'end_offset': 221}, {'id': 35224, 'label': '签订日期', 'start_offset': 641, 'end_offset': 654}, {'id': 35225, 'label': '金额', 'start_offset': 229, 'end_offset': 253}, {'id': 35226, 'label': '开户名称', 'start_offset': 460, 'end_offset': 462}, {'id': 35227, 'label': '开户行', 'start_offset': 466, 'end_offset': 476}, {'id': 35228, 'label': '账号', 'start_offset': 479, 'end_offset': 493}], 'relations': []}
    text = j['text']
    text = text.replace('\xa0',' ')
    entities = j['entities']
    print(list(text))
    for ent in entities:
        start = ent['start_offset']
        end = ent['end_offset']
        print(text[start:end])


def convert_format_bmes():
    labels = ['Title', 'JIA', '']
    pass

if __name__ == "__main__":
    split_text('data/data_src/old/origin_oldall.json', 'data/data_src/old/origin_oldall_split.json')
    divided_train_dev('data/data_src/old/origin_oldall_split.json', 'data/data_src/old/')
    # t()
    # convert_format('data/data_src/new/dev_300.json', 'data/data_src/cluener_format/dev_300.json')
    # convert_format('data/data_src/new/train_300.json', 'data/data_src/cluener_format/train_300.json')
    # convert_format_bmes()