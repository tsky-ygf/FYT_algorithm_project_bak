# -*- coding: utf-8 -*-
import os
import re
import json
import pandas as pd
import jieba
import numpy as np
from pyltp import Parser, Segmentor, Postagger, NamedEntityRecognizer

"""
特征匹配主要逻辑
"""

negative_word_list = ['不', '未', '没', '没有', '无', '非', '并未', '不再', '不能', '无法', '不足以', '不存在', '不能证明', '不认可','尚未'] # 否定词列表
negative_word = '(' + '|'.join(negative_word_list) + ')' # 否定词正则
no_negative_factor = ['被赡养方', '赡养方', '父方', '母方'] # 没有否定意义的特征

config_path = '../config/'
problem_suqius = {
    '婚姻家庭': [
        '离婚', '返还彩礼', '房产分割', '确认抚养权', '行使探望权', '支付抚养费', '增加抚养费', '减少抚养费',
        '支付赡养费', '确认婚姻无效', '财产分割', '夫妻共同债务', '确认遗嘱有效', '遗产继承'
    ]
}

problem_keyword_dict = {}
for problem, suqius in problem_suqius.items():
    df_keyword = pd.read_csv(config_path + problem + '关键词.csv', encoding='utf-8')
    df_keyword['keyword'] = '(' + df_keyword['keyword'] + ')'
    problem_keyword_dict[problem] = df_keyword['keyword'].groupby(df_keyword['factor'], sort=False).agg(lambda x: sorted(set(x)))


LTP_DATA_DIR = '../model/ltp'  # ltp模型目录的路径
seg_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性模型的路径
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体模型的路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

# 加载模型
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(seg_model_path, config_path + 'userdict.txt')  # 加载模型
postagger = Postagger()  # 初始化实例
postagger.load(pos_model_path)  # 加载模型
recognizer = NamedEntityRecognizer()  # 初始化实例
recognizer.load(ner_model_path)  # 加载模型
parser = Parser()  # 初始化实例
parser.load(par_model_path)


# 同义词典
with open(config_path + '同义词.json', 'r') as f:
    sim_dict = json.load(f)


def keyword_drop_duplicates(keyword_list):
    """
    剔除关键词列表里具有包含关系的关键词
    :param keyword_list: 关键词列表
    :return:
    """
    result_list = list(keyword_list)
    for i, k1 in enumerate(keyword_list):
        if k1 not in result_list or len(re.findall('(不|未|没有|无|非)', k1)) > 0:
            continue
        for j, k2 in enumerate(keyword_list[i + 1:]):
            if k2 not in result_list or len(re.findall('(不|未|没有|无|非)', k1)) > 0:
                continue
            if len(re.findall(k1, k2)) > 0:
                result_list.remove(k2)
            elif len(re.findall(k2, k1)) > 0:
                result_list.remove(k1)
                break
    return result_list


def keyword_split(keyword):
    """
    对关键词进行分词
    :param keyword:
    :return:
    """
    result_list = []
    r = ""
    for w in keyword:
        if len(re.findall('[\u4E00-\u9FA5]', w)) > 0:
            r += w
        else:
            if len(r) > 0:
                result_list.append(r)
                r = ""
            result_list.append(w)
    if len(r) > 0:
        result_list.append(r)
    result = []
    for r in result_list:
        result += list(segmentor.segment(r))
    return result


def keyword_list_expand(keyword_list, keep_org=True):
    """
    基于同义词表扩充关键词, 保持原关键词排序一致
    :param keyword_list: 关键词列表
    :param keep_org: 是否保留原有词
    :return:
    """
    if keyword_list != keyword_list or keyword_list is None:
        return keyword_list
    return_str = False
    if isinstance(keyword_list, str):
        keyword_list = [keyword_list]
        return_str = True
    result_list = []
    for keyword in keyword_list:
        ks = keyword_split(keyword)
        rs = list(ks)
        for i, k in enumerate(ks):
            if k in sim_dict:
                if keep_org:
                    rs[i] = '(' + '|'.join([k] + sim_dict[k]) + ')'
                else:
                    rs[i] = '(' + '|'.join(sim_dict[k]) + ')'
        result_list.append(''.join(rs))
    if return_str:
        result_list = result_list[0]
    return result_list


def written_correct(inputs):
    return inputs.replace('并末', '并未').replace('无儿无女', '无儿无女，')


def single_case_match(inputs, problem, use_tongyici=False):
    """
    匹配单独的一个输入(多个句子)，得到特征
    :param inputs: 用户输入的语句
    :return: 返回匹配到的特征和对应的短句
    """
    factor_keywords = problem_keyword_dict[problem]
    sentence_factor = {}
    if inputs is None or len(inputs.strip()) == 0:
        return sentence_factor

    sentence_list = [s for s in re.split('[。；，：,;:？！!?\s]', written_correct(inputs)) if len(s) > 0]
    sentence_list.reverse()

    for factor, keyword_list in factor_keywords.items():
        if use_tongyici:
            ks = keyword_list_expand(keyword_list)
        else:
            ks = keyword_list
        for keyword in ks:
            # print(keyword)
            if '方' in keyword:
                keyword = '(' + keyword + '|' + keyword.replace('方', '⽅') + ')'
            if '父' in keyword:
                keyword = '(' + keyword + '|' + keyword.replace('父', '⽗') + ')'
            keyword = '(' + keyword + ')'
            flag = 0
            if len(re.findall(keyword.replace('.*', '[^。；，：,;:\s]*'), inputs)) == 0:
                continue
            for index, sentence in enumerate(sentence_list):  # 针对每个句子去匹配
                flag = match_sentence_by_keyword(sentence, keyword)
                if flag==1 or flag==-1:
                    # print(keyword)
                    sentence_factor[factor] = [sentence, 1, index] if factor in no_negative_factor else [sentence, flag, index]
                    break
            if flag==1 or flag==-1:
                break

    sentence_factor = sorted(sentence_factor.items(), key=lambda x: x[1][2], reverse=True)
    sentence_factor = {sf[0]: [sf[1][0], sf[1][1]] for sf in sentence_factor}
    return sentence_factor


def match_sentence_by_keyword(sentence, key_word):
    """
    匹配关键词(key_word)。返回正向匹配（1）、负向匹配（-1）或没有匹配（0）
    :param sentence: 一整句话。如"xxx1，xxx2，xxx3，xxxx4。"
    :param key_word: 子案由因子下面的一个key_word
    :return:
    """
    # if sentence.startswith('如') or sentence.startswith('若') or len(re.findall('[如若]' + key_word, sentence)) > 0:
    #     return 0
    # if len(re.findall('(约定|要求|认为|应当|应该|如果).*' + key_word, sentence)) > 0:
    #     return 0
    flag_positive = len(re.findall(key_word, sentence)) > 0
    # 2.匹配负向
    if not flag_positive:
        return 0
    if len(re.findall(negative_word + '.*' + key_word, sentence)) > 0 \
            or len(re.findall(key_word + '.*' + negative_word, sentence)) > 0:
        if negative_match(key_word, sentence) == -1:
            return -1
    if len(re.findall(negative_word, key_word)) == 0 and '.*' in key_word:
        kl = key_word.split('.*')
        for i in range(len(kl) - 1):
            kw = '.*'.join([k + '.*' + negative_word if i == j else k for j, k in enumerate(kl)])
            if len(re.findall(kw, sentence)) > 0 and negative_match(key_word, sentence) == -1:
                return -1
    return 1


def negative_match(key_word, sentence):
    pattern = re.findall(key_word, sentence)[0][0]
    words = list(segmentor.segment(sentence))  # 分词 元芳你怎么看
    postags = list(postagger.postag(words))  # 词性标注
    arcs = parser.parse(words, postags)  # 句法分析
    for i in range(len(arcs)):
        if words[i] in negative_word_list and words[i] not in key_word and arcs[i].relation != 'HED':
            if words[arcs[i].head - 1] in pattern:
                return -1
            if arcs[arcs[i].head - 1].relation == 'VOB' and words[arcs[arcs[i].head - 1].head - 1] in pattern:
                return -1
        if words[i] in negative_word_list and words[i] not in key_word and arcs[i].relation == 'HED':
            for k, arc in enumerate(arcs):
                if arc.relation in ['SBV', 'VOB'] and arc.head == i + 1 and (words[k] in pattern or words[k] in ['证据', '证明']):
                    return -1
        if words[i] in negative_word_list and words[i] not in key_word and arcs[arcs[i].head - 1].relation == 'HED':
            for k, arc in enumerate(arcs):
                if arc.relation in ['SBV', 'VOB'] and arc.head == arcs[i].head and (words[k] in pattern or words[k] in ['证据', '证明']):
                    return -1
        if words[i] in pattern and words[arcs[i].head - 1] not in key_word and words[arcs[i].head - 1] in negative_word_list:
            return -1
    return 1


def get_feature_list(sentence, problem):
    sentence_factor = single_case_match(sentence, problem, use_tongyici=False)
    feature_list = np.zeros(len(problem_keyword_dict[problem]), dtype=int)
    factors = problem_keyword_dict[problem].index.tolist()
    for factor, flag in sentence_factor.items():
        feature_list[factors.index(factor)] = flag[1]
    return feature_list


if __name__=='__main__':
    print(get_feature_list('小孩两岁', '婚姻家庭'))
