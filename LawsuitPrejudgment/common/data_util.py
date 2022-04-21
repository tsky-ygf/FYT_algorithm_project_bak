# -*- coding: utf-8 -*-
import re
import numpy as np
from config_loader import negative_word_list, segmentor, postagger


def cut_words(sentence):
    words = segmentor.segment(sentence)
    result = []
    i = 0
    while i < len(words)-1:
        if words[i] + words[i+1] in negative_word_list:
            result.append(words[i] + words[i+1])
            i += 2
        else:
            result.append(words[i])
            i += 1
    if i==len(words)-1:
        result.append(words[i])
    return result


poses = {'n', 'v', 'p', 'd', 'nh', 'q', 'a', 'b', 'wp', 'r'}


def pos_filter(content):
    words = list(segmentor.segment(content))
    postags = list(postagger.postag(words))
    content = ''.join([w for i, w in enumerate(words) if postags[i] in poses])
    content = re.sub('[，。；：][，。；：]', '，', content)
    if content[0] in ['，', '。', '；', '：']:
        content = content[1:]
    return content


def pos_cut(content):
    words = list(segmentor.segment(content))
    postags = list(postagger.postag(words))
    content = ' '.join([w for i, w in enumerate(words) if postags[i] in poses])
    content = re.sub('[，。；：][，。；：]', '，', content)
    if content[0] in ['，', '。', '；', '：']:
        content = content[1:]
    return content


def chinese_filter(content):
    return ''.join([t for t in re.findall('[\u4E00-\u9FA5]', content)])


def text_underline(text):
    return text.replace('【', '<font color="red">').replace('】', '</font>') #.replace('【', '<strong>').replace('】', '</strong>')


problem_suqiu_correct ={
    '婚姻家庭_履行赡养义务': ('婚姻家庭', '支付赡养费'),
    '婚姻家庭_探望权': ('婚姻家庭', '行使探望权'),
    '婚姻家庭_抚养权': ('婚姻家庭', '确认抚养权'),
    '劳动纠纷_确认存在劳动关系': ('劳动社保', '确认劳动关系'),
    '劳动纠纷_支付经济补偿金': ('劳动社保', '经济补偿金或赔偿金'),
    '劳动纠纷_支付赔偿金': ('劳动社保', '经济补偿金或赔偿金'),
    '劳动纠纷_支付二倍工资': ('劳动社保', '支付双倍工资'),
    '劳动纠纷_支付加班工资': ('劳动社保', '支付加班工资'),
    '社保纠纷_赔偿养老保险待遇损失': ('劳动社保', '养老保险待遇'),
    '社保纠纷_赔偿医疗保险待遇损失': ('劳动社保', '医疗保险待遇'),
    '社保纠纷_失业保险待遇': ('劳动社保', '失业保险待遇'),
    '社保纠纷_生育保险待遇': ('劳动社保', '生育保险待遇'),
    '工伤赔偿_支付工伤保险待遇': ('劳动社保', '工伤赔偿'),
    '提供劳务者受害责任纠纷_赔偿损失': ('劳动社保', '劳务损害赔偿'),
    '提供劳务者致害责任纠纷_赔偿损失': ('劳动社保', '劳务损害赔偿'),
}


def suqiu_correct(suqing, problem, suqiu):
    if suqing!=suqing or suqing is None:
        return problem, suqiu
    if problem + '_' + suqiu in problem_suqiu_correct:
        return problem_suqiu_correct[problem + '_' + suqiu]
    if problem == '婚姻家庭':
        if suqiu=='财产分割':
            if len(re.findall('产权|别墅|房', suqing))>0 \
                    and len(re.findall('(拆迁|征用|征收|搬家|附着物|动迁|安置)[^。；：，,;:]*(款|补助|补偿|奖励|费)', suqing))==0:
                return '婚姻家庭', '房产分割'
            if len(re.findall('贷款|债务|偿还|欠款|利息', suqing))>0:
                return '婚姻家庭', '夫妻共同债务'
        if suqiu=='抚养费':
            if len(re.findall('增加|追加|(支付[^。；：，,;:]*原告|向原告[^。；：，,;:]*支付).*(调整|变更)', suqing))>0 \
                    and len(re.findall('(增加|追加|变更|调整)[^。；：，,;:]*(诉讼请求|诉请|请求)', suqing))==0:
                return '婚姻家庭', '增加抚养费'
            if len(re.findall('减少|原告[^。；：，,;:]*支付.*(调整|变更)', suqing))>0 \
                    and len(re.findall('(减少|变更|调整)[^。；：，,;:]*(诉讼请求|诉请|请求)', suqing))==0:
                return '婚姻家庭', '减少抚养费'
            else:
                return '婚姻家庭', '支付抚养费'
    if problem == '继承问题':
        if suqiu=='确认遗嘱效力':
            if len(re.findall('遗嘱|遗书|遗言', suqing))>0:
                return '婚姻家庭', '确认遗嘱有效'
            else:
                return problem, suqiu
        return '婚姻家庭', '遗产继承'
    return problem, suqiu


def date_filter(inputs, replacement=''):
    inputs = re.sub('[\d一二三四五六七八九零]{2,4}年[\d一二三四五六七八九零]{1,2}月[\d一二三四五六七八九零]{1,2}日', replacement, inputs)
    inputs = re.sub('[\d一二三四五六七八九零]{2,4}年[\d一二三四五六七八九零]{1,2}月', replacement, inputs)
    inputs = re.sub('[\d一二三四五六七八九零]{1,2}月[\d一二三四五六七八九零]{1,2}日', replacement, inputs)
    inputs = re.sub('[\d一二三四五六七八九零]{4}年', replacement, inputs)
    return inputs


def question_filter(inputs):
    inputs = re.sub('([^。；，：,;:？！!?\s]*(怎么|如何|吗|么)[^。；，：,;:？！!?\s]*([。；，：,;:？！!?\s]|$))', '', inputs)
    return inputs


def word_filter(data, column, filter_words):
    """
    将指定列中包含过滤词的数据过滤掉
    :param data: 数据，格式为pandas.DataFrame
    :param column: 列名
    :param filter_words: 过滤词
    :return: 新DataFrame
    """

    def _match(x):
        result = []
        for t in re.split('[。，；：,;:]', x):
            result.append(1 if len(re.findall(word, t)) == 0 else 0)
        return bool(min(result))

    for word in filter_words:
        data = data[data[column].apply(lambda x: _match(x))]
    return data


def repeat_filter(data, column, extra_columns=None):
    """
    column剔除非中文以及原告被告姓名后，重复内容删除
    :param data: 数据，格式为pandas.DataFrame
    :return: 新DataFrame
    """

    def _extract(row):
        yuangao = row['yuangao'] if 'yuangao' in row else None
        beigao = row['beigao'] if 'beiago' in row else None
        sentences = re.sub('[^\u4E00-\u9FA5]', '', row[column])
        # 过滤开头的固定性语句
        sentences = re.sub('(原告.{0,6}诉称|原告.{0,6}向本院提出.{0,6}诉讼请求)', '', sentences)
        # 过滤原告
        if yuangao == yuangao and yuangao is not None:
            sentences = sentences.replace(yuangao, '')
        # 过滤被告
        if beigao == beigao and beigao is not None:
            sentences = sentences.replace(beigao, '')
        return sentences

    if len(data) == 0:
        return data
    data['temp'] = data.apply(_extract, axis=1)
    columns = ['temp'] if extra_columns is None else ['temp'] + extra_columns
    data = data.drop_duplicates(subset=columns)
    data = data.drop('temp', axis=1)
    return data


def num_filter(inputs, replacement=''):
    inputs = re.sub('[\d\.]', replacement, inputs)
    return inputs


def cosine_matrix(_matrixA, _matrixB):
    _matrixA_matrixB = _matrixA * _matrixB.transpose()
    _matrixA_norm = np.sqrt(np.multiply(_matrixA,_matrixA).sum(axis=1))
    _matrixB_norm = np.sqrt(np.multiply(_matrixB,_matrixB).sum(axis=1))
    return np.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())


if __name__=='__main__':
    print(num_filter('原告系被告东河区二里半街道办事处二里半社区卫生服务站职工。原告从到期间垫付了应由被告单位缴纳的养老保险金22434.9元。', '#'))
