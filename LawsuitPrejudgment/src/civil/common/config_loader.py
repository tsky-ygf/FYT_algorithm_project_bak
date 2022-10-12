# -*- coding: utf-8 -*-
from asyncio.log import logger
from xmindparser import xmind_to_dict
import pandas as pd
import re
import os
from itertools import permutations
from pyltp import Parser, Segmentor, Postagger, NamedEntityRecognizer



########################################################################################################################
#
# 路径配置
#
########################################################################################################################
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.constants import KNOWLEDGE_FILE_PATH, MODEL_FILE_PATH

config_path = KNOWLEDGE_FILE_PATH
model_path = MODEL_FILE_PATH



########################################################################################################################
#
# 否定词配置
#
########################################################################################################################

negative_word_list = ['不', '未', '没', '没有', '无', '非', '并未', '不再', '不能', '无法', '不足以', '不至于', '不存在',
                      '不能证明', '不认可','尚未', '不行', '不到', '不满', '未满', '未到', '没到', '没满', '没法'] # 否定词列表
negative_word = '(' + '|'.join(negative_word_list) + ')' # 否定词正则
no_negative_factor = []



########################################################################################################################
#
# 句法分析模型
#
########################################################################################################################

LTP_DATA_DIR = model_path + 'ltp'  # ltp模型目录的路径
seg_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性模型的路径
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体模型的路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

# 加载模型

# 原初始化方式，在python3.6下运行
# segmentor = Segmentor()
# segmentor.load_with_lexicon(seg_model_path, config_path + 'negative') # TODO:应该是把否定词加入了词汇表
# postagger = Postagger()  # 初始化实例
# postagger.load(pos_model_path)  # 加载模型
# recognizer = NamedEntityRecognizer()  # 初始化实例
# recognizer.load(ner_model_path)  # 加载模型
# parser = Parser()  # 初始化实例
# parser.load(par_model_path)
segmentor = Segmentor(seg_model_path, config_path + 'negative')
postagger = Postagger(pos_model_path)
recognizer = NamedEntityRecognizer(ner_model_path)
parser = Parser(par_model_path)


########################################################################################################################
#
# 纠纷类型和诉求配置
#
########################################################################################################################

df_suqiu = pd.read_csv(config_path + '诉求配置.csv', encoding='utf-8')
df_suqiu['user_ps'] = df_suqiu['user_problem'] + '_' + df_suqiu['user_suqiu']
df_suqiu['logic_ps'] = df_suqiu['problem'] + '_' + df_suqiu['logic_suqiu']
df_suqiu['prob_ps'] = df_suqiu['problem'] + '_' + df_suqiu['prob_suqiu']
df_suqiu = df_suqiu[df_suqiu['status']==1]

# 用户看到的纠纷类型和诉求
temp = df_suqiu[['user_problem', 'user_suqiu']].drop_duplicates()
user_ps = temp['user_suqiu'].groupby(temp['user_problem'], sort=False).agg(lambda x: list(x))

# 用户纠纷诉求与评估理由纠纷诉求的对应关系
user_ps2logic_ps = df_suqiu['logic_ps'].groupby(df_suqiu['user_ps'], sort=False).agg(lambda x: list(x))

# 评估理由纠纷诉求
logic_ps = df_suqiu['logic_suqiu'].groupby(df_suqiu['problem'], sort=False).agg(lambda x: list(x))

# 概率纠纷诉求
prob_ps = df_suqiu['prob_suqiu'].groupby(df_suqiu['problem'], sort=False).agg(lambda x: list(x))

# 评估理由诉求前提
def _precondition_process(x):
    result = []
    for s in x.split('|'):
        r = {}
        for t in s.split('&'):
            r[t.split(':')[0]] = int(t.split(':')[1])
        result.append(r)
    return result

temp = df_suqiu[~df_suqiu['logic_precondition'].isna()]
logic_ps_prediction = temp['logic_precondition'].groupby(df_suqiu['logic_ps'], sort=False).agg(lambda x: list(x)[0])
logic_ps_prediction = logic_ps_prediction.apply(_precondition_process)

# 评估理由诉求不满足前提的结论
temp = df_suqiu[~df_suqiu['logic_result'].isna()]
logic_ps_result = temp['logic_result'].groupby(temp['logic_ps'], sort=False).agg(lambda x: list(x)[0])

# 评估理由诉求结论展示条件
temp = df_suqiu[~df_suqiu['logic_condition'].isna()]
logic_ps_condition = temp['logic_condition'].groupby(temp['logic_ps'], sort=False).agg(lambda x: list(x)[0])

# 评估理由诉求默认特征
temp = df_suqiu[~df_suqiu['logic_suqiu_factor'].isna()]
# TODO: 这行代码，在python3.6(pandas1.1.5)环境下正常，在python3.9(pandas1.5.0)环境下异常
logic_ps_factor = temp['logic_suqiu_factor'].groupby(temp['logic_ps'], sort=False).agg(lambda x: list(x)[0])
logic_ps_factor = logic_ps_factor.apply(lambda x: {s.split(':')[0]: int(s.split(':')[1]) for s in x.split(';')})

# 评估理由诉求默认法律建议
logic_ps_advice = df_suqiu['logic_suqiu_advice'].groupby(df_suqiu['logic_ps'], sort=False).agg(lambda x: list(x)[0])

# 评估理由诉求额外证据
df_suqiu['logic_suqiu_proof'] = df_suqiu['logic_suqiu_proof'].fillna('')
logic_ps_proof = df_suqiu['logic_suqiu_proof'].groupby(df_suqiu['logic_ps'], sort=False).agg(lambda x: list(x)[0])

# 概率诉求相关描述
prob_ps_desc = df_suqiu['prob_suqiu_desc'].groupby(df_suqiu['prob_ps'], sort=False).agg(lambda x: list(x)[0])

# 概率诉求转换关系
def _repeat_filter(lt):
    result = []
    for item in lt:
        if item not in result:
            result.append(item)
    return result

user_ps2prob_ps = df_suqiu['prob_ps'].groupby(df_suqiu['user_ps'], sort=False).agg(lambda x: list(x))
user_ps2prob_ps = user_ps2prob_ps.apply(_repeat_filter)
prob_ps2logic_ps = df_suqiu['logic_ps'].groupby(df_suqiu['prob_ps'], sort=False).agg(lambda x: list(x))
prob_ps2logic_ps = prob_ps2logic_ps.apply(_repeat_filter)


########################################################################################################################
#
# 标注关键词
#
########################################################################################################################

label_keyword = pd.read_csv(config_path + '标签关键词.csv', encoding='utf-8')

problem_anyou_list = label_keyword['anyou'].groupby(label_keyword['problem']).agg(lambda x: '|'.join(list(x)))
problem_anyou_list = problem_anyou_list.str.split('|')
problem_anyou_list = problem_anyou_list.apply(lambda x: list(set(x)))

label_anyou = label_keyword['anyou'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('anyou')
label_keyword = label_keyword.drop('anyou', axis=1).join(label_anyou)
label_keyword['ps'] = label_keyword['problem']+'_'+label_keyword['suqiu']
anyou_ps_dict = label_keyword['ps'].groupby(label_keyword['anyou']).agg(lambda x: sorted(set(x)))
anyou_db_dict = label_keyword['database'].groupby(label_keyword['anyou']).agg(lambda x: list(x)[0])

ps_positive_keyword = label_keyword['positive_keywords'].groupby(label_keyword['ps']).agg(lambda x: '('+'|'.join(set(x))+')')

temp = label_keyword[~label_keyword['negative_keywords'].isna()]
ps_negative_keyword = temp['negative_keywords'].groupby(temp['ps']).agg(lambda x: '('+'|'.join(set(x))+')')

temp = label_keyword[~label_keyword['panjue_keywords'].isna()]
ps_panjue_keyword = temp['panjue_keywords'].groupby(temp['ps']).agg(lambda x: '('+'|'.join(set(x))+')')


########################################################################################################################
#
# 逻辑树
#
########################################################################################################################

logic_dict = {}
for problem, suqius in logic_ps.items():
    for suqiu in suqius:
        logic_dict[problem + '_' + suqiu] = xmind_to_dict(config_path + problem + '/' + problem + '_' + suqiu + '.xmind')[0]['topic']



########################################################################################################################
#
# 特征相关配置
#
########################################################################################################################

question_factor_dict = {}  # 问题对应的特征
question_multiple_dict = {}  # 多选问题
question_answer_dict = {}  # 问题对应的所有答案
question_father_dict = {}  # 父特征替换后的问题对应的原问题
factor_question_dict = {}  # 特征对应的问题
factor_answer_dict = {}  # 特征对应的答案
factor_proof_dict = {}  # 特征对应的证据
factor_unimportant_dict = {}  # 不重要的特征（路径不支持继续提问）
factor_pre_dict = {}  # 特征对应的前置特征
factor_son_dict = {}  # 特征对应的子特征
factor_exclusion_dict = {}  # 特征互斥关系
factor_replace_dict = {}  # 用于在评估理由中展示的特征

for problem, suqius in logic_ps.items():
    df_factor = pd.read_csv(config_path + problem + '/' + problem + '特征.csv', encoding='utf-8')
    df_factor['question_answer'] = df_factor['question'] + ':' + df_factor['answer'].str.replace('|', ';')
    for suqiu in suqius:
        df = df_factor[df_factor['suqiu'] == suqiu]

        # 问题对应的特征
        question_factor_dict[problem + '_' + suqiu] = df['factor'].groupby(df['question_answer'], sort=False).agg(lambda x: sorted(set(x)))

        # 多选问题
        temp = df[df['multiple_choice'] == 1]
        question_multiple_dict[problem + '_' + suqiu] = temp['question_answer'].values

        # 问题对应的所有答案
        question_answer_dict[problem + '_' + suqiu] = df['answer'].groupby(df['question_answer'], sort=False).agg(lambda x: list(x)[0].split('|'))

        # 父特征替换后的问题对应的原问题
        temp = df[df['question_answer'].str.contains('[FF]')].drop_duplicates(subset=['question_answer'])
        temp_dict = {}
        for index, row in temp.iterrows():
            for father in row['pre_factor'].split('|'):
                temp_dict[row['question_answer'].replace('[FF]', father)] = row['question_answer']
        question_father_dict[problem + '_' + suqiu] = temp_dict

        # 特征对应的问题
        factor_question_dict[problem + '_' + suqiu] = df['question_answer'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0])

        # 特征对应的答案
        factor_answer_dict[problem + '_' + suqiu] = df['factor_answer'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0])

        # 特征对应的证据
        temp = df[~df['proof'].isna()]
        factor_proof_dict[problem + '_' + suqiu] = temp['proof'].groupby(temp['factor'], sort=False).agg(lambda x: list(x)[0])

        # 不重要的特征（路径不支持继续提问）
        temp = df[df['factor_unimportant']==1]
        factor_unimportant_dict[problem + '_' + suqiu] = temp['factor'].values

        # 特征对应的前置特征
        temp = df[~df['pre_factor'].isna()]
        factor_pre_dict[problem + '_' + suqiu] = temp['pre_factor'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0].split('|'))

        # 特征对应的子特征
        temp = df[~df['son_factor'].isna()]
        factor_son_dict[problem + '_' + suqiu] = temp['son_factor'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0].split('|'))

        # 特征互斥关系
        temp = df[~df['group_id'].isna()]
        factor_exclusion = temp['factor'].groupby(temp['group_id']).agg(lambda x: sorted(set(x)))
        factor_exclusion_dict[problem + '_' + suqiu] = {f: [t for t in v if t != f] for v in factor_exclusion.values for f in v}

        # 用于在评估理由中展示的特征
        factor_replace_dict[problem + '_' + suqiu] = df[df['replace'] == 1]['factor'].values



########################################################################################################################
#
# 特征关键词配置
#
########################################################################################################################

def _encoding_correct(keyword):
    if '方' in keyword:
        keyword = '(' + keyword + ')|(' + keyword.replace('方', '⽅') + ')'
    if '父' in keyword:
        keyword = '(' + keyword + ')|(' + keyword.replace('父', '⽗') + ')'
    keyword = '(' + keyword + ')'
    return keyword


df_keyword = pd.read_csv(config_path + '句式关键词.csv', encoding='utf-8')
sentence_keyword_dict = df_keyword['keyword'].groupby(df_keyword['sentence'], sort=False).agg(lambda x: list(x)[0])
sentence_keyword_dict = sentence_keyword_dict.apply(_encoding_correct)
sentence_has_neg_dict = df_keyword['has_negative'].groupby(df_keyword['sentence'], sort=False).agg(lambda x: list(x)[0])

# 对于多个【[xx][yy][zz]】和【[xx][yy]】【[zz]】相同，输出[fx, fy, fz]
def get_factor_base_keyword(sentences):
    keyword_list = []
    for sentence in sentences.split('|'):
        keywords = []
        for sub_sentence in re.findall('【([^【】]*)】', sentence):
            for st in re.findall('\[([^\[\]]*)\]', sub_sentence):
                if st=='没有':
                    continue
                if st in sentence_keyword_dict:
                    keywords.append(sentence_keyword_dict[st])
                else:
                    logger.warning(st)
                    # keywords.append("("+st+")")
        keyword_list.append(keywords)
    return keyword_list

# 对于【[x][y]】【[z]】输出(fx*fy)*zz，fz*(fx*fy)分别加入返回list
def get_factor_positive_keyword(sentences):
    keyword_list = []
    for sentence in sentences.split('|'):
        keywords = []
        for sub_sentence in re.findall('【([^【】]*)】', sentence):
            keyword = []
            for sub_st in re.findall('\[([^\[\]]*)\]', sub_sentence):
                if sub_st in sentence_keyword_dict:
                    keyword.append(sentence_keyword_dict[sub_st].replace('.*', '[^。；，：,;:？！!?\s]*'))
                else:
                    logger.warning(sub_st)
                    # keyword.append('(' + sub_st + ')')
            keywords.append('(' + '[^。；，：,;:？！!?\s]*'.join(keyword) + ')')

        for k in permutations(keywords):
            keyword_list.append('(' + '[^。？！?!]*?'.join(k) + ')')
    return keyword_list


def get_factor_negative_keyword(sentences):
    keyword_list = []
    for sentence in sentences.split('|'):
        sentence = re.findall('【([^【】]*)】', sentence)
        negative_sentence = []
        for i in range(len(sentence)):
            if '[没有]' not in sentence[i]:
                for sub_st in re.findall('\[([^\[\]]*)\]', sentence[i]):
                    if sub_st in sentence_has_neg_dict and sentence_has_neg_dict[sub_st]==1:
                        temp = sentence.copy()
                        temp[i] = temp[i].replace('['+sub_st+']', '[没有]['+sub_st+']')
                        negative_sentence.append(temp)
            elif '[没有]' in sentence[i]:
                for sub_st in re.findall('\[([^\[\]]*)\]', sentence[i]):
                    if sub_st in sentence_has_neg_dict and sentence_has_neg_dict[sub_st]==1:
                        temp = sentence.copy()
                        temp[i] = temp[i].replace('[没有]', '')
                        negative_sentence.append(temp)
                        break

        keywords = []
        for sentence in negative_sentence:
            keyword = []
            for sub_sentence in sentence:
                k = []
                for sub_st in re.findall('\[([^\[\]]*)\]', sub_sentence):
                    if sub_st in sentence_keyword_dict:
                        k.append(sentence_keyword_dict[sub_st].replace('.*', '[^。；，：,;:？！!?\s]*'))
                    else:
                        logger.warning(sub_st)
                        # k.append('('+sub_st+')')
                if len(k) > 0:
                    keyword.append('(' + '[^。；，：,;:？！!?\s]*'.join(k) + ')')
            if len(keyword) > 0:
                for k in permutations(keyword):
                    keywords.append('(' + '[^。？！?!]*?'.join(k) + ')')
        keyword_list += keywords
    return keyword_list


problem_bkw_dict = {}
problem_pkw_dict = {}
problem_nkw_dict = {}
suqiu_bkw_dict = {}
suqiu_pkw_dict = {}
suqiu_nkw_dict = {}
for problem, suqius in logic_ps.items():
    df_sentence = pd.read_csv(config_path + problem + '/' + problem + '特征.csv', encoding='utf-8')
    df_sentence['bkw'] = df_sentence['sentences'].apply(get_factor_base_keyword)
    df_sentence['pkw'] = df_sentence['sentences'].apply(get_factor_positive_keyword)
    df_sentence['nkw'] = df_sentence['sentences'].apply(get_factor_negative_keyword)
    problem_bkw_dict[problem] = df_sentence['bkw'].groupby(df_sentence['factor'], sort=False).agg(lambda x: list(x)[0])
    problem_pkw_dict[problem] = df_sentence['pkw'].groupby(df_sentence['factor'], sort=False).agg(lambda x: list(x)[0])
    problem_nkw_dict[problem] = df_sentence['nkw'].groupby(df_sentence['factor'], sort=False).agg(lambda x: list(x)[0])
    for suqiu in suqius:
        df = df_sentence[df_sentence['suqiu']==suqiu]
        suqiu_bkw_dict[problem + '_' + suqiu] = df['bkw'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0])
        suqiu_pkw_dict[problem + '_' + suqiu] = df['pkw'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0])
        suqiu_nkw_dict[problem + '_' + suqiu] = df['nkw'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0])



########################################################################################################################
#
# 候选问答配置
#
########################################################################################################################

candidate_question_dict = {}
candidate_factor_dict = {}
candidate_multiple_dict = {}
for problem, suqius in logic_ps.items():
    # 问题配置表
    df_question = pd.read_csv(config_path + problem + '/' + problem + '候选问答.csv', encoding='utf-8')
    df_question['question_answer'] = df_question['question'] + ':' + df_question['answer'].str.replace('|', ';')
    for suqiu in suqius:
        df = df_question[df_question['suqiu'] == suqiu]
        candidate_question_dict[problem + '_' + suqiu] = df['question_answer'].drop_duplicates().values[0]
        candidate_factor_dict[problem + '_' + suqiu] = df['factor_answer'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0])
        # 多选问题
        temp = df[df['multiple_choice'] == 1]
        candidate_multiple_dict[problem + '_' + suqiu] = temp['question_answer'].values


########################################################################################################################
#
# 特征权重配置
#
########################################################################################################################

factor_weight_dict = {}
for problem, suqius in logic_ps.items():
    df_weight = pd.read_csv(config_path + problem + '/' + problem + '因子权重.csv', encoding='utf-8')
    for suqiu in suqius:
        df = df_weight[(df_weight['suqiu']==suqiu)]
        # 特征对应频率
        factor_weight_dict[problem + '_' + suqiu] = df['weight'].groupby(df['factor'], sort=False).agg(lambda x: list(x)[0])