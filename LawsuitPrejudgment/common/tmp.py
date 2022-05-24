import pandas as pd
config_path = '../config/'

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