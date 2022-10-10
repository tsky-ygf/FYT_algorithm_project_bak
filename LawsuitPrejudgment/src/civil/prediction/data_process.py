# -*- coding: utf-8 -*-
import pymysql
import random
from LawsuitPrejudgment.src.civil.common import *
import configparser

config = configparser.ConfigParser()
config.read('../main/db_config.ini')
big_data_host = config.get('bigData', 'host')
big_data_port = config.get('bigData', 'port')
big_data_user = config.get('bigData', 'user')
big_data_password = config.get('bigData', 'password')
big_data_db = config.get('bigData', 'db')

def export_consult_data():
    connect_big_data = pymysql.connect(host='rm-bp18g0150979o8v4t.mysql.rds.aliyuncs.com',
                                       user='justice_user_01', password='justice_user_01_pd_!@#$',
                                       db='justice')
    sql1 = """
        select case_reason_name, case_fact_desc, date_format(gmt_create, '%Y%m%d') as date from justice.law_consult_record 
        where gmt_create > '2019-08-05' and gmt_create < '2019-08-12' 
        and case_fact_desc not like '2008年，我在周至邮政储蓄办理了存款业务%' and case_reason_name != '刑事'
    """
    data1 = pd.read_sql(sql1, con=connect_big_data)
    print("从数据库的表取出了:", len(data1))

    sql2 = """
        select question, answer, date_format(create_time, '%Y%m%d') as date from justice.law_advisory_record 
        where create_time > '2019-08-05' and create_time < '2019-08-12'
    """
    data2 = pd.read_sql(sql2, con=connect_big_data)
    print("从数据库的表取出了:", len(data2))

    connect_big_data.close()

    data1 = data1.drop_duplicates()
    data2 = data2.drop_duplicates()
    data1.to_csv('../data/评估.csv', index=False, encoding='utf-8')
    data2.to_csv('../data/咨询.csv', index=False, encoding='utf-8')


def export_test_data(problem_list):
    problem_cases = {
        '婚姻继承': ['婚姻家庭', '继承问题'],
        '劳动社保': ['劳动纠纷', '社保纠纷', '劳务纠纷', '工伤赔偿', '提供劳务者受害责任纠纷', '提供劳务者致害责任纠纷', '义务帮工人受害责任纠纷'],
        '借贷纠纷': ['借贷纠纷', '金融借款合同纠纷'],
        '交通事故': ['交通事故']
    }
    result = []
    connect_big_data = pymysql.connect(host='rm-bp18g0150979o8v4t.mysql.rds.aliyuncs.com',
                                       user='justice_user_01', password='justice_user_01_pd_!@#$',
                                       db='justice')
    for problem in problem_list:
        for case in problem_cases[problem]:
            sql = """
                select case_reason_name, case_fact_desc from justice.law_consult_record
                where case_reason_name = '%s' order by gmt_create desc limit 2000
            """%(case)
            data = pd.read_sql(sql, con=connect_big_data)
            data['problem'] = problem
            print("case：", case, "从数据库的表取出了:", len(data))
            data = data.sample(frac=1)[:20]
            result.append(data)
    result = pd.concat(result, sort=False)
    result = result.reset_index().drop('index', axis=1)
    connect_big_data.close()
    result[['problem','case_reason_name','case_fact_desc']].to_csv('../data/test20190815.csv', index=False, encoding='utf-8')


def export_data_from_database(problem_database, problem_anyous):
    for problem, anyous in problem_anyous.items():
        data = []
        # connect_big_data = pymysql.connect(host='rm-bp100iyd6uq3s5mtkbo.mysql.rds.aliyuncs.com',
        #                                    user='justice_user_03', password='justice_user_03_pd_!@#$',
        #                                    db='justice_big_data')
        connect_big_data = pymysql.connect(host='192.168.1.253', port=3366,
                                           user='justice_user_03', password='justice_user_03_pd_!@#$',
                                           db='justice_big_data')
        for anyou in anyous:
            sql1 = '''
                select f5 as serial, f12 as reference, f40 as litigant, f30 as result, f13 as viewpoint, f44 as fact
                from justice_big_data.%s
                where f30 is not null and f40 is not null and f13 is not null and f44 is not null and f12 = '%s' limit 20
            ''' % (problem_database[problem], anyou)
            temp = pd.read_sql(sql1, con=connect_big_data)
            print("案由：", anyou, "从数据库的表取出了:", len(temp))
            data.append(temp)
        connect_big_data.close()

        data = pd.concat(data, sort=False)
        data = data.reset_index().drop('index', axis=1)
        data['item_id'] = np.arange(len(data))
        data['item_title'] = np.arange(len(data))
        data = data[['item_id', 'item_title', 'serial', 'reference', 'litigant', 'fact', 'viewpoint', 'result']]
        data.to_csv('../data/labelling/'+problem+'数据.csv', index=False, encoding='utf-8')


def export_raw_data_from_database(problem_database, problem_anyous):
    for problem, anyous in problem_anyous.items():
        data = []
        connect_big_data = pymysql.connect(host='192.168.1.253', port=3366,
                                           user='justice_user_03', password='justice_user_03_pd_!@#$',
                                           db='justice_big_data')
        for anyou in anyous:
            sql1 = '''
                select f5 as serial, f12 as reference, f7 as raw_content
                from justice_big_data.%s where f12 = '%s' limit 100
            ''' % (problem_database[problem], anyou)
            temp = pd.read_sql(sql1, con=connect_big_data)
            print("案由：", anyou, "从数据库的表取出了:", len(temp))
            data.append(temp)
        data = pd.concat(data, sort=False)
        data = data.reset_index().drop('index', axis=1)
        connect_big_data.close()

        data["problem"] = problem

        data['litigant'] = data['raw_content'].apply(sucheng_extract)
        data['litigant'] = data['litigant'].apply(lambda x: x[0] if x is not None else None)
        data['fact'] = data['raw_content'].apply(chaming_extract)
        data['viewpoint'] = data['raw_content'].apply(renwei_extract)
        data['fatiao'] = data['raw_content'].apply(fatiao_extract)
        data['result'] = data['raw_content'].apply(panjue_extract)
        data['result'] = data['result'].apply(lambda x: x['判决'] if x is not None and '判决' in x else None)

        # 2.数据处理（过滤判决、诉求字数少的；过滤原告，目前不取原告是个人的。）
        data = data[data['litigant'].str.len() > 10]
        data = data[data['fact'].str.len() > 10]
        data = data[data['viewpoint'].str.len() > 10]
        data = data[data['result'].str.len() > 10]

        data = data.reset_index().drop('index', axis=1)
        result = []
        for anyou in anyous:
            result.append(data[data['reference']==anyou][:20])
        result = pd.concat(result, sort=False)
        result['item_id'] = np.arange(len(result))
        result['item_title'] = np.arange(len(result))
        result = result[['item_id', 'item_title', 'serial', 'reference', 'litigant', 'fact', 'viewpoint', 'result']]
        result.to_csv('../data/'+problem+'.csv', index=False, encoding='utf-8')


def wenshu_information_generate(target_file, start_index, num, problem):
    if problem not in problem_anyou_list.index:
        raise ValueError("ERROR. %s not in y_keyword table" % (problem))
    print(problem, ";problem_anyou_list[problem]:", problem_anyou_list[problem])

    data = []
    # 从数据库读取所有案由数据
    connect_big_data = pymysql.connect(host=big_data_host, port=int(big_data_port),
                                       user=big_data_user, password=big_data_password,
                                       db=big_data_db)
    for anyou in problem_anyou_list[problem]:
        print("anyou:", anyou)
        sql1 = '''
            select f1 as title, f2 as doc_id, f3 as province, f5 as serial, f12 as anyou, f14 as date, f41 as court, f7 as raw_content
            from justice_big_data.%s where f12 = '%s' limit %s, %s
        ''' % (anyou_db_dict[anyou], anyou, start_index, num)
        temp = pd.read_sql(sql1, con=connect_big_data)
        print("案由：", anyou, "从数据库的表取出了:", len(temp))
        data.append(temp)
    data = pd.concat(data, sort=False)
    data = data.reset_index().drop('index', axis=1)
    connect_big_data.close()

    # 去重
    if len(data) == 0:
        return

    data["problem"] = problem

    data['participant'] = data['raw_content'].apply(participant_extract)
    data['yuangao'] = data['participant'].apply(lambda x: '|'.join(x[1]) if x[1] is not None else None)
    data['beigao'] = data['participant'].apply(lambda x: '|'.join(x[2]) if x[2] is not None else None)
    data['participant'] = data['participant'].apply(
        lambda x: '|'.join('|'.join(':'.join(t) for t in p) for p in x[0]) if x[0] is not None else None)

    data['sucheng'] = data['raw_content'].apply(sucheng_extract)
    data['suqing'] = data['sucheng'].apply(lambda x: x[1] if x is not None else None)
    data['sucheng'] = data['sucheng'].apply(lambda x: x[0] if x is not None else None)

    data['biancheng'] = data['raw_content'].apply(biancheng_extract)
    data['biancheng'] = data['biancheng'].apply(lambda x: '|'.join(x) if x is not None else None)

    data['proof'] = data['raw_content'].apply(proof_extract)
    data['proof'] = data['proof'].apply(lambda x: '|'.join(x) if x is not None else None)

    data['chaming'] = data['raw_content'].apply(chaming_extract)

    data['renwei'] = data['raw_content'].apply(renwei_extract)

    data['fatiao'] = data['raw_content'].apply(fatiao_extract)
    data['fatiao'] = data['fatiao'].apply(fatiao_correct)

    data['panjue'] = data['raw_content'].apply(panjue_extract)
    data['panjue'] = data['panjue'].apply(lambda x: x['判决'] if x is not None and '判决' in x else None)

    print('data size: %s' % (len(data)))
    data = data[~((data['suqing'].isna()) | (data['panjue'].isna()))]
    print('none filter: %s' % (len(data)))

    data['label_string'] = data.apply(lambda row: get_label_string(problem, row['anyou'], row['panjue'], row['suqing']), axis=1)

    # 检查路径是否存在
    if '/' in target_file:
        target_folder = target_file[0:target_file.rfind("/")]
        if not os.path.exists(target_folder):
            print("target_folder:", target_folder)
            os.makedirs(target_folder)
    print("panjue_label_report.target_file:", target_file)

    # 数据保存
    data = data.sample(frac=1.0)
    data = data.reset_index().drop(['index', 'raw_content'], axis=1)
    print("length of result of data going to be save:", len(data))
    data.to_csv(target_file, index=False, encoding='utf-8')


def wenshu_information_multi_processing_generate(problem_list, data_path, process_num=20, number_example=10000):
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    for i, problem in enumerate(problem_list):
        # wenshu_information_generate(data_path + problem + '_文书信息.csv', 0, number_example, problem)
        pool = multiprocessing.Pool(processes=process_num)
        print(i, "problem:", problem)
        start_index = 0
        for k in range(process_num):
            print(k, "start_index:", start_index)
            pool.apply_async(wenshu_information_generate, args=(
            data_path + problem + '_文书信息' + str(k) + '.csv', start_index, number_example, problem))
            start_index += number_example
        pool.close()
        pool.join()


def transfer_file_data_to_training_data(data_path, problem_list, target_path):
    """
    :return:
    """
    filter_keywords = [
        '被告所在地.*管辖',
        '(违反.*程序|属.*程序不当|显属不当)',
        '(不符合|不属).*(受理|受案|审理).*(范围|条件|范畴)',
        '(不符合诚实信用原则|不能代替行政部门的职权|不符合.*法律.*起诉条件)',
        '超过.*(时效|时限|申请期限)', '期限届满',
        '撤回', '撤销', '撤诉', '上诉', '二审', '反诉', '移送', '调解', '和解',
    ]

    # 1. get data from table
    print("transfer_table_data_to_bert_format.started.")
    data = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            print(os.path.join(root, file))
            if len(re.findall('|'.join(problem_list), file))==0:
                continue
            temp_data = pd.read_csv(os.path.join(root, file), usecols=['doc_id', 'serial','problem','anyou','sucheng','suqing','biancheng','chaming','renwei','panjue','proof','label_string'])
            print('source data:', len(temp_data))
            temp_data = temp_data[~(temp_data['suqing'].isna())]
            temp_data = temp_data[~(temp_data['renwei'].isna())]
            temp_data = temp_data[~(temp_data['chaming'].isna())]
            temp_data = temp_data[(temp_data['chaming'].str.len() > 30)] # & (temp_data['chaming'].str.len() < 600)]
            temp_data = temp_data[~(temp_data['label_string'].isna())]

            temp_data = word_filter(temp_data, 'sucheng', ['撤诉申请'])
            temp_data = word_filter(temp_data, 'panjue', ['驳回.*起诉', '管辖权', '反诉', '移送'])
            temp_data = word_filter(temp_data, 'renwei', filter_keywords)
            print('data filter:', len(temp_data))

            if len(temp_data) == 0:
                continue

            data.append(temp_data)
    data = pd.concat(data, sort=False)
    print("length of data from z_train_law_labelled_data before drop_duplicates:", len(data))
    data = data.drop_duplicates(subset=['serial'])
    print("length of data from z_train_law_labelled_data after drop_duplicates:", len(data))
    data = repeat_filter(data, column='sucheng', extra_columns=['problem'])
    print("length of data from z_train_law_labelled_data after repeat_filter:", len(data))

    data['label_string'] = data['label_string'].apply(lambda x: ';'.join([t for t in x.split(';') if not t.endswith(':-1')]))
    data = data[data['label_string'].str.len()>0]
    data = data.reset_index().drop('index', axis=1)
    data = data.drop('label_string', axis=1).join(
        data['label_string'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('label_string'))
    data['ps'], data['label'] = zip(*data['label_string'].str.split(':'))
    data['label'] = data['label'].astype('int')
    data['problem'], data['suqiu'] = zip(*data['ps'].str.split('_'))
    data['suqiu_desc'] = data['ps'].apply(lambda x: prob_ps_desc[x])
    print('data split:', len(data))

    # 3. print number of postive and negative for each suqiu
    print("=================================================================================================")
    positive_num_sum = []
    negative_num_sum = []
    print("just for debug purposes: print number of samples for each suqiu")
    for problem, suqius in prob_ps.items():
        for suqiu in suqius:
            if problem not in problem_list:
                positive_num_sum.append(0)
                negative_num_sum.append(0)
                print(problem, suqiu, "1_num:", 0, ";0_num:", 0)
                continue
            positive_num = len(data[(data['problem'] == problem) & (data['suqiu'] == suqiu) & (data['label']==1)])
            negative_num = len(data[(data['problem'] == problem) & (data['suqiu'] == suqiu) & (data['label']==0)])
            print(problem, suqiu, "1_num:", positive_num, ";0_num:", negative_num)
            positive_num_sum.append(positive_num)
            negative_num_sum.append(negative_num)
    print("1_num_sum:", sum(positive_num_sum), ";0_num_sum:", sum(negative_num_sum))

    # save result data
    columns = ['problem','suqiu','suqiu_desc','chaming','doc_id','serial','anyou','sucheng','suqing','biancheng','renwei','panjue','proof','label']
    data = data[columns]
    result = []
    for problem, suqius in prob_ps.items():
        for suqiu in suqius:
            if problem not in problem_list:
                continue
            temp = data[(data['problem'] == problem) & (data['suqiu'] == suqiu)].copy()
            result.append(temp.sample(frac=1)[:50000])
    result = pd.concat(result, sort=False)
    result = result.sample(frac=1)
    result['chaming_fact'] = result['chaming'].apply(chaming_fact_extract)

    serials = result['serial'].drop_duplicates().values.tolist()
    valid_serials = random.sample(serials, int(len(serials)*0.2))
    test_serials = valid_serials[:len(valid_serials)//2]
    tmp_valid_serials = valid_serials[len(valid_serials)//2:]

    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)
    # 测试集保存
    result_test = result[result['serial'].isin(test_serials)]
    result_test.to_csv(target_path+'result_test.csv', index=False, encoding='utf-8')

    # 验证集保存
    result_valid = result[result['serial'].isin(tmp_valid_serials)]
    # history_valid = pd.read_csv(target_path + 'result_valid.csv', encoding='utf-8')
    # result_valid = pd.concat([result_valid, history_valid], sort=False)
    result_valid.to_csv(target_path+'result_valid.csv', index=False, encoding='utf-8')

    # 训练集保存
    result_train = result[~(result['serial'].isin(valid_serials))]
    # history_train = pd.read_csv(target_path + 'result_train.csv', encoding='utf-8')
    # result_train = pd.concat([result_train, history_train], sort=False)
    result_train.to_csv(target_path + 'result_train.csv', index=False, encoding='utf-8')

    print("transfer_table_data_to_bert_format.ended.")


def feature_x_multi_processing_generate(problem_list, data_raw_file, target_path, set_type, process_num=30):
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    # 读取训练原始数据
    print(data_raw_file)
    raw_data = pd.read_csv(data_raw_file, encoding='utf-8', usecols=['doc_id', 'serial', 'problem', 'suqiu', 'sucheng', 'biancheng', 'chaming', 'renwei', 'label'])

    # 匹配特征关键词
    for problem in problem_list:
        if len(raw_data[raw_data['problem']==problem]) == 0:
            print(problem + 'has not data!')
            continue
        data = raw_data[raw_data['problem']==problem].copy()
        # data['new_problem'] = data.apply(lambda row: suqiu_correct(row['suqing'], row['problem'], row['suqiu'])[0], axis=1)
        # data['new_suqiu'] = data.apply(lambda row: suqiu_correct(row['suqing'], row['problem'], row['suqiu'])[1], axis=1)

        data['chaming_filter'] = data['chaming'].apply(pos_filter)
        data['chaming_fact'] = data['chaming'].apply(chaming_fact_extract)
        data['chaming_fact'] = data['chaming_fact'].fillna('')
        data['renwei_fact'] = data['renwei'].apply(renwei_fact_extract)
        data['renwei_fact'] = data['renwei_fact'].fillna('')
        print(data.shape)
        data_x_kw = multi_processing_data((data['renwei_fact']+'。'+data['chaming_fact']).values,
                                          min(process_num, len(data)),
                                          problem, None)
        print("data_x_kw: ", data_x_kw.shape)
        columns = ['doc_id', 'serial', 'problem', 'suqiu', 'sucheng', 'biancheng', 'chaming_filter', 'chaming_fact', 'renwei_fact', 'label']
        for i, f in enumerate(problem_bkw_dict[problem].index):
            data['factor:' + problem + '_' + f] = data_x_kw[:, i]
            columns.append('factor:' + problem + '_' + f)

        data = data[columns]
        data.to_csv(target_path + problem + '_' + set_type + '.csv', index=False)


def factor_weight_calculate(problem_list, data_path):
    for problem in problem_list:
        result = []
        data = pd.read_csv(data_path + problem + '_train.csv', encoding='utf-8')
        print(problem, len(data))
        for suqiu in logic_ps[problem]:
            for f in suqiu_bkw_dict[problem+'_'+suqiu].index:
                count = abs(data['factor:'+problem+'_'+f].values).sum()
                result.append([suqiu, f, count])
        result = pd.DataFrame(result, columns=['suqiu', 'factor', 'weight'])
        result.to_csv(config_path + problem + '/' + problem + '因子权重.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    # export_consult_data()

    # export_test_data(['劳动社保', '交通事故', '借贷纠纷'])

    # export_raw_data_from_database({'知识产权': 'case_list_original_zhishichanquan'},
    #                               {'知识产权': [
    #                                   '著作权权属、侵权纠纷','侵害商标权纠纷','侵害作品信息网络传播权纠纷', '侵害其他著作财产权纠纷',
    #                                   '侵害外观设计专利权纠纷','侵害发明专利权纠纷','侵害实用新型专利权纠纷','不正当竞争纠纷',
    #                                   '商标权权属、侵权纠纷','侵害作品复制权纠纷','侵害计算机软件著作权纠纷','著作权权属纠纷',
    #                                   '侵害录音录像制作者权纠纷','专利权权属纠纷','侵害商业秘密纠纷','虚假宣传纠纷',
    #                                   '擅自使用知名商品特有名称、包装、装潢纠纷','擅自使用他人企业名称、姓名纠纷','专利申请权权属纠纷',
    #                                   '商标权权属纠纷','网络域名权属纠纷','专利权权属、侵权纠纷','计算机软件著作权权属纠纷'
    #                               ]})

    problem_list = eval(config.get('functionParameter', 'problem_list'))
    local_db_folder = config.get('functionParameter', 'local_db_folder')
    output_folder = config.get('functionParameter', 'output_folder')

    wenshu_information_multi_processing_generate(problem_list, local_db_folder, number_example=1000)
    transfer_file_data_to_training_data(local_db_folder, problem_list, output_folder)
    #  ['婚姻继承', '劳动社保', '交通事故', '借贷纠纷', '知识产权', '租赁合同']
    feature_x_multi_processing_generate(problem_list,
                                        output_folder+'/result_train.csv', output_folder, 'train')
    feature_x_multi_processing_generate(problem_list,
                                        output_folder+'/result_valid.csv', output_folder, 'valid')
    feature_x_multi_processing_generate(problem_list,
                                        output_folder + '/result_test.csv', output_folder, 'test')

    factor_weight_calculate(problem_list, output_folder)
