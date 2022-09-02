import re
import traceback
import logging
import logging.handlers

import pandas as pd
from flask import Flask, request
from flask_cors import CORS
import json

import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '3'

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


@app.route('/suqiureview', methods=["post"])
def get_suqiu_res():
    try:
        input_json = request.get_data()
        if input_json is not None:
            input_dict = json.loads(input_json.decode("utf-8"))
            text = input_dict['content']
            suqiu_type = get_suqiu_type(text)
            # TODO 关键词获取诉求，与suqiu_type的结果合并
            print(text)
            print(suqiu_type)

            return json.dumps(suqiu_type, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


@app.route('/situationreview', methods=["post"])
def get_situation_res():
    try:
        # input_str = request.get_data()
        # content, suqiu = input_str.split(str.encode('\t\t'))
        input_json = request.get_data()
        if input_json is not None:
            input_dict = json.loads(input_json.decode("utf-8"))
            text = input_dict['content']
            suqiu_type_user = input_dict['suqiu'].split(',')
            suqiu_pro_new = {}
            if len(input_dict) > 2:
                for suqiu_user_sub in suqiu_type_user:
                    suqiu_pro_new[suqiu_user_sub] = float(input_dict["suqiu_pro"][suqiu_user_sub])
            else:
                # suqiu_type = get_suqiu_type(text)
                for suqiu_user_sub in suqiu_type_user:
                    suqiu_pro_new[suqiu_user_sub] = float(1.0)

            situation_type = get_situation_type(suqiu_pro_new, text)
            print(text)
            print(suqiu_type_user)
            print(situation_type)
            return json.dumps(situation_type, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


def get_split_content(input_words):
    text_merge = []

    return text_merge


def get_suqiu_type(text):
    from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation

    situation = BasicBertSituation(
        config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config/config_suqiu_cls.yaml")
    suqiu_pro = situation.review_main(content=text, mode="text")
    #  如果概率高于阿尔法，则使用正则表达式检查，是否存在该诉求
    suqiu_pro_new = {}
    if len(suqiu_pro) == 0:
        suqiu_exist_list = have_suqiu_list(text)
        for exist_list_sub in suqiu_exist_list:
            suqiu_pro_new[exist_list_sub] = float(1)
    else:
        suqiu_pro_new = get_suqiupro_new(situation, suqiu_pro, text)
    return json.dumps({'suqiu_pro': suqiu_pro_new, "status": 0}, ensure_ascii=False)

def get_suqiupro_new(situation, suqiu_pro, text):
    suqiu_pro_new = {}
    for suqiu_pro_key, suqiu_pro_val in suqiu_pro.items():
        if suqiu_pro_val > float(situation.threshold):
            suqiu_exist_bool = have_suqiu_bool(suqiu_pro_key, text)
            if suqiu_exist_bool:
                suqiu_pro_new[suqiu_pro_key] = suqiu_pro_val
            else:
                suqiu_exist_list = have_suqiu_list(text)
                for exist_list_sub in suqiu_exist_list:
                    suqiu_pro_new[exist_list_sub] = float(1)
                break
        else:  # 如果概率低于阿尔法，则使用正则表达式依次匹配诉求
            suqiu_exist_list = have_suqiu_list(text)
            for exist_list_sub in suqiu_exist_list:
                suqiu_pro_new[exist_list_sub] = float(1)  # 规则匹配到的设置为1
            break
    return suqiu_pro_new

def have_suqiu_bool(suqiu_pro_key, text):

    reg_loan_csv = pd.read_csv('/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config/loan.csv', usecols=[4, 5, 6])
    reg_marr_csv = pd.read_csv('/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config/marrige.csv', usecols=[4, 5, 6])

    if suqiu_pro_key == '民间借贷' and re.search('|'.join([reg_loan_csv.iat[2, 0], reg_loan_csv.iat[2, 1], reg_loan_csv.iat[2, 2]]), text):
        return True
    elif suqiu_pro_key == '金融借贷' and re.search('|'.join([reg_loan_csv.iat[0, 0], reg_loan_csv.iat[0, 1], reg_loan_csv.iat[0, 2]]), text):
        return True
    elif suqiu_pro_key == '离婚' and re.search('|'.join([reg_marr_csv.iat[13, 0], reg_marr_csv.iat[13, 1], reg_marr_csv.iat[13, 2]]), text):
        return True
    elif suqiu_pro_key == '支付赡养费' and re.search('|'.join([reg_marr_csv.iat[10, 0], reg_marr_csv.iat[10, 1], reg_marr_csv.iat[10, 2]]), text):
        return True
    elif suqiu_pro_key == '减少抚养费' and re.search('|'.join([reg_marr_csv.iat[8, 0], reg_marr_csv.iat[8, 1], reg_marr_csv.iat[8, 2]]), text):
        return True
    elif suqiu_pro_key == '支付抚养费' and re.search('|'.join([reg_marr_csv.iat[6, 0], reg_marr_csv.iat[6, 1], reg_marr_csv.iat[6, 2]]), text):
        return True
    elif suqiu_pro_key == '增加抚养费' and re.search('|'.join([reg_marr_csv.iat[7, 0], reg_marr_csv.iat[7, 1], reg_marr_csv.iat[7, 2]]), text):
        return True
    elif suqiu_pro_key == '财产分割' and re.search('|'.join([reg_marr_csv.iat[3, 0], reg_marr_csv.iat[3, 1], reg_marr_csv.iat[3, 2]]), text):
        return True
    elif suqiu_pro_key == '确认婚姻无效' and re.search('|'.join([reg_marr_csv.iat[1, 0], reg_marr_csv.iat[1, 1], reg_marr_csv.iat[1, 2]]), text):
        return True
    elif suqiu_pro_key == '行使探望权' and re.search('|'.join([reg_marr_csv.iat[9, 0], reg_marr_csv.iat[9, 1], reg_marr_csv.iat[9, 2]]), text):
        return True
    elif suqiu_pro_key == '确认抚养权' and re.search('|'.join([reg_marr_csv.iat[5, 0], reg_marr_csv.iat[5, 1], reg_marr_csv.iat[5, 2]]), text):
        return True
    elif suqiu_pro_key == '返还彩礼' and re.search('|'.join([reg_marr_csv.iat[2, 0], reg_marr_csv.iat[2, 1], reg_marr_csv.iat[2, 2]]), text):
        return True
    elif suqiu_pro_key == '遗产继承' and re.search('|'.join([reg_marr_csv.iat[12, 0], reg_marr_csv.iat[12, 1], reg_marr_csv.iat[12, 2]]), text):
        return True
    elif suqiu_pro_key == '夫妻共同债务' and re.search('|'.join([reg_marr_csv.iat[0, 0], reg_marr_csv.iat[0, 1], reg_marr_csv.iat[0, 2]]), text):
        return True
    else:
        return False

def have_suqiu_list(text):
    suqiu_exist_list = []
    reg_loan_csv = pd.read_csv('/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config/loan.csv', usecols=[4, 5, 6])
    reg_marr_csv = pd.read_csv('/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config/marrige.csv', usecols=[4, 5, 6])

    if re.search('|'.join([reg_loan_csv.iat[2, 0], reg_loan_csv.iat[2, 1], reg_loan_csv.iat[2, 2]]), text):
        suqiu_exist_list.append('民间借贷')
    if re.search('|'.join([reg_loan_csv.iat[0, 0], reg_loan_csv.iat[0, 1], reg_loan_csv.iat[0, 2]]), text):
        suqiu_exist_list.append('金融借贷')
    if re.search('|'.join([reg_marr_csv.iat[13, 0], reg_marr_csv.iat[13, 1], reg_marr_csv.iat[13, 2]]), text):
        suqiu_exist_list.append('离婚')
    if re.search('|'.join([reg_marr_csv.iat[10, 0], reg_marr_csv.iat[10, 1], reg_marr_csv.iat[10, 2]]), text):
        suqiu_exist_list.append('支付赡养费')
    if re.search('|'.join([reg_marr_csv.iat[8, 0], reg_marr_csv.iat[8, 1], reg_marr_csv.iat[8, 2]]), text):
        suqiu_exist_list.append('减少抚养费')
    if re.search('|'.join([reg_marr_csv.iat[6, 0], reg_marr_csv.iat[6, 1], reg_marr_csv.iat[6, 2]]), text):
        suqiu_exist_list.append('支付抚养费')
    if re.search('|'.join([reg_marr_csv.iat[7, 0], reg_marr_csv.iat[7, 1], reg_marr_csv.iat[7, 2]]), text):
        suqiu_exist_list.append('增加抚养费')
    if re.search('|'.join([reg_marr_csv.iat[3, 0], reg_marr_csv.iat[3, 1], reg_marr_csv.iat[3, 2]]), text):
        suqiu_exist_list.append('财产分割')
    if re.search('|'.join([reg_marr_csv.iat[1, 0], reg_marr_csv.iat[1, 1], reg_marr_csv.iat[1, 2]]), text):
        suqiu_exist_list.append('确认婚姻无效')
    if re.search('|'.join([reg_marr_csv.iat[9, 0], reg_marr_csv.iat[9, 1], reg_marr_csv.iat[9, 2]]), text):
        suqiu_exist_list.append('行使探望权')
    if re.search('|'.join([reg_marr_csv.iat[5, 0], reg_marr_csv.iat[5, 1], reg_marr_csv.iat[5, 2]]), text):
        suqiu_exist_list.append('确认抚养权')
    if re.search('|'.join([reg_marr_csv.iat[2, 0], reg_marr_csv.iat[2, 1], reg_marr_csv.iat[2, 2]]), text):
        suqiu_exist_list.append('返还彩礼')
    if re.search('|'.join([reg_marr_csv.iat[12, 0], reg_marr_csv.iat[12, 1], reg_marr_csv.iat[12, 2]]), text):
        suqiu_exist_list.append('遗产继承')
    if re.search('|'.join([reg_marr_csv.iat[0, 0], reg_marr_csv.iat[0, 1], reg_marr_csv.iat[0, 2]]), text):
        suqiu_exist_list.append('夫妻共同债务')
    return suqiu_exist_list

def get_situation_type(suqiu_pro, text):
    situation_res = ''
    sit_pro = {}
    for suqiu_type, pro in suqiu_pro.items():
        from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation
        situation = BasicBertSituation(
            config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config"
                        "/config_situa_cls.yaml")
        res_pro = situation.review_main(content=suqiu_type+','+text+','+suqiu_type, mode="text")
        sit_pro[suqiu_type] = res_pro
    return json.dumps({'sit_pro': sit_pro, "status": 0},
                      ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7995, debug=True)  # , use_reloader=False)
