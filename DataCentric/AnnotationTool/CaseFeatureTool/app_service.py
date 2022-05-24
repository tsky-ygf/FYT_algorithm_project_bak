#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 14:05
# @Author  : Adolf
# @Site    : 
# @File    : app_service.py
# @Software: PyCharm
import json
import re
from pprint import pprint
import traceback
import logging
import logging.handlers
from loguru import logger
from flask import Flask, request, jsonify
from flask_cors import CORS
from DataCentric.AnnotationTool.CaseFeatureTool.entity_annotation import *
import datetime
import json

class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return json.JSONEncoder.default(self, obj)


"""
情形和特征标注工具服务
"""
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


@app.route('/getAnyou', methods=["post"])
def get_anyou():
    try:
        anyou_list = get_anyou_list()
        return json.dumps(
            {'AnyouList': anyou_list, "error_msg": "", "status": 0},
            ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


@app.route('/getCaseFeature', methods=["post"])
def get_case_feature():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            anyou = in_dict['anyou']
            anyou_list = get_case_feature_dict(anyou_name=anyou)
            return json.dumps(
                {'AnyouList': anyou_list, "error_msg": "", "status": 0},
                ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


@app.route('/getBaseData', methods=["post"])
def get_base_data():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            anyou = in_dict['anyou']
            base_data_dict = get_base_data_dict(anyou_name=anyou)
            return json.dumps(
                {"base_data": base_data_dict, "error_msg": "", "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "error:" + repr(e), "status": 1}, ensure_ascii=False)





@app.route('/getBaseAnnotation', methods=["post"])
def get_base_annotation():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            anyou = in_dict['anyou']
            sentence = in_dict["sentence"]
            content_html = in_dict.get("contentHtml", "")
            base_annotation_dict = get_base_annotation_dict(anyou_name=anyou, sentence=sentence)
            return json.dumps(
                {"base_data": {"contentHtml": content_html, "list": base_annotation_dict['data']},
                 "error_msg": "", "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "error:" + repr(e), "status": 1}, ensure_ascii=False)


@app.route('/insertAnnotationData', methods=["post"])
def do_insert_annotation_data():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            # data_id = in_dict['id']
            anyou_name = in_dict['anyou_name']
            source = in_dict['source']
            content_html = in_dict['contentHtml']
            insert_data = in_dict['insert_data']
            labelingperson = in_dict['labelingperson']
            # insert_data_list = insert_data['insert_data_list']
            print('data is coming')
            insert_data_to_mysql(anyou_name, source, labelingperson, insert_data)
            return json.dumps(
                {"insert_result": "success", "contentHtml": content_html, "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "error:" + repr(e), "status": 1}, ensure_ascii=False)

"""
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            anyou = in_dict['anyou']
            base_data_dict = get_base_data_dict(anyou_name=anyou)
            return json.dumps(
                {"base_data": base_data_dict, "error_msg": "", "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)"""
@app.route('/getSecondCheck',methods=['post'])
def is_second_check():
    # 二次审核
    in_json = request.get_data()
    if in_json:
        in_dict = json.loads(in_json.decode("utf-8"))
        anyou = in_dict['anyou']
        # print(f"anyou:{anyou}")
        second_check_data_dict = get_second_check(anyou)
        # print(f"run sql result:{second_check_data_dict}")
        if second_check_data_dict:
            return json.dumps(
                {"data_list": second_check_data_dict, "error_msg": "", "status": 0}, ensure_ascii=False,cls=DateEncoder)
        else:
            return json.dumps({"error_msg": "no check", "status": 1}, ensure_ascii=False,cls=DateEncoder)

@app.route('/getSecondCheckTrue',methods=['post'])
def get_true_check():
    # 二次审核 确认
    in_json = request.get_data()
    if in_json:
        in_dict = json.loads(in_json.decode("utf-8"))
        checkperson = in_dict['checkperson']
        id = in_dict['id']
        save_second_check_proson(id,checkperson)
        return json.dumps(
            {"data_list": "update success", "error_msg": "", "status": 0}, ensure_ascii=False,cls=DateEncoder)

@app.route('/getWorkCount',methods=['post'])
def get_work_count():
    # 获取当日工作量
    data_json = request.get_data()
    if data_json:
        data_dict = json.loads(data_json.decode('utf-8'))
        name = data_dict.get('name')
        pprint(f"decode_name:{name}")
        second_check_data_dict = get_day_work_count(name)
        pprint(second_check_data_dict)
        if second_check_data_dict:
            return json.dumps(
                               {"work_count": second_check_data_dict[0].get('num'), "error_msg": "", "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "this person have no work", "status": 1}, ensure_ascii=False, cls=DateEncoder)

@app.route('/getSourceContent',methods=['post'])
def get_source():
    data_json = request.get_data()
    if data_json:
        data_dict = json.loads(data_json.decode('utf-8'))
        key = data_dict.get('key')
        pprint(f"decode_key:{key}")
        source_content = get_source_content(key)
        # pprint(second_check_data_dict)
        if source_content:
            return json.dumps(
                               {"content": source_content[0].get('f13'), "error_msg": "", "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "have no data", "status": 1}, ensure_ascii=False, cls=DateEncoder)


@app.route('/loginCheck',methods=['post'])
def login_check():
    # 登录 校验
    data_json = request.get_data()
    if not data_json:
        logging.error(f"没有传入账号密码")
        return json.dumps({"error_msg": "no username or password", "status": 1}, ensure_ascii=False, cls=DateEncoder)
    try:
        data_dict = json.loads(data_json.decode('utf-8'))
    except:
        return json.dumps({"error_msg": "data is not dictionary", "status": 1}, ensure_ascii=False, cls=DateEncoder)
    print(f"type:{type(data_json)},data:{data_json}")
    username = data_dict.get("username")
    password = data_dict.get("password")
    print(f"username:{username},password:{password}")
    if check_username(username):
        select_password = get_login_password(username)
        if select_password == password:
            return json.dumps(
                {"content": "login success", "error_msg": "", "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "Password error", "status": 1}, ensure_ascii=False, cls=DateEncoder)
    else:
        return json.dumps({"error_msg": "There is no such user", "status": 1}, ensure_ascii=False, cls=DateEncoder)

@app.route('/registerUser',methods=['post'])
def register_user():
    # 注册用户
    data_json = request.get_data()
    if not data_json:
        logging.error(f"没有传入账号密码")
        return json.dumps({"error_msg": "no username or password", "status": 1}, ensure_ascii=False, cls=DateEncoder)
    try:
        data_dict = json.loads(data_json.decode('utf-8'))
    except:
        return json.dumps({"error_msg": "data is not dictionary", "status": 1}, ensure_ascii=False, cls=DateEncoder)
    print(f"type:{type(data_json)},data:{data_json}")
    username = data_dict.get("username")
    password = data_dict.get("password")
    if check_username(username):
        return json.dumps({"error_msg": "this user name is in use,repeat of user name", "status": 1}, ensure_ascii=False, cls=DateEncoder)
    if save_username_password(username=username,password=password):
        return json.dumps(
            {"content": "register success", "error_msg": "", "status": 0}, ensure_ascii=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6021, debug=False)  # , use_reloader=False)
