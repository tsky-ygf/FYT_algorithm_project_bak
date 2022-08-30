#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 15:01
# @Author  : Adolf
# @Site    : 
# @File    : criminal_prejudgment.py
# @Software: PyCharm
# from pprint import pprint, pformat
# from extraction.feature_extraction import init_extract
from pathlib import Path
import requests
from LawsuitPrejudgment.Criminal.basic_prejudgment import PrejudgmentPipeline

from autogluon.text import TextPredictor
import os
# from xmindparser import xmind_to_dict
from LawsuitPrejudgment.Criminal.parse_xmind import deal_xmind_to_dict
import pandas as pd
from pprint import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pd.set_option('display.max_columns', None)


class CriminalPrejudgment(PrejudgmentPipeline):
    def __init__(self, *args, **kwargs):
        super(CriminalPrejudgment, self).__init__(*args, **kwargs)

        # 加载案由预测模型
        self.predictor = TextPredictor.load(self.config.anyou_identify_model_path)
        # self.ie = init_extract(criminal_type=criminal_type)
        if self.config.situation_identify_model_path[:4] == "http":
            self.ie_url = self.config.situation_identify_model_path

        self.logger.info("加载预测模型成功")

    def anyou_identify(self):
        sentence = self.content["fact"]
        predictions = self.predictor.predict({"fact": [sentence]}, as_pandas=False)
        self.content["anyou"] = predictions[0]
        # self.logger.debug(self.content)

    def suqiu_identify(self):
        if self.config.prejudgment_type == "criminal":
            self.content["suqiu"] = "量刑推荐"

    def situation_identify(self):
        if self.content['anyou'] == "盗窃":
            criminal_type = 'theft'
        else:
            raise Exception("暂时不支持该数据格式")

        r = requests.post(self.ie_url, json={"criminal_type": criminal_type, "fact": self.content["fact"]})
        extract_result = r.json()['result'][0]
        self.content["event"] = dict()
        for key, values in extract_result.items():
            for value in values:
                self.content["event"]["事件"] = value["text"]
                relations = value["relations"]
                # self.logger.debug(relations)
                self.content["event"]["物品"] = relations.get("物品", [{"text": "一些"}])

                _thing = self.content["event"]["物品"]
                _thing = [one_thing['text'] for one_thing in _thing]
                _thing_str = "、".join(_thing)
                self.content["event"]["物品"] = _thing_str

                self.content["event"]["时间"] = relations.get("时间", [{"text": ""}])[0]["text"]
                self.content["event"]["地点"] = relations.get("地点", [{"text": ""}])[0]["text"]
                self.content["event"]["人物"] = relations.get("人物", [{"text": "嫌疑人"}])[0]["text"]
                self.content["event"]["总金额"] = relations.get("总金额", [{"text": ""}])[0]["text"]
                self.content["event"]["行为"] = relations.get("行为", [{"text": self.content['anyou']}])[0]["text"]
        # self.logger.debug(self.content)

        # if self.content["event"]["行为"] is not None:
        #     self.content["graph_process"]["情节"] = 1
        if self.content["event"]["总金额"] != "":
            self.content["graph_process"]["量刑"] = 1

    def parse_config_file(self):
        config_path = "LawsuitPrejudgment/Criminal/base_config"
        xmind_path = Path(config_path, self.content["anyou"], "base_logic.xmind")
        question_answers_path = Path(config_path, self.content["anyou"], "question_answers.csv")
        report_path = Path(config_path, self.content["anyou"], "report.csv")

        base_logic_dict = deal_xmind_to_dict(xmind_path)
        question_answers_df = pd.read_csv(question_answers_path)
        report_content = pd.read_csv(report_path)

        report_dict = dict()
        for index, row in report_content.iterrows():
            report_dict[row['reportID']] = row.to_dict()

        question_answers_dict = {}
        for index, row in question_answers_df.iterrows():
            question_answers_dict[row['circumstances']] = row.to_dict()

        self.content["base_logic_graph"] = base_logic_dict
        self.content['question_answers_config'] = question_answers_dict
        self.content['report_dict'] = report_dict

        self.content["graph_process"] = {key: 0 for key in base_logic_dict[self.content['anyou']].keys()}
        self.content["graph_process_content"] = {key: "" for key in base_logic_dict[self.content['anyou']].keys()}

    def get_question(self):
        self.logger.debug(self.content["question_answers"])

        # 通过问题将没有覆盖到的点进行点亮
        if len(self.content["question_answers"]) > 0:
            for key in self.content["question_answers"].keys():
                self.content["graph_process"][key] = 1
                if self.content["graph_process_content"][key] == "":
                    self.content["graph_process_content"][key] = self.content["question_answers"][key]["usr_answer"]

            if self.content["question_answers"]["前提"]["usr_answer"] == "是":
                self.content["graph_process"]["情节"] = 1
                self.content["graph_process"]["量刑"] = 1

                # for key,v

        for key, value in self.content["graph_process"].items():
            if value == 0:
                qa_dict = self.content["question_answers_config"][key]
                qa_dict.pop('circumstances')
                qa_dict["usr_answer"] = ""
                self.content["question_answers"][key] = qa_dict
                break

        # self.logger.debug(self.content["question_answers"])

    def generate_report(self):
        evaluation_report = dict()
        # evaluation_report["案件事实"] = "根据您的描述，【时间】，在【地点】，【人物】存在【行为】等行为，窃得【物品】、【物品】等财物，盗窃总金额为【总金额】。"
        # evaluation_report["评估理由"] = "【盗窃罪XMind评估理由】【量刑情节XMind评估理由】"
        # evaluation_report["法律建议"] = "【盗窃罪XMind法律建议】"
        # evaluation_report["法律依据"] = "【盗窃罪XMind法律依据】【量刑情节XMind法律依据】"
        self.logger.debug(self.content["event"])
        self.logger.debug(self.content["graph_process_content"])
        self.logger.debug(self.content["base_logic_graph"])
        # self.logger.debug(self.content)
        self.logger.info("开始生成报告")

        # for key,value in self.content["base_logic_graph"][self.content["anyou"]].items():
        case_num = self.content["base_logic_graph"][self.content["anyou"]]["情节"][
            "【" + self.content["graph_process_content"]["情节"] + "】"]

        sentencing_dict = self.content["base_logic_graph"][self.content["anyou"]]["量刑"]
        report_id = sentencing_dict["【" + self.content["graph_process_content"]["量刑"] + "】"][case_num]

        _thing = self.content["event"]["物品"]
        _time = self.content["event"]["时间"]

        _location = self.content["event"]["地点"]
        _person = self.content["event"]["人物"]
        _action = self.content["event"]["行为"]
        _amount = self.content["event"]["总金额"]

        if _time != "":
            _time += "，"

        if _location != "":
            _location = "在" + _location + "，"

        evaluation_report["案件事实"] = \
            f"根据您的描述，{_time}{_location}{_person}存在{_action}等行为，窃得{_thing}财物，盗窃金额为{_amount}。"

        evaluation_report["评估理由"] = self.content['report_dict'][report_id]["评估理由"]
        evaluation_report["法律建议"] = self.content['report_dict'][report_id]["法律建议"]
        evaluation_report["法律依据"] = self.content['report_dict'][report_id]["法律依据"]

        self.content['report_result'] = evaluation_report


if __name__ == '__main__':
    criminal_config = {"log_level": "debug",
                       "prejudgment_type": "criminal",
                       "anyou_identify_model_path": "model/gluon_model/accusation",
                       "situation_identify_model_path": "http://172.19.82.199:7777/information_result",
                       }
    criminal_pre_judgment = CriminalPrejudgment(**criminal_config)

    text = "浙江省诸暨市人民检察院指控，2019年7月22日10时30分许，被告人唐志强窜至诸暨市妇幼保健医院，在3楼21号病床床头柜内窃得被害人俞" \
           "某的皮包一只，内有现金￥1500元和银行卡、身份证等财物。"

    # question_answers = []

    # predict
    # criminal_pre_judgment(fact=text, question_answers=question_answers)
    input_dict = {"fact": text}
    # 第一次调用
    res = criminal_pre_judgment(**input_dict)
    # print(res)
    # 第二次调用
    res['question_answers']['前提']['usr_answer'] = "否"
    res2 = criminal_pre_judgment(**res)  # 传入上一次的结果
    # print(res2)
    # 第三次调用
    res2["question_answers"]["情节"]["usr_answer"] = "以上都没有"
    res3 = criminal_pre_judgment(**res2)

    # 第四次调用
    res3["question_answers"]["量刑"]["usr_answer"] = "3000（含）-40000（不含）"
    res4 = criminal_pre_judgment(**res3)

    pprint(res4["report_result"])
