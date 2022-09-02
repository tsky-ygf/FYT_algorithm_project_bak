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
import re
import os
import cn2an

# from xmindparser import xmind_to_dict
from LawsuitPrejudgment.Criminal.parse_xmind import deal_xmind_to_dict
import pandas as pd
from pprint import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pd.set_option("display.max_columns", None)


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
        # print(predictions[0])
        # self.logger.debug(self.content)

    def suqiu_identify(self):
        if self.config.prejudgment_type == "criminal":
            self.content["suqiu"] = "量刑推荐"

    def situation_identify(self):
        if self.content["anyou"] == "盗窃":
            criminal_type = "theft"
        elif self.content["anyou"] == "容留他人吸毒":
            criminal_type = "provide_drug"
        else:
            self.content["report_result"] = {
                "敬请期待": f"你的行为属于{self.content['anyou']}犯罪,目前还未上线，正在训练优化中，敬请期待！"
            }
            return

        r = requests.post(
            self.ie_url,
            json={"criminal_type": criminal_type, "fact": self.content["fact"]},
        )
        # extract_result = r.json()['result'][0]
        self.content["event"] = r.json()["result"]

        # self.logger.debug(self.content)

        # if self.content["event"]["行为"] is not None:
        #     self.content["graph_process"]["情节"] = 1

    def parse_config_file(self):
        config_path = "LawsuitPrejudgment/Criminal/base_config"
        xmind_path = Path(config_path, self.content["anyou"], "base_logic.xmind")
        question_answers_path = Path(
            config_path, self.content["anyou"], "question_answers.csv"
        )
        report_path = Path(config_path, self.content["anyou"], "report.csv")
        sentence_path = Path(config_path, self.content["anyou"], "sentences.csv")

        base_logic_dict = deal_xmind_to_dict(xmind_path)
        question_answers_df = pd.read_csv(question_answers_path)
        report_content = pd.read_csv(report_path)
        sentence_keywords = pd.read_csv(sentence_path)

        report_dict = dict()
        report_content.fillna("", inplace=True)
        for index, row in report_content.iterrows():
            report_dict[row["reportID"]] = row.to_dict()

        question_answers_dict = {}

        for index, row in question_answers_df.iterrows():
            question_answers_dict[row["circumstances"]] = row.to_dict()

        self.content["base_logic_graph"] = base_logic_dict
        self.content["question_answers_config"] = question_answers_dict
        self.content["report_dict"] = report_dict
        self.content["sentence_keywords"] = sentence_keywords

        self.content["graph_process"] = {
            key: 0 for key in base_logic_dict[self.content["anyou"]].keys()
        }
        self.content["graph_process_content"] = {
            key: "" for key in base_logic_dict[self.content["anyou"]].keys()
        }

    def match_graph(self):
        if (
            list(self.content["base_logic_graph"][self.content["anyou"]]["量刑"].keys())[
                0
            ]
            == "【量刑】"
        ):
            self.content["graph_process"]["量刑"] = 1
            self.content["graph_process_content"]["量刑"] = "量刑"

        # if self.content["anyou"] == "容留他人吸毒":
        self.logger.debug(self.content["event"])
        self.logger.debug(self.content["sentence_keywords"])

        for index, row in self.content["sentence_keywords"].iterrows():
            # schema_list = row["schema"].split("|")
            # for schema in schema_list:
            # if row["sentence"] in self.content["event"][schema]:
            if len(re.findall(row["sentences"], self.content["fact"])) > 0:
                self.content["graph_process"]["情节"] = 1
                self.content["graph_process_content"]["情节"] = row["crime_plot"]
                # break
            if self.content["graph_process"]["情节"] == 1:
                break

        if self.content["anyou"] == "盗窃" and self.content["event"]["总金额"] != "":
            self.content["graph_process"]["量刑"] = 1

            # output_amount = cn2an.cn2an(self.content["event"]["总金额"], "smart")
            output_amount = cn2an.transform(self.content["event"]["总金额"], "cn2an")
            output_amount = re.findall("\d+", output_amount)[0]
            output_amount = float(output_amount)

            if output_amount < 1500:
                self.content["graph_process_content"]["量刑"] = "0-1500（不含）"
            elif output_amount < 3000:
                self.content["graph_process_content"]["量刑"] = "1500（含）-3000（不含）"
            elif output_amount < 40000:
                self.content["graph_process_content"]["量刑"] = "3000（含）-40000（不含）"
            elif output_amount < 80000:
                self.content["graph_process_content"]["量刑"] = "40000（含）-80000（不含）"
            elif output_amount < 100000:
                self.content["graph_process_content"]["量刑"] = "80000（含）-100000（不含）"
            elif output_amount < 150000:
                self.content["graph_process_content"]["量刑"] = "100000（含）-150000（不含）"
            elif output_amount < 200000:
                self.content["graph_process_content"]["量刑"] = "150000（含）-200000（不含）"
            elif output_amount < 250000:
                self.content["graph_process_content"]["量刑"] = "200000（含）-250000（不含）"
            elif output_amount < 300000:
                self.content["graph_process_content"]["量刑"] = "250000（含）-300000（不含）"
            elif output_amount < 350000:
                self.content["graph_process_content"]["量刑"] = "300000（含）-350000（不含）"
            elif output_amount < 400000:
                self.content["graph_process_content"]["量刑"] = "350000（含）-400000（不含）"
            else:
                self.content["graph_process_content"]["量刑"] = "400000（含）以上"

    def get_question(self):
        self.logger.debug(self.content["question_answers"])

        # 通过问题将没有覆盖到的点进行点亮
        if len(self.content["question_answers"]) > 0:
            for key in self.content["question_answers"].keys():
                self.content["graph_process"][key] = 1
                if self.content["graph_process_content"][key] == "":
                    self.content["graph_process_content"][key] = self.content[
                        "question_answers"
                    ][key]["usr_answer"]

            if self.content["question_answers"]["前提"]["usr_answer"] == "是":
                self.content["graph_process"]["情节"] = 1
                self.content["graph_process"]["量刑"] = 1

        for key, value in self.content["graph_process"].items():
            if value == 0:
                qa_dict = self.content["question_answers_config"][key]
                qa_dict.pop("circumstances")
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

        # if self.content["question_answers"]["前提"]["usr_answer"] == "是":

        self.logger.debug(self.content["event"])
        self.logger.debug(self.content["graph_process_content"])
        self.logger.debug(self.content["base_logic_graph"])
        # self.logger.debug(self.content)
        self.logger.info("开始生成报告")

        # for key,value in self.content["base_logic_graph"][self.content["anyou"]].items():
        case_num = self.content["base_logic_graph"][self.content["anyou"]]["情节"][
            "【" + self.content["graph_process_content"]["情节"] + "】"
        ]

        sentencing_dict = self.content["base_logic_graph"][self.content["anyou"]]["量刑"]
        if self.content["question_answers"]["前提"]["usr_answer"] == "是":
            report_id = "report-1"
        else:
            if (
                case_num
                not in sentencing_dict[
                    "【" + self.content["graph_process_content"]["量刑"] + "】"
                ]
            ):
                case_num = "case0"

            report_id = sentencing_dict[
                "【" + self.content["graph_process_content"]["量刑"] + "】"
            ][case_num]

        _time = self.content["event"]["时间"]

        _location = self.content["event"]["地点"]
        _person = self.content["event"]["人物"]
        # _action = self.content["event"]["行为"]
        _action = self.content["graph_process_content"]["情节"]

        if _time != "":
            _time += "，"

        if _location != "":
            _location = "在" + _location + "，"

        if self.content["anyou"] == "盗窃":
            _thing = self.content["event"]["物品"]

            if "量刑" in self.content["question_answers"]:
                _amount = self.content["question_answers"]["量刑"]["usr_answer"]
            else:
                _amount = self.content["event"]["总金额"]

            evaluation_report[
                "案件事实"
            ] = f"根据您的描述，{_time}{_location}{_person}存在{_action}等情形，窃得{_thing}财物，盗窃金额为{_amount}。"

        elif self.content["anyou"] == "容留他人吸毒":
            _drug_name = self.content["event"]["毒品名称"]
            _drug_type = self.content["event"]["毒品种类"]

            _provide_count = self.content["event"]["容留次数"]
            _provided_person = self.content["event"]["被容留人"]

            evaluation_report[
                "案件事实"
            ] = f"根据您的描述，{_time}{_location}{_person}容留{_provided_person}吸食{_drug_type}（俗称{_drug_name}）{_provide_count}，存在{_action}等情形。"

        evaluation_report["涉嫌罪名"] = self.content["anyou"]
        evaluation_report["评估理由"] = self.content["report_dict"][report_id]["评估理由"]
        evaluation_report["法律依据"] = self.content["report_dict"][report_id]["法律依据"]
        evaluation_report["法律建议"] = self.content["report_dict"][report_id]["法律建议"]

        evaluation_report["相关类案"] = (
            self.content["report_dict"][report_id]["相似案例1"]
            + "|"
            + self.content["report_dict"][report_id]["相似案例2"]
        )

        evaluation_report["相关类案"] = evaluation_report["相关类案"].split("|")

        self.content["report_result"] = evaluation_report


if __name__ == "__main__":
    criminal_config = {
        "log_level": "info",
        "prejudgment_type": "criminal",
        "anyou_identify_model_path": "model/gluon_model/accusation",
        "situation_identify_model_path": "http://172.19.82.199:7777/information_result",
    }
    criminal_pre_judgment = CriminalPrejudgment(**criminal_config)

    # text = "浙江省诸暨市人民检察院指控，2019年7月22日10时30分许，被告人唐志强窜至诸暨市妇幼保健医院，在3楼21号病床床头柜内窃得被害人俞" \
    #        "某的皮包一只，内有现金￥1500元和银行卡、身份证等财物。"

    # input_dict = {"fact": text}
    # # 第一次调用
    # res = criminal_pre_judgment(**input_dict)
    # # print(res)
    # # 第二次调用
    # res['question_answers']['前提']['usr_answer'] = "否"
    # res2 = criminal_pre_judgment(**res)  # 传入上一次的结果
    # # print(res2)
    # # 第三次调用
    # res2["question_answers"]["情节"]["usr_answer"] = "以上都没有"
    # res3 = criminal_pre_judgment(**res2)
    #
    # # 第四次调用
    # res3["question_answers"]["量刑"]["usr_answer"] = "3000（含）-40000（不含）"
    # res4 = criminal_pre_judgment(**res3)
    #
    # pprint(res4["report_result"])

    # text = (
    #     "湖南省涟源市人民检察院指控，2014年8月至2015年1月，被告人刘某甲先后多次容留刘2某、刘某乙、刘1某、刘某丙、袁某等人在其位于本市"
    #     "安平镇田心村二组的家中吸食甲基苯丙胺（冰毒）和甲基苯丙胺片剂（麻古）。具体事实如下：1、2014年8月份的一天，被告人"
    #     "刘某甲容留刘某丙、刘1某等人在其家中卧室吸食甲基苯丙胺和甲基苯丙胺片剂。"
    # )

    text = (
        "2022年8月12日，罗某某利用螺丝刀撬开房间门锁进入某市某区某栋某单元某层某房间内，窃得现金50000元。2022年8月12日，趁邻居卢某家"
        "无人在家，从卢某家厨房后窗翻进其家，盗走现金50000元。"
    )
    input_dict = {"fact": text}
    # 第一次调用
    res = criminal_pre_judgment(**input_dict)
    pprint(res)
    # print('@@@@@@@@@@')

    # 第二次调用
    res["question_answers"]["前提"]["usr_answer"] = "否"
    res2 = criminal_pre_judgment(**res)  # 传入上一次的结果
    #
    # if "report_result" in res2:
    #     pprint(res2["question_answers"])
    #     pprint(res2["report_result"])
    # else:
    #     pprint(res2["question_answers"])
    # #
    #     pprint(res2["event"])
    # pprint(res2["question_answers"])
    # pprint(res2["report_result"])

    # 第三次调用
    # res2["question_answers"]["情节"]["usr_answer"] = "以上都没有"
    # res3 = criminal_pre_judgment(**res2)

    # pprint(res3["report_result"])
