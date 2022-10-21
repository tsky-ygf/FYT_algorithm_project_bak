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
from autogluon.text import TextPredictor
import re
import os
import cn2an
import time

# from xmindparser import xmind_to_dict
from LawsuitPrejudgment.src.common.basic_prejudgment_v2 import PrejudgmentPipeline
from LawsuitPrejudgment.src.criminal.utils.parse_xmind import deal_xmind_to_dict
import pandas as pd
from pprint import pprint

from LawsuitPrejudgment.src.criminal.nlu.feature_extraction import (
    init_extract,
    post_process_uie_results,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pd.set_option("display.max_columns", None)


class CriminalPrejudgmentV2(PrejudgmentPipeline):
    def __init__(self, *args, **kwargs):
        super(CriminalPrejudgmentV2, self).__init__(*args, **kwargs)
        t0 = time.time()
        # 加载案由预测模型
        self.predictor = TextPredictor.load(self.config.anyou_identify_model_path)

        # 加载信息抽取模型
        criminal_list = ["theft", "provide_drug"]
        self.predictor_dict = {}
        for criminal_type in criminal_list:
            self.predictor_dict[criminal_type] = init_extract(criminal_type=criminal_type)

        self.logger.info("加载模型耗时:{}".format(time.time() - t0))
        self.logger.info("加载预测模型成功")

    def anyou_identify(self):
        self.logger.info("预测案由！")
        sentence = self.context["fact"]
        predictions = self.predictor.predict({"fact": [sentence]}, as_pandas=False)
        self.context["anyou"] = predictions[0]
        # print(predictions[0])
        # self.logger.debug(self.context)

    def suqiu_identify(self):
        if self.config.prejudgment_type == "criminal":
            self.context["suqiu"] = "量刑推荐"

    def situation_identify(self):
        if self.context["anyou"] == "盗窃":
            criminal_type = "theft"
        elif self.context["anyou"] == "容留他人吸毒":
            criminal_type = "provide_drug"
        else:
            # self.context["report_result"] = {
            #     "敬请期待": f"目前仅支持盗窃罪和容留他人吸毒罪的预测，你的行为属于{self.context['anyou']}犯罪，目前还未上线，正在训练优化中，敬请期待！"
            # }
            self.context["report_result"] = {"敬请期待": "正在训练中"}
            return

        self.logger.info("模型抽取！")
        t0 = time.time()
        self.context["event"] = post_process_uie_results(predictor=self.predictor_dict[criminal_type],
                                                         criminal_type=criminal_type,
                                                         fact=self.context["fact"])
        self.logger.info("模型抽取耗时:", time.time() - t0)
        # self.logger.debug(self.context)

        # if self.context["event"]["行为"] is not None:
        #     self.context["graph_process"]["情节"] = 1

    def parse_anyou_config_file(self):
        if self.context["anyou"] not in ["盗窃", "容留他人吸毒"]:
            return

        config_path = "LawsuitPrejudgment/config/criminal"
        xmind_path = Path(config_path, self.context["anyou"], "base_logic.xmind")
        question_answers_path = Path(
            config_path, self.context["anyou"], "question_answers.csv"
        )
        report_path = Path(config_path, self.context["anyou"], "report.csv")
        sentence_path = Path(config_path, self.context["anyou"], "sentences.csv")

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

        self.context["base_logic_graph"] = base_logic_dict
        self.context["question_answers_config"] = question_answers_dict
        self.context["report_dict"] = report_dict
        self.context["sentence_keywords"] = sentence_keywords

        self.context["graph_process"] = {
            key: 0 for key in base_logic_dict[self.context["anyou"]].keys()
        }
        self.context["graph_process_content"] = {
            key: "" for key in base_logic_dict[self.context["anyou"]].keys()
        }

    def get_circumstance_of_question(self, question):

        for circumstance, info in self.context["question_answers_config"].items():
            if str(question) == str(info.get("question")) + ":" + str(info.get("answer")).replace("|", ";"):
                return circumstance, info
        return None, None

    def match_graph(self):
        if (
                list(self.context["base_logic_graph"][self.context["anyou"]]["量刑"].keys())[0] == "【量刑】"
        ):
            self.context["graph_process"]["量刑"] = 1
            self.context["graph_process_content"]["量刑"] = "量刑"

        # if self.context["anyou"] == "容留他人吸毒":
        self.logger.trace(self.context["event"])
        self.logger.trace(self.context["sentence_keywords"])

        for index, row in self.context["sentence_keywords"].iterrows():
            # schema_list = row["schema"].split("|")
            # for schema in schema_list:
            # if row["sentence"] in self.context["event"][schema]:
            if len(re.findall(row["sentences"], self.context["fact"])) > 0:
                self.context["graph_process"]["情节"] = 1
                self.context["graph_process_content"]["情节"] = row["crime_plot"]
                # break
            if self.context["graph_process"]["情节"] == 1:
                break

        if self.context["anyou"] == "盗窃" and self.context["event"]["总金额"] != "":
            self.context["graph_process"]["量刑"] = 1

            # output_amount = cn2an.cn2an(self.context["event"]["总金额"], "smart")
            output_amount = cn2an.transform(self.context["event"]["总金额"], "cn2an")
            output_amount = re.findall("\d+", output_amount)[0]
            output_amount = float(output_amount)

            self.logger.debug("output_amount", output_amount)
            if output_amount < 1500:
                self.context["graph_process_content"]["量刑"] = "0-1500（不含）"
            elif output_amount < 3000:
                self.context["graph_process_content"]["量刑"] = "1500（含）-3000（不含）"
            elif output_amount < 40000:
                self.context["graph_process_content"]["量刑"] = "3000（含）-40000（不含）"
            elif output_amount < 80000:
                self.context["graph_process_content"]["量刑"] = "40000（含）-80000（不含）"
            elif output_amount < 100000:
                self.context["graph_process_content"]["量刑"] = "80000（含）-100000（不含）"
            elif output_amount < 150000:
                self.context["graph_process_content"]["量刑"] = "100000（含）-150000（不含）"
            elif output_amount < 200000:
                self.context["graph_process_content"]["量刑"] = "150000（含）-200000（不含）"
            elif output_amount < 250000:
                self.context["graph_process_content"]["量刑"] = "200000（含）-250000（不含）"
            elif output_amount < 300000:
                self.context["graph_process_content"]["量刑"] = "250000（含）-300000（不含）"
            elif output_amount < 350000:
                self.context["graph_process_content"]["量刑"] = "300000（含）-350000（不含）"
            elif output_amount < 400000:
                self.context["graph_process_content"]["量刑"] = "350000（含）-400000（不含）"
            else:
                self.context["graph_process_content"]["量刑"] = "400000（含）以上"

    def update_by_question(self):
        # 通过问题将没有覆盖到的点进行点亮
        if len(self.context["question_answers"]) > 0:
            for key in self.context["question_answers"].keys():
                self.context["graph_process"][key] = 1
                if self.context["graph_process_content"][key] == "":
                    self.context["graph_process_content"][key] = self.context["question_answers"][key]["usr_answer"]
                    self.context["question_answers"][key]["status"] = 1

            if self.context["question_answers"]["前提"]["usr_answer"] == "是":
                self.context["graph_process"]["情节"] = 1
                self.context["graph_process"]["量刑"] = 1

    def get_question(self):
        self.logger.trace(self.context["question_answers"])

        # # 通过问题将没有覆盖到的点进行点亮
        # if len(self.context["question_answers"]) > 0:
        #     for key in self.context["question_answers"].keys():
        #         self.context["graph_process"][key] = 1
        #         if self.context["graph_process_content"][key] == "":
        #             self.context["graph_process_content"][key] = self.context["question_answers"][key]["usr_answer"]
        #             self.context["question_answers"][key]["status"] = 1
        #
        #     if self.context["question_answers"]["前提"]["usr_answer"] == "是":
        #         self.context["graph_process"]["情节"] = 1
        #         self.context["graph_process"]["量刑"] = 1

        self.logger.debug(self.context["graph_process"])
        self.logger.debug(self.context["question_answers"])

        for key, value in self.context["graph_process"].items():
            if value == 0:
                qa_dict = self.context["question_answers_config"][key]
                # qa_dict.pop("circumstances")
                qa_dict["usr_answer"] = ""
                qa_dict["status"] = 0
                self.context["question_answers"][key] = qa_dict
                return qa_dict
        # self.logger.debug(self.context["question_answers"])

    def get_report(self):
        if "report_result" in self.context:
            return self.context["report_result"]

        evaluation_report = dict()

        # evaluation_report["案件事实"] = "根据您的描述，【时间】，在【地点】，【人物】存在【行为】等行为，窃得【物品】、【物品】等财物，盗窃总金额为【总金额】。"
        # evaluation_report["评估理由"] = "【盗窃罪XMind评估理由】【量刑情节XMind评估理由】"
        # evaluation_report["法律建议"] = "【盗窃罪XMind法律建议】"
        # evaluation_report["法律依据"] = "【盗窃罪XMind法律依据】【量刑情节XMind法律依据】"

        # if self.context["question_answers"]["前提"]["usr_answer"] == "是":

        self.logger.debug(self.context["event"])
        self.logger.debug(self.context["graph_process_content"])
        self.logger.debug(self.context["base_logic_graph"])
        # self.logger.debug(self.context)
        self.logger.info("开始生成报告")

        # for key,value in self.context["base_logic_graph"][self.context["anyou"]].items():
        if self.context["question_answers"]["前提"]["usr_answer"] == "是":
            report_id = "report-1"
        else:

            case_num = self.context["base_logic_graph"][self.context["anyou"]]["情节"][
                "【" + self.context["graph_process_content"]["情节"] + "】"]
            sentencing_dict = self.context["base_logic_graph"][self.context["anyou"]]["量刑"]

            if (
                    case_num not in sentencing_dict["【" + self.context["graph_process_content"]["量刑"] + "】"]
            ):
                case_num = "case0"

            report_id = sentencing_dict[
                "【" + self.context["graph_process_content"]["量刑"] + "】"
                ][case_num]

        _time = self.context["event"].get("时间", "")

        _location = self.context["event"].get("地点", "")
        _person = self.context["event"].get("人物", "")
        # _action = self.context["event"]["行为"]
        _action = self.context["graph_process_content"]["情节"]
        if _action != "以上都没有":
            _action = f"存在{_action}等情形。"
        else:
            _action = ""

        if _time != "":
            _time += "，"

        if _location != "":
            _location = "在" + _location + "，"

        if self.context["anyou"] == "盗窃":
            _thing = self.context["event"]["物品"]

            if "量刑" in self.context["question_answers"]:
                _amount = self.context["question_answers"]["量刑"]["usr_answer"]
            else:
                _amount = self.context["event"]["总金额"]

            evaluation_report[
                "案件事实"
            ] = f"根据您的描述，{_time}{_location}{_person}{_action}窃得{_thing}财物，盗窃金额为{_amount}。"

        elif self.context["anyou"] == "容留他人吸毒":
            _drug_name = self.context["event"]["毒品名称"]
            if _drug_name != "毒品":
                _drug_name = f"（俗称{_drug_name}）"
            else:
                _drug_name = ""

            _drug_type = self.context["event"]["毒品种类"]

            _provide_count = self.context["event"]["容留次数"]
            _provided_person = self.context["event"]["被容留人"]

            evaluation_report[
                "案件事实"
            ] = f"根据您的描述，{_time}{_location}{_person}容留{_provided_person}吸食{_drug_type}{_drug_name}{_provide_count}。{_action}"

        evaluation_report["涉嫌罪名"] = self.context["anyou"]
        evaluation_report["评估理由"] = self.context["report_dict"][report_id]["评估理由"]
        evaluation_report["法律依据"] = self.context["report_dict"][report_id]["法律依据"]
        evaluation_report["法律建议"] = self.context["report_dict"][report_id]["法律建议"]

        evaluation_report["相关类案"] = (
                self.context["report_dict"][report_id]["相似案例1"]
                + "|"
                + self.context["report_dict"][report_id]["相似案例2"]
        )

        evaluation_report["相关类案"] = evaluation_report["相关类案"].split("|")

        self.context["report_result"] = evaluation_report

        return self.context["report_result"]

    def recover_context(self, **kwargs):
        self.dialogue_history = kwargs["dialogue_history"]
        self.dialogue_state = kwargs["dialogue_state"]
        self.context = dict()
        # 恢复context
        if kwargs["context"]:
            self.context = kwargs["context"]

        # 初始化context中的部分内容。一般在首次问答时需要。
        if "fact" not in self.context:
            self.context["fact"] = self.dialogue_history.user_input
            if not self.context["fact"]:
                self.context["fact"] = ""
        if "question_answers" not in self.context:
            self.context["question_answers"] = dict()

    def nlu(self, **kwargs):
        if "anyou" not in self.context:
            self.anyou_identify()

        if "suqiu" not in self.context:
            self.suqiu_identify()

        if "event" not in self.context:
            self.situation_identify()

    def update_context(self, **kwargs):
        # 已经产生结果，则直接返回。如不支持的犯罪类型。
        if "report_result" in self.context:
            return

        # update context by last question_answer
        if self.dialogue_history.question_answers:
            last_question_answer = self.dialogue_history.question_answers[-1]
            circumstance = last_question_answer["other"]["circumstances"]
            answer = last_question_answer["user_answer"][0] # TODO: 多选时怎么处理
            self.context["question_answers"][circumstance]["usr_answer"] = answer
        # if "base_logic_graph" not in self.context:# TODO: 能取消注释吗
        self.parse_anyou_config_file()
        self.match_graph()
        self.update_by_question()

    def decide_next_action(self, **kwargs) -> str:
        if "report_result" in self.context:
            return "report"

        for key, value in self.context["graph_process"].items():
            if value == 0:
                return "ask"
        return "report"

    def get_next_question(self):
        question_info = self.get_question()
        return {
            "question": question_info["question"],
            "candidate_answers": str(question_info["answer"]).split("|"),
            "question_type": "single" if str(question_info["multiplechoice"]) == "0" else "multiple",
            "other": {
                "circumstances": question_info["circumstances"]
            }
        }

    def generate_report(self, **kwargs):
        report_result = self.get_report()
        # TODO
        return report_result