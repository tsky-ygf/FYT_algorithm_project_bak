#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 17/10/2022 15:02 
@Desc    : None
"""
from Utils.logger import get_logger
from dataclasses import dataclass

from typing import List


@dataclass
class PrejudgmentConfig:
    log_level: str = "INFO"
    log_path: str = ""
    prejudgment_type: str = ""
    anyou_identify_model_path: str = ""
    situation_identify_model_path: List[str] = ""


class PrejudgmentPipeline:
    def __init__(self, *args, **kwargs) -> None:
        self.config = PrejudgmentConfig(*args, **kwargs)
        self.logger = get_logger(self.config.log_level, self.config.log_path)
        self.context = dict()
        self.dialogue_history = dict()
        self.dialogue_state = dict()

    def init_context(self):
        self.context = dict()

    def recover_context(self, **kwargs):
        raise NotImplemented

    def nlu(self, **kwargs):
        raise NotImplemented

    def update_context(self, **kwargs):
        raise NotImplemented

    def decide_next_action(self, **kwargs) -> str:
        raise NotImplemented

    def get_next_question(self):
        raise NotImplemented

    def generate_report(self, **kwargs):
        raise NotImplemented

    def __call__(self, **kwargs):
        self.recover_context(**kwargs)
        # self.context.update(kwargs)

        self.nlu()
        # if "question_answers" not in self.context:
        #     self.context["question_answers"] = dict()
        #
        # if "anyou" not in self.context:
        #     self.anyou_identify()
        #
        # if "suqiu" not in self.context:
        #     self.suqiu_identify()
        #
        # if "base_logic_graph" not in self.context:
        #     self.parse_config_file()
        #
        # if "event" not in self.context:
        #     self.situation_identify()
        #     if "report_result" in self.context:
        #         return self.context
        #     self.match_graph()

        self.update_context()

        # dialogue policy
        next_action_type = self.decide_next_action()

        if next_action_type == "ask":
            action_result = self.get_next_question()
        elif next_action_type == "report":
            action_result = self.generate_report()
        else:
            action_result = None
            # self.get_question()

        # for key, value in self.context["graph_process"].items():
        #     if value == 0:
        #         return self.context

        # self.generate_report()
        # self.logger.info(pformat(self.context))

        # return result
        self.dialogue_state.other = self.context
        return {
            "dialogue_history": self.dialogue_history,
            "dialogue_state": self.dialogue_state,
            "next_action": {
                "action_type": next_action_type,
                "content": action_result
            }
        }
