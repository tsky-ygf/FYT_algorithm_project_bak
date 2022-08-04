import json
import logging
import traceback

import requests
from typing import Dict
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier import SituationClassifier
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier_message import \
    SituationClassifierMessage


class HttpClient:
    def __init__(self, url, method="post"):
        self.url = url
        self.method = method

    def get_response_json(self, request_data: Dict):
        try:
            if self.method == "post":
                resp_json = requests.post(url=self.url, json=request_data).json()
            elif self.method == 'get':
                resp_json = requests.get(url=self.url, params=request_data).json()
            else:
                raise Exception("method应是post或get。传入了不支持的method:{}。".format(self.method))

            return json.loads(resp_json) if isinstance(resp_json, str) else resp_json
        except Exception:
            logging.exception(traceback.format_exc())
            return dict()
        pass


class DataTransferObject:
    def __init__(self):
        self.data_from_http = None
        self.claim_convert_dict = {
            "财产分割": "请求分割财产"
        }
        self.situation_convert_dict = {
            "夫妻关系存续期间所得生产、经营的收益": "财产为婚姻关系存续期间夫妻的共同财产"
        }

    @property
    def claim(self) -> str:
        claim_from_http = self.data_from_http.get("suqiu_type", "")
        return self.claim_convert_dict.get(claim_from_http, "")

    @property
    def situation(self) -> str:
        situation_from_http = self.data_from_http.get("situation", "")
        return self.situation_convert_dict.get(situation_from_http, "")

    @property
    def probability(self):
        return self.data_from_http.get("probability")

    def convert_response_format(self, data_from_http: Dict) -> Dict:
        self.data_from_http = data_from_http
        return {
            "claim": self.claim,
            "situation": self.situation,
            "probability": self.probability,
            "status": self.data_from_http.get("status")
        }


class HttpBasedSituationClassifier(SituationClassifier):
    """ 通过外部的http服务，识别法律情形。 """

    def __init__(self, http_client: HttpClient, dto: DataTransferObject = None):
        """

        Args:
            http_client: http客户端。
            dto: 转换数据格式的类: http接口数据格式 -> 业务数据格式。
        """
        self.http_client = http_client
        self.dto = dto

    def classify_situations(self, message: SituationClassifierMessage):
        resp_json = self.http_client.get_response_json(message.to_dict())
        if self.dto:
            resp_json = self.dto.convert_response_format(resp_json)
        return resp_json
