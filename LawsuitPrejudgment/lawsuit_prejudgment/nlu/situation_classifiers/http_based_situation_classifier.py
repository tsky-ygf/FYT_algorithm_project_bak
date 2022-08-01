import logging
import traceback

import requests
from typing import Dict
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier import SituationClassifier
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier_message import \
    SituationClassifierMessage


class HttpClient:
    def __init__(self, url, method="post"):
        if method not in ["post", "get"]:
            raise Exception("method应是post或get。传入了不支持的method:{}。".format(method))
        self.url = url
        self.method = method

    def get_response_json(self, request_data: Dict):
        try:
            if self.method == "post":
                return requests.post(url=self.url, json=request_data).json()
            elif self.method == "get":
                return requests.get(url=self.url, params=request_data).json()
        except Exception:
            logging.exception(traceback.format_exc())
            return dict()
        pass


class HttpBasedSituationClassifier(SituationClassifier):
    """ 通过外部的http服务，识别法律情形。 """

    def __init__(self, http_client: HttpClient):
        self.http_client = http_client

    def classify_situations(self, message: SituationClassifierMessage):
        return self.http_client.get_response_json(message.to_dict())
