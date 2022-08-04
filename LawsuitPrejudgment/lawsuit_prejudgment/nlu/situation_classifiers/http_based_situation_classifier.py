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


class HttpBasedSituationClassifier(SituationClassifier):
    """ 通过外部的http服务，识别法律情形。 """

    def __init__(self, http_client: HttpClient):
        self.http_client = http_client

    def classify_situations(self, message: SituationClassifierMessage):
        return self.http_client.get_response_json(message.to_dict())
