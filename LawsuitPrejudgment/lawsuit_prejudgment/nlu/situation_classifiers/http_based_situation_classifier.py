from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier import SituationClassifier


class HttpBasedSituationClassifier(SituationClassifier):
    """ 通过外部的http服务，识别法律情形。 """

    def __init__(self, http_client):
        self.http_client = http_client

    def classify_situations(self, text):
        pass
