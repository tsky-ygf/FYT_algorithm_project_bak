from LawsuitPrejudgment.lawsuit_prejudgment.nlu.factor_extractors.factor_extractor import FactorExtractor
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier import SituationClassifier
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier_message import \
    SituationClassifierMessage


class FactorBasedSituationClassifier(SituationClassifier):
    """ 通过法律要素，识别法律情形。 """

    def __init__(self, factor_extractor: FactorExtractor):
        self.factor_extractor = factor_extractor

    def classify_situations(self, message: SituationClassifierMessage):
        pass
