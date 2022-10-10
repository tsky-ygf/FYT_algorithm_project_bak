import abc

from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier_message import \
    SituationClassifierMessage


class SituationClassifier(abc.ABC):
    """ 抽象基类：用于法律情形识别 """

    @abc.abstractmethod
    def classify_situations(self, message: SituationClassifierMessage):
        pass
