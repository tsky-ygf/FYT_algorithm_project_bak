import abc


class SituationClassifier(abc.ABC):
    """ 抽象基类：用于法律情形识别 """

    @abc.abstractmethod
    def classify_situations(self, text):
        pass
