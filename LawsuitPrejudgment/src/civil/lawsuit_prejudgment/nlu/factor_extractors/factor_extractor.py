import abc


class FactorExtractor(abc.ABC):
    """ 抽象基类：用于法律要素抽取 """

    @abc.abstractmethod
    def extract_factors(self, text):
        pass
