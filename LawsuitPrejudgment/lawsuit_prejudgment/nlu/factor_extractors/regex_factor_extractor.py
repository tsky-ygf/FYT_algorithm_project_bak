import re
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.factor_extractors.factor_extractor import FactorExtractor
from LawsuitPrejudgment.lawsuit_prejudgment.shared.nlu.constants import (
    FACTOR_TYPE,
    FACTOR_START,
    FACTOR_END,
    FACTOR_MENTION,
    FACTOR_CONFIDENCE
)


class RegexFactorExtractor(FactorExtractor):
    """ 通过正则匹配(人工运营的词组、同义词和规则)的方式，抽取法律要素 """

    def __init__(self, patterns):
        self.patterns = patterns if patterns else []

    def extract_factors(self, text):
        factors = []

        for pattern in self.patterns:
            matches = re.finditer(pattern["pattern"], text)

            for match in matches:
                start_index = match.start()
                end_index = match.end()
                factors.append(
                    {
                        FACTOR_TYPE: pattern["name"],
                        FACTOR_START: start_index,
                        FACTOR_END: end_index,
                        FACTOR_MENTION: text[start_index:end_index],
                        FACTOR_CONFIDENCE: None
                    }
                )
        return factors
