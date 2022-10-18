
import time
from dataclasses import dataclass

from Corrector.src.corrector import Model

@dataclass
class CommonModelArgs:#模型参数
    pass

def init_model():#模型初始化
    common_model_args = CommonModelArgs()
    print('=' * 50, '模型初始化...', '=' * 50)
    print(time.localtime())
    model = Model()
    return model


model = init_model()

def get_corrected_contract_result(text: str):
    result = model.process(text)
    return result