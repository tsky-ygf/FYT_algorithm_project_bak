#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : server.py
# @Software: PyCharm
# import _io
# import time

import traceback
import uvicorn
from fastapi import FastAPI
import json
from pydantic import BaseModel, Field

from Corrector.server_use.server import get_corrected_contract_result

app = FastAPI()


class TextInput(BaseModel):
    text: str = Field(default="真麻烦你了。 希望你们好好的跳无", description="输入的文本")
    class Config:
        schema_extra = {
            "example": {
                "text": "真麻烦你了。 希望你们好好的跳无"
            }
        }


class TextResult(BaseModel):
    result: dict


@app.post("/get_corrected_contract_result", response_model=TextResult)
async def _get_corrected_contract_result(text_input: TextInput):
    """
    获取纠错的结果及详情

    请求参数：

    | Param             | Type  | Description  |
    |-------------------|-------|--------------|
    |        text       |  str  |    输入文本   |

    响应参数：

    | Param  | Type  | Description  |
    |--------|-------|--------------|
    | result | Dict  |  合同审核结果  |

    result的内容如下:
    * corrected_pred: string, 纠错后的结果
    * msg: string, 错误信息
    * success: boolean, 服务调用是否成功
    * detail_info: List[Tuple], 具体纠错信息
        * 元素0: string, 原始字
        * 元素1: string, 纠错后的字
        * 元素2: int, 在输入文本中的起始位置
        * 元素3: int, 在输入文本中的结束为止
    """
    result = {'corrected_pred': '', 'detail_info': [], 'msg': '', 'success': True}
    try:
        tgt_pred, pred_detail_list = get_corrected_contract_result(text_input.text)
        result['corrected_pred'] = tgt_pred
        result['detail_info'] = pred_detail_list
    except:
        result['msg'] = traceback.format_exc()
        result['success'] = False

    result = json.dumps(result, ensure_ascii=False)
    return result


if __name__ == "__main__":
    # 日志设置
    uvicorn.run('OnlineServer.ContractCorrector.server:app', host="0.0.0.0", port=8150, reload=False, workers=1)
