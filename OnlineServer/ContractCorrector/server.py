#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : server.py
# @Software: PyCharm
# import _io
# import time

from sre_constants import SUCCESS
import traceback
from construct import Enum
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

from enum import IntEnum
class ErrType(IntEnum):
    SPELLERR = 1#拼写纠错

class TextResult(BaseModel):
    corrected_pred: str
    msg: str
    success: int
    detail_info: list[dict]
    type: IntEnum


@app.post("/get_corrected_contract_result", response_model=TextResult)
async def _get_corrected_contract_result(text_input: TextInput):
    """
    获取纠错的结果及详情

    请求参数：

    | Param             | Type  | Description  |
    |-------------------|-------|--------------|
    |        text       |  str  |    输入文本   |

    响应参数：

    |     Param      |    Type    | Description  |
    |----------------|------------|--------------|
    | corrected_pred |    str     |    纠错结果   |
    |      msg       |    str     |     信息     |
    |    success     |  boolean   |    是否成功   |
    |   detail_info  | list[dict] |  具体纠错信息  |
    |      type      |  IntEnum   |    错误类型   |


    result的内容如下:
    * corrected_pred: string, 纠错后的结果
    * msg: string, 错误信息
    * success: boolean, 服务调用是否成功
    * detail_info: List[Dict], 具体纠错信息
        * ori_char: string, 原始字
        * new_char: string, 纠错后的字
        * start: int, 在输入文本中的起始位置
        * end: int, 在输入文本中的结束为止
    * type: IntEnum, 错误类型
    """
    result = {'corrected_pred': '', 'detail_info': [], 'msg': '', 'success': True, "type": ErrType.SPELLERR}
    try:
        tgt_pred, pred_detail_list = get_corrected_contract_result(text_input.text)
        result['corrected_pred'] = tgt_pred
        result['detail_info'] = pred_detail_list
    except:
        result['msg'] = traceback.format_exc()
        result['success'] = False

    # import pdb
    # pdb.set_trace()
    # result = json.dumps(result, ensure_ascii=False)
    # print(type(result))
    return result


if __name__ == "__main__":
    # 日志设置
    uvicorn.run('OnlineServer.ContractCorrector.server:app', host="0.0.0.0", port=8150, reload=False, workers=1)
