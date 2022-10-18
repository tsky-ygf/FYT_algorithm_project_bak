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
    text: str = Field(default="", description="输入的文本")


@app.post("/get_corrected_contract_result")
async def _get_corrected_contract_result(text_input: TextInput):
    """
    获取纠错的结果及详情

    参数设定；

    @text: 输入文本
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
    uvicorn.run('OnlineServer.ContractCorrector.server:app', host="0.0.0.0", port=6598, reload=False, workers=1)
