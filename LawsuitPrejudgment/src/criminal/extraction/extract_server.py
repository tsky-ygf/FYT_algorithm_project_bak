#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/25 17:37
# @Author  : Adolf
# @Site    : 
# @File    : extract_server.py
# @Software: PyCharm
import os
import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel


from LawsuitPrejudgment.src.criminal.extraction.feature_extraction import (
    init_extract,
    post_process_uie_results,
)

criminal_list = ["theft", "provide_drug"]
predictor_dict = {}
for criminal_type in criminal_list:
    # model_path = "model/uie_model/export_cpu/{}/inference".format(criminal_type)
    predictor_dict[criminal_type] = init_extract(criminal_type=criminal_type)

app = FastAPI()


class Item(BaseModel):
    criminal_type: str = "theft"
    fact: str = "我偷了同事的3000元"


@app.post("/information_result")
async def get_information_result(item: Item):
    # result = predictor_dict[_criminal_type].predict([_fact])
    logger.info(item)
    result = post_process_uie_results(
        predictor_dict[item.criminal_type], item.criminal_type, item.fact
    )
    logger.info(result)
    return {"result": result}


if __name__ == "__main__":
    # 日志设置
    dir_log = "log/lawsuit_prejudgment"
    path_log = os.path.join(dir_log, 'criminal_extract.log')
    # 路径，每日分割时间，是否异步记录，日志是否序列化，编码格式，最长保存日志时间
    logger.add(path_log, rotation='0:00', enqueue=True, serialize=False, encoding="utf-8", retention="10 days")
    logger.debug("服务器重启！")
    uvicorn.run('LawsuitPrejudgment.Criminal.extraction.extract_server:app', host="0.0.0.0", port=7777, reload=False,
                workers=4)
