#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    :
# @File    : server.py
# @Software: PyCharm
# import _io
# import time

import pymysql
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"参数不对{request.method} {request.url}")
    return JSONResponse({"code": "400", "error_msg": exc.errors(),"status": 1})

@app.post("/getNews")
async def get_news(category: str = Body(1, title='专栏名称', embed=True)):
    print(category)
    tabName = get_tabName(category)
    res = get_content(tabName)
    return {'res_data': res, "error_msg": "", "status": 0}

def get_tabName(category):
    if category == "税法专栏":
        return "swj_hot_news"
    elif category == "司法专栏":
        return "sfj_hot_news"
    elif category == "金融专栏":
        return "banking_hot_news"
    elif category == "市场监督":
        return "scjd_hot_news"
    elif category == "法院专栏":
        return "fy_hot_news"
    elif category == "公安专栏":
        return "ga_hot_news"
    elif category == "文旅专栏":
        return "wl_hot_news"
    elif category == "环保专栏":
        return "hb_hot_news"
    elif category == "交通专栏":
        return "jt_hot_news"
    elif category == "科技专栏":
        return "kj_hot_news"
    else:
        return ""

def get_content(tabName):
    select_sql = f"""select url,htmlContent,title,pubDate,source,content from {tabName} order by pubDate desc limit 30"""
    conn = pymysql.connect(host='172.19.82.227',db='hot_news',user='root',password='Nblh@2022',cursorclass=pymysql.cursors.DictCursor)
    curs = conn.cursor()
    curs.execute(select_sql)
    res = curs.fetchall()
    curs.close()
    conn.close()
    if res:
        return res
    else:
        return ''

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=7000)
if __name__ == "__main__":
    # 日志设置
    uvicorn.run('OnlineServer.CommonLaws.server:app', host="0.0.0.0", port=8149, reload=False, workers=1)
