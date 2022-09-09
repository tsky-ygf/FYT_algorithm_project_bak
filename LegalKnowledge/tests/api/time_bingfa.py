#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 18:54
# @Author  : Adolf
# @Site    :
# @File    : time_bingfa.py
# @Software: PyCharm
import base64
import os
import urllib
import numpy as np
import requests, time, json, threading, random


class Presstest(object):
    """
    并发压力测试
    """

    def __init__(self, press_url):
        self.press_url = press_url

    def test_interface(self):
        """压测接口"""
        global INDEX
        INDEX += 1

        global ERROR_NUM
        global TIME_LENS
        try:
            start = time.time()
            # payload = {"criminal_type": "theft", "fact": "我偷了同事的3000元", }
            payload = {"news_id": 31}
            response_content = self.do_request(self.press_url, payload)
            result = json.loads(response_content)
            end = time.time()
            TIME_LENS.append(end - start)
            print("end")
        except Exception as e:
            ERROR_NUM += 1
            print(e)

    def test_onework(self):
        """一次并发处理单个任务"""
        i = 0
        while i < ONE_WORKER_NUM:
            i += 1
            self.test_interface()
        time.sleep(LOOP_SLEEP)

    def do_request(self, url, payload):
        """通用http获取webapi请求结果方法"""

        headers = {
            "Content-Type": "application/json; charset=UTF-8",
        }
        request = urllib.request.Request(
            url, json.dumps(payload).encode("utf-8"), headers=headers
        )
        retry_num = 0
        while retry_num < 3:
            response = urllib.request.urlopen(request, timeout=300)
            if not response or response.status == 421:
                time.sleep(1)
                retry_num = retry_num + 1
                continue
            else:
                break
        response_content = response.read()
        if hasattr(response_content, "decode"):
            response_content = response_content.decode("utf-8")
        return response_content

    def run(self):
        """使用多线程进程并发测试"""
        t1 = time.time()
        Threads = []

        for i in range(THREAD_NUM):
            t = threading.Thread(target=self.test_onework, name="T" + str(i))
            t.setDaemon(True)
            Threads.append(t)

        for t in Threads:
            t.start()
        for t in Threads:
            t.join()
        t2 = time.time()

        print("===============压测结果===================")
        print("URL:", self.press_url)
        print("任务数量:", THREAD_NUM, "*", ONE_WORKER_NUM, "=", THREAD_NUM * ONE_WORKER_NUM)
        print("总耗时(秒):", t2 - t1)
        print("每次请求耗时(秒):", (t2 - t1) / (THREAD_NUM * ONE_WORKER_NUM))
        print("每秒承载请求数:", 1 / ((t2 - t1) / (THREAD_NUM * ONE_WORKER_NUM)))
        print("错误数量:", ERROR_NUM)
        print(INDEX)


if __name__ == "__main__":
    # LEGAL_KNOWLEDGE_SERVICE_URL = "http://101.69.229.138:8120"
    # press_url = "http://172.19.82.199:7777/information_result"
    press_url = "http://47.111.0.124:8110/recommend_laws"
    TIME_LENS = []
    INDEX = 0
    THREAD_NUM = 100  # 并发线程总数
    ONE_WORKER_NUM = 100  # 每个线程的循环次数
    LOOP_SLEEP = 0  # 每次请求时间间隔(秒)
    ERROR_NUM = 0  # 出错数

    obj = Presstest(press_url)
    obj.run()
