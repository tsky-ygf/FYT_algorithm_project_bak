#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 10:16
# @Author  : Adolf
# @Site    : 
# @File    : extract_show.py
# @Software: PyCharm
import pandas as pd
import streamlit as st
from LawsuitPrejudgment.Criminal.extraction.feature_extraction import get_xing7_result
from LawsuitPrejudgment.Criminal.extraction.drug_extraction import get_drug_result


class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.radio(
            'Go To',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()


def base_extract():
    text = st.text_area(value="诸暨市人民检察院指控：2012年10月25日1时许，被告人郑小明伙同彭小清窜到诸暨市暨阳街道跨湖路99号永"
                              "鑫花园，撬断围栏进入小区，由被告人郑小明望风，彭小清通过下水道进入小区20幢5单元被害人杨燕家，窃得"
                              "部分财物，后两人翻墙逃跑。被告人郑小明在逃跑途中被民警抓获。被告人郑小明于2009年5月21日因犯盗窃"
                              "罪被江西省抚州市临川区人民法院判处有期徒刑4年6个月，于2012年9月7日刑满释放。被告人郑小明对公诉机关指控"
                              "的事实和罪名及证据均无异议，未提出辩解。", height=300, label="请输入裁判文书内容",
                        key="text")
    run = st.button("抽取")
    if run:
        res = get_xing7_result(text)[0]
        # st.write(res)
        for key, value in res.items():
            st.markdown(f'## {key}')
            for content in value:
                st.markdown(f'- {content["text"]}')
            st.write('-' * 50 + '分割线' + '-' * 50)


def drug_extract():
    st.title("抽取毒品相关内容")
    text = st.text_area(value="公诉机关指控，2015年1月13日22时30分许至23时30分，被告人陈某先后在重庆市江北区北城旺角X栋X楼、负X楼"
                              "附近，两次将共计净重1.33克的海洛因贩卖给左某。公诉机关当庭举示了相应证据证明其指控，据此认为"
                              "被告人陈某的行为触犯了《中华人民共和国刑法》××××、××的规定，已构成贩卖毒品罪，提请对其依法判处。"
                              "公诉机关指控，2012年初，被告人高某在苏州工业园区，以人民币2000元的价格向顾某出售甲基苯丙胺。",
                        height=300, label="请输入裁判文书内容",
                        key="text")
    run = st.button("抽取")

    span_map = dict()
    start_use = []
    if run:
        res_relations, res_span = get_drug_result(text)
        # st.write(res)
        for one_span, content in res_span.items():
            for one_content in content:
                span_map[one_content["start"]] = one_span
        # st.write(res_relations)
        # st.write(span_map)
        # exit()
        res_relations_list = res_relations["被告人"]
        # st.write(res_relations_list)
        beigao_list = []
        xingwei_list = []
        drug_type_list = []
        drug_amount_list = []
        drug_quantity_list = []

        for one_relation in res_relations_list:
            beigao = one_relation["text"]
            for one_xw, content in one_relation["relations"].items():
                xingwei = one_xw
                type_flag = 0
                amount_flag = 0
                quantity_flag = 0
                for one_content in content:
                    if one_content["start"] in start_use:
                        continue
                    else:
                        start_use.append(one_content["start"])
                    if span_map[one_content["start"]] == "毒品种类":
                        if type_flag == 0:
                            drug_type_list.append(one_content["text"])
                            type_flag = 1
                        else:
                            drug_type_list[-1] += "#" + one_content["text"]
                    if span_map[one_content["start"]] == "毒品金额":
                        if amount_flag == 0:
                            drug_amount_list.append(one_content["text"])
                            amount_flag = 1
                        else:
                            drug_amount_list[-1] += "#" + one_content["text"]
                    if span_map[one_content["start"]] == "毒品数量":
                        if quantity_flag == 0:
                            drug_quantity_list.append(one_content["text"])
                            quantity_flag = 1
                        else:
                            drug_quantity_list[-1] += "#" + one_content["text"]

                if type_flag == 0:
                    drug_type_list.append("无")
                if amount_flag == 0:
                    drug_amount_list.append("无")
                if quantity_flag == 0:
                    drug_quantity_list.append("无")
                beigao_list.append(beigao)
                xingwei_list.append(xingwei)

        res_dict = {"主体人": beigao_list,
                    "犯罪行为": xingwei_list,
                    "毒品种类": drug_type_list,
                    "毒品金额": drug_amount_list,
                    "毒品数量": drug_quantity_list}

        try:
            res_df = pd.DataFrame(res_dict)
            res_df = res_df.loc[res_df["毒品种类"] != "无"]
            st.table(res_df)
            st.write(res_relations)
        except Exception as e:
            print(e)
            st.write(res_relations)


app = MultiApp()
app.add_app("初步信息抽取", base_extract)
app.add_app("毒品内容抽取", drug_extract)
app.run()
