#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 16:46
# @Author  : Adolf
# @Site    : 
# @File    : show_page.py
# @Software: PyCharm
import streamlit as st
from OnlineServer.ContractReview.show import contract_review_main
from OnlineServer.LawsuitPrejudgment.testing_page.show import civil_prejudgment_testing_page
from OnlineServer.ProfessionalSearch.show import search


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


def welcome():
    st.title("欢迎来到法域通测试页面！")


app = MultiApp()
app.add_app("首页", welcome)
app.add_app("合同智审测试服务", contract_review_main)
app.add_app("民事预判测试服务", civil_prejudgment_testing_page)
app.add_app("检索测试服务", search)
app.run()
