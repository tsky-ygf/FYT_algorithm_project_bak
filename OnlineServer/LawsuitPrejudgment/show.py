#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 15:28 
@Desc    : None
"""
import streamlit as st

from OnlineServer.LawsuitPrejudgment.testing_page.administrative.show import administrative_prejudgment_testing_page
from OnlineServer.LawsuitPrejudgment.testing_page.civil.show import civil_prejudgment_testing_page
from OnlineServer.LawsuitPrejudgment.testing_page.criminal.show import criminal_prejudgment_testing_page


def lawsuit_prejudgment_testing_page():
    tab1, tab2, tab3 = st.tabs(["民事", "刑事", "行政"])

    with tab1:
        civil_prejudgment_testing_page()

    with tab2:
        criminal_prejudgment_testing_page()

    with tab3:
        administrative_prejudgment_testing_page()
