#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/4 17:10
# @Author  : Adolf
# @Site    : 
# @File    : base_graph.py
# @Software: PyCharm
import json
import pandas as pd
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from pprint import pprint

# %matplotlib inline

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# origin_df = pd.read_csv("data/law/law_lib/item_test.csv")
with open("data/law/law_lib/statistics_json.json", "r") as f:
    statistics_json = json.load(f)
pprint(statistics_json)
# print(origin_df[:10])
# for key, value in statistics_json.items():
law_item_index_dict = {}

G = nx.Graph()
for key, value in statistics_json.items():
    node1 = key.split("#")[0]
    node2 = key.split("#")[1]

    if node1 not in law_item_index_dict:
        law_item_index_dict[node1] = 'node_{}'.format(len(law_item_index_dict))

    if node2 not in law_item_index_dict:
        law_item_index_dict[node2] = 'node_{}'.format(len(law_item_index_dict))

    G.add_edge(law_item_index_dict[node1], law_item_index_dict[node2], weight=value)

# nx.draw_networkx(G)
# plt.show()
pos = nx.spring_layout(G)
#
Gdegree = nx.degree(G)
Gdegree = dict(Gdegree)
Gdegree = pd.DataFrame({'name': list(Gdegree.keys()), 'degree': list(Gdegree.values())})
# # node
nx.draw_networkx_nodes(G, pos, alpha=0.6, node_size=Gdegree.degree*10)
#
nx.draw_networkx_labels(G, pos, font_size=10)
plt.axis('off')
plt.title('law item graph')
plt.show()
