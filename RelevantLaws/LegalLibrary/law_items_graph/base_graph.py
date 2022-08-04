#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/4 17:10
# @Author  : Adolf
# @Site    : 
# @File    : base_graph.py
# @Software: PyCharm
import pandas as pd
import networkx as nx
import itertools
import matplotlib.pyplot as plt

origin_df = pd.read_csv("data/law/law_lib/item_test.csv")
demo_df = origin_df[:100]
print(demo_df)
# print(origin_df[:10])

G = nx.Graph()
for i in demo_df.index:
    # print(demo_df.law_items[i])
    law_list = demo_df.law_items[i].split("|")
    # print(law_list)
    for e in itertools.combinations(law_list, 2):
        G.add_edge(e[0], e[1])
    # G.add_edge(law_list[0], law_list[1], weight=1)
    # break

pos = nx.spring_layout(G)

Gdegree = nx.degree(G)
Gdegree = dict(Gdegree)
Gdegree = pd.DataFrame({'name': list(Gdegree.keys()), 'degree': list(Gdegree.values())})
# node
nx.draw_networkx_nodes(G, pos, alpha=0.6, node_size=Gdegree.degree * 100)

nx.draw_networkx_labels(G, pos, font_size=10)
plt.axis('off')
plt.title('红楼梦社交网络')
plt.show()
