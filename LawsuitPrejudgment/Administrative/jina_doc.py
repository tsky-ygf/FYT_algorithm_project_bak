#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 18:02
# @Author  : Adolf
# @Site    : 
# @File    : jina_doc.py
# @Software: PyCharm
from docarray import Document, DocumentArray

# break large text into smaller chunks
doc = Document(uri="https://www.gutenberg.org/files/1342/1342-0.txt").load_uri_to_text()

# apply feature hashing to embed the DocumentArray
docs = DocumentArray(Document(text=s.strip()) for s in doc.text.split('\n') if s.strip())
docs.apply(lambda doc: doc.embed_feature_hashing())

# query sentence
query = (Document(text="she entered the room").embed_feature_hashing().match(docs, limit=5, exclude_self=True,
                                                                             metric="jaccard", use_scipy=True))

# fetch the output
output = query.matches[:, ('text', 'scores__jaccard')][0]

# print the results
for i in (output):
    print(i)
