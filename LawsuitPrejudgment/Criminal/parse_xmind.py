#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 17:32
# @Author  : Adolf
# @Site    :
# @File    : parse_xmind.py
# @Software: PyCharm
import pandas as pd
from xmindparser import xmind_to_dict


def traversal_xmind(root, root_string, list_container):
    """
    功能：递归dictionary文件得到容易写入Excel形式的格式。
    注意：root string都用str来处理中文字符
    @param root: 将xmind处理后的dictionary文件
    @param root_string: xmind根标题
    @param list_container: 用于存储最终结果的list
    """
    if isinstance(root, dict):
        if "title" in root.keys() and "topics" in root.keys():
            traversal_xmind(root["topics"], str(root_string), list_container)
        if "title" in root.keys() and "topics" not in root.keys():
            traversal_xmind(root["title"], str(root_string), list_container)
    elif isinstance(root, list):
        for son_root in root:
            if "makers" in son_root and "callout" in son_root:
                traversal_xmind(
                    son_root,
                    str(root_string)
                    + "&"
                    + son_root["title"]
                    + "&"
                    + str(son_root["makers"][0])
                    + "&"
                    + str(son_root["callout"][0]),
                    list_container,
                )
            elif "callout" in son_root and "makers" not in son_root:
                traversal_xmind(
                    son_root,
                    str(root_string)
                    + "&"
                    + son_root["title"]
                    + "&"
                    + str(son_root["callout"][0]),
                    list_container,
                )
            elif "makers" in son_root and "callout" not in son_root:
                traversal_xmind(
                    son_root,
                    str(root_string)
                    + "&"
                    + son_root["title"]
                    + "&"
                    + str(son_root["makers"][0])
                    + "&"
                    + "",
                    list_container,
                )
            else:
                traversal_xmind(
                    son_root, str(root_string) + "&" + son_root["title"], list_container
                )

    elif isinstance(root, str):
        list_container.append(str(root_string))  # 此处是一步骤多结果时，多结果合并


def get_case(root):
    root_string = root["title"]
    list_container = []
    traversal_xmind(root, root_string, list_container)
    return list_container


def deal_xmind_to_df(xmind_path):
    root = xmind_to_dict(xmind_path)[0]["topic"]
    case = get_case(root)
    case = [con.split("&") for con in case]
    # print(_case)
    case_df = pd.DataFrame(case)
    # print(case_df)
    return case_df


if __name__ == "__main__":
    file_path = "LawsuitPrejudgment/Criminal/base_config/盗窃/base_logic.xmind"
    print(deal_xmind_to_df(file_path))
