#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 17:32
# @Author  : Adolf
# @Site    :
# @File    : parse_xmind.py
# @Software: PyCharm
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


def maker_judgment(makers):
    maker = 4
    if "1" in makers:
        maker = 0
    elif "2" in makers:
        maker = 1
    elif "3" in makers:
        maker = 2
    return maker


def deal_with_list_dict(_case):
    print(_case)
    for con in _case:
        one_con = con.split("&")
        print(one_con)
        x = 0
        break


if __name__ == "__main__":
    root_ = xmind_to_dict("LawsuitPrejudgment/Criminal/base_config/盗窃/base_logic.xmind")[0]["topic"]
    # print(root_)
    case = get_case(root_)
    deal_with_list_dict(case)
