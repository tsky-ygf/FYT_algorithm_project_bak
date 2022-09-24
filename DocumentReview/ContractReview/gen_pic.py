#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/3 17:41
# @Author  : Adolf
# @Site    : 
# @File    : gen_pic.py
# @Software: PyCharm
import json
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path


def create_pic(texts,
               img_path,
               size=None,
               margin=50,
               background_RGB=None,
               font_type="DocumentReview/ContractReview/方正小标宋简体.ttf",
               font_RGB=None):
    if font_RGB is None:
        font_RGB = [0, 0, 0]
    if background_RGB is None:
        background_RGB = [255, 255, 255]
    if size is None:
        size = [320, 451]

    size = tuple(size)
    backgroundRGB = tuple(background_RGB)
    fontRGB = tuple(font_RGB)

    image = Image.new('RGB', size, backgroundRGB)  # 设置画布大小及背景色
    Iwidth, Iheight = image.size  # 获取画布高宽

    # 计算字节数，GBK编码下汉字双字，英文单字。都转为双字计算
    # size = len(text.encode('utf-8')) / 2
    # 计算字体大小，每两个字号按字节长度翻倍。
    # fontSize = math.ceil((Iwidth - (margin * 2)) / size)
    # print(size)
    fontSize = 20

    font = ImageFont.truetype(font_type, fontSize)  # 设置字体及字号
    draw = ImageDraw.Draw(image)

    for i in range(len(texts)):
        text = texts[i]
        fwidth, fheight = draw.textsize(text, font)  # 获取文字高宽
        owidth, oheight = font.getoffset(text)

        fontx = (Iwidth - fwidth - owidth) / 2
        fonty = (Iheight - fheight - oheight) / 3 + (fheight + oheight) * i

        draw.text((fontx, fonty), text, fontRGB, font)
    image.save(img_path)  # 保存图片


def create_pic_v2(texts,
                  img_path,
                  size=None,
                  background_RGB=None,
                  font_type="DocumentReview/ContractReview/方正小标宋简体.ttf",
                  font_RGB=None):
    if font_RGB is None:
        font_RGB = [0, 0, 0]
    if background_RGB is None:
        background_RGB = [255, 255, 255]
    if size is None:
        size = [320, 451]

    size = tuple(size)
    backgroundRGB = tuple(background_RGB)
    fontRGB = tuple(font_RGB)

    image = Image.new('RGB', size, backgroundRGB)  # 设置画布大小及背景色
    Iwidth, Iheight = image.size  # 获取画布高宽

    # 计算字节数，GBK编码下汉字双字，英文单字。都转为双字计算
    # size = len(text.encode('utf-8')) / 2
    # 计算字体大小，每两个字号按字节长度翻倍。
    # fontSize = math.ceil((Iwidth - (margin * 2)) / size)
    # print(size)
    fontSize = 20

    font1 = ImageFont.truetype(font_type, fontSize)  # 设置字体及字号
    font2 = ImageFont.truetype(font_type, 15)

    draw = ImageDraw.Draw(image)

    if "(" in texts or "（" in texts:
        texts = texts.replace('（', '(').replace('）', ')')
        text_list = texts.split("(")
        text1 = text_list[0]
        text2 = "(" + text_list[1]

        fwidth1, fheight1 = draw.textsize(text1, font1)  # 获取文字高宽
        owidth1, oheight1 = font1.getoffset(text1)

        fontx = (Iwidth - fwidth1 - owidth1) / 2
        fonty = (Iheight - fheight1 - oheight1) / 3

        draw.text((fontx, fonty), text1, fontRGB, font1)

        fwidth2, fheight2 = draw.textsize(text2, font2)  # 获取文字高宽
        owidth2, oheight2 = font2.getoffset(text1)

        fontx2 = (Iwidth - fwidth2 - owidth2) / 2
        fonty2 = fonty + fheight1 + oheight1 + 5

        draw.text((fontx2, fonty2), text2, fontRGB, font2)

    else:
        fwidth, fheight = draw.textsize(texts, font1)  # 获取文字高宽
        owidth, oheight = font1.getoffset(texts)

        fontx = (Iwidth - fwidth - owidth) / 2
        fonty = (Iheight - fheight - oheight) / 3

        draw.text((fontx, fonty), texts, fontRGB, font1)
    image.save(img_path)  # 保存图片


def get_csv():
    doc_list = []
    for doc in Path("data/DocData/合同模板20类").rglob("*.docx"):
        print(doc.stem)
        doc_list.append(doc.stem)

    res_dict = {"文件": doc_list}
    df = pd.DataFrame(res_dict)
    df.to_csv("data/DocData/合同模板20类.csv", index=False)


# main()
# get_csv()
def main():
    ori_df = pd.read_csv("data/DocData/补充图片.csv")
    # print(ori_df)
    for index, row in ori_df.iterrows():
        file = row['文件']
        file_list = file.split("|")
        # print(file_list)
        # if len(file_list) > 1:
        #     print(file_list)
        # continue
        pic_name = ''.join(file_list)
        create_pic(file_list, f"data/DocData/buchong/{pic_name}.jpg")


def main_v2():
    for doc in Path("data/word").rglob("*.docx"):
        doc_name = doc.stem
        pic_name = doc.stem
        # doc_name = doc_name.replace('(', '|').replace(')', '')
        # file_list = doc_name.split("|")
        # create_pic(file_list, f"data/DocData/buchong/{pic_name}.jpg")
        create_pic_v2(doc_name, f"data/DocData/buchong/{pic_name}.jpg")


if __name__ == "__main__":
    # create_pic(["二手车买卖居间服务合同（深圳市）"], "DocumentReview/ContractReview/test.jpg")
    # create_pic_v2("意见书(离婚案件)", "DocumentReview/ContractReview/test.jpg")
    # main()
    main_v2()
