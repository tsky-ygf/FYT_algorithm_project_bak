#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/3 17:41
# @Author  : Adolf
# @Site    : 
# @File    : gen_pic.py
# @Software: PyCharm
import math
from PIL import Image, ImageFont, ImageDraw


def create_pic(text,
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
    size = len(text.encode('utf-8')) / 2
    # 计算字体大小，每两个字号按字节长度翻倍。
    fontSize = math.ceil((Iwidth - (margin * 2)) / size)

    font = ImageFont.truetype(font_type, fontSize)  # 设置字体及字号
    draw = ImageDraw.Draw(image)

    fwidth, fheight = draw.textsize(text, font)  # 获取文字高宽
    owidth, oheight = font.getoffset(text)

    fontx = (Iwidth - fwidth - owidth) / 2
    fonty = (Iheight - fheight - oheight) / 2

    draw.text((fontx, fonty), text, fontRGB, font)
    image.save(img_path)  # 保存图片


if __name__ == "__main__":
    create_pic("为什么是我呢为什么是我呢为什么是我呢", "DocumentReview/ContractReview/test.jpg")
