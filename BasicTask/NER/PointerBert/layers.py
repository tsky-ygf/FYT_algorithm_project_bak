#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 10:50
# @Author  : Czq
# @File    : layers.py
# @Software: PyCharm
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        print("pe shape", pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x, bias=0):
        x = x +self.pe[:, bias:(x.size(1)+bias)].clone().detach().requires_grad_(False)
        return self.dropout(x)


if __name__ == "__main__":
    pe = PositionalEncoding(200, 0, 1000)
    h = torch.rand((16,512,200))
    y = pe(h)
    print(y.shape)


    pass
