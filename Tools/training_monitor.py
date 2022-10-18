#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 10:23
# @Author  : Adolf
# @Site    : 
# @File    : training_monitor.py
# @Software: PyCharm
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_json(file_path):
    """
    加载json文件
    :param file_path:
    :return:
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


def save_json(data, file_path):
    """
    保存成json文件
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


class TrainingMonitor:
    paths: dict

    def __init__(self, file_dir, arch, add_test=False):
        if isinstance(file_dir, Path):
            pass
        else:
            file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)

        self.arch = arch
        self.file_dir = file_dir
        self.H = {}
        self.add_test = add_test
        self.json_path = file_dir / (arch + "_training_monitor.json")

    def reset(self, start_at):
        if start_at > 0:
            if self.json_path is not None:
                if self.json_path.exists():
                    self.H = load_json(self.json_path)
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:start_at]

    def epoch_step(self, logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            # np.float32会报错
            if not isinstance(v, np.float):
                v = round(float(v), 4)
            l.append(v)
            self.H[k] = l

        # 写入文件
        if self.json_path is not None:
            save_json(data=self.H, file_path=self.json_path)

        # 保存train图像
        if len(self.H["loss"]) == 1:
            self.paths = {key: self.file_dir / (self.arch + f'_{key.upper()}') for key in self.H.keys()}

        if len(self.H["loss"]) > 1:
            # 指标变化
            # 曲线
            # 需要成对出现
            keys = [key for key, _ in self.H.items() if '_' not in key]
            for key in keys:
                N = np.arange(0, len(self.H[key]))
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, self.H[key], label=f"train_{key}")
                plt.plot(N, self.H[f"valid_{key}"], label=f"valid_{key}")
                if self.add_test:
                    plt.plot(N, self.H[f"test_{key}"], label=f"test_{key}")
                plt.legend()
                plt.xlabel("Epoch #")
                plt.ylabel(key)
                plt.title(f"Training {key} [Epoch {len(self.H[key])}]")
                plt.savefig(str(self.paths[key]))
                plt.close()
