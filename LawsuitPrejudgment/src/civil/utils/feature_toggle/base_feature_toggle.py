#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 13:56 
@Desc    :
    功能开关的基类。
    代码参考:https://github.com/vwt-digital/feature-toggles/blob/develop/featuretoggles/__init__.py。
    与原版相比做了一些改动，用法见测试用例LawsuitPrejudgment/tests/test_feature_toggles.py。
"""
import inspect
import os
from dataclasses import dataclass

import yaml

import logging

logger = logging.getLogger("Feature Toggles")


@dataclass(frozen=True)
class ToggleItem:
    enabled: bool
    name: str
    description: str = ""


class BaseFeatureToggle:
    """ Base class for toggle features."""
    def __init__(self, document):
        if os.path.isfile(document):
            with open(document, 'r', encoding="utf-8") as f:
                self._toggle_config = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            self._toggle_config = yaml.load(document, Loader=yaml.SafeLoader)

        if not hasattr(self, '__annotations__'):
            raise Exception("No toggles are declared")

        not_declared = set(self._toggle_config) - set(self.__annotations__)
        if not_declared:
            raise Exception(f"The following toggles are not declared: {not_declared}")

        not_configured = set(self.__annotations__) - set(self._toggle_config)
        if not_configured:
            raise Exception(f"The following toggles are not configured: {not_configured}")

        for toggle in self._toggle_config:
            self.__setattr__(toggle, ToggleItem(**self._toggle_config.get(toggle)))

    def __getattribute__(self, attr):
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        if not attr.startswith("_") and attr in self._toggle_config:
            logger.info(f"Checking toggle {attr} in {calframe[1][1]}:{calframe[1][2]} {calframe[1][3]}()")
        return super().__getattribute__(attr)