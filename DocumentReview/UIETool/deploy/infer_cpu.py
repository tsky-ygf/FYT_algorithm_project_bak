# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pprint import pprint

from uie_predictor import UIEPredictor
from dataclasses import dataclass


# def parse_args():"
#     parser = argparse.ArgumentParser()
#     # Required parameters
#     parser.add_argument(
#         "--model_path_prefix",
#         type=str,
#         required=True,
#         help="The path prefix of inference model to be used.",
#     )
#     parser.add_argument(
#         "--position_prob",
#         default=0.5,
#         type=float,
#         help="Probability threshold for start/end index probabiliry.",
#     )
#     parser.add_argument(
#         "--max_seq_len",
#         default=512,
#         type=int,
#         help=
#         "The maximum input sequence length. Sequences longer than this will be split automatically.",
#     )
#     parser.add_argument("--batch_size",
#                         default=4,
#                         type=int,
#                         help="Batch size per CPU for inference.")
#     args = parser.parse_args()
#     return args

class InferArgs:
    model_path_prefix = ""
    position_prob = 0.5
    max_seq_len = 512
    batch_size = 4
    device = "cpu"
    schema = []


def cpu_infer(model_path_prefix, texts, schema):
    args = InferArgs()

    args.model_path_prefix = model_path_prefix
    args.schema = schema
    predictor = UIEPredictor(args)

    print("-----------------------------")
    outputs = predictor.predict(texts)
    pprint(outputs)


if __name__ == "__main__":
    model_path = "model/uie_model/export_cpu/maimai/inference"
    texts_ =["北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"]
    schema1 = ['法院', '原告', '被告']

    cpu_infer(model_path, texts_, schema1)
