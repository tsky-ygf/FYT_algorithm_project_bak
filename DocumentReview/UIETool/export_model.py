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

import os

import paddle

from DocumentReview.UIETool.model import UIE

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def export_model_onnx(model_path, output_path):
    model = UIE.from_pretrained(model_path)
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='input_ids'),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='token_type_ids'),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='pos_ids'),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='att_mask'),
        ])
    # Save in static graph model.
    save_path = os.path.join(output_path, "inference")
    paddle.jit.save(model, save_path)


if __name__ == '__main__':
    for export_type in ['fangwuzulin', 'jiekuan', 'jietiao', 'laodong', 'laowu', 'maimai', 'caigou', 'baomi',
                        'yibanzulin']:
        export_model_onnx(
            model_path="model/uie_model/new/{}/model_best".format(export_type),
            output_path="model/uie_model/export_cpu/{}".format(export_type),
        )
    print("Done.")
