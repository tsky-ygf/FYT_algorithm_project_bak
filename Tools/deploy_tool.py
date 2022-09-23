import torch
from onnxruntime import InferenceSession
from transformers import AutoTokenizer


def pt2onnx(model_ckpt_path, out_model_path, in_shape):
    """
    ckpt转onnx
    :param model_ckpt_path: ckpt文件路径
    :param out_model_path: onnx模型输出路径，例如，model/onnx_model/test_model.onnx
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_ckpt_path, map_location=device)
    model.eval()

    input_data_shape = torch.rand(in_shape, device=device).long()
    torch.onnx._export(model, input_data_shape, out_model_path)


def infer_onnx_example(lang_model_path, onnx_model_path, content_test):
    """
    推理
    :param lang_model_path: 语言模型路径
    :param onnx_model_path: onnx模型路径
    :param content_test: 文本内容
    """
    session = InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    tokenizer = AutoTokenizer.from_pretrained(lang_model_path)
    # ONNX Runtime expects NumPy arrays as input
    inputs = tokenizer(
        content_test, padding="max_length", max_length=512, return_tensors="np"
    )
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    outputs = session.run(
        output_names=[label_name], input_feed={input_name: inputs["input_ids"]}
    )
    print(outputs)
    predictions = torch.sigmoid(torch.tensor(outputs[0])) > 0.5
    predictions = predictions.detach().cpu().numpy().astype(int)
    probability = torch.sigmoid(torch.tensor(outputs)).detach().cpu().numpy()[0]
    print(predictions, probability)


if __name__ == "__main__":
    model_ckpt_path = "model/onnx_model/test/ckpt/reuters_chinese-roberta-wwm-ext_FL_json_epoch3para_ckpt"
    out_model_path = "model/onnx_model/test/output/test_model.onnx"
    in_shape = (1, 512) # (bathsize, max_length)

    onnx_model_path = "model/onnx_model/test_model.onnx"
    lang_model_path = "model/language_model/chinese-roberta-wwm-ext"
    content_test = "经审理查明，被告向原告分别于2017年12月12日借款22000元，2017年12月29日借款28000元，2018年2月27日借款25000元，" \
                   "2018年3月15日借款20000元，2018年4月30日借款45000元，2017年8月13日借款8000元，共计148000元。并出具了欠条，" \
                   "借款后被告没有偿还，后原告诉至法院。;"

    pt2onnx(model_ckpt_path, out_model_path, in_shape)
    infer_onnx_example(lang_model_path, onnx_model_path, content_test)
    # exp_one(in_shape=in_shape)
