#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:26
# @Author  : Adolf
# @Site    : 
# @File    : parse_argument.py
# @Software: PyCharm
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict


# 解析模型训练文件
def parse_config_file(config_path):
    with open(config_path, 'r') as f:
        res_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        if "BASE_PATH" in res_dict.keys():
            print(res_dict["BASE_PATH"])
            with open(res_dict["BASE_PATH"], 'r') as f2:
                update_dict = yaml.load(f2.read(), Loader=yaml.FullLoader)
                # print(update_dict)
                res_dict.update(update_dict)
    return res_dict


@dataclass
class LogArguments:
    """
    Arguments pertaining to which log we're going to run and logging.
    """
    is_debug: bool = field(
        default=False, metadata={"help": "Whether to run the model in debug mode."}
    )

    log_level: Optional[str] = field(
        default="INFO",
        metadata={"help": "The level of the log."},
    )

    log_file: Optional[str] = field(
        default=None,
        metadata={"help": "The file of the log."},
    )

    tensorboard_log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory to store tensorboard logs."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    num_labels: Optional[int] = field(
        default=2, metadata={"help": "Number of labels to use in the model."})

    dropout: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout rate."})

    hidden_size: Optional[int] = field(
        default=768, metadata={"help": "Hidden size."})

    feature_dim: Optional[int] = field(
        default=128, metadata={"help": "Feature dim."})

    bert_emb_size: Optional[int] = field(
        default=768, metadata={"help": "bert embedding size"}
    )

    def __post_init__(self):
        if self.config_name is None:
            self.config_name = self.model_name_or_path

        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."})

    dev_data_path: Optional[str] = field(
        default=None, metadata={"help": "The input dev data file (a csv or JSON file)."})

    test_data_path: Optional[str] = field(
        default=None, metadata={"help": "The input test data file (a csv or JSON file)."})

    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."})

    label_mapping_path: Optional[str] = field(
        default=None,
        metadata={"help": "The label mapping file path."})

    max_length: Optional[int] = field(
        default=128, metadata={"help": "Max length."})


@dataclass
class TrainingArguments:
    train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/CPU for training."})

    eval_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/CPU for evaluation."})

    lr_scheduler_type: str = field(
        default="linear", metadata=
        {"help": "The scheduler type to use.can be one of 'linear', 'cosine', "
                 "'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'."})

    num_warmup_steps: Optional[int] = field(
        default=None, metadata={"help": "Linear warmup over warmup_steps."})

    warmup_ratio: Optional[float] = field(
        default=None, metadata={"help": "Linear warmup over warmup_ratio."})

    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})

    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay if we apply some."})

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for Adam."})

    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."})

    max_train_steps: Optional[int] = field(
        default=None, metadata={"help": "If set, overrides num_train_epochs."})

    eval_ever_epoch: bool = field(
        default=True, metadata={"help": "Whether to evaluate every epoch."})

    eval_every_number_of_epochs: int = field(
        default=1, metadata={"help": "Evaluate every number of epochs."})

    early_stopping_patience: int = field(
        default=5, metadata={"help": "Early stopping patience."})

    do_adv: bool = field(
        default=False, metadata={"help": "Whether to use adversarial training."})

    adv_name: Optional[str] = field(
        default="word_embeddings", metadata={"help": "The name of the adversarial training."})

    adv_epsilon: Optional[int] = field(
        default=1.0, metadata={"help": "The epsilon of the adversarial training."})

    max_grad_norm: Optional[int] = field(
        default=1.0, metadata={"help": "The max grad norm of the training."})
    # accelerator_params: Optional[Dict] = field(
    #     default_factory={"fp16": True}, metadata={"help": "The accelerator params."})

    accelerator_params: Optional[Dict] = field(
        default=None, metadata={"help": "The accelerator params."})

    metrics_name: Optional[str] = field(
        default="Accuracy", metadata={"help": "The metrics name."})
