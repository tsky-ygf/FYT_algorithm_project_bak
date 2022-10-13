#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 13:47
# @Author  : Adolf
# @Site    : 
# @File    : train_tool.py
# @Software: PyCharm
# import sys
import os
import shutil
import torch
import math
import json
from tqdm.auto import tqdm

import torch.utils.data as data
from transformers import get_scheduler, default_data_collator

from accelerate import Accelerator
from Utils.logger import get_logger
from Tools.parse_argument import parse_config_file
from Tools.parse_argument import (
    LogArguments,
    TrainingArguments,
    DataTrainingArguments,
    ModelArguments
)

from torch.utils.tensorboard import SummaryWriter

from Tools.data_pipeline import BaseDataset, InputExample


class FGM:
    """
    Example
    # 初始化
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    """

    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class BaseTrainTool:
    def __init__(self, config_path):
        """

        :param config_path: config file path
        """
        self.config = parse_config_file(config_path)

        # self.create_examples = data_func
        # self.prepare_input = prepare_input

        self.log_args = LogArguments(**self.config["LogArguments"])
        self.model_args = ModelArguments(**self.config["ModelArguments"])
        self.train_args = TrainingArguments(**self.config["TrainingArguments"])
        self.data_train_args = DataTrainingArguments(**self.config["DataTrainingArguments"])

        self.logger = get_logger(level=self.log_args.log_level, logger_file=self.log_args.log_file)

        self.logger.info(self.config)
        # exit()

        if self.log_args.tensorboard_log_dir is not None and os.path.exists(self.log_args.tensorboard_log_dir):
            shutil.rmtree(self.log_args.tensorboard_log_dir)
            self.writer = SummaryWriter(log_dir=self.log_args.tensorboard_log_dir)

        self.accelerator = self.init_accelerator()

        self.tokenizer, self.model = self.init_model()
        self.train_dataset, self.eval_dataset = self.init_dataset()

        self.train_dataloader, self.eval_dataloader = self.init_dataloader()

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.train_args.gradient_accumulation_steps)
        self.train_args.max_train_steps = self.train_args.num_train_epochs * self.num_update_steps_per_epoch
        self.process_bar = tqdm(range(self.train_args.max_train_steps),
                                disable=not self.accelerator.is_local_main_process)

        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.eval_dataloader,
                                     self.lr_scheduler)

        if self.train_args.do_adv:
            self.fgm = FGM(self.model, emb_name=self.train_args.adv_name, epsilon=self.train_args.adv_epsilon)

        self.completed_steps = 0

    # 目前都是使用默认参数
    def init_accelerator(self):
        if self.train_args.accelerator_params:
            self.logger.info(self.train_args.accelerator_params)
            accelerator = Accelerator(**self.train_args.accelerator_params)
        else:
            self.logger.info("Use accelerator default parameters")
            accelerator = Accelerator()
        return accelerator

    def init_model(self, *args, **kwargs):
        raise NotImplemented

    def create_examples(self, data_path, mode):
        """Creates examples for the training and dev sets."""
        self.logger.trace(data_path, mode)
        with open(data_path, 'rb') as f:
            lines = json.load(f)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (mode, i)
            text = line['text']
            label = line['label']
            examples.append(InputExample(guid=guid, texts=[text], label=label))
        return examples

    def prepare_input(self, example, mode="train"):
        text = example.texts[0]
        label = example.label

        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                max_length=self.data_train_args.max_length,
                                padding="max_length",
                                truncation=True,
                                return_offsets_mapping=False,
                                return_tensors="pt")
        inputs['labels'] = label

        return inputs

    def init_dataset(self, *args, **kwargs):
        data_dir_dict = {'train': self.data_train_args.train_data_path,
                         'dev': self.data_train_args.dev_data_path,
                         'test': self.data_train_args.test_data_path}

        train_dataset = BaseDataset(data_dir_dict,
                                    mode='train',
                                    create_examples=self.create_examples,
                                    is_debug=self.log_args.is_debug,
                                    prepare_input=self.prepare_input)

        eval_dataset = BaseDataset(data_dir_dict,
                                   mode='dev',
                                   create_examples=self.create_examples,
                                   is_debug=self.log_args.is_debug,
                                   prepare_input=self.prepare_input)

        return train_dataset, eval_dataset

    def data_collator(self, *args, **kwargs):
        self.logger.trace("Use default data collator")
        return default_data_collator(*args, **kwargs)

    def init_dataloader(self):
        train_dataloader = data.DataLoader(
            dataset=self.train_dataset, shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.train_args.train_batch_size,
        )

        eval_dataloader = data.DataLoader(
            dataset=self.eval_dataset,
            collate_fn=self.data_collator,
            batch_size=self.train_args.eval_batch_size)

        return train_dataloader, eval_dataloader

    def init_optimizer(self):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.train_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.train_args.learning_rate = float(self.train_args.learning_rate)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.train_args.learning_rate)

        return optimizer

    def init_lr_scheduler(self):
        if self.train_args.num_warmup_steps is not None:
            num_warmup_steps = self.train_args.num_warmup_steps
        elif self.train_args.warmup_ratio is not None:
            num_warmup_steps = int(self.train_args.max_train_steps * self.train_args.warmup_ratio)
        else:
            num_warmup_steps = 0
        lr_scheduler = get_scheduler(
            name=self.train_args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.train_args.max_train_steps,
        )
        return lr_scheduler

    def cal_output_loss(self, batch, **kwargs):
        for key, value in batch.items():
            batch[key] = value.squeeze()
        outputs, loss = self.model(**batch)
        # loss = outputs.loss
        return outputs, loss

    def post_process_function(self, batch, outputs):
        raise NotImplemented

    def compute_metrics(self):
        raise NotImplemented

    def train_epoch(self, epoch):

        # self.lr_scheduler.step()
        for step, batch in enumerate(self.train_dataloader):
            self.model.train()
            _, loss = self.cal_output_loss(batch)

            loss /= self.train_args.gradient_accumulation_steps
            # self.accelerator.backward(loss, retain_graph=False)
            self.accelerator.backward(loss)

            if self.train_args.do_adv:
                self.fgm.attack()
                _, loss_adv = self.cal_output_loss(batch)
                loss_adv /= self.train_args.gradient_accumulation_steps
                self.accelerator.backward(loss_adv)
                self.fgm.restore()

            # if "log_path" in self.config:
            if hasattr(self, "writer"):
                self.writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=self.completed_steps)
                self.writer.add_scalar(tag="lr", scalar_value=self.optimizer.param_groups[0]["lr"],
                                       global_step=self.completed_steps)

            if step % self.train_args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                # if hasattr(self, 'max_grad_norm'):
                if self.train_args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_args.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                self.process_bar.update(1)
                self.completed_steps += 1

            if step % int(self.num_update_steps_per_epoch / 3) == 0:
                self.logger.info(
                    f"\nTrain epoch:{epoch}======> epoch_setps:{self.num_update_steps_per_epoch}"
                    f"======> step:{step}"
                    # f"epoch:{self.completed_steps / self.num_update_steps_per_epoch}"
                    f"======> loss: {loss.item():.4f}"
                    f"======> learning_rate:{self.optimizer.state_dict()['param_groups'][0]['lr']}")

                # for name, parms in self.model.named_parameters():
                #     self.logger.info(
                #         f'-->name:{name}'
                #         f'-->grad_requirs:{parms.requires_grad}'
                #         f'-->weight:{torch.mean(parms.data)}'
                #         f'-->grad_value:{torch.mean(parms.grad)}'
                #     )

            if self.completed_steps >= self.train_args.max_train_steps:
                return True

        return False

    def eval_epoch(self):

        eval_loss_res = 0

        for step, batch in enumerate(self.eval_dataloader):
            self.model.eval()
            with torch.no_grad():
                output, eval_loss = self.cal_output_loss(batch)
                eval_loss_res += eval_loss.item()

                self.post_process_function(batch, output)
                # TODO: evaluate

        eval_loss_res /= len(self.eval_dataloader)
        return eval_loss_res

    def save_model(self, model_path):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(model_path, save_function=self.accelerator.save)

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(model_path)

    def run(self):
        best_eval_loss = float("inf")
        patience = 0

        if not os.path.exists(self.data_train_args.output_dir):
            os.makedirs(self.data_train_args.output_dir)
        else:
            shutil.rmtree(self.data_train_args.output_dir)
            os.makedirs(self.data_train_args.output_dir)

        for epoch in range(self.train_args.num_train_epochs):
            # self.logger.info("epoch:{}=====patience:{}".format(epoch, patience))
            if patience > self.train_args.early_stopping_patience:
                break
            self.train_epoch(epoch)
            torch.cuda.empty_cache()
            if epoch % self.train_args.early_stopping_patience == 0 and self.train_args.eval_ever_epoch:  # and epoch > 0:
                eval_loss = self.eval_epoch()
                self.logger.info("epoch:{}======>eval_loss: {}".format(epoch, eval_loss))
                if hasattr(self, 'writer'):
                    self.writer.add_scalar(tag="eval_loss", scalar_value=eval_loss, global_step=epoch)

                if eval_loss < best_eval_loss:
                    # self.save_model(
                    #     model_path=self.config["output_dir"] + "/epoch_{}_score_{}/".format(epoch, eval_loss))
                    torch.save(self.model.state_dict(),
                               self.data_train_args.output_dir + "/epoch_{}_score_{:.4f}.bin".format(epoch, eval_loss))
                    best_eval_loss = eval_loss
                    patience = 0
                    self.save_model(model_path=self.data_train_args.output_dir + "/best")
                else:
                    patience += 1

        self.save_model(model_path=self.data_train_args.output_dir + "/final")
