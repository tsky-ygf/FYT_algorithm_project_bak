#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 13:47
# @Author  : Adolf
# @Site    : 
# @File    : train_tool.py
# @Software: PyCharm
import sys
import torch
import math
from tqdm.auto import tqdm

import torch.utils.data as data
from transformers import get_scheduler, default_data_collator
from accelerate import Accelerator
from Utils.logger import get_module_logger
from Utils.parse_file import parse_config_file


class BaseTrainTool:
    def __init__(self, config_path):
        self.config = parse_config_file(config_path=config_path)
        self.logger = get_module_logger(module_name="Train", level=self.config.get("log_level", "INFO"))

        self.accelerator = self.init_accelerator()

        self.tokenizer, self.model = self.init_model()
        self.train_dataset, self.eval_dataset = self.init_dataset()
        self.train_dataloader, self.eval_dataloader = self.init_dataloader()

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config["gradient_accumulation_steps"])
        self.config["max_train_steps"] = self.config["num_train_epochs"] * self.num_update_steps_per_epoch
        self.process_bar = tqdm(range(self.config["max_train_steps"]),
                                disable=not self.accelerator.is_local_main_process)

        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.eval_dataloader)

        self.completed_steps = 0

    # 目前都是使用默认参数
    def init_accelerator(self):
        if 'accelerator_params' in self.config:
            self.logger.info(self.config['accelerator_params'])
        else:
            self.logger.info("Use accelerator default parameters")
        accelerator = Accelerator()
        return accelerator

    def init_model(self, *args, **kwargs):
        raise NotImplemented

    def init_dataset(self, *args, **kwargs):
        raise NotImplemented

    def data_collator(self, *args, **kwargs):
        self.logger.info("Use default data collator")
        return default_data_collator(*args, **kwargs)

    def init_dataloader(self):
        train_dataloader = data.DataLoader(
            dataset=self.train_dataset, shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.config["train_batch_size"]
        )

        eval_dataloader = data.DataLoader(
            dataset=self.eval_dataset,
            collate_fn=self.data_collator,
            batch_size=self.config["eval_batch_size"])

        return train_dataloader, eval_dataloader

    def init_optimizer(self):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"])

        return optimizer

    def init_lr_scheduler(self):
        lr_scheduler = get_scheduler(
            name=self.config["lr_scheduler_type"],
            optimizer=self.optimizer,
            num_warmup_steps=self.config["num_warmup_steps"],
            num_training_steps=self.config["max_train_steps"],
        )
        return lr_scheduler

    def cal_loss(self, *args, **kwargs):
        raise NotImplementedError

    def train_epoch(self):
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            loss = self.cal_loss(batch)

            loss = loss / self.config["gradient_accumulation_steps"]
            self.accelerator.backward(loss, retain_graph=False)

            if step % self.config["gradient_accumulation_steps"] == 0 or step == len(self.train_dataloader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.process_bar.update(1)
                self.completed_steps += 1

            if self.completed_steps % 500 == 0:
                self.logger.info(
                    f"epoch:{step / self.num_update_steps_per_epoch}"
                    f"=====> step:{step} =====> loss: {loss}"
                    f"======> learning_rate:{self.optimizer.state_dict()['param_groups'][0]['lr']}")

            if self.completed_steps >= self.config["max_train_steps"]:
                return True

        return False

    def eval_epoch(self):
        self.model.eval()
        eval_loss_res = 0

        for step, batch in enumerate(self.eval_dataloader):
            eval_loss = self.cal_loss(batch)
            eval_loss_res += eval_loss.item()

        eval_loss_res /= len(self.eval_dataloader)
        return eval_loss_res

    def save_model(self, model_path):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(model_path, save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(model_path)

    def train_main(self):
        best_eval_loss = float("inf")
        patience = 0
        for epoch in range(self.config["num_train_epochs"]):
            if patience > self.config["early_stop_patience"]:
                break
            self.train_epoch()
            if epoch % self.config["eval_every_number_of_epoch"] == 0 and epoch > 0:
                # torch.cuda.empty_cache()
                eval_loss = self.eval_epoch()

                if eval_loss < best_eval_loss:
                    if "output_dir" not in self.config:
                        self.logger.error("=========== no model save path ================")
                        sys.exit(1)

                    self.save_model(
                        model_path=self.config["output_dir"] + "/epoch_{}_score_{}/".format(epoch, eval_loss))
                    best_eval_loss = eval_loss
                    patience = 0
            else:
                patience += 1

        self.save_model(model_path=self.config["output_dir"] + "/final/")
        torch.cuda.empty_cache()
