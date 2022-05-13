#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 13:47
# @Author  : Adolf
# @Site    : 
# @File    : train_tool.py
# @Software: PyCharm
import sys
import os
import shutil
import torch
import math
from tqdm.auto import tqdm

import torch.utils.data as data
from transformers import get_scheduler, default_data_collator
from accelerate import Accelerator
from Utils.logger import get_module_logger
from Utils.parse_file import parse_config_file

from torch.utils.tensorboard import SummaryWriter


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
        self.config = parse_config_file(config_path=config_path)
        self.logger = get_module_logger(module_name="Train", level=self.config.get("log_level", "INFO"))

        if os.path.exists(self.config["log_path"]):
            shutil.rmtree(self.config["log_path"])
        self.writer = SummaryWriter(log_dir=self.config["log_path"])

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
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.eval_dataloader,
                                     self.lr_scheduler)

        if self.config["do_adv"]:
            self.fgm = FGM(self.model, emb_name=self.config["adv_name"], epsilon=self.config["adv_epsilon"])

        self.completed_steps = 0

    # 目前都是使用默认参数
    def init_accelerator(self):
        if 'accelerator_params' in self.config:
            self.logger.info(self.config['accelerator_params'])
            accelerator = Accelerator(**self.config['accelerator_params'])
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
            batch_size=self.config["train_batch_size"],
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
        if "num_warmup_steps" in self.config:
            num_warmup_steps = self.config["num_warmup_steps"]
        elif "warmup_ratio" in self.config:
            num_warmup_steps = int(self.config["max_train_steps"] * self.config["warmup_ratio"])
        else:
            num_warmup_steps = 0
        lr_scheduler = get_scheduler(
            name=self.config["lr_scheduler_type"],
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.config["max_train_steps"],
        )
        return lr_scheduler

    def cal_loss(self, *args, **kwargs):
        raise NotImplementedError

    def train_epoch(self, epoch):
        self.model.train()
        # self.lr_scheduler.step()
        for step, batch in enumerate(self.train_dataloader):
            loss = self.cal_loss(batch)

            loss = loss / self.config["gradient_accumulation_steps"]
            # self.accelerator.backward(loss, retain_graph=False)
            self.accelerator.backward(loss)

            if self.config["do_adv"]:
                self.fgm.attack()
                loss_adv = self.cal_loss(batch)
                loss_adv = loss_adv / self.config["gradient_accumulation_steps"]
                self.accelerator.backward(loss_adv)
                self.fgm.restore()

            self.writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=self.completed_steps)
            self.writer.add_scalar(tag="lr", scalar_value=self.optimizer.param_groups[0]["lr"],
                                   global_step=self.completed_steps)

            if step % self.config["gradient_accumulation_steps"] == 0 or step == len(self.train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                self.process_bar.update(1)
                self.completed_steps += 1

            if step % int(self.num_update_steps_per_epoch / 3) == 0:
                self.logger.info(
                    f"Train epoch:{epoch}======> epoch_setps:{self.num_update_steps_per_epoch}"
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

        if "output_dir" not in self.config:
            # self.logger.error("=========== no model save path ================")
            raise Exception("==========no model save path============")

        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])
        else:
            shutil.rmtree(self.config["output_dir"])
            os.makedirs(self.config["output_dir"])

        for epoch in range(self.config["num_train_epochs"]):
            # self.logger.info("epoch:{}=====patience:{}".format(epoch, patience))
            if patience > self.config["early_stop_patience"]:
                break
            self.train_epoch(epoch)
            torch.cuda.empty_cache()
            if epoch % self.config["eval_every_number_of_epoch"] == 0:  # and epoch > 0:
                # torch.cuda.empty_cache()
                eval_loss = self.eval_epoch()
                self.logger.info("epoch:{}======>eval_loss: {}".format(epoch, eval_loss))
                self.writer.add_scalar(tag="eval_loss", scalar_value=eval_loss, global_step=epoch)

                if eval_loss < best_eval_loss:
                    # self.save_model(
                    #     model_path=self.config["output_dir"] + "/epoch_{}_score_{}/".format(epoch, eval_loss))
                    torch.save(self.model.state_dict(),
                               self.config["output_dir"] + "/epoch_{}_score_{:.4f}.bin".format(epoch, eval_loss))
                    best_eval_loss = eval_loss
                    patience = 0
                    self.save_model(model_path=self.config["output_dir"] + "/final")
                else:
                    patience += 1

