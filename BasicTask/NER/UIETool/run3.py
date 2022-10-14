#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    : run.py
# @Time        : 2022/10/13 21:57
# @Author      : Czq
import os

if __name__ == "__main__":
    os.system("export CUDA_VISIBLE_DEVICES=3")
    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/laowu/origin.json \
      --task_type 'ext' --save_dir data/doccano_data/laowu/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
    --train_path data/doccano_data/laowu/train.txt  \
    --dev_path data/doccano_data/laowu/dev.txt\
     --save_dir model/uie_model/new/laowu/\
      --learning_rate 1e-5\
        --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")

    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/yibanzulin/origin.json \
          --task_type 'ext' --save_dir data/doccano_data/yibanzulin/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
        --train_path data/doccano_data/yibanzulin/train.txt  \
        --dev_path data/doccano_data/yibanzulin/dev.txt\
         --save_dir model/uie_model/new/yibanzulin/\
          --learning_rate 1e-5\
            --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")

    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/laodongbaomi/origin.json \
          --task_type 'ext' --save_dir data/doccano_data/laodongbaomi/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
        --train_path data/doccano_data/laodongbaomi/train.txt  \
        --dev_path data/doccano_data/laodongbaomi/dev.txt\
         --save_dir model/uie_model/new/laodongbaomi/\
          --learning_rate 1e-5\
            --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")

    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/shangyebaomi/origin.json \
          --task_type 'ext' --save_dir data/doccano_data/shangyebaomi/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
        --train_path data/doccano_data/shangyebaomi/train.txt  \
        --dev_path data/doccano_data/shangyebaomi/dev.txt\
         --save_dir model/uie_model/new/shangyebaomi/\
          --learning_rate 1e-5\
            --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")