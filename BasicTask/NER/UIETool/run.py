#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    : run.py
# @Time        : 2022/10/13 21:57
# @Author      : Czq
import os

if __name__ == "__main__":
    os.system("export CUDA_VISIBLE_DEVICES=0")
    os.system("export PYTHONPATH=$(pwd):$PYTHONPATH")
    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/maimai/origin.json \
      --task_type 'ext' --save_dir data/doccano_data/maimai/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
    --train_path data/doccano_data/maimai/train.txt  \
    --dev_path data/doccano_data/maimai/dev.txt\
     --save_dir model/uie_model/new/maimai/\
      --learning_rate 1e-5\
        --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")

    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/fangwuzulin/origin.json \
          --task_type 'ext' --save_dir data/doccano_data/fangwuzulin/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
        --train_path data/doccano_data/fangwuzulin/train.txt  \
        --dev_path data/doccano_data/fangwuzulin/dev.txt\
         --save_dir model/uie_model/new/fangwuzulin/\
          --learning_rate 1e-5\
            --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")

    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/jietiao/origin.json \
          --task_type 'ext' --save_dir data/doccano_data/jietiao/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
        --train_path data/doccano_data/jietiao/train.txt  \
        --dev_path data/doccano_data/jietiao/dev.txt\
         --save_dir model/uie_model/new/jietiao/\
          --learning_rate 1e-5\
            --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")