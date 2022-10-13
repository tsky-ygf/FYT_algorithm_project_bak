#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    : run.py
# @Time        : 2022/10/13 21:57
# @Author      : Czq
import os

if __name__ == "__main__":
    os.system("export CUDA_VISIBLE_DEVICES=2")
    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/caigou/origin.json \
      --task_type 'ext' --save_dir data/doccano_data/caigou/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
    --train_path data/doccano_data/caigou/train.txt  \
    --dev_path data/doccano_data/caigou/dev.txt\
     --save_dir model/uie_model/new/caigou/\
      --learning_rate 1e-5\
        --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")

    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/jiekuan/origin.json \
          --task_type 'ext' --save_dir data/doccano_data/jiekuan/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
        --train_path data/doccano_data/jiekuan/train.txt  \
        --dev_path data/doccano_data/jiekuan/dev.txt\
         --save_dir model/uie_model/new/jiekuan/\
          --learning_rate 1e-5\
            --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")

    os.system("python BasicTask/NER/UIETool/doccano.py --doccano_file data/doccano_data/laodong/origin.json \
          --task_type 'ext' --save_dir data/doccano_data/laodong/ --splits 0.8 0.2 0")
    os.system("python BasicTask/NER/UIETool/finetune.py \
        --train_path data/doccano_data/laodong/train.txt  \
        --dev_path data/doccano_data/laodong/dev.txt\
         --save_dir model/uie_model/new/laodong/\
          --learning_rate 1e-5\
            --batch_size 16 --max_seq_len 512 --num_epochs 100  --model 'uie-base' --seed 1000  --logging_steps 100 --valid_steps 500 --device 'gpu'")