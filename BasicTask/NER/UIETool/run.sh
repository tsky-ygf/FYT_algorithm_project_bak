export CUDA_VISIBLE_DEVICES=0
export contract_type="maimai"
export doccano_file="./data/data_src/common_1013/maimai.jsonl"
export save_dir="data/doccano_data/maimai/"
python BasicTask/NER/UIETool/doccano.py --doccano_file $doccano_file  --task_type "ext"  --save_dir $save_dir  --splits 0.8 0.2 0
export PYTHONPATH=$(pwd):$PYTHONPATH
export train_path=data/doccano_data/maimai/train.txt
export dev_path=data/doccano_data/maimai/dev.txt
export save_dir=model/uie_model/new/maimai/
python BasicTask/NER/UIETool/finetune.py --train_path $train_path  --dev_path $dev_path --save_dir $save_dir --learning_rate 1e-5  --batch_size 16 --max_seq_len 512 --num_epochs 100  --model "uie-base" --seed 1000  --logging_steps 100 --valid_steps 500 --device "gpu"