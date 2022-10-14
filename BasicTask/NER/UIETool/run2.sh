#export PYTHONPATH=$(pwd):$PYTHONPATH
#nohup sh BasicTask/NER/UIETool/run.sh  > log/uie/run2.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2
export contract_type=jietiao
python BasicTask/NER/UIETool/doccano.py --doccano_file data/data_src/common_1013/${contract_type}.jsonl \
      --task_type "ext" --save_dir data/doccano_data/${contract_type}/ --splits 0.8 0.2 0

python BasicTask/NER/UIETool/finetune.py \
    --train_path "data/doccano_data/${contract_type}/train.txt" \
    --dev_path "data/doccano_data/${contract_type}/dev.txt" \
    --save_dir "model/uie_model/new/${contract_type}/" \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model "uie-base" \
    --seed 1000 \
    --logging_steps 100 \
    --valid_steps 500 \
    --device "gpu"

export contract_type=laowu
python BasicTask/NER/UIETool/doccano.py --doccano_file data/data_src/common_1013/${contract_type}.jsonl \
      --task_type "ext" --save_dir data/doccano_data/${contract_type}/ --splits 0.8 0.2 0

python BasicTask/NER/UIETool/finetune.py \
    --train_path "data/doccano_data/${contract_type}/train.txt" \
    --dev_path "data/doccano_data/${contract_type}/dev.txt" \
    --save_dir "model/uie_model/new/${contract_type}/" \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model "uie-base" \
    --seed 1000 \
    --logging_steps 100 \
    --valid_steps 500 \
    --device "gpu"
