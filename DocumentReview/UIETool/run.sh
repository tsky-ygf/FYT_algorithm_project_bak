#python DocumentReview/UIETool/doccano.py --doccano_file data/doccano_data/maimai/origin.json \
#      --task_type "ext" --save_dir data/doccano_data/maimai/ --splits 0.7 0.3 0

export CUDA_VISIBLE_DEVICES=1;
python DocumentReview/UIETool/finetune.py \
    --train_path "data/doccano_data/maimai/train.txt" \
    --dev_path "data/doccano_data/maimai/dev.txt" \
    --save_dir "model/uie_model/new/maimai/" \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model "uie-base" \
    --seed 1000 \
    --logging_steps 100 \
    --valid_steps 500 \
    --device "gpu"