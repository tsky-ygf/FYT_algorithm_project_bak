#python DocumentReview/UIETool/doccano.py --doccano_file data/doccano_data/labor/labor.json \
#      --task_type "ext" --save_dir data/doccano_data/labor/ --splits 0.8 0.2 0

export CUDA_VISIBLE_DEVICES=2;
python DocumentReview/UIETool/finetune.py \
    --train_path "data/doccano_data/labor/train.txt" \
    --dev_path "data/doccano_data/labor/dev.txt" \
    --save_dir "model/uie_model/labor" \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model "uie-base" \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device "gpu"