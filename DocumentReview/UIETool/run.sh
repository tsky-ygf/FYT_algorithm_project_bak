python DocumentReview/UIETool/doccano.py --doccano_file data/doccano_data/caigou/origin.json \
      --task_type "ext" --save_dir data/doccano_data/caigou/ --splits 0.8 0.2 0

export CUDA_VISIBLE_DEVICES=1;
python DocumentReview/UIETool/finetune.py \
    --train_path "data/doccano_data/caigou/train.txt" \
    --dev_path "data/doccano_data/caigou/dev.txt" \
    --save_dir "model/uie_model/new/caigou/" \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model "uie-base" \
    --seed 1000 \
    --logging_steps 100 \
    --valid_steps 500 \
    --device "gpu"