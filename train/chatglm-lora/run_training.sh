CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path  THUDM/chatglm-6b \
    --data_path alpaca_data_cleaned.json \
    --lora_rank 8 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir chatglm-6b-lora