#!/bin/bash
cd src/cls-rl
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="debug_log_2b_instruct_sat_direct.txt"
#Qwen/Qwen2-VL-2B-Instruct
PYTHONIOENCODING=utf-8 python -m torch.distributed.run --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_sat_direct.py \
    --output_dir  'Qwen2-VL-2B-Instruct-SAT-generation4-Direct/'\
    --model_name_or_path  Qwen/Qwen2-VL-2B-Instruct\
    --dataset_name afdsafas/Caltech-4shot-b2n \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-Instruct-SAT-Direct\
    --save_steps 10000 \
    --save_only_model true \
    --num_generations 4   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance

