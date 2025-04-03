#!/bin/bash
cd src/cls-rl
DATASET="cars"  # imagenet, caltech, dtd, eurosat, food, flowers, pets, cars, sun, ucf, fgvc
MAX_PROMPT_LENGTH=2048  # 2048 for cars and 1024 for other datasets
NUM_TRAIN_EPOCHS=50  # 10 for imagenet, 25 for sun, 100 for eurosat, and 50 for other datasets

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="debug_log_2b_instruct-no-thinking_${DATASET}-4shot-fewshot-ep${NUM_TRAIN_EPOCHS}.txt"
OUTPUT_DIR="Qwen2-VL-2B-Instruct-no-thinking-${DATASET}-4shot-fewshot-ep${NUM_TRAIN_EPOCHS}-generation4/"

PYTHONIOENCODING=utf-8 python -m torch.distributed.run --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_direct.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name "afdsafas/${DATASET}-4shot-fewshot" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --run_name "Qwen2-VL-2B-Instruct-${DATASET}-4shot-fewshot-ep${NUM_TRAIN_EPOCHS}" \
    --save_steps 10000 \
    --save_only_model true \
    --num_generations 4  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance
