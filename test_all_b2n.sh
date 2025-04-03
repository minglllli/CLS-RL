dataset_list=('Imagenet' 'Caltech101' 'DescribableTextures' 'EuroSAT' 'Food101' 'OxfordFlowers' \
              'OxfordPets' 'StanfordCars' 'SUN397' 'UCF101' 'FGVCAircraft')


PROMPT_TYPE=normal
files=(
    "Qwen2-VL-2B-Instruct-dtd-b2n-4shot-ep50-generation4"
)

for file in "${files[@]}"; do
    # change model path to your own
    MODEL_PATH="/mnt/petrelfs/liming/CLS-RL/src/cls-rl/${file}"
    LOWER_MODEL_PATH=$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

    if echo "$LOWER_MODEL_PATH" | grep -q "imagenet"; then
        index=0
    elif echo "$LOWER_MODEL_PATH" | grep -q "caltech"; then
        index=1
    elif echo "$LOWER_MODEL_PATH" | grep -q "dtd"; then
        index=2
    elif echo "$LOWER_MODEL_PATH" | grep -q "eurosat"; then
        index=3
    elif echo "$LOWER_MODEL_PATH" | grep -q "food"; then
        index=4
    elif echo "$LOWER_MODEL_PATH" | grep -q "flowers"; then
        index=5
    elif echo "$LOWER_MODEL_PATH" | grep -q "pets"; then
        index=6
    elif echo "$LOWER_MODEL_PATH" | grep -q "cars"; then
        index=7
    elif echo "$LOWER_MODEL_PATH" | grep -q "sun"; then
        index=8
    elif echo "$LOWER_MODEL_PATH" | grep -q "ucf"; then
        index=9
    elif echo "$LOWER_MODEL_PATH" | grep -q "fgvc"; then
        index=10
    fi

    echo "Index: ${index}"
    ft_dataset=${dataset_list[$index]}

    MODEL_NAME=$(basename ${MODEL_PATH})

    LOG_FILE="test_logs/log_${PROMPT_TYPE}.txt"

    MODEL_ID=${index}
    echo "$(date '+%Y-%m-%d %H:%M:%S') Running with MODEL_ID=${MODEL_ID}" | tee -a ${LOG_FILE}
    #change test_qwen2vl_b2n_base.py to test_qwen2vl_b2n_new.py for testing new class
    python /mnt/petrelfs/liming/CLS-RL/src/eval/test_qwen2vl_b2n_base.py \
        --model ${MODEL_ID} \
        --prompt ${PROMPT_TYPE} \
        --model_path ${MODEL_PATH} \
        --ft_dataset ${ft_dataset}
done
