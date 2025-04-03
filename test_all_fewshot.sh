
dataset_list=('Imagenet' 'Caltech101' 'DescribableTextures' 'EuroSAT' 'Food101' 'OxfordFlowers' \
              'OxfordPets' 'StanfordCars' 'SUN397' 'UCF101' 'FGVCAircraft' 'combine')

# normal means reasoning prompt, direct means no-thinking prompt
PROMPT_TYPE=normal # normal or direct


MODEL_PATH='/mnt/petrelfs/liming/CLS-RL/src/cls-rl/Qwen2-VL-2B-Instruct-eurosat-4shot-fewshot-ep100-generation4'
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

echo ${index}
ft_dataset=${dataset_list[$index]}
echo ${ft_dataset}

MODEL_NAME=$(basename ${MODEL_PATH})


LOG_FILE="test_logs/log_${PROMPT_TYPE}_${MODEL_NAME}.txt"

for MODEL_ID in {1..10}; do
  #echo "Running with MODEL_ID=${MODEL_ID}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') Running with MODEL_ID=${MODEL_ID}" | tee -a ${LOG_FILE}
  python /mnt/petrelfs/liming/CLS-RL/src/eval/test_qwen2vl_fewshot.py \
    --model ${MODEL_ID} \
    --prompt ${PROMPT_TYPE} \
    --model_path ${MODEL_PATH} \
    --ft_dataset ${ft_dataset}
done