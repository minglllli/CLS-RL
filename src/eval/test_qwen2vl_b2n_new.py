from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.multiprocessing import Process, set_start_method, Manager

import torch
import sys
import json
from tqdm import tqdm
import re
import math
from PIL import Image
import  random
import os
from dataset_prefix import dataset_prefix
import argparse
from transformers.utils.logging import disable_progress_bar

#from src.eval.test_qwen2vl_imagenet_zero_shot import all_outputs

disable_progress_bar()


def extract_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_str, re.DOTALL)

    if match:
        return match.group(1).strip()
    return None


def init_model(model_path, gpu_id):
    """init a model(args.model_path) on a specific gpu"""
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu_id}",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    return model, processor


def answer_a_batch_question_qwen(batch_messages, model, processor,ft_method):
    """ let qwen answer a batch of questions """
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = []
    for x in image_inputs:
        img_temp = x.resize((384, 384), Image.Resampling.LANCZOS)
        inputs.append(img_temp)
    inputs = processor(
        text=text,
        images=inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    #if ft_method=='sft':
    #    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=128)  # do_sample=False
    #else:
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024)  # do_sample=False
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return batch_output_text

def multi_gpu_inference(prompts, gpu_ids, model_path, batch_size,ft_method):
    """ let each gpu (along with a model) answer a chunk of questions """
    set_start_method("spawn", force=True)
    manager = Manager()
    gpu_id2result = manager.dict()

    gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    chunk_size = len(prompts) // num_gpus
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_gpus - 1 else len(prompts)
        chunk = prompts[start_idx: end_idx]
        process = Process(target=infer_on_single_gpu, args=(model_path, gpu_id, chunk, batch_size, gpu_id2result,ft_method))
        process.start()
        processes.append(process)

    # for process in tqdm.auto.tqdm(processes, desc="Inference progress", position=num_gpus, leave=True):
    for process in processes:
        process.join()

    all_predicts = []
    for gpu_id in gpu_ids:
        all_predicts.extend(gpu_id2result[gpu_id])

    return all_predicts


def infer_on_single_gpu(model_path, device_id, chunk_of_tested_messages, batch_size, results=None,ft_method=None):
    """init model on this single gpu and let it answer asign chunk of questions"""
    model, processor = init_model(model_path, device_id)

    ### split batch
    responses = []
    batch_messages_list = [chunk_of_tested_messages[start: start + batch_size]
                           for start in range(0, len(chunk_of_tested_messages), batch_size)]

    for batch_messages in tqdm(batch_messages_list, desc=f"GPU {device_id} progress", position=device_id,
                                         leave=False):
        batch_output_text = answer_a_batch_question_qwen(batch_messages, model, processor,ft_method)

        responses.extend(batch_output_text)

    results[device_id] = responses
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='0')
    parser.add_argument('--model_path', type=str, default='0')
    parser.add_argument('--prompt', type=str, default='0')
    parser.add_argument('--ft_dataset', type=str, default='0')
    all_gpu = ",".join(map(str, range(torch.cuda.device_count())))
    print(f'all gpus: {all_gpu}')
    parser.add_argument("--gpu_ids", default=all_gpu, help="comma-separated list of GPU IDs to use")
    args = parser.parse_args()

    #print(args)
    # input
    dataset_list = ['Imagenet', 'Caltech101', 'DescribableTextures', 'EuroSAT', 'Food101', 'OxfordFlowers', 'OxfordPets', 'StanfordCars', 'SUN397', 'UCF101', 'FGVCAircraft']
    dataset = dataset_list[int(args.model)]
    MODEL_PATH = args.model_path
    BSZ = 32  # reduce it if GPU OOM
    prompt = args.prompt
    if MODEL_PATH=='Qwen/Qwen2-VL-2B-Instruct':
        ft_method = 'zeroshot'
        args.ft_dataset = 'Qwen'
    elif 'sft' in MODEL_PATH.lower():
        ft_method = 'sft'
    elif 'direct' in  MODEL_PATH.lower():
        ft_method = 'direct'
    else:
        ft_method = 'r1'
    prompt = args.prompt

    test_setting = 'new'
    OUTPUT_PATH = 'results_b2n/'+test_setting+'_'+ ft_method+'_'+args.prompt+'/'+test_setting+'_'+ ft_method+'_'+args.ft_dataset+'_'+args.prompt+"_Qwen2-VL-2B-Instruct-" + dataset + "-b2n.json"
    if not os.path.exists(
            'results_b2n/' + test_setting + '_' + ft_method +  '_' + args.prompt):
        # Create the directory
        os.makedirs(
            'results_b2n/' + test_setting + '_' + ft_method + '_' + args.prompt)
    if os.path.exists(OUTPUT_PATH):
        print(f'{dataset}: file exist')
        sys.exit(0)

    prefix, path_prefix = dataset_prefix(dataset)


    data_file_path = "/mnt/petrelfs/liming/CLS-RL/src/eval/prompts/datasets_b2n/"+prefix+"_b2n_"+test_setting+"_test.jsonl"

    f = open(data_file_path, "r", encoding="utf-8")
    data_file_list = [json.loads(line) for line in f]
    data_file_list = [item for item in data_file_list if item["split"] == "test" or item["split"] == 'valid']
    print(len(data_file_list))
    random.seed(100)
    random.shuffle(data_file_list)
    data = data_file_list
    # QUESTION_TEMPLATE = "{Question}\n Please directly output the final answer (name)."
    if prompt == 'normal':
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    elif prompt == 'direct':
        QUESTION_TEMPLATE = "{Question} Please directly output the answer name."

    messages = []
    print(len(data))
    for i in data:
        class_name = i['label']
        image_path = i['image']
        image = image_path
        if dataset == 'Imagenet':
            # l = len('/pasteur/u/yuhuiz/data/ImageNet/imagenet/val/')
            image = path_prefix + image.split("/")[-1]
        elif dataset == 'SUN397':
            image = path_prefix + image.split("/")[-1]
        else:
            image = path_prefix + image
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=i['problem'])
                }
            ]
        }]
        messages.append(message)

    all_predicts = multi_gpu_inference(messages, args.gpu_ids, args.model_path, BSZ,ft_method)
    all_outputs = all_predicts

    final_output = []
    correct_number = 0

    for input_example, model_output in zip(data, all_outputs):
        original_output = model_output
        ground_truth = input_example['label']
        content_match = extract_answer(original_output)
        answer = content_match if content_match else original_output.strip()

        yes = 0
        gt = ground_truth
        if answer.lower() == gt.lower() or gt.lower() in answer.lower():
            correct_number += 1
            yes = 1
        else:
            if '_' in gt:
                gt = gt.replace('_', ' ')
                if answer.lower() == gt.lower() or gt.lower() in answer.lower():
                    correct_number += 1
                    yes = 1

        # Create a result dictionary for this example
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer': answer,
            "correct": yes

        }
        final_output.append(result)

    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print(f"\n{dataset}: {test_setting} Accuracy: {accuracy:.2f}%")
    # Save results to a JSON file
    output_path = OUTPUT_PATH
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")


    with open("results_b2n/output_map.txt", "a") as f:  # 以写模式 ("w") 打开文件
        f.write(f"{output_path}:  {MODEL_PATH}\n")
    print(f"Results saved to {output_path}")
