# Think or Not Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning
🔍 Overview

CLS-RL explores fine-tuning Multimodal Large Language Models (MLLMs) for image classification using rule-based reinforcement learning (RL). We introduce CLS-RL, which leverages verifiable signals (class names) for fine-tuning. CLS-RL demonstrates a "free-lunch" phenomenon, showing cross-dataset improvement, and we further introduce No-Thinking-RL, which optimizes performance by removing the thinking process during training.

[![arXiv](https://img.shields.io/badge/arXiv-2503.13939-b31b1b.svg)](http://arxiv.org/abs/2503.16188)

Key features of CLS-RL:
* **Rule-based Reinforcement Learning:** Fine-tunes MLLMs using verifiable reward losses instead of token-level losses, guiding models to explore diverse reasoning.
  
* **"Free-Lunch" Phenomenon:** Demonstrates that fine-tuning with CLS-RL on one dataset can improve performance on other, distinct datasets.

<table>
 <tr>
  <td width="50%">
   <img src="r1_improve_heatmap.png" alt="">
  </td>
  <td width="50%">
   <img src="direct_improve_heatmap.png" alt="">
  </td>
 </tr>
</table>

* **No-Thinking Variant:** Introduces No-Thinking-RL, which removes the thinking process during training, leading to improved performance and reduced training time.
<img src="compare_difference_heatmap.png" alt="Alt text" width="600" height="600">

* **No-Thinking-RL on CVBench:** We follow [VisualThinker-R1-Zero](https://github.com/turningpoint-ai/VisualThinker-R1-Zero) to fine-tune Qwen2-VL-2B-Instruct on SAT and then test on CVBench. We found that No-Thinking-RL surpasses VisualThinker-R1-Zero by 6% accuracy.
  
| Model                | Total Accuracy | Count Acc | Relation Acc | Depth Acc | Distance Acc |
|----------------------|---------------|-----------|--------------|-----------|--------------|
| No-Thinking-RL      | 0.7676        | 0.6967    | 0.8446       | 0.8067    | 0.7383       |
| VisualThinker-R1-Zero | 0.7036       | 0.6612    | 0.8338       | 0.6850    | 0.6367       |



## 🛠️ Installation

```bash
git clone https://github.com/minglllli/CLS-RL.git
cd CLS-RL
conda create -n cls-rl python=3.11
conda activate cls-rl
bash setup.sh
```

## Training
### Classification 

#### 1. CLS-RL Training
##### Fewshot training
Change the dataset name in fewshot.sh, then run
```bash
bash fewshot.sh
```
##### base-eo-new training
Change the dataset name in b2n.sh, then run
```bash
bash b2n.sh
```
#### 2. No-Thinking-RL Training
##### Fewshot training
Change the dataset name in fewshot_no-think.sh, then run
```bash
bash fewshot_no-think.sh
```
##### base-eo-new training
Change the dataset name in b2n_no-thinking.sh, then run
```bash
bash b2n_no-thinking.sh
```

#### 3. SFT Training
##### Fewshot training
Change the dataset name in fewshot_no-think.sh, then run
```bash
bash fewshot_no-think.sh
```
##### base-to-new training
Change the dataset name in b2n_no-thinking.sh, then run
```bash
bash b2n_no-thinking.sh
```

### SAT
Follow [VisualThinker-R1-Zero](https://github.com/turningpoint-ai/VisualThinker-R1-Zero) to prepare the dataset.
#### 1. VisualThinker-R1-Zero Training
```bash
bash run_sat.sh
```
#### 2. No-Thinking-RL Training
```bash
bash run_sat_no-think.sh
```
We provide the fine-tuned checkpoints below.

Fine-tuned model of thinking RL (VisualThinker-R1-Zero). [🤗 Model on Hugging Face](https://huggingface.co/afdsafas/Qwen2-VL-2B-Instruct-SAT-generation4)

Fine-tuned model of No-Thinking-RL. [🤗 Model on Hugging Face](https://huggingface.co/afdsafas/Qwen2-VL-2B-Instruct-SAT-generation4-Direct)
## Inference
We provide the [results](https://drive.google.com/drive/folders/1NXpGfWj1oazoK4SwzoDBqpV59Fjyi2od?usp=sharing) of classfication and CVBench for reference.
'direct' means no-thinking-rl. 
### Classification
Download the [test prompt](https://drive.google.com/drive/folders/1Quqh2H3TkqX91Rc2YUdnPXixzDDFf1dy?usp=sharing) to src/eval/prompts/. 
Follow [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download datasets and write the dataset path to src/eval/data_prefix.py.
#### Fewshot and Cross-Dataset
Change the MODEL_PATH in test_all_fewshot.sh and then run
```bash
bash test_all_fewshot.sh
```

#### Base-to-New
Change the MODEL_PATH in test_all_b2n.sh and then run
```bash
bash test_all_b2n.sh
```

### CVBench
```bash
bash test_cvbench.sh
```


## 📌 Open-source Plan

- [x] Release the fine-tuning and inference code.
- [x] Release the dataset.
- [x] Release the evaluation results.
- [x] Release the checkpoints.

## Acknowledgements

Our code is based on **R1-V**. We thank the authors of **R1-V** for their open-source contributions.  
🔗 [R1-V GitHub Repository](https://github.com/Deep-Agent/R1-V)

## Citation
```
@misc{li2025clsrl,
      title={CLS-RL: Image Classification with Rule-Based Reinforcement Learning}, 
      author={Ming Li, Shitian Zhao, Jike Zhong, Yuxiang Lai, Kaipeng Zhang},
      year={2025},
      eprint={2503.16188},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={http://arxiv.org/2503.16188}, 
}
```
