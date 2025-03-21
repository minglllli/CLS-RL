# CLS-RL: Image Classification with Rule-Based Reinforcement Learning
🔍 Overview

CLS-RL explores fine-tuning Multimodal Large Language Models (MLLMs) for image classification using rule-based reinforcement learning (RL). We introduce CLS-RL, which leverages verifiable signals (class names) for fine-tuning. CLS-RL demonstrates a "free-lunch" phenomenon, showing cross-dataset improvement, and we further introduce No-Thinking-CLS-RL, which optimizes performance by removing the thinking process during training.

[![arXiv](https://img.shields.io/badge/arXiv-2503.13939-b31b1b.svg)](http://arxiv.org/abs/2503.16188)

Key features of CLS-RL:
* **Rule-based Reinforcement Learning:** Fine-tunes MLLMs using verifiable reward losses instead of token-level losses, guiding models to explore diverse reasoning.
* **"Free-Lunch" Phenomenon:** Demonstrates that fine-tuning with CLS-RL on one dataset can improve performance on other, distinct datasets.
* **No-Thinking Variant:** Introduces No-Thinking-CLS-RL, which removes the thinking process during training, leading to improved performance and reduced training time.



The code is coming soon!


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
