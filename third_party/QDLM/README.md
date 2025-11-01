# Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs

<h5 align="center"> 

[![arXiv](https://img.shields.io/badge/QDLM-2508.14896-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2508.14896)
[![GitHub](https://img.shields.io/badge/GitHub-Code-green?logo=github)](https://github.com/FelixMessi/QDLM)
[![License](https://img.shields.io/badge/⚖️%20Code%20License-MIT-blue)](https://github.com/FelixMessi/QDLM/blob/main/LICENSE)
 <br>

</h5>


Welcome to the official code repository for "[**Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs**](https://arxiv.org/abs/2508.14896)".

Your star means a lot to us in developing this project! ⭐⭐⭐


## 📰 News
* [2025/10/15] 🔥 We release the code for quantizing dLLMs!
* [2025/08/20] 🚀 Our paper is available on arXiv!



## 👀 Introduction


- We present the **first systematic study** on quantizing diffusion-based language models (dLLMs).

- This repository implements **state-of-the-art post-training quantization (PTQ)** methods for dLLM, including **GPTQ**, **AWQ**, **SmoothQuant**, **QuaRot**, and **DuQuant**.

- We comprehensively investigate the **impact of quantization** on dLLMs across **four key dimensions**: bit-width, quantization method, task category, and model architecture.



## 🔧 Installation
```bash
conda create -n qdlm python=3.10 -y
conda activate qdlm
git clone https://github.com/FelixMessi/QDLM
pip install --upgrade qdlm 
pip install -r requirements.txt
pip install math-verify==0.8.0 antlr4-python3-runtime==4.11.0 sympy==1.14.0
cd ./lm-evaluation-harness && pip install -e .
```

To run evaluation for QuaRot, please download and install the [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform/releases) with your cuda version.



## ⚙️ Usage
Please refer to the `scripts` folder for running different weight-only quantization methods (AWQ, GPTQ) and weight–activation quantization methods (SmoothQuant, QuaRot, DuQuant).

Please download the LLaDA-base/LLaDA-Instruct or Dream models and replace the `MODEL_PATH` with your specific paths.

Detailed usage instructions are provided in the corresponding shell scripts.



## 📂 Contact
If you have further questions, please open an issue or contact haokun.lin@cripac.ia.ac.cn or xuhb2001@gmail.com.

Discussions and potential collaborations are also welcome.



## 🙏 Acknowledgement
This repo is built upon the following projects: [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [AWQ](https://github.com/mit-han-lab/llm-awq), [QuaRot](https://github.com/spcl/QuaRot), [DuQuant](https://github.com/Hsu1023/DuQuant), and [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).


We thank the authors for their codes.



## 📝 Citation
Please cite our work if you use our code or discuss our findings in your own research:
```bibtex
@article{lin2025quantization,
  title={Quantization meets dllms: A systematic study of post-training quantization for diffusion llms},
  author={Lin, Haokun and Xu, Haobo and Wu, Yichen and Guo, Ziyu and Zhang, Renrui and Lu, Zhichao and Wei, Ying and Zhang, Qingfu and Sun, Zhenan},
  journal={arXiv preprint arXiv:2508.14896},
  year={2025}
}
```



## 🧠 Related Work
Explore our additional research on **Post-training Quantization** and **Network Pruning**:

- **[DuQuant]** [DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs](https://arxiv.org/abs/2406.01721)
- **[IntactKV]** [IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact](https://arxiv.org/abs/2403.01241)
- **[LRQ-DiT]** [LRQ-DiT: Log-Rotation Post-Training Quantization of Diffusion Transformers for Image and Video Generation](https://www.arxiv.org/abs/2508.03485)
- **[DopQ-ViT]** [DopQ-ViT: Towards Distribution-Friendly and Outlier-Aware Post-Training Quantization for Vision Transformers](https://arxiv.org/abs/2408.03291)
- **[RIA]** [Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models](https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning)
- **[MoPE-CLIP]** [MoPE-CLIP: Structured Pruning for Efficient Vision-Language Models with Module-wise Pruning Error Metric](https://arxiv.org/abs/2403.07839)